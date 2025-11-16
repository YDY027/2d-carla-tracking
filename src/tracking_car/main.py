import argparse
import carla
import queue
import random
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional

# ==============================================================================
# 内置卡尔曼滤波与SORT跟踪器（无第三方依赖）
# ==============================================================================
class KalmanFilter:
    """卡尔曼滤波器：用于目标位置和速度的预测与更新"""
    def __init__(self, dt: float = 0.05):
        """
        初始化卡尔曼滤波器
        Args:
            dt: 时间步长（与CARLA仿真步长保持一致）
        """
        self.dt = dt
        # 状态向量：[x1, y1, x2, y2, vx, vy, vw, vh]（边界框坐标 + 速度）
        self.x = np.zeros(8, dtype=np.float32)
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],
            [0, 1, 0, 0, 0, self.dt, 0, 0],
            [0, 0, 1, 0, 0, 0, self.dt, 0],
            [0, 0, 0, 1, 0, 0, 0, self.dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        # 观测矩阵（仅观测边界框坐标）
        self.H = np.eye(4, 8, dtype=np.float32)
        # 过程噪声协方差矩阵（速度项权重更高）
        self.Q = np.diag([1, 1, 1, 1, 10, 10, 10, 10]).astype(np.float32)
        # 观测噪声协方差矩阵
        self.R = np.diag([10, 10, 10, 10]).astype(np.float32)
        # 状态协方差矩阵
        self.P = np.eye(8, dtype=np.float32) * 100

    def predict(self) -> np.ndarray:
        """预测下一时刻状态（仅返回边界框坐标）"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        根据观测值更新状态
        Args:
            z: 观测到的边界框 [x1, y1, x2, y2]
        Returns:
            更新后的边界框坐标
        """
        z = z.astype(np.float32)
        y = z - self.H @ self.x  # 残差
        S = self.H @ self.P @ self.H.T + self.R  # 残差协方差
        K = self.P @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益
        self.x = self.x + K @ y  # 状态更新
        self.P = (np.eye(8) - K @ self.H) @ self.P  # 协方差更新
        return self.x[:4]

class Track:
    """单目标跟踪实例：管理单个目标的生命周期和状态"""
    def __init__(self, track_id: int, bbox: np.ndarray):
        """
        初始化跟踪实例
        Args:
            track_id: 唯一跟踪ID
            bbox: 初始边界框 [x1, y1, x2, y2]
        """
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.bbox = bbox.astype(np.float32)
        self.kf.x[:4] = self.bbox  # 初始化卡尔曼滤波器状态
        self.hits = 1  # 连续检测命中次数
        self.age = 0  # 跟踪总帧数
        self.time_since_update = 0  # 距离上次更新的帧数

    def predict(self) -> np.ndarray:
        """预测边界框位置"""
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray) -> None:
        """根据新检测结果更新边界框"""
        self.bbox = self.kf.update(bbox)
        self.hits += 1
        self.time_since_update = 0

class SimpleSORT:
    """简易SORT跟踪器：基于IOU匹配的多目标跟踪"""
    def __init__(self, max_age: int = 2, min_hits: int = 2, iou_threshold: float = 0.3):
        """
        初始化SORT跟踪器
        Args:
            max_age: 目标最大未更新帧数（超过则删除）
            min_hits: 确认跟踪目标所需的最小连续命中次数
            iou_threshold: IOU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []  # 活跃跟踪列表
        self.next_id = 1  # 下一个可用跟踪ID

    def update(self, detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新跟踪器状态
        Args:
            detections: 检测结果数组，格式为 [N, 5] = [x1, y1, x2, y2, score]
        Returns:
            tracked_boxes: 跟踪到的边界框 [M, 4]
            track_ids: 对应的跟踪ID [M,]
        """
        # 步骤1：预测所有活跃跟踪目标的位置
        for track in self.tracks:
            track.predict()

        # 步骤2：计算检测框与跟踪框的IOU矩阵
        if len(self.tracks) == 0 or len(detections) == 0:
            # 无跟踪目标或无检测结果
            self._add_new_tracks(detections)
            return np.array([]), np.array([])

        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = detections[:, :4].astype(np.float32)
        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)

        # 步骤3：IOU匹配（贪心算法）
        matches = self._greedy_iou_matching(iou_matrix)

        # 步骤4：更新匹配到的跟踪目标
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx][:4])

        # 步骤5：添加新检测到的目标（未匹配到现有跟踪）
        unmatched_det_idxs = set(range(len(detections))) - set(det_idx for _, det_idx in matches)
        for det_idx in unmatched_det_idxs:
            self.tracks.append(Track(self.next_id, detections[det_idx][:4]))
            self.next_id += 1

        # 步骤6：删除长时间未更新的跟踪目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 步骤7：筛选出确认的跟踪目标（满足最小命中次数）
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        tracked_boxes = np.array([t.bbox for t in confirmed_tracks], dtype=np.int32)
        track_ids = np.array([t.track_id for t in confirmed_tracks], dtype=np.int32)

        return tracked_boxes, track_ids

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算两个边界框集合的IOU矩阵"""
        n = len(boxes1)
        m = len(boxes2)
        iou_matrix = np.zeros((n, m), dtype=np.float32)
        
        for i in range(n):
            for j in range(m):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])
        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算单个IOU值"""
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2

        # 计算交集区域
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        
        # 避免除零
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 1e-6 else 0.0

    def _greedy_iou_matching(self, iou_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """贪心IOU匹配算法"""
        matches = []
        used_tracks = set()
        used_dets = set()

        # 按IOU值降序排序所有可能的匹配对
        n_dets, n_tracks = iou_matrix.shape
        flat_indices = np.argsort(iou_matrix.flatten())[::-1]
        
        for idx in flat_indices:
            det_idx = idx // n_tracks
            track_idx = idx % n_tracks
            
            if (det_idx not in used_dets and 
                track_idx not in used_tracks and 
                iou_matrix[det_idx, track_idx] > self.iou_threshold):
                matches.append((track_idx, det_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)
                
                # 所有检测和跟踪都已匹配，提前退出
                if len(used_dets) == n_dets or len(used_tracks) == n_tracks:
                    break

        return matches

# ==============================================================================
# 工具函数（独立实现，无需外部依赖）
# ==============================================================================
def draw_bounding_boxes(image: np.ndarray, boxes: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """
    在图像上绘制边界框和跟踪ID
    Args:
        image: 输入图像（BGR格式）
        boxes: 边界框数组 [N, 4] = [x1, y1, x2, y2]
        ids: 跟踪ID数组 [N,]
    Returns:
        绘制后的图像
    """
    display_img = image.copy()
    for i, (box, track_id) in enumerate(zip(boxes, ids)):
        x1, y1, x2, y2 = box
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, image.shape[1]-1))
        y1 = max(0, min(y1, image.shape[0]-1))
        x2 = max(0, min(x2, image.shape[1]-1))
        y2 = max(0, min(y2, image.shape[0]-1))
        
        # 绘制边界框（绿色，线宽2）
        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 绘制ID背景框
        id_text = f"ID: {track_id}"
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            display_img, (int(x1), int(y1)-20), 
            (int(x1) + text_size[0], int(y1)), 
            (0, 255, 0), -1
        )
        # 绘制跟踪ID
        cv2.putText(
            display_img, id_text, (int(x1), int(y1)-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
    return display_img

def clear_actors(world: carla.World) -> None:
    """
    清理CARLA世界中的所有车辆和传感器
    Args:
        world: CARLA世界对象
    """
    print("清理场景中的Actor...")
    for actor in world.get_actors():
        if actor.type_id.startswith(('vehicle.', 'sensor.')):
            try:
                actor.destroy()
            except Exception as e:
                print(f"销毁Actor失败: {e}")
    print("Actor清理完成")

def camera_callback(image: carla.Image, image_queue: queue.Queue) -> None:
    """
    相机数据回调函数：将CARLA图像转换为OpenCV格式并放入队列
    Args:
        image: CARLA图像对象
        image_queue: 图像存储队列
    """
    try:
        # 将CARLA原始数据转换为BGR图像（去除Alpha通道）
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (image.height, image.width, 4))
        img_bgr = img_array[:, :, :3]  # BGR格式（OpenCV默认）
        image_queue.put(img_bgr)
    except Exception as e:
        print(f"相机回调函数错误: {e}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CARLA多目标跟踪（YOLOv5 + 内置SORT）")
    parser.add_argument("--host", default="localhost", help="CARLA服务器地址（默认：localhost）")
    parser.add_argument("--port", type=int, default=2000, help="CARLA服务器端口（默认：2000）")
    parser.add_argument("--num_npcs", type=int, default=30, help="NPC车辆数量（默认：30）")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="YOLO检测置信度阈值（默认：0.4）")
    parser.add_argument("--iou-thres", type=float, default=0.3, help="SORT IOU匹配阈值（默认：0.3）")  # 修复：添加等号
    args = parser.parse_args()

    # 初始化CARLA客户端
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)  # 延长超时时间
        world = client.get_world()
        print(f"成功连接到CARLA服务器（{args.host}:{args.port}）")
    except Exception as e:
        print(f"CARLA服务器连接失败: {e}")
        return

    # 配置CARLA仿真设置（同步模式）
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 与卡尔曼滤波器时间步长一致
        world.apply_settings(settings)
        print("已启用CARLA同步模式")
    except Exception as e:
        print(f"配置仿真设置失败: {e}")
        return

    # 清理现有Actor
    clear_actors(world)

    # 获取蓝图库和生成点
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("地图没有可用的生成点，退出程序")
        return

    # 生成主车辆（Ego Vehicle）
    try:
        ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        ego_spawn_point = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        print(f"主车辆生成成功（ID: {ego_vehicle.id}）")
    except Exception as e:
        print(f"主车辆生成失败: {e}")
        clear_actors(world)
        return

    # 生成RGB相机（挂载到主车辆）
    try:
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # 相机参数配置
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')  # 与仿真步长一致
        # 相机安装位置（车辆前部上方）
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.0),
            carla.Rotation(pitch=-5.0)  # 轻微下倾，扩大视野
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        print(f"相机生成成功（ID: {camera.id}）")
    except Exception as e:
        print(f"相机生成失败: {e}")
        clear_actors(world)
        return

    # 初始化图像队列和YOLO检测器
    image_queue = queue.Queue(maxsize=5)  # 限制队列大小，避免内存溢出
    camera.listen(lambda img: camera_callback(img, image_queue))

    # 生成NPC车辆
    print(f"正在生成 {args.num_npcs} 辆NPC车辆...")
    npc_count = 0
    for _ in range(args.num_npcs):
        # 筛选4轮车辆蓝图
        npc_bps = [bp for bp in bp_lib.filter('vehicle') if int(bp.get_attribute('number_of_wheels')) == 4]
        if not npc_bps:
            print("没有找到可用的车辆蓝图")
            break
        
        npc_bp = random.choice(npc_bps)
        # 随机选择生成点（避免重叠）
        spawn_point = random.choice(spawn_points)
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True)
            npc_count += 1
    print(f"成功生成 {npc_count} 辆NPC车辆")

    # 初始化YOLO检测器和SORT跟踪器
    try:
        print("加载YOLOv5模型...")
        detector = YOLO("yolov5s.pt")  # 轻量版模型，适合实时跟踪
        print("YOLOv5模型加载完成")
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        clear_actors(world)
        return

    tracker = SimpleSORT(
        max_age=3,  # 允许3帧未更新（更稳健）
        min_hits=2,
        iou_threshold=args.iou_thres
    )

    # 创建显示窗口
    cv2.namedWindow("CARLA Vehicle Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CARLA Vehicle Tracking", 1280, 720)

    # 主仿真循环
    print("开始跟踪... 按 'q' 或 'ESC' 退出程序")
    try:
        while True:
            # 推进仿真（同步模式必须调用tick()）
            world.tick()

            # 更新 spectator 视角（跟随主车辆）
            try:
                ego_transform = ego_vehicle.get_transform()
                spectator_transform = carla.Transform(
                    ego_transform.location + carla.Location(x=-8.0, z=12.0),
                    carla.Rotation(pitch=-45.0, yaw=ego_transform.rotation.yaw)
                )
                world.get_spectator().set_transform(spectator_transform)
            except Exception as e:
                print(f"更新视角失败: {e}")
                continue

            # 处理相机图像
            if not image_queue.empty():
                image = image_queue.get_nowait()  # 非阻塞获取图像

                # 步骤1：YOLO目标检测（仅保留车辆类）
                results = detector.predict(
                    image,
                    conf=args.conf_thres,
                    verbose=False,  # 禁用详细输出
                    device="auto"  # 自动选择GPU/CPU
                )

                # 步骤2：筛选车辆类检测结果（COCO数据集）
                # 车辆类：2=car, 5=bus, 7=truck
                vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
                detections = []
                for result in results:
                    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                        cls_id = int(cls)
                        if cls_id in vehicle_classes:
                            detections.append([*box.cpu().numpy(), conf.cpu().numpy()])

                # 步骤3：多目标跟踪
                tracked_boxes, track_ids = np.array([]), np.array([])
                if detections:
                    tracked_boxes, track_ids = tracker.update(np.array(detections))

                # 步骤4：绘制并显示结果
                display_img = draw_bounding_boxes(image, tracked_boxes, track_ids)
                cv2.imshow("CARLA Vehicle Tracking", display_img)

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # q/ESC退出
                print("用户请求退出程序")
                break

    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"主循环错误: {e}")
    finally:
        # 资源清理（关键步骤，避免CARLA服务器残留）
        print("开始清理资源...")
        camera.stop()
        clear_actors(world)
        # 关闭同步模式
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # 关闭窗口
        cv2.destroyAllWindows()
        print("程序正常退出")

if __name__ == "__main__":
    main()