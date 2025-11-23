import argparse
import carla
import queue
import random
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 内置卡尔曼滤波与SORT跟踪器（优化版）
# ==============================================================================
class KalmanFilter:
    """卡尔曼滤波器：用于目标位置和速度的预测与更新（优化版）"""
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
        # 过程噪声协方差矩阵（优化权重，适应车辆运动特性）
        self.Q = np.diag([1, 1, 1, 1, 5, 5, 5, 5]).astype(np.float32)
        # 观测噪声协方差矩阵（根据实际检测精度调整）
        self.R = np.diag([5, 5, 5, 5]).astype(np.float32)
        # 状态协方差矩阵（初始值优化）
        self.P = np.eye(8, dtype=np.float32) * 50

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
        
        # 数值稳定性处理
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = self.P @ self.H.T @ S_inv  # 卡尔曼增益
        self.x = self.x + K @ y  # 状态更新
        self.P = (np.eye(8) - K @ self.H) @ self.P  # 协方差更新
        return self.x[:4]

class Track:
    """单目标跟踪实例：管理单个目标的生命周期和状态（优化版）"""
    def __init__(self, track_id: int, bbox: np.ndarray, img_shape: Tuple[int, int]):
        """
        初始化跟踪实例
        Args:
            track_id: 唯一跟踪ID
            bbox: 初始边界框 [x1, y1, x2, y2]
            img_shape: 图像尺寸 (height, width)
        """
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.img_shape = img_shape
        # 边界框坐标裁剪，确保在图像范围内
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox  # 初始化卡尔曼滤波器状态
        self.hits = 1  # 连续检测命中次数
        self.age = 0  # 跟踪总帧数
        self.time_since_update = 0  # 距离上次更新的帧数
        self.confidence = 0.0  # 最新检测置信度

    def _clip_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """裁剪边界框坐标，确保在图像范围内"""
        height, width = self.img_shape
        x1 = max(0, min(bbox[0], width - 1))
        y1 = max(0, min(bbox[1], height - 1))
        x2 = max(x1 + 1, min(bbox[2], width - 1))
        y2 = max(y1 + 1, min(bbox[3], height - 1))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """预测边界框位置（包含裁剪）"""
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray, confidence: float = 0.0) -> None:
        """根据新检测结果更新边界框"""
        bbox_clipped = self._clip_bbox(bbox)
        self.bbox = self.kf.update(bbox_clipped)
        self.bbox = self._clip_bbox(self.bbox)  # 再次裁剪确保安全
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence

class SimpleSORT:
    """简易SORT跟踪器：基于IOU匹配的多目标跟踪（优化版）"""
    def __init__(self, 
                 max_age: int = 3, 
                 min_hits: int = 2, 
                 iou_threshold: float = 0.3,
                 img_shape: Tuple[int, int] = (480, 640)):
        """
        初始化SORT跟踪器
        Args:
            max_age: 目标最大未更新帧数（超过则删除）
            min_hits: 确认跟踪目标所需的最小连续命中次数
            iou_threshold: IOU匹配阈值
            img_shape: 图像尺寸 (height, width)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.img_shape = img_shape
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
        if len(self.tracks) == 0:
            # 无跟踪目标，直接添加所有检测
            self._add_new_tracks(detections)
            return np.array([]), np.array([])
        
        if len(detections) == 0:
            # 无检测结果，只保留未过期的跟踪
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
            tracked_boxes = np.array([t.bbox for t in confirmed_tracks], dtype=np.int32)
            track_ids = np.array([t.track_id for t in confirmed_tracks], dtype=np.int32)
            return tracked_boxes, track_ids

        # 步骤3：计算IOU矩阵
        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = detections[:, :4].astype(np.float32)
        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)

        # 步骤4：IOU匹配（贪心算法）
        matches = self._greedy_iou_matching(iou_matrix)

        # 步骤5：更新匹配到的跟踪目标
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx][:4],
                detections[det_idx][4]
            )

        # 步骤6：添加新检测到的目标（未匹配到现有跟踪）
        unmatched_det_idxs = set(range(len(detections))) - set(det_idx for _, det_idx in matches)
        for det_idx in unmatched_det_idxs:
            self.tracks.append(Track(
                self.next_id,
                detections[det_idx][:4],
                self.img_shape
            ))
            self.next_id += 1

        # 步骤7：删除长时间未更新的跟踪目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 步骤8：筛选出确认的跟踪目标（满足最小命中次数）
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        tracked_boxes = np.array([t.bbox for t in confirmed_tracks], dtype=np.int32)
        track_ids = np.array([t.track_id for t in confirmed_tracks], dtype=np.int32)

        return tracked_boxes, track_ids

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算两个边界框集合的IOU矩阵（向量化优化）"""
        n = len(boxes1)
        m = len(boxes2)
        iou_matrix = np.zeros((n, m), dtype=np.float32)
        
        if n == 0 or m == 0:
            return iou_matrix
        
        # 向量化计算IOU，提升效率
        boxes1 = boxes1[:, np.newaxis, :]  # [n, 1, 4]
        boxes2 = boxes2[np.newaxis, :, :]  # [1, m, 4]
        
        # 计算交集
        inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        
        # 计算面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # 计算IOU
        union_area = area1 + area2 - inter_area
        iou_matrix = np.where(union_area > 1e-6, inter_area / union_area, 0.0)
        
        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算单个IOU值（备用）"""
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
        """贪心IOU匹配算法（优化版）"""
        matches = []
        used_tracks = set()
        used_dets = set()

        n_dets, n_tracks = iou_matrix.shape
        
        # 只考虑IOU大于阈值的匹配对
        valid_mask = iou_matrix > self.iou_threshold
        if not np.any(valid_mask):
            return matches
        
        # 按IOU值降序排序所有有效匹配对
        valid_indices = np.where(valid_mask)
        valid_iou_values = iou_matrix[valid_mask]
        sorted_indices = np.argsort(valid_iou_values)[::-1]
        
        for idx in sorted_indices:
            det_idx = valid_indices[0][idx]
            track_idx = valid_indices[1][idx]
            
            if det_idx not in used_dets and track_idx not in used_tracks:
                matches.append((track_idx, det_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)
                
                # 所有检测和跟踪都已匹配，提前退出
                if len(used_dets) == n_dets or len(used_tracks) == n_tracks:
                    break

        return matches

    def _add_new_tracks(self, detections: np.ndarray) -> None:
        """添加新的跟踪目标"""
        for det in detections:
            self.tracks.append(Track(
                self.next_id,
                det[:4],
                self.img_shape
            ))
            self.next_id += 1

# ==============================================================================
# 工具函数（增强版）
# ==============================================================================
class FPSCounter:
    """FPS计数器"""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.times = []
        self.fps = 0.0

    def update(self) -> float:
        """更新FPS计数"""
        current_time = time.time()
        self.times.append(current_time)
        
        # 保持窗口大小
        if len(self.times) > self.window_size:
            self.times.pop(0)
        
        # 计算FPS
        if len(self.times) >= 2:
            delta_time = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / delta_time
        
        return self.fps

def draw_bounding_boxes(image: np.ndarray, 
                       boxes: np.ndarray, 
                       ids: np.ndarray,
                       fps: float = 0.0,
                       detection_count: int = 0) -> np.ndarray:
    """
    在图像上绘制边界框、跟踪ID、FPS和检测计数（增强版）
    Args:
        image: 输入图像（BGR格式）
        boxes: 边界框数组 [N, 4] = [x1, y1, x2, y2]
        ids: 跟踪ID数组 [N,]
        fps: 当前FPS
        detection_count: 检测到的目标数量
    Returns:
        绘制后的图像
    """
    display_img = image.copy()
    height, width = display_img.shape[:2]
    
    # 绘制统计信息背景框
    stats_text = f"FPS: {fps:.1f} | Tracks: {len(boxes)} | Detections: {detection_count}"
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(
        display_img, (10, 10), 
        (10 + text_size[0] + 10, 10 + text_size[1] + 10), 
        (0, 0, 0), -1, cv2.LINE_AA
    )
    cv2.putText(
        display_img, stats_text, (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
    )
    
    # 绘制边界框和ID
    for i, (box, track_id) in enumerate(zip(boxes, ids)):
        x1, y1, x2, y2 = box
        # 确保坐标在图像范围内（双重保险）
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width - 1))
        y2 = max(y1 + 1, min(int(y2), height - 1))
        
        # 为不同ID分配不同颜色（基于ID的哈希值）
        color = tuple((hash(track_id) % 256 for _ in range(3)))
        color = (color[1], color[2], color[0])  # BGR格式
        
        # 绘制边界框
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # 绘制ID背景框
        id_text = f"ID: {track_id}"
        id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            display_img, (x1, y1 - 20), 
            (x1 + id_text_size[0], y1), 
            color, -1, cv2.LINE_AA
        )
        
        # 绘制跟踪ID
        cv2.putText(
            display_img, id_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
    
    return display_img

def clear_actors(world: carla.World, exclude: List[int] = None) -> None:
    """
    清理CARLA世界中的所有车辆和传感器（增强版）
    Args:
        world: CARLA世界对象
        exclude: 要排除的Actor ID列表
    """
    logger.info("清理场景中的Actor...")
    exclude_ids = set(exclude) if exclude else set()
    
    for actor in world.get_actors():
        if actor.type_id.startswith(('vehicle.', 'sensor.')) and actor.id not in exclude_ids:
            try:
                if actor.is_alive:
                    actor.destroy()
                    logger.debug(f"销毁Actor: {actor.type_id} (ID: {actor.id})")
            except Exception as e:
                logger.warning(f"销毁Actor失败 (ID: {actor.id}): {e}")
    
    logger.info("Actor清理完成")

def camera_callback(image: carla.Image, image_queue: queue.Queue) -> None:
    """
    相机数据回调函数：将CARLA图像转换为OpenCV格式并放入队列（增强版）
    Args:
        image: CARLA图像对象
        image_queue: 图像存储队列
    """
    try:
        # 将CARLA原始数据转换为BGR图像（去除Alpha通道）
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img_array = np.reshape(img_array, (image.height, image.width, 4))
        img_bgr = img_array[:, :, :3]  # BGR格式（OpenCV默认）
        
        # 图像预处理：轻微高斯模糊，减少噪声
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        
        # 队列满时丢弃最旧的图像
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        
        image_queue.put(img_bgr)
        
    except Exception as e:
        logger.error(f"相机回调函数错误: {e}")

def spawn_npc_vehicles(world: carla.World, 
                      num_npcs: int, 
                      spawn_points: List[carla.Transform]) -> int:
    """
    生成NPC车辆（优化版，避免重叠）
    Args:
        world: CARLA世界对象
        num_npcs: 要生成的NPC数量
        spawn_points: 生成点列表
    Returns:
        成功生成的NPC数量
    """
    logger.info(f"正在生成 {num_npcs} 辆NPC车辆...")
    bp_lib = world.get_blueprint_library()
    
    # 筛选4轮车辆蓝图（排除特殊车辆）
    npc_bps = [bp for bp in bp_lib.filter('vehicle') 
              if int(bp.get_attribute('number_of_wheels')) == 4 
              and not bp.id.endswith(('firetruck', 'ambulance', 'police'))]
    
    if not npc_bps:
        logger.warning("没有找到可用的车辆蓝图")
        return 0
    
    npc_count = 0
    used_spawn_points = set()
    
    # 尝试生成NPC，避免使用相同的生成点
    for _ in range(num_npcs):
        if len(used_spawn_points) >= len(spawn_points):
            logger.warning("可用生成点已耗尽，停止生成NPC")
            break
        
        # 随机选择未使用的生成点
        while True:
            spawn_point = random.choice(spawn_points)
            spawn_point_id = id(spawn_point)  # 使用内存地址作为唯一标识
            if spawn_point_id not in used_spawn_points:
                used_spawn_points.add(spawn_point_id)
                break
        
        # 随机选择车辆蓝图
        npc_bp = random.choice(npc_bps)
        
        # 随机调整车辆颜色
        if npc_bp.has_attribute('color'):
            color = random.choice(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', color)
        
        # 尝试生成车辆
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True)
            npc_count += 1
            logger.debug(f"生成NPC车辆: {npc_bp.id} (ID: {npc.id})")
    
    logger.info(f"成功生成 {npc_count} 辆NPC车辆")
    return npc_count

# ==============================================================================
# 主函数（优化版，CPU兼容）
# ==============================================================================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CARLA多目标跟踪（YOLOv5 + 优化版SORT）")
    parser.add_argument("--host", default="localhost", help="CARLA服务器地址（默认：localhost）")
    parser.add_argument("--port", type=int, default=2000, help="CARLA服务器端口（默认：2000）")
    parser.add_argument("--num_npcs", type=int, default=10, help="NPC车辆数量（CPU推荐：10，默认：10）")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="YOLO检测置信度阈值（CPU推荐：0.5，默认：0.5）")
    parser.add_argument("--iou-thres", type=float, default=0.3, help="SORT IOU匹配阈值（默认：0.3）")
    parser.add_argument("--img-width", type=int, default=640, help="相机图像宽度（默认：640）")
    parser.add_argument("--img-height", type=int, default=480, help="相机图像高度（默认：480）")
    parser.add_argument("--max-age", type=int, default=3, help="SORT最大未更新帧数（默认：3）")
    parser.add_argument("--min-hits", type=int, default=2, help="SORT最小命中次数（默认：2）")
    args = parser.parse_args()

    # 初始化CARLA客户端
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)  # 延长超时时间
        world = client.get_world()
        logger.info(f"成功连接到CARLA服务器（{args.host}:{args.port}）")
    except Exception as e:
        logger.error(f"CARLA服务器连接失败: {e}")
        return

    # 配置CARLA仿真设置（同步模式）
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 与卡尔曼滤波器时间步长一致
        world.apply_settings(settings)
        logger.info("已启用CARLA同步模式")
    except Exception as e:
        logger.error(f"配置仿真设置失败: {e}")
        return

    # 清理现有Actor
    clear_actors(world)

    # 获取蓝图库和生成点
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("地图没有可用的生成点，退出程序")
        return
    logger.info(f"找到 {len(spawn_points)} 个生成点")

    # 生成主车辆（Ego Vehicle）
    ego_vehicle = None
    try:
        ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        # 为主车辆设置特殊颜色
        ego_bp.set_attribute('color', '255,0,0')  # 红色
        ego_spawn_point = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        logger.info(f"主车辆生成成功（ID: {ego_vehicle.id}）")
    except Exception as e:
        logger.error(f"主车辆生成失败: {e}")
        clear_actors(world)
        return

    # 生成RGB相机（挂载到主车辆）
    camera = None
    try:
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # 相机参数配置
        camera_bp.set_attribute('image_size_x', str(args.img_width))
        camera_bp.set_attribute('image_size_y', str(args.img_height))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')  # 与仿真步长一致
        camera_bp.set_attribute('shutter_speed', '10000')  # 减少运动模糊
        # 相机安装位置（车辆前部上方）
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.0),
            carla.Rotation(pitch=-5.0)  # 轻微下倾，扩大视野
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        logger.info(f"相机生成成功（ID: {camera.id}）")
    except Exception as e:
        logger.error(f"相机生成失败: {e}")
        clear_actors(world, exclude=[ego_vehicle.id] if ego_vehicle else None)
        return

    # 初始化图像队列、FPS计数器
    image_queue = queue.Queue(maxsize=10)  # 增大队列容量，避免丢帧
    camera.listen(lambda img: camera_callback(img, image_queue))
    fps_counter = FPSCounter(window_size=15)

    # 生成NPC车辆
    spawn_npc_vehicles(world, args.num_npcs, spawn_points)

    # 初始化YOLO检测器和SORT跟踪器
    try:
        logger.info("加载YOLOv5模型...")
        # 优先尝试加载本地模型，如果没有则自动下载
        try:
            detector = YOLO("yolov5s.pt")
        except Exception:
            detector = YOLO("ultralytics/yolov5s")
        logger.info("YOLOv5模型加载完成")
    except Exception as e:
        logger.error(f"YOLO模型加载失败: {e}")
        clear_actors(world, exclude=[ego_vehicle.id, camera.id] if ego_vehicle and camera else None)
        return

    # 初始化SORT跟踪器
    tracker = SimpleSORT(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_thres,
        img_shape=(args.img_height, args.img_width)
    )

    # 创建显示窗口（强制置顶+居中，解决找不到窗口问题）
    cv2.namedWindow("CARLA Vehicle Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    # 放大窗口（适配屏幕）
    cv2.resizeWindow("CARLA Vehicle Tracking", 1920, 1080)
    # 强制窗口置顶（Windows系统有效）
    try:
        cv2.setWindowProperty("CARLA Vehicle Tracking", cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
    # 移动窗口到屏幕中央
    screen_center_x = (cv2.getWindowImageRect("CARLA Vehicle Tracking")[2] - 1920) // 2
    screen_center_y = (cv2.getWindowImageRect("CARLA Vehicle Tracking")[3] - 1080) // 2
    cv2.moveWindow("CARLA Vehicle Tracking", screen_center_x, screen_center_y)

    # 车辆类别映射（COCO数据集）
    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    logger.info("开始跟踪... 按 'q' 或 'ESC' 退出程序 | 按 's' 保存截图")

    # 主仿真循环
    try:
        while True:
            # 推进仿真（同步模式必须调用tick()）
            world.tick()

            # 更新 spectator 视角（跟随主车辆）
            try:
                if ego_vehicle and ego_vehicle.is_alive:
                    ego_transform = ego_vehicle.get_transform()
                    spectator_transform = carla.Transform(
                        ego_transform.location + carla.Location(x=-8.0, z=12.0),
                        carla.Rotation(pitch=-45.0, yaw=ego_transform.rotation.yaw)
                    )
                    world.get_spectator().set_transform(spectator_transform)
            except Exception as e:
                logger.warning(f"更新视角失败: {e}")
                continue

            # 处理相机图像
            if not image_queue.empty():
                # 非阻塞获取最新图像（丢弃旧帧，只保留最新）
                while image_queue.qsize() > 1:
                    try:
                        image_queue.get_nowait()
                    except queue.Empty:
                        break
                
                image = image_queue.get_nowait()
                current_fps = fps_counter.update()

                # 步骤1：YOLO目标检测（CPU兼容版）
                results = detector.predict(
                    image,
                    conf=args.conf_thres,
                    verbose=False,  # 禁用详细输出
                    device="cpu",  # 强制CPU运行（关键修复）
                    half=False,    # CPU不支持半精度（关键修复）
                    agnostic_nms=True,  # 类别无关NMS
                    vid_stride=2  # 跳帧推理，提升CPU性能
                )

                # 步骤2：筛选车辆类检测结果
                detections = []
                detection_count = 0
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                            cls_id = int(cls)
                            if cls_id in vehicle_classes:
                                det_box = box.cpu().numpy()
                                # 过滤过小的边界框（减少误检测，减轻CPU负担）
                                w = det_box[2] - det_box[0]
                                h = det_box[3] - det_box[1]
                                if w > 30 and h > 30:  # CPU模式下增大最小尺寸，提升性能
                                    detections.append([*det_box, conf.cpu().numpy()])
                                    detection_count += 1

                # 步骤3：多目标跟踪
                tracked_boxes, track_ids = np.array([]), np.array([])
                if detections:
                    tracked_boxes, track_ids = tracker.update(np.array(detections))

                # 步骤4：绘制并显示结果
                display_img = draw_bounding_boxes(
                    image,
                    tracked_boxes,
                    track_ids,
                    fps=current_fps,
                    detection_count=detection_count
                )
                cv2.imshow("CARLA Vehicle Tracking", display_img)

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # q/ESC退出
                logger.info("用户请求退出程序")
                break
            elif key == ord('s'):  # s键保存当前帧
                if 'display_img' in locals():
                    filename = f"tracking_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, display_img)
                    logger.info(f"截图已保存: {filename}")

    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"主循环错误: {e}", exc_info=True)
    finally:
        # 资源清理（关键步骤，避免CARLA服务器残留）
        logger.info("开始清理资源...")
        
        # 停止相机
        if camera:
            camera.stop()
            logger.info("相机已停止")
        
        # 清理Actor
        exclude_ids = []
        if ego_vehicle:
            exclude_ids.append(ego_vehicle.id)
        if camera:
            exclude_ids.append(camera.id)
        clear_actors(world, exclude=exclude_ids)
        
        # 关闭同步模式
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            logger.info("已关闭CARLA同步模式")
        except Exception as e:
            logger.error(f"关闭同步模式失败: {e}")
        
        # 关闭窗口
        cv2.destroyAllWindows()
        logger.info("程序正常退出")

if __name__ == "__main__":
    main()