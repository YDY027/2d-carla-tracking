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
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 内置卡尔曼滤波与SORT跟踪器（优化版）
# ==============================================================================
class KalmanFilter:
    """卡尔曼滤波器：用于目标位置和速度的预测与更新（优化版）"""
    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.x = np.zeros(8, dtype=np.float32)
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
        self.H = np.eye(4, 8, dtype=np.float32)
        self.Q = np.diag([1, 1, 1, 1, 5, 5, 5, 5]).astype(np.float32)
        self.R = np.diag([5, 5, 5, 5]).astype(np.float32)
        self.P = np.eye(8, dtype=np.float32) * 50

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

class Track:
    """单目标跟踪实例（优化版）"""
    def __init__(self, track_id: int, bbox: np.ndarray, img_shape: Tuple[int, int]):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.img_shape = img_shape
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confidence = 0.0

    def _clip_bbox(self, bbox: np.ndarray) -> np.ndarray:
        height, width = self.img_shape
        x1 = max(0, min(bbox[0], width - 1))
        y1 = max(0, min(bbox[1], height - 1))
        x2 = max(x1 + 1, min(bbox[2], width - 1))
        y2 = max(y1 + 1, min(bbox[3], height - 1))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: np.ndarray, confidence: float = 0.0) -> None:
        bbox_clipped = self._clip_bbox(bbox)
        self.bbox = self.kf.update(bbox_clipped)
        self.bbox = self._clip_bbox(self.bbox)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence

class SimpleSORT:
    """简易SORT跟踪器（优化版）"""
    def __init__(self, 
                 max_age: int = 3, 
                 min_hits: int = 2, 
                 iou_threshold: float = 0.3,
                 img_shape: Tuple[int, int] = (480, 640)):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.img_shape = img_shape
        self.tracks: List[Track] = []
        self.next_id = 1

    def update(self, detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            self._add_new_tracks(detections)
            return np.array([]), np.array([])
        
        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
            return np.array([t.bbox for t in confirmed_tracks], dtype=np.int32), np.array([t.track_id for t in confirmed_tracks], dtype=np.int32)

        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        det_boxes = detections[:, :4].astype(np.float32)
        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)
        matches = self._greedy_iou_matching(iou_matrix)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx][:4], detections[det_idx][4])

        unmatched_det_idxs = set(range(len(detections))) - set(det_idx for _, det_idx in matches)
        for det_idx in unmatched_det_idxs:
            self.tracks.append(Track(self.next_id, detections[det_idx][:4], self.img_shape))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        return np.array([t.bbox for t in confirmed_tracks], dtype=np.int32), np.array([t.track_id for t in confirmed_tracks], dtype=np.int32)

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        n, m = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n, m), dtype=np.float32)
        if n == 0 or m == 0:
            return iou_matrix
        boxes1 = boxes1[:, np.newaxis, :]
        boxes2 = boxes2[np.newaxis, :, :]
        inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union_area = area1 + area2 - inter_area
        iou_matrix = np.where(union_area > 1e-6, inter_area / union_area, 0.0)
        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 1e-6 else 0.0

    def _greedy_iou_matching(self, iou_matrix: np.ndarray) -> List[Tuple[int, int]]:
        matches = []
        used_tracks = set()
        used_dets = set()
        n_dets, n_tracks = iou_matrix.shape
        valid_mask = iou_matrix > self.iou_threshold
        if not np.any(valid_mask):
            return matches
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
                if len(used_dets) == n_dets or len(used_tracks) == n_tracks:
                    break
        return matches

    def _add_new_tracks(self, detections: np.ndarray) -> None:
        for det in detections:
            self.tracks.append(Track(self.next_id, det[:4], self.img_shape))
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
        current_time = time.time()
        self.times.append(current_time)
        if len(self.times) > self.window_size:
            self.times.pop(0)
        if len(self.times) >= 2:
            delta_time = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / delta_time
        return self.fps

def draw_bounding_boxes(image: np.ndarray, 
                       boxes: np.ndarray, 
                       ids: np.ndarray,
                       fps: float = 0.0,
                       detection_count: int = 0) -> np.ndarray:
    """绘制边界框、ID、FPS统计"""
    display_img = image.copy()
    height, width = display_img.shape[:2]
    stats_text = f"FPS: {fps:.1f} | Tracks: {len(boxes)} | Detections: {detection_count}"
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(display_img, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(display_img, stats_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    for i, (box, track_id) in enumerate(zip(boxes, ids)):
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width - 1))
        y2 = max(y1 + 1, min(int(y2), height - 1))
        color = tuple((hash(track_id) % 256 for _ in range(3)))
        color = (color[1], color[2], color[0])
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        id_text = f"ID: {track_id}"
        id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(display_img, (x1, y1 - 20), (x1 + id_text_size[0], y1), color, -1, cv2.LINE_AA)
        cv2.putText(display_img, id_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return display_img

def clear_actors(world: carla.World, exclude: List[int] = None) -> None:
    """清理CARLA场景中的Actor"""
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
    """相机回调：转换图像格式并放入队列"""
    try:
        expected_shape = (image.height, image.width, 4)
        img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        if img_array.size != np.prod(expected_shape):
            logger.warning(f"图像数据异常，跳过该帧（期望尺寸：{expected_shape}，实际长度：{img_array.size}）")
            return
        img_array = np.reshape(img_array, expected_shape)
        img_bgr = img_array[:, :, :3]
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
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
    """生成NPC车辆（避免重叠）"""
    logger.info(f"正在生成 {num_npcs} 辆NPC车辆...")
    bp_lib = world.get_blueprint_library()
    npc_bps = [bp for bp in bp_lib.filter('vehicle') 
              if int(bp.get_attribute('number_of_wheels')) == 4 
              and not bp.id.endswith(('firetruck', 'ambulance', 'police'))]
    if not npc_bps:
        logger.warning("没有找到可用的车辆蓝图")
        return 0
    npc_count = 0
    used_spawn_points = set()
    for _ in range(num_npcs):
        if len(used_spawn_points) >= len(spawn_points):
            logger.warning("可用生成点已耗尽，停止生成NPC")
            break
        while True:
            spawn_point = random.choice(spawn_points)
            spawn_point_id = id(spawn_point)
            if spawn_point_id not in used_spawn_points:
                used_spawn_points.add(spawn_point_id)
                break
        npc_bp = random.choice(npc_bps)
        if npc_bp.has_attribute('color'):
            color = random.choice(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', color)
        npc = world.try_spawn_actor(npc_bp, spawn_point)
        if npc:
            npc.set_autopilot(True)
            npc_count += 1
            logger.debug(f"生成NPC车辆: {npc_bp.id} (ID: {npc.id})")
    logger.info(f"成功生成 {npc_count} 辆NPC车辆")
    return npc_count

# ==============================================================================
# 主函数（最终稳定版：无lerp依赖+YOLO兼容所有版本）
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="CARLA多目标跟踪（YOLOv5 + 优化版SORT）")
    parser.add_argument("--host", default="localhost", help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA服务器端口")
    parser.add_argument("--num_npcs", type=int, default=10, help="NPC车辆数量")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="YOLO检测置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.3, help="SORT IOU匹配阈值")
    parser.add_argument("--img-width", type=int, default=640, help="相机图像宽度")
    parser.add_argument("--img-height", type=int, default=480, help="相机图像高度")
    parser.add_argument("--max-age", type=int, default=5, help="SORT最大未更新帧数")
    parser.add_argument("--min-hits", type=int, default=3, help="SORT最小命中次数")
    args = parser.parse_args()

    # 连接CARLA服务器
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        world = client.get_world()
        logger.info(f"成功连接到CARLA服务器（{args.host}:{args.port}）")
    except Exception as e:
        logger.error(f"CARLA服务器连接失败: {e}")
        return

    # 配置同步模式
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        logger.info("已启用CARLA同步模式")
    except Exception as e:
        logger.error(f"配置仿真设置失败: {e}")
        return

    # 清理场景
    clear_actors(world)

    # 获取蓝图和生成点
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logger.error("地图没有可用的生成点，退出程序")
        return
    logger.info(f"找到 {len(spawn_points)} 个生成点")

    # 生成主车辆
    ego_vehicle = None
    try:
        ego_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        ego_bp.set_attribute('color', '255,0,0')
        ego_spawn_point = random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot(True)
        logger.info(f"主车辆生成成功（ID: {ego_vehicle.id}）")
    except Exception as e:
        logger.error(f"主车辆生成失败: {e}")
        clear_actors(world)
        return

    # 生成相机
    camera = None
    try:
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.img_width))
        camera_bp.set_attribute('image_size_y', str(args.img_height))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')
        camera_bp.set_attribute('shutter_speed', '10000')
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.0),
            carla.Rotation(pitch=-5.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        logger.info(f"相机生成成功（ID: {camera.id}）")
    except Exception as e:
        logger.error(f"相机生成失败: {e}")
        clear_actors(world, exclude=[ego_vehicle.id] if ego_vehicle else None)
        return

    # 初始化队列和FPS计数器
    image_queue = queue.Queue(maxsize=3)
    camera.listen(lambda img: camera_callback(img, image_queue))
    fps_counter = FPSCounter(window_size=15)

    # 生成NPC
    spawn_npc_vehicles(world, args.num_npcs, spawn_points)

    # 加载YOLO模型（兼容所有ultralytics版本，移除YOLO.list()依赖）
    try:
        logger.info("加载YOLOv5模型...")
        # 直接加载模型：本地有则用本地，没有则自动下载（ultralytics库自带自动下载逻辑）
        detector = YOLO("yolov5s.pt")
        logger.info("YOLOv5模型加载完成")
    except Exception as e:
        logger.error(f"YOLO模型加载失败: {e}")
        clear_actors(world, exclude=[ego_vehicle.id, camera.id] if ego_vehicle and camera else None)
        return

    tracker = SimpleSORT(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=0.25,
        img_shape=(args.img_height, args.img_width)
    )

    # 创建小窗口（800x480，置顶+居中）
    cv2.namedWindow("CARLA Vehicle Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("CARLA Vehicle Tracking", 800, 480)
    try:
        cv2.setWindowProperty("CARLA Vehicle Tracking", cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
    screen_center_x = (cv2.getWindowImageRect("CARLA Vehicle Tracking")[2] - 800) // 2
    screen_center_y = (cv2.getWindowImageRect("CARLA Vehicle Tracking")[3] - 480) // 2
    cv2.moveWindow("CARLA Vehicle Tracking", screen_center_x, screen_center_y)

    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    logger.info("开始跟踪... 按 'q' 或 'ESC' 退出程序 | 按 's' 保存截图")

    # 主循环
    try:
        while True:
            world.tick()

            # 视角跟随（手动平滑，无lerp依赖，100%稳定）
            try:
                if ego_vehicle and ego_vehicle.is_alive:
                    ego_loc = ego_vehicle.get_transform().location
                    ego_rot = ego_vehicle.get_transform().rotation
                    spectator = world.get_spectator()
                    current_loc = spectator.get_transform().location
                    current_rot = spectator.get_transform().rotation

                    # 目标视角参数（跟随主车辆，45度俯视）
                    target_loc = {
                        'x': ego_loc.x - 8.0,
                        'y': ego_loc.y,
                        'z': ego_loc.z + 12.0
                    }
                    target_rot = {
                        'pitch': -45.0,
                        'yaw': ego_rot.yaw,
                        'roll': 0.0
                    }

                    # 手动线性插值（平滑过渡，避免抖动）
                    smooth = 0.3
                    new_loc = carla.Location(
                        x=current_loc.x * (1 - smooth) + target_loc['x'] * smooth,
                        y=current_loc.y * (1 - smooth) + target_loc['y'] * smooth,
                        z=current_loc.z * (1 - smooth) + target_loc['z'] * smooth
                    )
                    new_rot = carla.Rotation(
                        pitch=current_rot.pitch * (1 - smooth) + target_rot['pitch'] * smooth,
                        yaw=current_rot.yaw * (1 - smooth) + target_rot['yaw'] * smooth,
                        roll=current_rot.roll * (1 - smooth) + target_rot['roll'] * smooth
                    )

                    spectator.set_transform(carla.Transform(new_loc, new_rot))
            except Exception as e:
                logger.warning(f"更新视角失败: {e}")
                continue

            # 处理图像和跟踪
            if not image_queue.empty():
                while image_queue.qsize() > 1:
                    try:
                        image_queue.get_nowait()
                    except queue.Empty:
                        break
                image = image_queue.get_nowait()
                current_fps = fps_counter.update()

                # YOLO检测（CPU优化）
                results = detector.predict(
                    image,
                    conf=args.conf_thres,
                    verbose=False,
                    device="cpu",
                    half=False,
                    agnostic_nms=True,
                    vid_stride=3,
                    imgsz=480,
                    iou=0.45
                )

                # 筛选车辆检测结果
                detections = []
                detection_count = 0
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                            cls_id = int(cls)
                            if cls_id in vehicle_classes:
                                det_box = box.cpu().numpy()
                                w, h = det_box[2] - det_box[0], det_box[3] - det_box[1]
                                if w > 30 and h > 30:
                                    detections.append([*det_box, conf.cpu().numpy()])
                                    detection_count += 1

                # SORT跟踪
                tracked_boxes, track_ids = tracker.update(np.array(detections)) if detections else (np.array([]), np.array([]))

                # 绘制显示
                display_img = draw_bounding_boxes(image, tracked_boxes, track_ids, current_fps, detection_count)
                cv2.imshow("CARLA Vehicle Tracking", display_img)

            # 键盘事件
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                logger.info("用户请求退出程序")
                break
            elif key == ord('s') and 'display_img' in locals():
                filename = f"tracking_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, display_img)
                logger.info(f"截图已保存: {filename}")

    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"主循环错误: {e}", exc_info=True)
    finally:
        # 资源清理
        logger.info("开始清理资源...")
        if camera:
            camera.stop()
        exclude_ids = [ego_vehicle.id, camera.id] if ego_vehicle and camera else []
        clear_actors(world, exclude=exclude_ids)
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception as e:
            logger.error(f"关闭同步模式失败: {e}")
        cv2.destroyAllWindows()
        logger.info("程序正常退出")

if __name__ == "__main__":
    main()