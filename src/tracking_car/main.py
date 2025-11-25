import argparse
import carla
import queue
import random
import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
import os
import yaml
from dataclasses import dataclass, field

# 简化日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置类（精简）
# ==============================================================================
@dataclass
class Config:
    host: str = "localhost"
    port: int = 2000
    num_npcs: int = 10
    img_width: int = 640
    img_height: int = 480
    conf_thres: float = 0.5
    iou_thres: float = 0.3
    max_age: int = 5
    min_hits: int = 3
    yolo_imgsz_max: int = 480  # 确保是32的整数倍
    yolo_iou: float = 0.45
    kf_dt: float = 0.05
    max_speed: float = 50.0
    window_width: int = 800
    window_height: int = 480
    smooth_alpha: float = 0.2
    fps_window_size: int = 15

    @classmethod
    def from_yaml(cls, yaml_path: str = None) -> "Config":
        try:
            # 自动定位config.yaml（main.py同级目录）
            yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if not os.path.exists(yaml_path):
                logger.warning("配置文件不存在，使用默认配置")
                return cls()

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f.read().strip().replace("\t", "  "))
                if not isinstance(data, dict):
                    logger.warning("配置文件格式错误，使用默认配置")
                    return cls()

            # 过滤有效参数并转换类型
            valid_keys = set(cls.__dataclass_fields__.keys())
            data = {k: v for k, v in data.items() if k in valid_keys}
            for k, v in data.items():
                try:
                    if cls.__dataclass_fields__[k].type == int:
                        data[k] = int(v)
                    elif cls.__dataclass_fields__[k].type == float:
                        data[k] = float(v)
                except:
                    del data[k]

            logger.info("配置文件加载成功")
            return cls(**data)
        except Exception as e:
            logger.warning(f"加载配置失败：{e}，使用默认配置")
            return cls()

# ==============================================================================
# 核心算法（修复KalmanFilter属性）
# ==============================================================================
class KalmanFilter:
    def __init__(self, dt=0.05, max_speed=50.0):
        self.dt = dt
        self.max_speed = max_speed  # 恢复缺失的属性
        self.x = np.zeros(8, dtype=np.float32)
        self.F = np.array([[1,0,0,0,self.dt,0,0,0],[0,1,0,0,0,self.dt,0,0],[0,0,1,0,0,0,self.dt,0],[0,0,0,1,0,0,0,self.dt],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], dtype=np.float32)
        self.H = np.eye(4, 8, dtype=np.float32)
        self.Q = np.diag([1,1,1,1,5,5,5,5]).astype(np.float32)
        self.R = np.diag([5,5,5,5]).astype(np.float32)
        self.P = np.eye(8, dtype=np.float32) * 50

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z):
        z = z.astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.pinv(S) if np.linalg.det(S) == 0 else np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

    def update_noise_covariance(self, speed):
        speed_factor = min(1.0, speed / self.max_speed)  # 现在能正常访问max_speed
        self.Q = np.diag([1+speed_factor*4]*4 + [5+speed_factor*20]*4).astype(np.float32)

class Track:
    def __init__(self, track_id, bbox, img_shape, kf_config):
        self.track_id = track_id
        self.kf = KalmanFilter(dt=kf_config["dt"], max_speed=kf_config["max_speed"])
        self.img_shape = img_shape
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.cls_id = None

    def _clip_bbox(self, bbox):
        h, w = self.img_shape
        return np.array([max(0, min(bbox[0], w-1)), max(0, min(bbox[1], h-1)), 
                        max(bbox[0]+1, min(bbox[2], w-1)), max(bbox[1]+1, min(bbox[3], h-1))], dtype=np.float32)

    def predict(self):
        # 修复速度计算逻辑（避免数组维度错误）
        prev_center_x = (self.kf.x[0] + self.kf.x[2]) / 2
        curr_center_x = (self.bbox[0] + self.bbox[2]) / 2
        speed = abs(curr_center_x - prev_center_x)  # 简化速度计算，避免维度问题
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self.age += 1
        self.time_since_update += 1
        self.kf.update_noise_covariance(speed)
        return self.bbox

    def update(self, bbox, cls_id):
        self.bbox = self.kf.update(self._clip_bbox(bbox))
        self.hits += 1
        self.time_since_update = 0
        self.cls_id = cls_id

class SimpleSORT:
    def __init__(self, config):
        self.max_age = config.max_age
        self.min_hits = config.min_hits
        self.iou_threshold = config.iou_thres
        self.img_shape = (config.img_height, config.img_width)
        self.kf_config = {"dt": config.kf_dt, "max_speed": config.max_speed}
        self.tracks = []
        self.next_id = 1

    def _compute_iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

    def update(self, detections):
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det[:4], self.img_shape, self.kf_config))
                self.next_id += 1
            return np.array([]), np.array([]), np.array([])

        if len(detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._format_results([t for t in self.tracks if t.hits >= self.min_hits])

        iou_matrix = np.array([[self._compute_iou(t.bbox, d[:4]) for t in self.tracks] for d in detections])
        matches = []
        used_dets = set()
        used_tracks = set()

        for det_idx in range(len(detections)):
            track_idx = np.argmax(iou_matrix[det_idx])
            if iou_matrix[det_idx][track_idx] > self.iou_threshold and track_idx not in used_tracks:
                matches.append((track_idx, det_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx][:4], int(detections[det_idx][5]))

        for det_idx in set(range(len(detections))) - used_dets:
            self.tracks.append(Track(self.next_id, detections[det_idx][:4], self.img_shape, self.kf_config))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self._format_results([t for t in self.tracks if t.hits >= self.min_hits])

    def _format_results(self, tracks):
        return (np.array([t.bbox.astype(int) for t in tracks]),
                np.array([t.track_id for t in tracks]),
                np.array([t.cls_id if t.cls_id is not None else -1 for t in tracks]))

# ==============================================================================
# 工具函数（精简+修复）
# ==============================================================================
class FPSCounter:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.times = []
        self.fps = 0.0

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)
        if len(self.times) >= 2:
            self.fps = (len(self.times)-1) / (self.times[-1] - self.times[0])
        return self.fps

def draw_bounding_boxes(image, boxes, ids, cls_ids, fps=0.0, detection_count=0):
    display_img = image.copy()
    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    # 绘制FPS栏
    cv2.rectangle(display_img, (10,10), (200,40), (0,0,0), -1)
    cv2.putText(display_img, f"FPS:{fps:.1f} | Tracks:{len(boxes)}", (15,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    # 绘制跟踪框
    for box, track_id, cls_id in zip(boxes, ids, cls_ids):
        x1,y1,x2,y2 = box
        color = (hash(track_id)%256, hash(track_id+1)%256, hash(track_id+2)%256)
        cv2.rectangle(display_img, (x1,y1), (x2,y2), color, 2)
        cls_name = vehicle_classes.get(cls_id, "Unknown")
        label = f"ID:{track_id} | {cls_name}"
        cv2.rectangle(display_img, (x1,y1-20), (x1+len(label)*10, y1), color, -1)
        cv2.putText(display_img, label, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return display_img

def clear_actors(world, exclude=None):
    exclude_ids = set(exclude) if exclude else set()
    for actor in world.get_actors():
        if actor.type_id.startswith(('vehicle.', 'sensor.')) and actor.id not in exclude_ids:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass

def camera_callback(image, queue):
    try:
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        if queue.full():
            queue.get_nowait()
        queue.put(cv2.GaussianBlur(img_array[:,:,:3], (3,3), 0))
    except:
        pass

def spawn_npc_vehicles(world, num_npcs, spawn_points):
    npc_bps = [bp for bp in world.get_blueprint_library().filter('vehicle') 
               if int(bp.get_attribute('number_of_wheels')) == 4 and not bp.id.endswith(('firetruck', 'ambulance', 'police'))]
    npc_count = 0
    used_spawns = set()
    for _ in range(num_npcs):
        if len(used_spawns) >= len(spawn_points):
            break
        spawn_point = random.choice(spawn_points)
        spawn_key = (round(spawn_point.location.x,2), round(spawn_point.location.y,2), round(spawn_point.location.z,2))
        if spawn_key not in used_spawns:
            used_spawns.add(spawn_key)
            npc = world.try_spawn_actor(random.choice(npc_bps), spawn_point)
            if npc:
                npc.set_autopilot(True)
                npc_count += 1
    logger.info(f"生成NPC车辆：{npc_count} 辆")
    return npc_count

# ==============================================================================
# 主函数（修复YOLO尺寸+窗口响应）
# ==============================================================================
def main():
    # 解析参数 + 加载配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--host", help="CARLA主机")
    parser.add_argument("--port", type=int, help="CARLA端口")
    parser.add_argument("--conf-thres", type=float, help="检测置信度")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    # 命令行参数覆盖
    if args.host: config.host = args.host
    if args.port: config.port = args.port
    if args.conf_thres: config.conf_thres = args.conf_thres

    # 连接CARLA
    try:
        client = carla.Client(config.host, config.port)
        client.set_timeout(15.0)
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.kf_dt
        world.apply_settings(settings)
        logger.info(f"连接CARLA成功：{config.host}:{config.port}")
    except Exception as e:
        logger.error(f"CARLA连接失败：{e}")
        return

    # 初始化资源
    ego_vehicle = None
    camera = None
    exclude_actors = []
    try:
        clear_actors(world)
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise ValueError("无可用生成点")

        # 生成主车辆
        ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        ego_bp.set_attribute('color', '255,0,0')
        ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
        ego_vehicle.set_autopilot(True)
        exclude_actors.append(ego_vehicle.id)

        # 生成相机
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.img_width))
        camera_bp.set_attribute('image_size_y', str(config.img_height))
        camera_bp.set_attribute('fov', '90')
        camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0), carla.Rotation(pitch=-5.0)), attach_to=ego_vehicle)
        exclude_actors.append(camera.id)

        # 初始化队列、FPS计数器、模型、跟踪器
        image_queue = queue.Queue(maxsize=3)
        camera.listen(lambda img: camera_callback(img, image_queue))
        fps_counter = FPSCounter(window_size=config.fps_window_size)
        detector = YOLO("yolov5s.pt")
        tracker = SimpleSORT(config)

        # 生成NPC
        spawn_npc_vehicles(world, config.num_npcs, spawn_points)

        # 可视化窗口（优化响应）
        cv2.namedWindow("CARLA Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("CARLA Tracking", config.window_width, config.window_height)
        cv2.setWindowProperty("CARLA Tracking", cv2.WND_PROP_TOPMOST, 1)

        # 主循环（优化窗口响应）
        while True:
            world.tick()
            
            # 视角跟随（简化逻辑，避免卡顿）
            if ego_vehicle and ego_vehicle.is_alive:
                try:
                    ego_tf = ego_vehicle.get_transform()
                    spectator = world.get_spectator()
                    spectator.set_transform(carla.Transform(
                        ego_tf.location + carla.Location(x=-8.0, z=12.0),
                        carla.Rotation(pitch=-45.0, yaw=ego_tf.rotation.yaw)
                    ))
                except:
                    pass

            # 获取图像 + 检测 + 跟踪（优化YOLO尺寸）
            try:
                image = image_queue.get(timeout=2.0)
                h, w = image.shape[:2]
                
                # 修复YOLO尺寸：确保是32的整数倍（核心！消除警告）
                def make_divisible(x, divisor=32):
                    return (x + divisor - 1) // divisor * divisor
                ratio = min(config.yolo_imgsz_max / w, config.yolo_imgsz_max / h)
                imgsz_w = make_divisible(int(w * ratio))
                imgsz_h = make_divisible(int(h * ratio))
                
                # YOLO推理（使用修复后的尺寸）
                results = detector.predict(
                    image,
                    conf=config.conf_thres,
                    verbose=False,
                    device="cpu",
                    agnostic_nms=True,
                    imgsz=(imgsz_h, imgsz_w),  # 注意：YOLO的imgsz是(高,宽)
                    iou=config.yolo_iou
                )

                # 解析检测结果
                detections = []
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            if cls_id in {2,5,7}:  # 只保留车辆类
                                detections.append([*box.xyxy[0].cpu().numpy(), float(box.conf[0]), cls_id])
                detections = np.array(detections, dtype=np.float32)

                # 跟踪更新 + 绘制
                tracked_boxes, tracked_ids, tracked_cls = tracker.update(detections)
                display_img = draw_bounding_boxes(image, tracked_boxes, tracked_ids, tracked_cls, fps_counter.update(), len(detections))
                cv2.imshow("CARLA Tracking", display_img)

                # 键盘控制（优化响应）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q/ESC退出
                    break
                elif key == ord('s'):  # 保存截图
                    cv2.imwrite(f"track_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png", display_img)
                    logger.info("截图保存成功")

            except queue.Empty:
                logger.warning("未获取到图像，跳过该帧")
                continue
            except Exception as e:
                logger.warning(f"帧处理失败：{str(e)[:50]}")
                continue

    except Exception as e:
        logger.error(f"程序异常：{e}")
    finally:
        # 资源清理（确保释放）
        if camera:
            camera.stop()
            try: camera.destroy()
            except: pass
        if ego_vehicle:
            try: ego_vehicle.destroy()
            except: pass
        clear_actors(world, exclude_actors)
        # 恢复CARLA设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        # 关闭窗口
        cv2.destroyAllWindows()
        logger.info("程序退出")

if __name__ == "__main__":
    main()