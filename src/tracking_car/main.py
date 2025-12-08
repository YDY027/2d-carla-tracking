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
import sys
import yaml
from dataclasses import dataclass, field
import threading
from scipy.optimize import linear_sum_assignment
import torch

# ==============================================================================
# 0. 环境检测与全局配置
# ==============================================================================
# 跨平台兼容
PLATFORM = sys.platform
IS_WINDOWS = PLATFORM.startswith('win')
IS_LINUX = PLATFORM.startswith('linux')
# 日志配置
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 预设天气配置（CARLA官方天气参数）
# 预设天气配置（兼容所有CARLA版本，仅保留基础参数）
WEATHER_PRESETS = {
    'clear': carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=0.0, sun_azimuth_angle=180.0, sun_altitude_angle=75.0,
        fog_density=0.0, fog_distance=0.0, fog_falloff=1.0,
        wetness=0.0, scattering_intensity=0.0
    ),
    'rain': carla.WeatherParameters(
        cloudiness=80.0, precipitation=80.0, precipitation_deposits=50.0,
        wind_intensity=30.0, sun_azimuth_angle=180.0, sun_altitude_angle=45.0,
        fog_density=20.0, fog_distance=50.0, fog_falloff=0.8,
        wetness=80.0, scattering_intensity=0.5
    ),
    'fog': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=10.0, sun_azimuth_angle=180.0, sun_altitude_angle=30.0,
        fog_density=70.0, fog_distance=20.0, fog_falloff=0.5,
        wetness=10.0, scattering_intensity=0.8
    ),
    'night': carla.WeatherParameters(
        cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=0.0, sun_azimuth_angle=0.0, sun_altitude_angle=-90.0,  # 太阳高度角为负=夜晚
        fog_density=10.0, fog_distance=100.0, fog_falloff=0.7,
        wetness=0.0, scattering_intensity=1.0
    ),
    'cloudy': carla.WeatherParameters(
        cloudiness=90.0, precipitation=0.0, precipitation_deposits=0.0,
        wind_intensity=20.0, sun_azimuth_angle=180.0, sun_altitude_angle=60.0,
        fog_density=10.0, fog_distance=100.0, fog_falloff=0.9,
        wetness=0.0, scattering_intensity=0.3
    ),
    'snow': carla.WeatherParameters(
        cloudiness=90.0, precipitation=90.0, precipitation_deposits=80.0,  # 低版本用precipitation模拟雪
        wind_intensity=40.0, sun_azimuth_angle=180.0, sun_altitude_angle=20.0,
        fog_density=30.0, fog_distance=30.0, fog_falloff=0.6,
        wetness=50.0, scattering_intensity=0.7
    )
}

# ==============================================================================
# 1. 配置类（新增天气相关配置）
# ==============================================================================
@dataclass
class Config:
    # CARLA基础配置
    host: str = "localhost"
    port: int = 2000
    num_npcs: int = 10
    img_width: int = 640
    img_height: int = 480
    
    # 检测/跟踪核心配置
    conf_thres: float = 0.5
    iou_thres: float = 0.3
    max_age: int = 5
    min_hits: int = 3
    
    # YOLO优化配置
    yolo_model: str = "yolov8n.pt"
    yolo_imgsz_max: int = 320
    yolo_iou: float = 0.45
    
    # 卡尔曼滤波配置
    kf_dt: float = 0.05
    max_speed: float = 50.0
    
    # 可视化配置
    window_width: int = 1280
    window_height: int = 720
    smooth_alpha: float = 0.2
    fps_window_size: int = 15
    display_fps: int = 30
    
    # 轨迹/行为分析配置
    track_history_len: int = 20
    track_line_width: int = 2
    track_alpha: float = 0.6
    stop_speed_thresh: float = 1.0
    stop_frames_thresh: int = 5
    overtake_speed_ratio: float = 1.5
    overtake_dist_thresh: float = 50.0
    
    # 新增：天气配置
    default_weather: str = "clear"  # 默认天气
    auto_adjust_detection: bool = True  # 天气自适应检测
    
    @classmethod
    def from_yaml(cls, yaml_path: str = None) -> "Config":
        try:
            yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
            if not os.path.exists(yaml_path):
                logger.warning("配置文件不存在，使用默认配置")
                return cls()

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f.read().strip().replace("\t", "  "))
            if not isinstance(data, dict):
                logger.warning("配置文件格式错误，使用默认配置")
                return cls()

            valid_keys = set(cls.__dataclass_fields__.keys())
            data = {k: v for k, v in data.items() if k in valid_keys}
            # 类型转换
            for k, v in data.items():
                try:
                    field_type = cls.__dataclass_fields__[k].type
                    if field_type == int:
                        data[k] = int(v)
                    elif field_type == float:
                        data[k] = float(v)
                    elif field_type == bool:
                        data[k] = bool(v)
                except:
                    del data[k]

            logger.info("配置文件加载成功")
            return cls(**data)
        except Exception as e:
            logger.warning(f"加载配置失败：{e}，使用默认配置")
            return cls()

# ==============================================================================
# 2. 图像增强（天气自适应）
# ==============================================================================
class WeatherImageEnhancer:
    """天气自适应图像增强器"""
    def __init__(self, config):
        self.config = config
        self.current_weather = "clear"
        # 不同天气的增强参数
        self.enhance_params = {
            'clear': {'brightness': 1.0, 'contrast': 1.0, 'gamma': 1.0},
            'rain': {'brightness': 1.1, 'contrast': 1.2, 'gamma': 0.9, 'dehaze': True},
            'fog': {'brightness': 1.3, 'contrast': 1.4, 'gamma': 0.8, 'dehaze': True},
            'night': {'brightness': 1.5, 'contrast': 1.3, 'gamma': 0.7, 'denoise': True},
            'cloudy': {'brightness': 1.2, 'contrast': 1.1, 'gamma': 1.0},
            'snow': {'brightness': 1.1, 'contrast': 1.3, 'gamma': 0.9, 'dehaze': True}
        }
    
    def set_weather(self, weather_name):
        """设置当前天气"""
        if weather_name in WEATHER_PRESETS:
            self.current_weather = weather_name
            logger.info(f"切换天气：{weather_name}，自动调整图像增强参数")
    
    def enhance(self, image):
        """根据天气增强图像"""
        if not self.config.auto_adjust_detection:
            return image
        
        params = self.enhance_params.get(self.current_weather, self.enhance_params['clear'])
        enhanced = image.copy()
        
        # 1. 亮度/对比度调整
        alpha = params['contrast']  # 对比度
        beta = int(params['brightness'] * 255 - 255)  # 亮度
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # 2. Gamma校正
        gamma = params['gamma']
        inv_gamma = 1.0 / gamma
        gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, gamma_table)
        
        # 3. 去雾（雾/雨/雪天）
        if params.get('dehaze', False):
            enhanced = self._dehaze(enhanced)
        
        # 4. 去噪（夜晚）
        if params.get('denoise', False):
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def _dehaze(self, image):
        """快速去雾算法"""
        # 简化版暗通道去雾
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(gray, np.ones((7,7), np.uint8))
        atmospheric_light = np.max(image[dark_channel < 10])
        
        # 透射率估计
        t = 1 - 0.1 * (gray / atmospheric_light)
        t = np.clip(t, 0.1, 1.0)
        
        # 去雾
        dehazed = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            dehazed[:,:,c] = (image[:,:,c].astype(np.float32) - atmospheric_light) / t + atmospheric_light
        dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
        
        return dehazed

# ==============================================================================
# 3. 核心算法（卡尔曼滤波+SORT跟踪）
# ==============================================================================
class KalmanFilter:
    def __init__(self, dt=0.05, max_speed=50.0):
        self.dt = dt
        self.max_speed = max_speed
        self.x = np.zeros(8, dtype=np.float32)  # [x1,y1,x2,y2,vx1,vy1,vx2,vy2]
        # 状态转移矩阵
        self.F = np.array([
            [1,0,0,0,self.dt,0,0,0],
            [0,1,0,0,0,self.dt,0,0],
            [0,0,1,0,0,0,self.dt,0],
            [0,0,0,1,0,0,0,self.dt],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ], dtype=np.float32)
        # 观测矩阵
        self.H = np.eye(4, 8, dtype=np.float32)
        # 过程噪声协方差
        self.Q = np.diag([1,1,1,1,5,5,5,5]).astype(np.float32)
        # 观测噪声协方差
        self.R = np.diag([5,5,5,5]).astype(np.float32)
        # 状态协方差矩阵
        self.P = np.eye(8, dtype=np.float32) * 50

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z):
        z = z.astype(np.float32)
        y = z - self.H @ self.x  # 残差
        S = self.H @ self.P @ self.H.T + self.R  # 残差协方差
        # 避免奇异矩阵
        S_inv = np.linalg.pinv(S) if np.linalg.det(S) < 1e-6 else np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv  # 卡尔曼增益
        # 更新状态
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]

    def update_noise_covariance(self, speed):
        speed_factor = min(1.0, speed / self.max_speed)
        self.Q = np.diag([1+speed_factor*4]*4 + [5+speed_factor*20]*4).astype(np.float32)

class Track:
    def __init__(self, track_id, bbox, img_shape, kf_config, config):
        self.track_id = track_id
        self.kf = KalmanFilter(dt=kf_config["dt"], max_speed=kf_config["max_speed"])
        self.img_shape = img_shape
        self.config = config
        
        # 严格的边界检查
        if len(bbox) != 4:
            raise ValueError(f"bbox维度错误：期望4维，实际{len(bbox)}维")
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox
        
        # 轨迹记录
        self.track_history = []
        self._update_track_history()
        
        # 行为分析相关
        self.speed_history = []
        self.is_stopped = False
        self.stop_frame_count = 0
        self.is_overtaking = False
        self.overtake_frame_count = 0
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.cls_id = None

    def _clip_bbox(self, bbox):
        h, w = self.img_shape
        return np.array([
            max(0, min(bbox[0], w-1)),
            max(0, min(bbox[1], h-1)),
            max(bbox[0]+1, min(bbox[2], w-1)),
            max(bbox[1]+1, min(bbox[3], h-1))
        ], dtype=np.float32)

    def _update_track_history(self):
        """更新轨迹历史"""
        center_x = (self.bbox[0] + self.bbox[2]) / 2
        center_y = (self.bbox[1] + self.bbox[3]) / 2
        self.track_history.append((center_x, center_y))
        if len(self.track_history) > self.config.track_history_len:
            self.track_history.pop(0)

    def _calculate_speed(self):
        """计算目标速度"""
        if len(self.track_history) < 2:
            return 0.0
        prev_center = self.track_history[-2]
        curr_center = self.track_history[-1]
        speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center)) / self.kf.dt
        self.speed_history.append(speed)
        if len(self.speed_history) > 5:
            self.speed_history.pop(0)
        return np.mean(self.speed_history)

    def _analyze_behavior(self, ego_center):
        """分析目标行为"""
        # 停车检测
        current_speed = self._calculate_speed()
        if current_speed < self.config.stop_speed_thresh:
            self.stop_frame_count += 1
            self.is_stopped = self.stop_frame_count >= self.config.stop_frames_thresh
        else:
            self.stop_frame_count = 0
            self.is_stopped = False
        
        # 超车检测
        if ego_center is None or len(self.track_history) < 2:
            self.is_overtaking = False
            return
        
        target_center = self.track_history[-1]
        ego_center_np = np.array(ego_center)
        target_center_np = np.array(target_center)
        
        dist = np.linalg.norm(target_center_np - ego_center_np)
        if dist > self.config.overtake_dist_thresh:
            self.overtake_frame_count = 0
            self.is_overtaking = False
            return
        
        ego_speed = 0.0
        if hasattr(self, 'ego_speed') and self.ego_speed > 0:
            if current_speed > self.ego_speed * self.config.overtake_speed_ratio:
                self.overtake_frame_count += 1
                self.is_overtaking = self.overtake_frame_count >= 3
            else:
                self.overtake_frame_count = 0
                self.is_overtaking = False

    def predict(self):
        # 优化速度计算
        prev_center = np.array([(self.kf.x[0]+self.kf.x[2])/2, (self.kf.x[1]+self.kf.x[3])/2])
        curr_center = np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
        pixel_speed = np.linalg.norm(curr_center - prev_center) / self.kf.dt
        max_pixel_speed = max(self.img_shape) / self.kf.dt
        speed = min(1.0, pixel_speed / max_pixel_speed) * self.kf.max_speed
        
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self._update_track_history()
        self.age += 1
        self.time_since_update += 1
        self.kf.update_noise_covariance(speed)
        return self.bbox

    def update(self, bbox, cls_id, ego_center=None):
        if len(bbox) != 4:
            raise ValueError(f"更新bbox维度错误：期望4维，实际{len(bbox)}维")
        self.bbox = self.kf.update(self._clip_bbox(bbox))
        self._update_track_history()
        self.hits += 1
        self.time_since_update = 0
        self.cls_id = cls_id
        self._analyze_behavior(ego_center)

class SimpleSORT:
    def __init__(self, config):
        self.max_age = config.max_age
        self.min_hits = config.min_hits
        self.iou_threshold = config.iou_thres
        self.img_shape = (config.img_height, config.img_width)
        self.kf_config = {"dt": config.kf_dt, "max_speed": config.max_speed}
        self.config = config
        self.tracks = []
        self.next_id = 1
        self.ego_center = None

    def _compute_iou(self, box1, box2):
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections, ego_center=None):
        """更新跟踪器"""
        self.ego_center = ego_center

        # 1. 严格的格式校验
        valid_detections = []
        if detections is not None and len(detections) > 0:
            for det in detections:
                if len(det) >= 6:
                    x1,y1,x2,y2,conf,cls_id = det[:6]
                    if (isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and
                        isinstance(x2, (int, float)) and isinstance(y2, (int, float)) and
                        conf > 0 and x2 > x1 and y2 > y1):
                        valid_detections.append([x1,y1,x2,y2,conf,int(cls_id)])
        
        valid_detections = np.array(valid_detections, dtype=np.float32)

        # 2. 预测所有轨迹
        for track in self.tracks:
            try:
                track.predict()
            except Exception as e:
                logger.warning(f"轨迹预测失败: {str(e)[:30]}")

        # 3. 无轨迹时初始化
        if len(self.tracks) == 0 and len(valid_detections) > 0:
            for det in valid_detections:
                try:
                    self.tracks.append(Track(self.next_id, det[:4], self.img_shape, self.kf_config, self.config))
                    self.next_id += 1
                except Exception as e:
                    logger.warning(f"轨迹初始化失败: {str(e)[:30]}")
            return np.array([]), np.array([]), np.array([])

        # 4. 无检测时清理过期轨迹
        if len(valid_detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return self._format_results([t for t in self.tracks if t.hits >= self.min_hits])

        # 5. 匈牙利算法匹配
        try:
            iou_matrix = np.array([[self._compute_iou(t.bbox, d[:4]) for t in self.tracks] for d in valid_detections])
            cost_matrix = 1 - iou_matrix
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.warning(f"匹配算法失败: {str(e)[:30]}")
            track_indices, det_indices = [], []

        matches = []
        used_dets = set()
        used_tracks = set()

        # 6. 筛选有效匹配
        for t_idx, d_idx in zip(track_indices, det_indices):
            try:
                if (t_idx < len(self.tracks) and d_idx < len(valid_detections) and
                    self._compute_iou(self.tracks[t_idx].bbox, valid_detections[d_idx][:4]) > self.iou_threshold):
                    matches.append((t_idx, d_idx))
                    used_dets.add(d_idx)
                    used_tracks.add(t_idx)
            except Exception as e:
                continue

        # 7. 更新匹配的轨迹
        for track_idx, det_idx in matches:
            try:
                self.tracks[track_idx].update(valid_detections[det_idx][:4], 
                                             int(valid_detections[det_idx][5]), 
                                             self.ego_center)
            except Exception as e:
                logger.warning(f"轨迹更新失败: {str(e)[:30]}")

        # 8. 新增未匹配的检测
        for det_idx in set(range(len(valid_detections))) - used_dets:
            try:
                self.tracks.append(Track(self.next_id, valid_detections[det_idx][:4], self.img_shape, self.kf_config, self.config))
                self.next_id += 1
            except Exception as e:
                logger.warning(f"新增轨迹失败: {str(e)[:30]}")

        # 9. 清理过期轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 10. 返回有效轨迹
        valid_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        return self._format_results(valid_tracks)

    def _format_results(self, tracks):
        if not tracks:
            return np.array([]), np.array([]), np.array([])
        try:
            boxes = np.array([t.bbox.astype(int) for t in tracks])
            ids = np.array([t.track_id for t in tracks])
            cls = np.array([t.cls_id if t.cls_id is not None else -1 for t in tracks])
            return boxes, ids, cls
        except Exception as e:
            logger.warning(f"结果格式化失败: {str(e)[:30]}")
            return np.array([]), np.array([]), np.array([])

# ==============================================================================
# 4. 推理线程类
# ==============================================================================
class DetectionThread(threading.Thread):
    def __init__(self, detector, config, enhancer, input_queue, output_queue, device="cpu"):
        super().__init__(daemon=True)
        self.detector = detector
        self.config = config
        self.enhancer = enhancer  # 新增：图像增强器
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.device = device

    def run(self):
        while self.running:
            try:
                image = self.input_queue.get(timeout=1.0)
                if image is None or len(image.shape) != 3 or image.shape[2] != 3:
                    self.output_queue.put((None, np.array([])))
                    continue
                
                # 新增：天气自适应图像增强
                image_enhanced = self.enhancer.enhance(image)
                
                h, w = image.shape[:2]
                # 确保尺寸是32的整数倍
                def make_divisible(x, divisor=32):
                    return (x + divisor - 1) // divisor * divisor
                ratio = min(self.config.yolo_imgsz_max / w, self.config.yolo_imgsz_max / h)
                imgsz_w = make_divisible(int(w * ratio))
                imgsz_h = make_divisible(int(h * ratio))
                
                # YOLO推理
                try:
                    results = self.detector.predict(
                        image_enhanced,
                        conf=self.config.conf_thres,
                        verbose=False,
                        device=self.device,
                        agnostic_nms=True,
                        imgsz=(imgsz_h, imgsz_w),
                        iou=self.config.yolo_iou
                    )
                except Exception as e:
                    logger.warning(f"YOLO推理失败: {str(e)[:30]}")
                    self.output_queue.put((image, np.array([])))
                    continue

                # 解析检测结果
                detections = []
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            try:
                                cls_id = int(box.cls[0])
                                if cls_id in {2,5,7}:  # 仅保留车辆类
                                    xyxy = box.xyxy[0].cpu().numpy()
                                    conf = float(box.conf[0])
                                    detections.append([*xyxy, conf, cls_id])
                            except Exception as e:
                                continue
                
                self.output_queue.put((image, np.array(detections, dtype=np.float32)))
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"推理线程错误: {str(e)[:50]}")
                self.output_queue.put((None, np.array([])))

    def stop(self):
        self.running = False

# ==============================================================================
# 5. 帧缓存+固定帧率管理器
# ==============================================================================
class FrameBuffer:
    def __init__(self, default_size=(480, 640, 3)):
        self.default_frame = np.zeros(default_size, dtype=np.uint8)
        cv2.putText(self.default_frame, "Initializing...", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.current_frame = self.default_frame.copy()
        self.lock = threading.Lock()

    def update(self, frame):
        """线程安全更新帧"""
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            with self.lock:
                self.current_frame = frame.copy()

    def get(self):
        """获取当前帧"""
        return self.current_frame.copy()

class FixedRateDisplay:
    def __init__(self, fps=30):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_display_time = time.time()

    def wait(self):
        """等待直到达到目标帧率"""
        elapsed = time.time() - self.last_display_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        self.last_display_time = time.time()

# ==============================================================================
# 6. 工具函数
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

def draw_bounding_boxes(image, boxes, ids, cls_ids, tracks, fps=0.0, detection_count=0, config=None, current_weather="clear"):
    """绘制跟踪框+轨迹+行为标签+天气信息"""
    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if config is None:
        config = Config()
    
    display_img = np.empty_like(image)
    display_img[:] = image
    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    
    # 绘制FPS、跟踪数、天气信息
    overlay = display_img.copy()
    cv2.rectangle(overlay, (10,10), (450,40), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
    
    # 统计行为数量
    stop_count = sum(1 for t in tracks if t.is_stopped)
    overtake_count = sum(1 for t in tracks if t.is_overtaking)
    cv2.putText(display_img, 
                f"FPS:{fps:.1f} | Weather:{current_weather} | Tracks:{len(boxes)} | Dets:{detection_count} | Stop:{stop_count} | Overtake:{overtake_count}", 
                (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, lineType=cv2.LINE_AA)
    
    # 绘制跟踪框+轨迹+行为标签
    if len(boxes) > 0 and len(ids) > 0 and len(cls_ids) > 0 and len(tracks) > 0:
        min_len = min(len(boxes), len(ids), len(cls_ids), len(tracks))
        for i in range(min_len):
            try:
                box = boxes[i]
                track_id = ids[i]
                cls_id = cls_ids[i]
                track = tracks[i]
                
                if len(box) != 4:
                    continue
                
                x1,y1,x2,y2 = box
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # 固定颜色
                color = (
                    (track_id * 59) % 256,
                    (track_id * 127) % 256,
                    (track_id * 199) % 256
                )
                cv2.rectangle(display_img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2, lineType=cv2.LINE_AA)
                
                # 绘制轨迹
                if len(track.track_history) >= 2:
                    track_overlay = display_img.copy()
                    for j in range(1, len(track.track_history)):
                        pt1 = (int(track.track_history[j-1][0]), int(track.track_history[j-1][1]))
                        pt2 = (int(track.track_history[j][0]), int(track.track_history[j][1]))
                        alpha = j / len(track.track_history) * config.track_alpha
                        line_width = int(j / len(track.track_history) * config.track_line_width) + 1
                        cv2.line(track_overlay, pt1, pt2, color, line_width, lineType=cv2.LINE_AA)
                    cv2.addWeighted(track_overlay, alpha, display_img, 1-alpha, 0, display_img)
                
                # 构建标签
                cls_name = vehicle_classes.get(cls_id, "Unknown")
                behavior_tags = []
                if track.is_stopped:
                    behavior_tags.append("STOP")
                if track.is_overtaking:
                    behavior_tags.append("OVERTAKE")
                behavior_str = " | " + " | ".join(behavior_tags) if behavior_tags else ""
                label = f"ID:{track_id} | {cls_name}{behavior_str}"
                
                # 绘制标签背景
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                overlay = display_img.copy()
                cv2.rectangle(overlay, (int(x1),int(y1)-20), (int(x1)+label_size[0]+20, int(y1)), color, -1)
                cv2.addWeighted(overlay, 0.8, display_img, 0.2, 0, display_img)
                cv2.putText(display_img, label, (int(x1)+5, int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, lineType=cv2.LINE_AA)
            except Exception as e:
                continue
    
    return display_img

def clear_actors(world, exclude=None):
    """优化的actor清理函数"""
    exclude_ids = set(exclude) if exclude else set()
    actors = world.get_actors()
    batch_size = 10
    
    vehicle_actors = [a for a in actors if a.type_id.startswith('vehicle.') and a.id not in exclude_ids]
    sensor_actors = [a for a in actors if a.type_id.startswith('sensor.') and a.id not in exclude_ids]
    
    # 销毁车辆
    for i in range(0, len(vehicle_actors), batch_size):
        batch = vehicle_actors[i:i+batch_size]
        for actor in batch:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                logger.warning(f"销毁车辆失败: {str(e)[:30]}")
    
    # 销毁传感器
    for i in range(0, len(sensor_actors), batch_size):
        batch = sensor_actors[i:i+batch_size]
        for actor in batch:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as e:
                logger.warning(f"销毁传感器失败: {str(e)[:30]}")

def camera_callback(image, queue):
    try:
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        img_rgb = cv2.GaussianBlur(img_array[:,:,:3], (3,3), 0)
        
        if queue.full():
            try:
                queue.get_nowait()
            except:
                pass
        queue.put(img_rgb)
    except Exception as e:
        logger.warning(f"相机回调错误: {str(e)[:30]}")

def spawn_npc_vehicles(world, num_npcs, spawn_points):
    """优化的NPC生成函数"""
    npc_bps = [
        bp for bp in world.get_blueprint_library().filter('vehicle') 
        if int(bp.get_attribute('number_of_wheels')) == 4 
        and not bp.id.endswith(('firetruck', 'ambulance', 'police'))
    ]
    
    if not npc_bps:
        logger.warning("无可用车辆蓝图")
        return 0
    
    npc_count = 0
    used_spawns = set()
    max_attempts = num_npcs * 3
    
    for _ in range(max_attempts):
        if npc_count >= num_npcs or len(used_spawns) >= len(spawn_points):
            break
        
        spawn_point = random.choice(spawn_points)
        spawn_key = (round(spawn_point.location.x,2), round(spawn_point.location.y,2), round(spawn_point.location.z,2))
        
        if spawn_key not in used_spawns:
            used_spawns.add(spawn_key)
            npc = world.try_spawn_actor(random.choice(npc_bps), spawn_point)
            if npc:
                npc.set_autopilot(True)
                npc_count += 1
    
    logger.info(f"生成NPC车辆：{npc_count} 辆 (目标: {num_npcs})")
    return npc_count

# ==============================================================================
# 7. 主函数（新增天气切换逻辑）
# ==============================================================================
def main():
    # 解析参数 + 加载配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--host", help="CARLA主机")
    parser.add_argument("--port", type=int, help="CARLA端口")
    parser.add_argument("--conf-thres", type=float, help="检测置信度")
    parser.add_argument("--weather", help="初始天气 (clear/rain/fog/night/cloudy/snow)")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    # 命令行参数覆盖配置
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.conf_thres:
        config.conf_thres = args.conf_thres
    if args.weather and args.weather in WEATHER_PRESETS:
        config.default_weather = args.weather

    # 连接CARLA
    try:
        client = carla.Client(config.host, config.port)
        client.set_timeout(15.0)
        world = client.get_world()
        
        # 设置同步模式
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
    det_thread = None
    
    try:
        # 清理现有actor
        clear_actors(world)
        
        # 获取生成点
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise ValueError("地图无可用生成点")

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
        camera_transform = carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.0),
            carla.Rotation(pitch=-5.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        exclude_actors.append(camera.id)

        # 初始化天气增强器
        enhancer = WeatherImageEnhancer(config)
        # 设置初始天气
        initial_weather = config.default_weather
        world.set_weather(WEATHER_PRESETS[initial_weather])
        enhancer.set_weather(initial_weather)

        # 初始化队列、帧缓存、固定帧率器
        image_queue = queue.Queue(maxsize=3)
        camera.listen(lambda img: camera_callback(img, image_queue))
        frame_buffer = FrameBuffer(default_size=(config.img_height, config.img_width, 3))
        display_controller = FixedRateDisplay(fps=config.display_fps)
        
        # 初始化FPS计数器、跟踪器
        fps_counter = FPSCounter(window_size=config.fps_window_size)
        tracker = SimpleSORT(config)
        
        # 初始化YOLO推理线程
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用YOLO模型: {config.yolo_model}, 推理设备: {device}")

        detector = YOLO(config.yolo_model)
        try:
            detector.fuse()
            detector.to(device)
        except Exception as e:
            logger.warning(f"模型优化失败: {e}，使用默认配置")

        det_input_queue = queue.Queue(maxsize=2)
        det_output_queue = queue.Queue(maxsize=2)
        # 传入图像增强器
        det_thread = DetectionThread(detector, config, enhancer, det_input_queue, det_output_queue, device)
        det_thread.start()

        # 生成NPC车辆
        spawn_npc_vehicles(world, config.num_npcs, spawn_points)

        # 初始化可视化窗口
        cv2.namedWindow("CARLA Tracking (Weather Adaptive)", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("CARLA Tracking (Weather Adaptive)", config.window_width, config.window_height)
        cv2.setWindowProperty("CARLA Tracking (Weather Adaptive)", cv2.WND_PROP_TOPMOST, 1)
        
        # 预显示默认帧
        initial_frame = frame_buffer.get()
        initial_frame_resized = cv2.resize(initial_frame, (config.window_width, config.window_height), 
                                          interpolation=cv2.INTER_LINEAR)
        cv2.imshow("CARLA Tracking (Weather Adaptive)", initial_frame_resized)
        cv2.waitKey(1)

        # 打印操作提示
        logger.info("\n========== 操作提示 ==========")
        logger.info("Q/ESC: 退出程序")
        logger.info("S: 保存截图")
        logger.info("1: 晴天 (clear)")
        logger.info("2: 雨天 (rain)")
        logger.info("3: 雾天 (fog)")
        logger.info("4: 夜晚 (night)")
        logger.info("5: 多云 (cloudy)")
        logger.info("6: 雪天 (snow)")
        logger.info("==============================")

        # 主循环
        current_weather = config.default_weather
        while True:
            world.tick()
            
            # 视角跟随主车辆
            if ego_vehicle and ego_vehicle.is_alive:
                try:
                    ego_tf = ego_vehicle.get_transform()
                    spectator = world.get_spectator()
                    spectator_transform = carla.Transform(
                        ego_tf.location + carla.Location(x=-8.0, z=12.0),
                        carla.Rotation(pitch=-45.0, yaw=ego_tf.rotation.yaw)
                    )
                    spectator.set_transform(spectator_transform)
                except Exception as e:
                    logger.warning(f"视角更新失败: {str(e)[:30]}")

            # 核心帧处理逻辑
            current_display_frame = frame_buffer.get()
            try:
                # 1. 非阻塞获取相机图像
                try:
                    image = image_queue.get_nowait()
                    if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
                        ego_center = (image.shape[1] / 2, image.shape[0] / 2)
                        tracker.ego_center = ego_center
                        if not det_input_queue.full():
                            det_input_queue.put(image.copy())
                except queue.Empty:
                    pass

                # 2. 非阻塞获取推理结果
                try:
                    img, detections = det_output_queue.get_nowait()
                    if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                        ego_center = (img.shape[1] / 2, img.shape[0] / 2)
                        tracked_boxes, tracked_ids, tracked_cls = tracker.update(detections, ego_center)
                        # 绘制结果（传入当前天气）
                        display_img = draw_bounding_boxes(
                            img, tracked_boxes, tracked_ids, tracked_cls,
                            tracker.tracks,
                            fps_counter.update(), len(detections),
                            config, current_weather
                        )
                        frame_buffer.update(display_img)
                        current_display_frame = display_img
                except queue.Empty:
                    current_display_frame = frame_buffer.get()
                    cv2.putText(current_display_frame, "Detecting...", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, lineType=cv2.LINE_AA)

                # 3. 固定帧率刷新窗口
                if current_display_frame is not None and len(current_display_frame.shape) == 3:
                    display_frame_resized = cv2.resize(current_display_frame, 
                                                      (config.window_width, config.window_height),
                                                      interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("CARLA Tracking (Weather Adaptive)", display_frame_resized)
                display_controller.wait()

                # 4. 键盘控制（新增天气切换）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q/ESC退出
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # S保存截图
                    save_path = f"track_screenshot_{current_weather}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(save_path, frame_buffer.get())
                    logger.info(f"截图已保存: {save_path}")
                # 天气切换快捷键
                elif key == ord('1'):
                    current_weather = "clear"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)
                elif key == ord('2'):
                    current_weather = "rain"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)
                elif key == ord('3'):
                    current_weather = "fog"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)
                elif key == ord('4'):
                    current_weather = "night"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)
                elif key == ord('5'):
                    current_weather = "cloudy"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)
                elif key == ord('6'):
                    current_weather = "snow"
                    world.set_weather(WEATHER_PRESETS[current_weather])
                    enhancer.set_weather(current_weather)

            except Exception as e:
                logger.warning(f"帧处理失败: {str(e)[:50]}")
                continue

    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
    finally:
        # 停止推理线程
        if det_thread:
            det_thread.stop()
            det_thread.join(timeout=2.0)
        
        # 清理资源
        if camera:
            camera.stop()
            try:
                camera.destroy()
            except:
                pass
        
        if ego_vehicle:
            try:
                ego_vehicle.destroy()
            except:
                pass
        
        # 清理剩余actor
        clear_actors(world, exclude_actors)
        
        # 恢复CARLA设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        # 关闭窗口
        cv2.destroyAllWindows()
        logger.info("程序已安全退出")

if __name__ == "__main__":
    main()