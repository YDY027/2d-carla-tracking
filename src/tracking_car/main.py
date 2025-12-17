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
import csv
from datetime import datetime
import open3d as o3d
from numba import njit
from loguru import logger
import psutil
import json
from sklearn.cluster import DBSCAN

# ==============================================================================
# 0. 环境检测与全局配置
# ==============================================================================
# 跨平台兼容
PLATFORM = sys.platform
IS_WINDOWS = PLATFORM.startswith('win')
IS_LINUX = PLATFORM.startswith('linux')

# 结构化日志配置（替换原生logging）
logger.remove()
logger.add(
    sink=sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    sink=f"track_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    rotation="100 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    serialize=False
)

# 预设天气配置（兼容所有CARLA版本）
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
        wind_intensity=0.0, sun_azimuth_angle=0.0, sun_altitude_angle=-90.0,
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
        cloudiness=90.0, precipitation=90.0, precipitation_deposits=80.0,
        wind_intensity=40.0, sun_azimuth_angle=180.0, sun_altitude_angle=20.0,
        fog_density=30.0, fog_distance=30.0, fog_falloff=0.6,
        wetness=50.0, scattering_intensity=0.7
    )
}

# 行为类型定义
BEHAVIOR_TYPES = {
    0: "normal",
    1: "stopped",
    2: "overtaking",
    3: "lane_changing",
    4: "braking",
    5: "accelerating",
    6: "turning",
    7: "dangerous"
}

# ==============================================================================
# 1. 配置类（新增多传感器/行为分析/数据记录配置）
# ==============================================================================
@dataclass
class Config:
    # CARLA基础配置
    host: str = "localhost"
    port: int = 2000
    num_npcs: int = 20
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
    yolo_quantize: bool = False  # 是否使用量化模型
    
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
    
    # 新增：行为分析扩展配置
    lane_change_thresh: float = 0.5  # 变道检测阈值（横向位移）
    brake_accel_thresh: float = 2.0  # 刹车/加速加速度阈值
    turn_angle_thresh: float = 15.0  # 转弯角度阈值（度）
    danger_dist_thresh: float = 10.0  # 危险距离阈值（米）
    predict_frames: int = 10  # 轨迹预测帧数（未来1-3秒）
    
    # 新增：天气配置
    default_weather: str = "clear"
    auto_adjust_detection: bool = True
    
    # 新增：多传感器配置
    use_lidar: bool = True  # 是否启用LiDAR
    lidar_channels: int = 32
    lidar_range: float = 100.0
    lidar_points_per_second: int = 500000
    fuse_lidar_vision: bool = True  # 是否融合LiDAR和视觉检测
    
    # 新增：数据记录配置
    record_data: bool = True  # 是否记录离线数据
    record_dir: str = "track_records"
    record_format: str = "csv"  # csv/hdf5/json
    record_fps: int = 10  # 记录帧率（避免数据量过大）
    save_screenshots: bool = False  # 是否自动保存截图
    
    # 新增：3D可视化配置
    use_3d_visualization: bool = False
    pcd_view_size: int = 800  # 点云可视化窗口大小
    
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
            'rain': {'brightness': 1.1, 'contrast': 1.2, 'gamma': 0.9, 'dehaze': True, 'derain': True},
            'fog': {'brightness': 1.3, 'contrast': 1.4, 'gamma': 0.8, 'dehaze': True},
            'night': {'brightness': 1.5, 'contrast': 1.3, 'gamma': 0.7, 'denoise': True},
            'cloudy': {'brightness': 1.2, 'contrast': 1.1, 'gamma': 1.0},
            'snow': {'brightness': 1.1, 'contrast': 1.3, 'gamma': 0.9, 'dehaze': True, 'desnow': True}
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
        
        # 空图像保护
        if image is None or len(image.shape) != 3 or image.size == 0:
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
        
        # 4. 去雨（雨天）
        if params.get('derain', False):
            enhanced = self._derain(enhanced)
        
        # 5. 去雪（雪天）
        if params.get('desnow', False):
            enhanced = self._desnow(enhanced)
        
        # 6. 去噪（夜晚）
        if params.get('denoise', False):
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
    
    def _dehaze(self, image):
        """快速去雾算法"""
        # 空图像保护
        if image is None or len(image.shape) != 3 or image.size == 0:
            return image
            
        # 简化版暗通道去雾
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(gray, np.ones((7,7), np.uint8))
        
        # 空数组保护
        non_zero_pixels = dark_channel[dark_channel < 10]
        if len(non_zero_pixels) == 0:
            atmospheric_light = 255.0
        else:
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
    
    def _derain(self, image):
        """快速去雨算法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 形态学操作检测雨丝
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        rain_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        # 修复雨丝区域
        derained = cv2.inpaint(image, (255 - rain_mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        return derained
    
    def _desnow(self, image):
        """快速去雪算法"""
        # 高斯模糊+阈值分割检测雪花
        blur = cv2.GaussianBlur(image, (5,5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, snow_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # 修复雪花区域
        desnowed = cv2.inpaint(image, snow_mask, 5, cv2.INPAINT_NS)
        return desnowed

# ==============================================================================
# 3. LiDAR处理模块（新增）
# ==============================================================================
class LiDARProcessor:
    """LiDAR点云处理与目标检测"""
    def __init__(self, config):
        self.config = config
        self.point_cloud_queue = queue.Queue(maxsize=2)
        self.lidar_data = None
        self.lidar_transform = None
        
    def lidar_callback(self, point_cloud):
        """LiDAR回调函数"""
        try:
            # 解析点云数据
            data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
            # x, y, z, intensity
            self.lidar_data = data[:, :3]
            self.lidar_transform = point_cloud.transform
            
            if self.point_cloud_queue.full():
                try:
                    self.point_cloud_queue.get_nowait()
                except:
                    pass
            self.point_cloud_queue.put((self.lidar_data.copy(), self.lidar_transform))
        except Exception as e:
            logger.warning(f"LiDAR回调错误: {str(e)[:30]}")
    
    def detect_objects_from_pointcloud(self, min_points=50, cluster_eps=0.8, min_cluster_size=30):
        """从点云聚类检测目标"""
        if self.lidar_data is None or len(self.lidar_data) < min_points:
            return []
        
        # 1. 地面分割（简单版：基于z轴高度）
        ground_mask = self.lidar_data[:, 2] < -1.0  # 地面点
        non_ground_points = self.lidar_data[~ground_mask]
        
        if len(non_ground_points) < min_points:
            return []
        
        # 2. DBSCAN聚类
        clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_size).fit(non_ground_points[:, :2])
        labels = clustering.labels_
        
        # 3. 提取聚类边界框
        detected_boxes = []
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_points = non_ground_points[labels == label]
            if len(cluster_points) < min_cluster_size:
                continue
            
            # 计算3D边界框
            min_x, min_y, min_z = cluster_points.min(axis=0)
            max_x, max_y, max_z = cluster_points.max(axis=0)
            
            # 转换为2D检测框（简化：投影到图像平面）
            detected_boxes.append({
                '3d_bbox': [min_x, min_y, min_z, max_x, max_y, max_z],
                'center': [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2],
                'size': [max_x-min_x, max_y-min_y, max_z-min_z],
                'num_points': len(cluster_points)
            })
        
        return detected_boxes
    
    def project_3d_to_2d(self, point_3d, camera_intrinsic, camera_extrinsic):
        """3D点投影到2D图像平面"""
        try:
            # 构建变换矩阵
            from scipy.spatial.transform import Rotation as R
            
            # 相机外参（世界到相机）
            cam_rot = R.from_quat([
                camera_extrinsic.rotation.x,
                camera_extrinsic.rotation.y,
                camera_extrinsic.rotation.z,
                camera_extrinsic.rotation.w
            ])
            cam_trans = np.array([
                camera_extrinsic.location.x,
                camera_extrinsic.location.y,
                camera_extrinsic.location.z
            ])
            
            # 点云到相机坐标系
            point_cam = cam_rot.apply(point_3d - cam_trans)
            
            # 剔除相机后方的点
            if point_cam[2] < 0.1:
                return None
            
            # 投影到图像平面
            fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
            cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
            
            u = fx * point_cam[0] / point_cam[2] + cx
            v = fy * point_cam[1] / point_cam[2] + cy
            
            return (int(u), int(v))
        except Exception as e:
            logger.warning(f"3D到2D投影错误: {str(e)[:30]}")
            return None
    
    def get_3d_visualization(self):
        """获取3D点云可视化"""
        if self.lidar_data is None:
            return None
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.lidar_data)
        
        # 颜色映射（z轴高度）
        z_min, z_max = self.lidar_data[:, 2].min(), self.lidar_data[:, 2].max()
        colors = (self.lidar_data[:, 2] - z_min) / (z_max - z_min + 1e-6)
        color_map = np.zeros((len(colors), 3))
        color_map[:, 0] = 1 - colors  # 蓝色（低）到红色（高）
        color_map[:, 2] = colors
        pcd.colors = o3d.utility.Vector3dVector(color_map)
        
        return pcd

# ==============================================================================
# 4. 数据记录模块（新增）
# ==============================================================================
class DataRecorder:
    """离线数据记录器"""
    def __init__(self, config):
        self.config = config
        self.record_dir = os.path.join(config.record_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.frame_count = 0
        self.record_files = {}
        
        # 创建记录目录
        if config.record_data:
            os.makedirs(self.record_dir, exist_ok=True)
            os.makedirs(os.path.join(self.record_dir, "screenshots"), exist_ok=True)
            logger.info(f"数据记录目录: {self.record_dir}")
            
            # 初始化记录文件
            self._init_record_files()
    
    def _init_record_files(self):
        """初始化记录文件"""
        # 跟踪结果记录
        if self.config.record_format == "csv":
            csv_path = os.path.join(self.record_dir, "track_results.csv")
            self.record_files['tracks'] = open(csv_path, 'w', newline='', encoding='utf-8')
            self.track_writer = csv.writer(self.record_files['tracks'])
            self.track_writer.writerow([
                'timestamp', 'frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2',
                'cls_id', 'cls_name', 'behavior', 'speed', 'confidence'
            ])
        
        # 系统性能记录
        perf_path = os.path.join(self.record_dir, "performance.csv")
        self.record_files['performance'] = open(perf_path, 'w', newline='', encoding='utf-8')
        self.perf_writer = csv.writer(self.record_files['performance'])
        self.perf_writer.writerow([
            'timestamp', 'frame_id', 'fps', 'cpu_usage', 'memory_usage',
            'gpu_usage', 'detection_count', 'track_count'
        ])
        
        # 配置文件备份
        config_path = os.path.join(self.record_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.__dict__, f, indent=2, default_flow_style=False)
    
    def record_track_data(self, tracks, detections, fps):
        """记录跟踪数据"""
        if not self.config.record_data:
            return
        
        # 控制记录帧率
        if self.frame_count % (self.config.display_fps // self.config.record_fps) != 0:
            self.frame_count += 1
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # 记录跟踪结果
        if tracks and len(tracks) > 0:
            for i, track in enumerate(tracks):
                try:
                    behavior = "normal"
                    if track.is_stopped:
                        behavior = "stopped"
                    elif track.is_overtaking:
                        behavior = "overtaking"
                    elif track.is_lane_changing:
                        behavior = "lane_changing"
                    elif track.is_braking:
                        behavior = "braking"
                    elif track.is_dangerous:
                        behavior = "dangerous"
                    
                    speed = track._calculate_speed() if hasattr(track, '_calculate_speed') else 0.0
                    
                    self.track_writer.writerow([
                        timestamp, self.frame_count, track.track_id,
                        track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3],
                        track.cls_id, self._get_cls_name(track.cls_id),
                        behavior, speed, track.conf if hasattr(track, 'conf') else 0.0
                    ])
                except Exception as e:
                    logger.warning(f"记录跟踪数据失败: {str(e)[:30]}")
        
        # 记录性能数据
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        self.perf_writer.writerow([
            timestamp, self.frame_count, fps,
            cpu_usage, memory_usage, gpu_usage,
            len(detections) if detections is not None else 0,
            len(tracks) if tracks is not None else 0
        ])
        
        # 刷新文件
        for f in self.record_files.values():
            f.flush()
        
        self.frame_count += 1
    
    def save_screenshot(self, image, weather):
        """保存截图"""
        if not self.config.save_screenshots or image is None:
            return
        
        screenshot_path = os.path.join(
            self.record_dir, "screenshots",
            f"screenshot_{weather}_{self.frame_count:06d}.png"
        )
        cv2.imwrite(screenshot_path, image)
    
    def _get_cls_name(self, cls_id):
        """获取类别名称"""
        cls_map = {2: "Car", 5: "Bus", 7: "Truck", -1: "Unknown"}
        return cls_map.get(cls_id, "Unknown")
    
    def close(self):
        """关闭记录文件"""
        if self.config.record_data:
            for f in self.record_files.values():
                try:
                    f.close()
                except:
                    pass
            logger.info(f"数据记录完成，文件保存在: {self.record_dir}")

# ==============================================================================
# 5. 核心算法（增强行为分析+轨迹预测）
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
        
        # 严格的边界检查 + 空值保护
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"bbox维度错误：期望4维，实际{len(bbox) if bbox is not None else 0}维")
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox
        
        # 轨迹记录
        self.track_history = []
        self._update_track_history()
        
        # 扩展行为分析相关
        self.speed_history = []
        self.accel_history = []
        self.heading_history = []
        self.lateral_displacement = []
        
        # 行为状态
        self.is_stopped = False
        self.is_overtaking = False
        self.is_lane_changing = False
        self.is_braking = False
        self.is_accelerating = False
        self.is_turning = False
        self.is_dangerous = False
        
        self.stop_frame_count = 0
        self.overtake_frame_count = 0
        self.lane_change_frame_count = 0
        self.brake_frame_count = 0
        self.turn_frame_count = 0
        
        # 轨迹预测
        self.predicted_trajectory = []
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.cls_id = None
        self.conf = 0.0

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
        
        # 记录横向位移（用于变道检测）
        if len(self.track_history) > 1:
            prev_x = self.track_history[-2][0]
            self.lateral_displacement.append(abs(center_x - prev_x))
        
        if len(self.track_history) > self.config.track_history_len:
            self.track_history.pop(0)
        if len(self.lateral_displacement) > 10:
            self.lateral_displacement.pop(0)

    def _calculate_speed(self):
        """计算目标速度（空数组保护）"""
        # 轨迹长度不足保护
        if len(self.track_history) < 2:
            return 0.0
        
        prev_center = self.track_history[-2]
        curr_center = self.track_history[-1]
        speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center)) / self.kf.dt
        self.speed_history.append(speed)
        
        # 计算加速度
        if len(self.speed_history) > 1:
            accel = (speed - self.speed_history[-2]) / self.kf.dt
            self.accel_history.append(accel)
        
        # 限制历史长度
        if len(self.speed_history) > 5:
            self.speed_history.pop(0)
        if len(self.accel_history) > 5:
            self.accel_history.pop(0)
        
        # 空数组保护
        return np.mean(self.speed_history) if self.speed_history else 0.0

    def _calculate_heading(self):
        """计算目标航向角"""
        if len(self.track_history) < 3:
            return 0.0
        
        # 计算最近3个点的航向
        dx = self.track_history[-1][0] - self.track_history[-3][0]
        dy = self.track_history[-1][1] - self.track_history[-3][1]
        heading = np.degrees(np.arctan2(dy, dx))
        self.heading_history.append(heading)
        
        if len(self.heading_history) > 5:
            self.heading_history.pop(0)
        
        return heading

    def _predict_trajectory(self):
        """预测未来轨迹（基于卡尔曼滤波）"""
        self.predicted_trajectory = []
        if len(self.track_history) < 5:
            return
        
        # 使用卡尔曼滤波预测未来N帧
        temp_kf = KalmanFilter(dt=self.kf.dt, max_speed=self.kf.max_speed)
        temp_kf.x = self.kf.x.copy()
        temp_kf.P = self.kf.P.copy()
        
        for _ in range(self.config.predict_frames):
            pred_bbox = temp_kf.predict()
            center_x = (pred_bbox[0] + pred_bbox[2]) / 2
            center_y = (pred_bbox[1] + pred_bbox[3]) / 2
            self.predicted_trajectory.append((center_x, center_y))

    def _analyze_behavior(self, ego_center):
        """增强版行为分析"""
        current_speed = self._calculate_speed()
        current_heading = self._calculate_heading()
        
        # 1. 停车检测
        if current_speed < self.config.stop_speed_thresh:
            self.stop_frame_count += 1
            self.is_stopped = self.stop_frame_count >= self.config.stop_frames_thresh
        else:
            self.stop_frame_count = 0
            self.is_stopped = False
        
        # 2. 超车检测
        if ego_center is not None and len(self.track_history) >= 2:
            target_center = self.track_history[-1]
            ego_center_np = np.array(ego_center)
            target_center_np = np.array(target_center)
            
            dist = np.linalg.norm(target_center_np - ego_center_np)
            if dist < self.config.overtake_dist_thresh:
                ego_speed = getattr(self, 'ego_speed', 0.0)
                if current_speed > ego_speed * self.config.overtake_speed_ratio:
                    self.overtake_frame_count += 1
                    self.is_overtaking = self.overtake_frame_count >= 3
                else:
                    self.overtake_frame_count = 0
                    self.is_overtaking = False
            else:
                self.overtake_frame_count = 0
                self.is_overtaking = False
        
        # 3. 变道检测
        if len(self.lateral_displacement) >= 5:
            avg_lateral = np.mean(self.lateral_displacement[-5:])
            if avg_lateral > self.config.lane_change_thresh:
                self.lane_change_frame_count += 1
                self.is_lane_changing = self.lane_change_frame_count >= 3
            else:
                self.lane_change_frame_count = 0
                self.is_lane_changing = False
        
        # 4. 刹车/加速检测
        if len(self.accel_history) >= 3:
            avg_accel = np.mean(self.accel_history[-3:])
            if avg_accel < -self.config.brake_accel_thresh:
                self.brake_frame_count += 1
                self.is_braking = self.brake_frame_count >= 2
                self.is_accelerating = False
            elif avg_accel > self.config.brake_accel_thresh:
                self.is_accelerating = True
                self.is_braking = False
                self.brake_frame_count = 0
            else:
                self.is_braking = False
                self.is_accelerating = False
                self.brake_frame_count = 0
        
        # 5. 转弯检测
        if len(self.heading_history) >= 3:
            heading_diff = np.abs(self.heading_history[-1] - self.heading_history[-3])
            if heading_diff > self.config.turn_angle_thresh:
                self.turn_frame_count += 1
                self.is_turning = self.turn_frame_count >= 2
            else:
                self.turn_frame_count = 0
                self.is_turning = False
        
        # 6. 危险行为检测（过近）
        if ego_center is not None:
            target_center = self.track_history[-1]
            dist = np.linalg.norm(np.array(target_center) - np.array(ego_center))
            self.is_dangerous = dist < self.config.danger_dist_thresh
        
        # 7. 预测未来轨迹
        self._predict_trajectory()

    def predict(self):
        # 空值保护：计算速度前先判断轨迹
        if len(self.track_history) >= 2:
            prev_center = np.array([(self.kf.x[0]+self.kf.x[2])/2, (self.kf.x[1]+self.kf.x[3])/2])
            curr_center = np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
            pixel_speed = np.linalg.norm(curr_center - prev_center) / self.kf.dt
            max_pixel_speed = max(self.img_shape) / self.kf.dt
            speed = min(1.0, pixel_speed / max_pixel_speed) * self.kf.max_speed
        else:
            speed = 0.0
        
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self._update_track_history()
        self.age += 1
        self.time_since_update += 1
        self.kf.update_noise_covariance(speed)
        return self.bbox

    def update(self, bbox, cls_id, conf=0.0, ego_center=None):
        # 空值保护
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"更新bbox维度错误：期望4维，实际{len(bbox) if bbox is not None else 0}维")
        
        self.bbox = self.kf.update(self._clip_bbox(bbox))
        self._update_track_history()
        self.hits += 1
        self.time_since_update = 0
        self.cls_id = cls_id
        self.conf = conf
        self._analyze_behavior(ego_center)

@njit
def _compute_iou_numba(box1, box2):
    """Numba加速的IOU计算"""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2-inter_x1) * max(0, inter_y2-inter_y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

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
        self.ego_speed = 0.0

    def update(self, detections, ego_center=None, lidar_detections=None):
        """更新跟踪器（支持LiDAR-视觉融合）"""
        self.ego_center = ego_center

        # 1. 空检测结果直接返回
        if detections is None or len(detections) == 0:
            # 如果有LiDAR检测，使用LiDAR结果
            if lidar_detections and len(lidar_detections) > 0 and self.config.fuse_lidar_vision:
                # 将LiDAR检测转换为2D框
                lidar_2d_detections = self._lidar_to_2d_detections(lidar_detections)
                detections = lidar_2d_detections
            
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return np.array([]), np.array([]), np.array([])

        # 2. 严格的格式校验
        valid_detections = []
        for det in detections:
            if len(det) >= 6:
                x1,y1,x2,y2,conf,cls_id = det[:6]
                if (isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and
                    isinstance(x2, (int, float)) and isinstance(y2, (int, float)) and
                    conf > 0 and x2 > x1 and y2 > y1):
                    valid_detections.append([x1,y1,x2,y2,conf,int(cls_id)])
        
        valid_detections = np.array(valid_detections, dtype=np.float32)

        # 3. 仍为空则返回
        if len(valid_detections) == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return np.array([]), np.array([]), np.array([])

        # 4. 预测所有轨迹
        for track in self.tracks:
            try:
                track.predict()
            except Exception as e:
                logger.warning(f"轨迹预测失败: {str(e)[:30]}")

        # 5. 无轨迹时初始化
        if len(self.tracks) == 0:
            for det in valid_detections:
                try:
                    self.tracks.append(Track(self.next_id, det[:4], self.img_shape, self.kf_config, self.config))
                    self.next_id += 1
                except Exception as e:
                    logger.warning(f"轨迹初始化失败: {str(e)[:30]}")
            return np.array([]), np.array([]), np.array([])

        # 6. 匈牙利算法匹配（Numba加速）
        try:
            iou_matrix = np.array([[_compute_iou_numba(t.bbox, d[:4]) for t in self.tracks] for d in valid_detections])
            cost_matrix = 1 - iou_matrix
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        except Exception as e:
            logger.warning(f"匹配算法失败: {str(e)[:30]}")
            track_indices, det_indices = [], []

        matches = []
        used_dets = set()
        used_tracks = set()

        # 7. 筛选有效匹配
        for t_idx, d_idx in zip(track_indices, det_indices):
            try:
                if (t_idx < len(self.tracks) and d_idx < len(valid_detections) and
                    _compute_iou_numba(self.tracks[t_idx].bbox, valid_detections[d_idx][:4]) > self.iou_threshold):
                    matches.append((t_idx, d_idx))
                    used_dets.add(d_idx)
                    used_tracks.add(t_idx)
            except Exception as e:
                continue

        # 8. 更新匹配的轨迹
        for track_idx, det_idx in matches:
            try:
                self.tracks[track_idx].update(
                    valid_detections[det_idx][:4], 
                    int(valid_detections[det_idx][5]),
                    valid_detections[det_idx][4],  # 置信度
                    self.ego_center
                )
                # 设置自车速度用于超车检测
                self.tracks[track_idx].ego_speed = self.ego_speed
            except Exception as e:
                logger.warning(f"轨迹更新失败: {str(e)[:30]}")

        # 9. 新增未匹配的检测
        for det_idx in set(range(len(valid_detections))) - used_dets:
            try:
                self.tracks.append(Track(self.next_id, valid_detections[det_idx][:4], self.img_shape, self.kf_config, self.config))
                self.next_id += 1
            except Exception as e:
                logger.warning(f"新增轨迹失败: {str(e)[:30]}")

        # 10. 清理过期轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 11. 返回有效轨迹（空值保护）
        valid_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        return self._format_results(valid_tracks)

    def _lidar_to_2d_detections(self, lidar_detections):
        """将LiDAR检测转换为2D检测框（简化版）"""
        detections = []
        for det in lidar_detections:
            # 简单转换：基于3D中心生成2D框
            center_3d = det['center']
            # 假设比例转换
            x1 = center_3d[0] * 10 + self.img_shape[1]/2
            y1 = center_3d[1] * 10 + self.img_shape[0]/2
            x2 = x1 + det['size'][0] * 5
            y2 = y1 + det['size'][1] * 5
            
            detections.append([x1, y1, x2, y2, 0.8, 2])  # 置信度0.8，类别car
        return np.array(detections)

    def _format_results(self, tracks):
        """格式化结果（全空值保护）"""
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
# 6. 推理线程类（优化）
# ==============================================================================
class DetectionThread(threading.Thread):
    def __init__(self, detector, config, enhancer, input_queue, output_queue, device="cpu"):
        super().__init__(daemon=True)
        self.detector = detector
        self.config = config
        self.enhancer = enhancer
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.device = device

    def run(self):
        while self.running:
            try:
                image = self.input_queue.get(timeout=1.0)
                
                # 1. 空图像/无效图像保护
                if image is None or len(image.shape) != 3 or image.shape[2] != 3 or image.size == 0:
                    self.output_queue.put((None, np.array([])))
                    continue
                
                # 2. 天气自适应图像增强
                image_enhanced = self.enhancer.enhance(image)
                
                # 3. 尺寸计算（空值保护）
                h, w = image.shape[:2]
                if w == 0 or h == 0:
                    self.output_queue.put((image, np.array([])))
                    continue
                
                # 确保尺寸是32的整数倍
                def make_divisible(x, divisor=32):
                    return (x + divisor - 1) // divisor * divisor
                ratio = min(self.config.yolo_imgsz_max / w, self.config.yolo_imgsz_max / h)
                imgsz_w = make_divisible(int(w * ratio))
                imgsz_h = make_divisible(int(h * ratio))
                
                # 4. YOLO推理（异常保护）
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

                # 5. 解析检测结果（全空值保护）
                detections = []
                for r in results:
                    # boxes为空保护
                    if not hasattr(r, 'boxes') or r.boxes is None or len(r.boxes) == 0:
                        continue
                    
                    for box in r.boxes:
                        try:
                            # 空值保护
                            if box.cls is None or box.conf is None or box.xyxy is None:
                                continue
                            
                            cls_id = int(box.cls[0])
                            # 仅保留车辆类
                            if cls_id in {2,5,7}:
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                # 检测框合法性检查
                                if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1] and conf > 0:
                                    detections.append([*xyxy, conf, cls_id])
                        except Exception as e:
                            continue
                
                # 6. 转为数组（空值保护）
                detections_np = np.array(detections, dtype=np.float32) if detections else np.array([])
                self.output_queue.put((image, detections_np))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"推理线程错误: {str(e)[:50]}")
                self.output_queue.put((None, np.array([])))

    def stop(self):
        self.running = False

# ==============================================================================
# 7. 工具类（增强可视化+性能监控）
# ==============================================================================
class FrameBuffer:
    def __init__(self, default_size=(480, 640, 3)):
        self.default_frame = np.zeros(default_size, dtype=np.uint8)
        cv2.putText(self.default_frame, "Initializing...", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.current_frame = self.default_frame.copy()
        self.lock = threading.Lock()

    def update(self, frame):
        """线程安全更新帧（空值保护）"""
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3 and frame.size > 0:
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

class FPSCounter:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.times = []
        self.fps = 0.0

    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)
        # 空值保护
        if len(self.times) >= 2:
            self.fps = (len(self.times)-1) / (self.times[-1] - self.times[0])
        return self.fps

class PerformanceMonitor:
    """系统性能监控"""
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.fps_history = []
        self.max_history = 30
    
    def update(self, fps):
        """更新性能数据"""
        # CPU使用率
        cpu = psutil.cpu_percent()
        # 内存使用率
        memory = psutil.virtual_memory().percent
        # GPU使用率
        gpu = 0.0
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            except:
                gpu = 0.0
        
        # 更新历史
        self.cpu_history.append(cpu)
        self.memory_history.append(memory)
        self.gpu_history.append(gpu)
        self.fps_history.append(fps)
        
        # 限制历史长度
        if len(self.cpu_history) > self.max_history:
            self.cpu_history.pop(0)
            self.memory_history.pop(0)
            self.gpu_history.pop(0)
            self.fps_history.pop(0)
        
        return {
            'cpu': cpu,
            'memory': memory,
            'gpu': gpu,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0.0
        }

def draw_bounding_boxes(image, boxes, ids, cls_ids, tracks, fps=0.0, detection_count=0, config=None, current_weather="clear", perf_data=None):
    """增强版可视化绘制"""
    # 空图像保护
    if image is None or len(image.shape) != 3 or image.shape[2] != 3 or image.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    if config is None:
        config = Config()
    
    display_img = np.empty_like(image)
    display_img[:] = image
    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    
    # 绘制顶部信息栏
    overlay = display_img.copy()
    cv2.rectangle(overlay, (10,10), (800, 80), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
    
    # 统计行为数量
    stop_count = sum(1 for t in tracks if t.is_stopped) if tracks else 0
    overtake_count = sum(1 for t in tracks if t.is_overtaking) if tracks else 0
    lane_change_count = sum(1 for t in tracks if t.is_lane_changing) if tracks else 0
    brake_count = sum(1 for t in tracks if t.is_braking) if tracks else 0
    danger_count = sum(1 for t in tracks if t.is_dangerous) if tracks else 0
    
    # 基础信息
    info_line1 = f"FPS:{fps:.1f} | Weather:{current_weather} | Tracks:{len(boxes)} | Dets:{detection_count}"
    info_line2 = f"Stop:{stop_count} | Overtake:{overtake_count} | LaneChange:{lane_change_count} | Brake:{brake_count} | Danger:{danger_count}"
    
    # 性能信息
    if perf_data:
        info_line3 = f"CPU:{perf_data['cpu']:.1f}% | MEM:{perf_data['memory']:.1f}% | GPU:{perf_data['gpu']:.1f}%"
    else:
        info_line3 = ""
    
    cv2.putText(display_img, info_line1, (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, lineType=cv2.LINE_AA)
    cv2.putText(display_img, info_line2, (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, lineType=cv2.LINE_AA)
    if info_line3:
        cv2.putText(display_img, info_line3, (15,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, lineType=cv2.LINE_AA)
    
    # 空值保护
    if boxes is None or ids is None or cls_ids is None or tracks is None:
        return display_img
    
    min_len = min(len(boxes), len(ids), len(cls_ids), len(tracks))
    if min_len == 0:
        return display_img
    
    # 绘制跟踪框+轨迹+行为标签+预测轨迹
    for i in range(min_len):
        try:
            box = boxes[i]
            track_id = ids[i]
            cls_id = cls_ids[i]
            track = tracks[i]
            
            # 空值保护
            if box is None or len(box) != 4:
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
            
            # 绘制历史轨迹
            if track and len(track.track_history) >= 2:
                track_overlay = display_img.copy()
                for j in range(1, len(track.track_history)):
                    pt1 = (int(track.track_history[j-1][0]), int(track.track_history[j-1][1]))
                    pt2 = (int(track.track_history[j][0]), int(track.track_history[j][1]))
                    alpha = j / len(track.track_history) * config.track_alpha
                    line_width = int(j / len(track.track_history) * config.track_line_width) + 1
                    cv2.line(track_overlay, pt1, pt2, color, line_width, lineType=cv2.LINE_AA)
                cv2.addWeighted(track_overlay, alpha, display_img, 1-alpha, 0, display_img)
            
            # 绘制预测轨迹（红色虚线）
            if track and len(track.predicted_trajectory) >= 2:
                pred_overlay = display_img.copy()
                for j in range(1, len(track.predicted_trajectory)):
                    pt1 = (int(track.predicted_trajectory[j-1][0]), int(track.predicted_trajectory[j-1][1]))
                    pt2 = (int(track.predicted_trajectory[j][0]), int(track.predicted_trajectory[j][1]))
                    cv2.line(pred_overlay, pt1, pt2, (0,0,255), 2, lineType=cv2.LINE_AA, shift=0)
                cv2.addWeighted(pred_overlay, 0.5, display_img, 0.5, 0, display_img)
            
            # 构建行为标签
            cls_name = vehicle_classes.get(cls_id, "Unknown")
            behavior_tags = []
            if track.is_stopped:
                behavior_tags.append("STOP")
            if track.is_overtaking:
                behavior_tags.append("OVERTAKE")
            if track.is_lane_changing:
                behavior_tags.append("LANE_CHANGE")
            if track.is_braking:
                behavior_tags.append("BRAKE")
            if track.is_accelerating:
                behavior_tags.append("ACCEL")
            if track.is_turning:
                behavior_tags.append("TURN")
            if track.is_dangerous:
                behavior_tags.append("DANGER!")
            
            behavior_str = " | " + " | ".join(behavior_tags) if behavior_tags else ""
            speed = track._calculate_speed() if hasattr(track, '_calculate_speed') else 0.0
            label = f"ID:{track_id} | {cls_name} | {speed:.1f}px/s{behavior_str}"
            
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
    """相机回调（空值保护）"""
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
                # 兼容低版本CARLA的autopilot设置
                try:
                    npc.set_autopilot(True, tm_port=8000)
                except TypeError:
                    npc.set_autopilot(True)
                npc_count += 1
    
    logger.info(f"生成NPC车辆：{npc_count} 辆 (目标: {num_npcs})")
    return npc_count

# ==============================================================================
# 8. 主函数（完整）
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
        client.set_timeout(20.0)
        world = client.get_world()
        
        # 启动交通管理器
        try:
            tm = client.get_trafficmanager(8000)
            tm.set_global_distance_to_leading_vehicle(2.0)
            tm.set_respawn_dormant_vehicles(True)
            tm.set_hybrid_physics_mode(True)
            tm.set_hybrid_physics_radius(50.0)
            tm.global_percentage_speed_difference(0)
            logger.info("✅ 交通管理器已启动（端口8000）")
        except Exception as e:
            logger.error(f"❌ 交通管理器启动失败：{e}")
        
        # 设置同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        logger.info("✅ CARLA同步模式已启用")
        
    except Exception as e:
        logger.error(f"CARLA连接失败：{e}")
        return

    # 初始化资源
    ego_vehicle = None
    camera = None
    lidar = None
    lidar_processor = None
        # 初始化资源
    ego_vehicle = None
    camera = None
    lidar = None
    lidar_processor = None
    detection_thread = None
    data_recorder = DataRecorder(config)
    perf_monitor = PerformanceMonitor()
    
    try:
        # 设置天气
        world.set_weather(WEATHER_PRESETS[config.default_weather])
        weather_enhancer = WeatherImageEnhancer(config)
        weather_enhancer.set_weather(config.default_weather)
        current_weather = config.default_weather
        
        # 生成自车
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            logger.error("❌ 无可用生成点")
            return
        
        ego_bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
        ego_bp.set_attribute('color', '255,0,0')
        ego_vehicle = world.spawn_actor(ego_bp, spawn_points[0])
        ego_vehicle.set_autopilot(True, tm_port=8000)
        logger.info(f"✅ 自车生成成功 (ID: {ego_vehicle.id})")
        
        # 生成NPC车辆
        spawn_npc_vehicles(world, config.num_npcs, spawn_points[1:])
        
        # 初始化相机传感器
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(config.img_width))
        camera_bp.set_attribute('image_size_y', str(config.img_height))
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')
        
        # 相机安装位置（自车顶部）
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.8))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera_queue = queue.Queue(maxsize=1)
        camera.listen(lambda img: camera_callback(img, camera_queue))
        logger.info(f"✅ 相机传感器启动成功 (ID: {camera.id})")
        
        # 初始化LiDAR传感器（如果启用）
        if config.use_lidar:
            lidar_processor = LiDARProcessor(config)
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', str(config.lidar_channels))
            lidar_bp.set_attribute('range', str(config.lidar_range))
            lidar_bp.set_attribute('points_per_second', str(config.lidar_points_per_second))
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('sensor_tick', '0.05')
            
            lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
            lidar.listen(lidar_processor.lidar_callback)
            logger.info(f"✅ LiDAR传感器启动成功 (ID: {lidar.id})")
        
        # 初始化YOLO检测器
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✅ 使用设备: {device}")
        
        # 加载YOLO模型（支持量化）
        model = YOLO(config.yolo_model)
        if config.yolo_quantize and device == "cuda":
            model = model.quantize()
            logger.info("✅ YOLO模型已量化")
        
        # 初始化推理线程
        input_queue = queue.Queue(maxsize=2)
        output_queue = queue.Queue(maxsize=2)
        detection_thread = DetectionThread(
            detector=model,
            config=config,
            enhancer=weather_enhancer,
            input_queue=input_queue,
            output_queue=output_queue,
            device=device
        )
        detection_thread.start()
        logger.info("✅ 推理线程已启动")
        
        # 初始化跟踪器
        tracker = SimpleSORT(config)
        
        # 可视化工具初始化
        frame_buffer = FrameBuffer((config.img_height, config.img_width, 3))
        fps_counter = FPSCounter(config.fps_window_size)
        display_controller = FixedRateDisplay(config.display_fps)
        cv2.namedWindow("CARLA Object Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA Object Tracking", config.window_width, config.window_height)
        
        # 3D可视化窗口（如果启用）
        vis = None
        if config.use_3d_visualization and config.use_lidar:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="LiDAR Point Cloud", width=config.pcd_view_size, height=config.pcd_view_size)
            logger.info("✅ 3D点云可视化窗口已启动")
        
        # 主循环
        logger.info("🚀 开始跟踪（按ESC退出）")
        frame_count = 0
        while True:
            # 同步CARLA世界
            world.tick()
            
            # 获取相机图像
            try:
                img_rgb = camera_queue.get(timeout=1.0)
                frame_buffer.update(img_rgb)
            except queue.Empty:
                logger.warning("⚠️ 相机队列空，使用缓存帧")
                img_rgb = frame_buffer.get()
            
            # 提交图像到推理线程
            if not input_queue.full():
                input_queue.put(img_rgb.copy())
            
            # 获取检测结果
            detections = np.array([])
            try:
                _, detections = output_queue.get_nowait()
            except queue.Empty:
                pass
            
            # 获取LiDAR检测结果（如果启用）
            lidar_detections = []
            if config.use_lidar and lidar_processor:
                lidar_detections = lidar_processor.detect_objects_from_pointcloud()
            
            # 计算自车中心（图像中心）
            ego_center = (config.img_width // 2, config.img_height // 2)
            
            # 更新跟踪器
            boxes, ids, cls_ids = tracker.update(detections, ego_center, lidar_detections)
            
            # 更新FPS和性能监控
            fps = fps_counter.update()
            perf_data = perf_monitor.update(fps)
            
            # 绘制可视化结果
            display_img = draw_bounding_boxes(
                img_rgb, boxes, ids, cls_ids, tracker.tracks,
                fps=fps, detection_count=len(detections),
                config=config, current_weather=current_weather,
                perf_data=perf_data
            )
            
            # 显示图像
            cv2.imshow("CARLA Object Tracking", display_img)
            
            # 3D点云可视化（如果启用）
            if config.use_3d_visualization and vis and lidar_processor:
                pcd = lidar_processor.get_3d_visualization()
                if pcd:
                    vis.clear_geometries()
                    vis.add_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
            
            # 数据记录
            data_recorder.record_track_data(tracker.tracks, detections, fps)
            if config.save_screenshots and frame_count % 30 == 0:
                data_recorder.save_screenshot(display_img, current_weather)
            
            # 控制显示帧率
            display_controller.wait()
            
            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                logger.info("🛑 用户按下ESC，退出程序")
                break
            elif key == ord('w'):  # 切换天气
                weather_list = list(WEATHER_PRESETS.keys())
                current_idx = weather_list.index(current_weather)
                current_weather = weather_list[(current_idx + 1) % len(weather_list)]
                world.set_weather(WEATHER_PRESETS[current_weather])
                weather_enhancer.set_weather(current_weather)
                logger.info(f"🌤️ 切换天气到: {current_weather}")
            
            frame_count += 1
        
    except KeyboardInterrupt:
        logger.info("🛑 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 主循环错误: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        logger.info("🧹 清理资源...")
        
        # 停止推理线程
        if detection_thread:
            detection_thread.stop()
            detection_thread.join(timeout=2.0)
        
        # 关闭3D可视化窗口
        if vis:
            vis.destroy_window()
        
        # 关闭CV窗口
        cv2.destroyAllWindows()
        
        # 停止数据记录
        data_recorder.close()
        
        # 销毁传感器和车辆
        if lidar:
            lidar.stop()
            lidar.destroy()
        if camera:
            camera.stop()
            camera.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        
        # 清理所有NPC
        clear_actors(world)
        
        # 恢复CARLA设置
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        logger.info("✅ 资源清理完成，程序退出")

if __name__ == "__main__":
    main()