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
import threading
from scipy.optimize import linear_sum_assignment
import torch  # 新增：用于设备检测和模型优化

# 简化日志配置
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 配置类（新增轨迹/行为分析配置）
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
    yolo_imgsz_max: int = 320  # 降低尺寸提升速度
    yolo_iou: float = 0.45
    kf_dt: float = 0.05
    max_speed: float = 50.0
    window_width: int = 1280
    window_height: int = 720
    smooth_alpha: float = 0.2
    fps_window_size: int = 15
    display_fps: int = 30  # 固定显示帧率
    
    # 新增：轨迹可视化配置
    track_history_len: int = 20  # 轨迹最大长度
    track_line_width: int = 2    # 轨迹线宽
    track_alpha: float = 0.6     # 轨迹透明度
    
    # 新增：行为分析配置
    stop_speed_thresh: float = 1.0  # 停车速度阈值（像素/帧）
    stop_frames_thresh: int = 5     # 判定停车的连续帧数
    overtake_speed_ratio: float = 1.5  # 超车速度比（目标/自车）
    overtake_dist_thresh: float = 50   # 超车判定距离（像素）

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
# 核心算法（新增轨迹记录+行为分析）
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
        self.config = config  # 新增：保存配置
        
        # 严格的边界检查
        if len(bbox) != 4:
            raise ValueError(f"bbox维度错误：期望4维，实际{len(bbox)}维")
        self.bbox = self._clip_bbox(bbox.astype(np.float32))
        self.kf.x[:4] = self.bbox
        
        # 新增：轨迹记录
        self.track_history = []  # 存储目标中心坐标 [(x, y), ...]
        self._update_track_history()
        
        # 新增：行为分析相关
        self.speed_history = []  # 速度历史
        self.is_stopped = False  # 是否停车
        self.stop_frame_count = 0  # 连续停车帧数
        self.is_overtaking = False  # 是否正在超车
        self.overtake_frame_count = 0  # 连续超车帧数
        
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
        """更新轨迹历史（仅保留最新N帧）"""
        center_x = (self.bbox[0] + self.bbox[2]) / 2
        center_y = (self.bbox[1] + self.bbox[3]) / 2
        self.track_history.append((center_x, center_y))
        # 限制轨迹长度
        if len(self.track_history) > self.config.track_history_len:
            self.track_history.pop(0)

    def _calculate_speed(self):
        """计算目标速度（像素/帧）"""
        if len(self.track_history) < 2:
            return 0.0
        prev_center = self.track_history[-2]
        curr_center = self.track_history[-1]
        speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center)) / self.kf.dt
        self.speed_history.append(speed)
        # 限制速度历史长度
        if len(self.speed_history) > 5:
            self.speed_history.pop(0)
        return np.mean(self.speed_history)  # 平滑速度

    def _analyze_behavior(self, ego_center):
        """分析目标行为：停车/超车"""
        # 1. 停车检测
        current_speed = self._calculate_speed()
        if current_speed < self.config.stop_speed_thresh:
            self.stop_frame_count += 1
            self.is_stopped = self.stop_frame_count >= self.config.stop_frames_thresh
        else:
            self.stop_frame_count = 0
            self.is_stopped = False
        
        # 2. 超车检测
        if ego_center is None or len(self.track_history) < 2:
            self.is_overtaking = False
            return
        
        # 计算目标与自车的相对位置和速度
        target_center = self.track_history[-1]
        ego_center_np = np.array(ego_center)
        target_center_np = np.array(target_center)
        
        # 距离判断
        dist = np.linalg.norm(target_center_np - ego_center_np)
        if dist > self.config.overtake_dist_thresh:
            self.overtake_frame_count = 0
            self.is_overtaking = False
            return
        
        # 速度判断（目标速度 > 自车速度 * 阈值）
        ego_speed = 0.0
        if hasattr(self, 'ego_speed') and self.ego_speed > 0:
            if current_speed > self.ego_speed * self.config.overtake_speed_ratio:
                self.overtake_frame_count += 1
                self.is_overtaking = self.overtake_frame_count >= 3  # 连续3帧判定超车
            else:
                self.overtake_frame_count = 0
                self.is_overtaking = False

    def predict(self):
        # 优化速度计算：考虑xy轴+时间维度
        prev_center = np.array([(self.kf.x[0]+self.kf.x[2])/2, (self.kf.x[1]+self.kf.x[3])/2])
        curr_center = np.array([(self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2])
        pixel_speed = np.linalg.norm(curr_center - prev_center) / self.kf.dt
        max_pixel_speed = max(self.img_shape) / self.kf.dt
        speed = min(1.0, pixel_speed / max_pixel_speed) * self.kf.max_speed
        
        self.bbox = self.kf.predict()
        self.bbox = self._clip_bbox(self.bbox)
        self._update_track_history()  # 新增：预测后更新轨迹
        self.age += 1
        self.time_since_update += 1
        self.kf.update_noise_covariance(speed)
        return self.bbox

    def update(self, bbox, cls_id, ego_center=None):
        """新增ego_center参数，用于行为分析"""
        if len(bbox) != 4:
            raise ValueError(f"更新bbox维度错误：期望4维，实际{len(bbox)}维")
        self.bbox = self.kf.update(self._clip_bbox(bbox))
        self._update_track_history()  # 新增：更新后更新轨迹
        self.hits += 1
        self.time_since_update = 0
        self.cls_id = cls_id
        
        # 新增：行为分析
        self._analyze_behavior(ego_center)

class SimpleSORT:
    def __init__(self, config):
        self.max_age = config.max_age
        self.min_hits = config.min_hits
        self.iou_threshold = config.iou_thres
        self.img_shape = (config.img_height, config.img_width)
        self.kf_config = {"dt": config.kf_dt, "max_speed": config.max_speed}
        self.config = config  # 保存配置
        self.tracks = []
        self.next_id = 1
        
        # 新增：自车中心（用于超车检测）
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
        """
        安全的更新函数：严格校验检测结果格式
        detections: np.array 每行格式 [x1,y1,x2,y2,conf,cls_id]
        ego_center: tuple (x, y) 自车中心坐标（新增）
        """
        # 新增：更新自车中心
        self.ego_center = ego_center

        # 1. 严格的格式校验
        valid_detections = []
        if detections is not None and len(detections) > 0:
            for det in detections:
                # 检查维度和数值有效性
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

        # 5. 匈牙利算法匹配（安全版）
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

        # 7. 更新匹配的轨迹（新增ego_center）
        for track_idx, det_idx in matches:
            try:
                self.tracks[track_idx].update(valid_detections[det_idx][:4], 
                                             int(valid_detections[det_idx][5]), 
                                             self.ego_center)
            except Exception as e:
                logger.warning(f"轨迹更新失败: {str(e)[:30]}")

        # 8. 新增未匹配的检测（新增config参数）
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
# 推理线程类（优化YOLO设备选择）
# ==============================================================================
class DetectionThread(threading.Thread):
    def __init__(self, detector, config, input_queue, output_queue, device="cpu"):
        super().__init__(daemon=True)
        self.detector = detector
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.device = device  # 新增：指定推理设备

    def run(self):
        while self.running:
            try:
                image = self.input_queue.get(timeout=1.0)
                # 图像格式校验
                if image is None or len(image.shape) != 3 or image.shape[2] != 3:
                    self.output_queue.put((None, np.array([])))
                    continue
                
                h, w = image.shape[:2]
                
                # 确保尺寸是32的整数倍
                def make_divisible(x, divisor=32):
                    return (x + divisor - 1) // divisor * divisor
                ratio = min(self.config.yolo_imgsz_max / w, self.config.yolo_imgsz_max / h)
                imgsz_w = make_divisible(int(w * ratio))
                imgsz_h = make_divisible(int(h * ratio))
                
                # YOLO推理（带异常捕获，优化设备）
                try:
                    results = self.detector.predict(
                        image,
                        conf=self.config.conf_thres,
                        verbose=False,
                        device=self.device,  # 使用指定设备
                        agnostic_nms=True,
                        imgsz=(imgsz_h, imgsz_w),
                        iou=self.config.yolo_iou
                    )
                except Exception as e:
                    logger.warning(f"YOLO推理失败: {str(e)[:30]}")
                    self.output_queue.put((image, np.array([])))
                    continue

                # 解析检测结果（安全版）
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
# 帧缓存+固定帧率管理器（无修改）
# ==============================================================================
class FrameBuffer:
    def __init__(self, default_size=(480, 640, 3)):
        self.default_frame = np.zeros(default_size, dtype=np.uint8)
        cv2.putText(self.default_frame, "Initializing...", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.current_frame = self.default_frame.copy()
        self.lock = threading.Lock()

    def update(self, frame):
        """线程安全更新帧（带格式校验）"""
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            with self.lock:
                self.current_frame = frame.copy()

    def get(self):
        """获取当前帧（无锁快速读取）"""
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
# 工具函数（新增轨迹绘制+行为标签）
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

def draw_bounding_boxes(image, boxes, ids, cls_ids, tracks, fps=0.0, detection_count=0, config=None):
    """新增：轨迹绘制+行为标签"""
    # 1. 图像格式校验
    if image is None or len(image.shape) != 3 or image.shape[2] != 3:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if config is None:
        config = Config()
    
    # 2. 预分配画布，避免频繁copy
    display_img = np.empty_like(image)
    display_img[:] = image  # 更快的复制方式
    vehicle_classes = {2: "Car", 5: "Bus", 7: "Truck"}
    
    # 3. 绘制FPS和跟踪数（半透明背景，减少闪烁）
    overlay = display_img.copy()
    cv2.rectangle(overlay, (10,10), (350,40), (0,0,0), -1)  # 加宽背景以显示行为统计
    cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)  # 半透明叠加
    
    # 新增：统计行为数量
    stop_count = sum(1 for t in tracks if t.is_stopped)
    overtake_count = sum(1 for t in tracks if t.is_overtaking)
    cv2.putText(display_img, 
                f"FPS:{fps:.1f} | Tracks:{len(boxes)} | Dets:{detection_count} | Stop:{stop_count} | Overtake:{overtake_count}", 
                (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, lineType=cv2.LINE_AA)
    
    # 4. 绘制跟踪框+轨迹+行为标签（带严格格式校验）
    if len(boxes) > 0 and len(ids) > 0 and len(cls_ids) > 0 and len(tracks) > 0:
        min_len = min(len(boxes), len(ids), len(cls_ids), len(tracks))
        for i in range(min_len):
            try:
                box = boxes[i]
                track_id = ids[i]
                cls_id = cls_ids[i]
                track = tracks[i]
                
                # 校验box格式
                if len(box) != 4:
                    continue
                
                x1,y1,x2,y2 = box
                # 校验坐标有效性
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # 固定颜色（基于ID的哈希，避免闪烁）
                color = (
                    (track_id * 59) % 256,
                    (track_id * 127) % 256,
                    (track_id * 199) % 256
                )
                # 抗锯齿绘制框
                cv2.rectangle(display_img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2, lineType=cv2.LINE_AA)
                
                # 新增：绘制轨迹
                if len(track.track_history) >= 2:
                    # 创建轨迹叠加层（透明效果）
                    track_overlay = display_img.copy()
                    # 绘制轨迹线（最新点更亮）
                    for j in range(1, len(track.track_history)):
                        pt1 = (int(track.track_history[j-1][0]), int(track.track_history[j-1][1]))
                        pt2 = (int(track.track_history[j][0]), int(track.track_history[j][1]))
                        # 轨迹渐变（越新越粗/越亮）
                        alpha = j / len(track.track_history) * config.track_alpha
                        line_width = int(j / len(track.track_history) * config.track_line_width) + 1
                        cv2.line(track_overlay, pt1, pt2, color, line_width, lineType=cv2.LINE_AA)
                    # 叠加轨迹到主图像
                    cv2.addWeighted(track_overlay, alpha, display_img, 1-alpha, 0, display_img)
                
                # 构建标签（新增行为信息）
                cls_name = vehicle_classes.get(cls_id, "Unknown")
                behavior_tags = []
                if track.is_stopped:
                    behavior_tags.append("STOP")
                if track.is_overtaking:
                    behavior_tags.append("OVERTAKE")
                behavior_str = " | " + " | ".join(behavior_tags) if behavior_tags else ""
                label = f"ID:{track_id} | {cls_name}{behavior_str}"
                
                # 绘制标签背景（半透明）
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                overlay = display_img.copy()
                # 加宽标签背景以容纳行为信息
                cv2.rectangle(overlay, (int(x1),int(y1)-20), (int(x1)+label_size[0]+20, int(y1)), color, -1)
                cv2.addWeighted(overlay, 0.8, display_img, 0.2, 0, display_img)
                cv2.putText(display_img, label, (int(x1)+5, int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, lineType=cv2.LINE_AA)
            except Exception as e:
                continue
    
    return display_img

def clear_actors(world, exclude=None):
    """优化的actor清理函数，避免内存泄漏"""
    exclude_ids = set(exclude) if exclude else set()
    actors = world.get_actors()
    batch_size = 10  # 分批销毁
    
    # 分类销毁
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
        # 转换图像格式并去噪
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        img_rgb = cv2.GaussianBlur(img_array[:,:,:3], (3,3), 0)
        
        # 队列满时丢弃旧数据
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
    max_attempts = num_npcs * 3  # 最大尝试次数
    
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
# 主函数（完整优化版）
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

    # 命令行参数覆盖配置
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.conf_thres:
        config.conf_thres = args.conf_thres

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

        # 初始化队列、帧缓存、固定帧率器
        image_queue = queue.Queue(maxsize=3)
        camera.listen(lambda img: camera_callback(img, image_queue))
        frame_buffer = FrameBuffer(default_size=(config.img_height, config.img_width, 3))
        display_controller = FixedRateDisplay(fps=config.display_fps)
        
        # 初始化FPS计数器、跟踪器
        fps_counter = FPSCounter(window_size=config.fps_window_size)
        tracker = SimpleSORT(config)
        
        # 初始化YOLO推理线程（优化版）
        # 自动选择设备和模型
        model_name = "yolov5su.pt"  # 升级为Ultralytics优化版
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用YOLO模型: {model_name}, 推理设备: {device}")

        # 初始化检测器并优化
        detector = YOLO(model_name)
        try:
            detector.fuse()  # 融合层加速（仅GPU有效）
            detector.to(device)
        except Exception as e:
            logger.warning(f"模型优化失败: {e}，使用默认配置")

        det_input_queue = queue.Queue(maxsize=2)
        det_output_queue = queue.Queue(maxsize=2)
        det_thread = DetectionThread(detector, config, det_input_queue, det_output_queue, device)
        det_thread.start()

        # 生成NPC车辆
        spawn_npc_vehicles(world, config.num_npcs, spawn_points)

        # 初始化可视化窗口（Windows兼容版）
        cv2.namedWindow("CARLA Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("CARLA Tracking", config.window_width, config.window_height)
        cv2.setWindowProperty("CARLA Tracking", cv2.WND_PROP_TOPMOST, 1)
        
        # 预显示默认帧，避免窗口空白
        initial_frame = frame_buffer.get()
        initial_frame_resized = cv2.resize(initial_frame, (config.window_width, config.window_height), 
                                          interpolation=cv2.INTER_LINEAR)
        cv2.imshow("CARLA Tracking", initial_frame_resized)
        cv2.waitKey(1)

        # 主循环
        logger.info("程序启动成功，按Q/ESC退出，S保存截图")
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

            # 核心帧处理逻辑（安全版）
            current_display_frame = frame_buffer.get()
            try:
                # 1. 非阻塞获取相机图像
                try:
                    image = image_queue.get_nowait()
                    # 图像格式校验
                    if image is not None and len(image.shape) == 3 and image.shape[2] == 3:
                        # 计算自车中心（图像中心，用于超车检测）
                        ego_center = (image.shape[1] / 2, image.shape[0] / 2)
                        tracker.ego_center = ego_center
                        # 将图像放入推理队列
                        if not det_input_queue.full():
                            det_input_queue.put(image.copy())
                except queue.Empty:
                    pass

                # 2. 非阻塞获取推理结果
                try:
                    img, detections = det_output_queue.get_nowait()
                    # 严格的格式校验
                    if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
                        # 计算自车中心
                        ego_center = (img.shape[1] / 2, img.shape[0] / 2)
                        # 更新跟踪器（传入自车中心）
                        tracked_boxes, tracked_ids, tracked_cls = tracker.update(detections, ego_center)
                        # 绘制结果（传入轨迹列表和配置）
                        display_img = draw_bounding_boxes(
                            img, tracked_boxes, tracked_ids, tracked_cls,
                            tracker.tracks,  # 新增：传入完整轨迹信息
                            fps_counter.update(), len(detections),
                            config  # 新增：传入配置
                        )
                        # 更新帧缓存
                        frame_buffer.update(display_img)
                        current_display_frame = display_img
                except queue.Empty:
                    # 无推理结果时，在缓存帧上绘制提示（不替换缓存）
                    current_display_frame = frame_buffer.get()
                    cv2.putText(current_display_frame, "Detecting...", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, lineType=cv2.LINE_AA)

                # 3. 固定帧率刷新窗口（统一尺寸）
                if current_display_frame is not None and len(current_display_frame.shape) == 3:
                    display_frame_resized = cv2.resize(current_display_frame, 
                                                      (config.window_width, config.window_height),
                                                      interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("CARLA Tracking", display_frame_resized)
                display_controller.wait()

                # 4. 键盘控制（独立处理，避免阻塞）
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q/ESC退出
                    logger.info("用户请求退出")
                    break
                elif key == ord('s'):  # S保存截图
                    save_path = f"track_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(save_path, frame_buffer.get())
                    logger.info(f"截图已保存: {save_path}")

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