# 2D-CARLA-Tracking（优化增强版）
基于CARLA模拟器的2D目标跟踪系统，集成YOLOv5su目标检测、轻量级SORT跟踪算法，新增轨迹可视化与车辆行为分析（停车/超车检测），实现交通参与者的实时感知与跟踪。

## 项目简介
本项目是轻量、高效的2D目标跟踪框架，基于CARLA仿真环境提供真实交通场景数据：
- 采用YOLOv5su模型（Ultralytics优化版）实现高精度目标检测，性能优于原版YOLOv5s
- 内置简易SORT跟踪器（卡尔曼滤波+匈牙利算法IOU匹配），无第三方跟踪库依赖
- 新增轨迹可视化（渐变轨迹线）、行为分析（停车/超车检测）核心功能
- 支持GPU加速、异步推理、固定帧率显示，避免卡顿与闪烁
- 全流程异常捕获，鲁棒性强，适配Windows/Ubuntu双平台

## 环境配置
### 支持平台
Windows 10/11、Ubuntu 20.04/22.04

### 软件依赖
- Python 3.7-3.12（推荐3.9+，兼容3.7+版本）
- CARLA Simulator 0.9.10+（需提前安装并启动）
- PyTorch 1.18.0+（支持CUDA更佳，用于YOLO模型GPU加速）

### 安装步骤
1. 安装Python 3.7-3.12环境（推荐3.9版本，兼容性最佳）
2. 克隆本仓库：
```shell
git clone https://github.com/haruDT/2d-carla-tracking.git
cd 2d-carla-tracking
```
3. 安装依赖包（使用阿里云镜像加速，避免下载超时）：
```shell
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```
4. 安装CARLA Simulator：
   - 参考[CARLA官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/)下载对应平台的CARLA安装包
   - 解压后启动CARLA服务器（默认端口2000）：
     ```shell
     # Windows系统
     CarlaUE4.exe -windowed -ResX=800 -ResY=600
     
     # Ubuntu系统
     ./CarlaUE4.sh -windowed -ResX=800 -ResY=600
     ```

## 快速启动
### 基础运行（默认参数）
1. 确保CARLA服务器已启动（端口默认2000，启动后请勿关闭）
2. 运行主程序：
```shell
python main.py
```
- 默认生成10辆NPC车辆，红色为主车辆（带相机传感器）
- 自动弹出跟踪可视化窗口（置顶显示，支持鼠标缩放）

### 自定义参数示例
```shell
# 指定CARLA服务器地址、端口、NPC车辆数量
python main.py --host=localhost --port=2001 --num_npcs=50

# 调整检测置信度阈值（降低阈值可提高检测率，但可能增加误检）
python main.py --conf-thres=0.4

# 加载自定义配置文件（推荐使用，灵活调整所有参数）
python main.py --config=my_config.yaml
```

### 交互操作说明
| 按键        | 功能描述                                  |
|-------------|-------------------------------------------|
| Q 或 ESC    | 退出程序，自动清理CARLA仿真资源            |
| S           | 保存当前跟踪画面截图（自动命名为`track_screenshot_时间戳.png`） |

## 核心功能（优化新增特性）
| 功能模块          | 详细说明                                                                 |
|-------------------|--------------------------------------------------------------------------|
| CARLA环境交互     | 自动生成主车辆/NPC、挂载RGB相机、同步仿真步长控制，支持视角跟随主车辆      |
| 目标检测          | YOLOv5su模型，精准检测3类车辆（轿车/公交/卡车），支持置信度阈值配置        |
| 多目标跟踪        | 卡尔曼滤波预测目标运动轨迹，匈牙利算法IOU匹配，稳定分配唯一跟踪ID          |
| 轨迹可视化        | 绘制目标历史运动轨迹（渐变效果，越新轨迹越亮），支持长度/透明度/线宽自定义 |
| 行为分析          | 自动检测车辆行为：停车（低速持续5帧）、超车（近距离+速度比1.5倍）          |
| 实时可视化        | 半透明标签显示「ID+车型+行为状态」，顶部状态栏显示FPS/跟踪数/停车数/超车数 |
| 性能优化          | 异步推理线程避免阻塞、GPU加速提升帧率、帧缓存机制解决闪烁问题              |

## 配置自定义（config.yaml）
支持通过配置文件灵活调整参数（项目根目录新建`config.yaml`文件），示例内容：
```yaml
# CARLA连接配置
host: localhost
port: 2000
num_npcs: 15  # NPC车辆数量

# 检测与跟踪参数
conf_thres: 0.5    # 检测置信度阈值
iou_thres: 0.3     # IOU匹配阈值
max_age: 5         # 轨迹最大过期帧数
min_hits: 3        # 最小命中帧数（稳定跟踪后才显示）

# YOLO推理参数
yolo_imgsz_max: 320  # 推理图像最大尺寸（越小越快）
yolo_iou: 0.45       # YOLO NMS阈值

# 轨迹可视化配置
track_history_len: 25  # 轨迹最大长度（帧数）
track_line_width: 2    # 轨迹线宽
track_alpha: 0.7       # 轨迹透明度（0-1）

# 行为分析配置
stop_speed_thresh: 1.2    # 停车速度阈值（像素/帧）
stop_frames_thresh: 5     # 判定停车的连续帧数
overtake_speed_ratio: 1.6 # 超车速度比（目标速度/自车速度）
overtake_dist_thresh: 45  # 超车判定距离（像素）

# 显示配置
window_width: 1280  # 可视化窗口宽度
window_height: 720  # 可视化窗口高度
display_fps: 30     # 固定显示帧率（避免画面闪烁）
```

## 代码结构
```
2d-carla-tracking/
├── main.py               # 主程序（环境初始化、检测跟踪流程、可视化）
├── requirements.txt      # 项目依赖包清单（含版本约束）
├── config.yaml           # 自定义配置文件（可选，需手动新建）
├── README.md             # 项目说明文档（本文档）
└── track_screenshot_xxx.png  # 自动保存的跟踪截图（运行后生成）
```

## 常见问题排查
1. **CARLA连接失败**：检查CARLA服务器是否启动、端口是否匹配（默认2000）、防火墙是否拦截
2. **无检测结果**：确保YOLOv5su模型已自动下载（首次运行需联网），或手动下载后放到项目根目录
3. **GPU加速失效**：确认PyTorch已安装CUDA版本，运行`python -c "import torch; print(torch.cuda.is_available())"`验证
4. **窗口闪烁**：无需手动调整，程序内置固定帧率和帧缓存机制，自动稳定显示
5. **内存泄漏**：程序退出时会自动清理CARLA Actor资源，请勿直接关闭终端（需按Q/ESC退出）

## 扩展方向
1. 扩展目标类别：支持自行车、摩托车、行人等交通参与者检测
2. 升级跟踪算法：集成DeepSORT（外观特征匹配）、OCSORT（更优运动模型）
3. 模型性能优化：更换YOLOv8/YOLOv9模型，或对模型进行量化压缩
4. 数据记录功能：保存检测/跟踪结果（bbox、ID、行为标签）为JSON/CSV格式
5. 更多行为分析：增加跟车、变道、逆行等交通行为检测逻辑
6. ROS集成：将跟踪结果发布为ROS话题，对接规划/控制模块

## 参考资源
- [CARLA官方文档](https://carla.readthedocs.io/)
- [Ultralytics YOLO官方仓库](https://github.com/ultralytics/ultralytics)
- [SORT跟踪算法原论文](https://arxiv.org/abs/1602.00763)
- [卡尔曼滤波原理与实现](https://github.com/OpenHUTB/neuro)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
```
