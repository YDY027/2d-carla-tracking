# 2D-CARLA-Tracking

基于CARLA模拟器的2D目标跟踪系统，集成YOLO目标检测与轻量级SORT跟踪算法，实现对车辆、行人等交通参与者的实时感知与跟踪。

## 项目简介
本项目旨在构建一个轻量、高效的2D目标跟踪框架，基于CARLA仿真环境提供真实交通场景数据，通过YOLO系列模型完成目标检测，结合自定义简易SORT跟踪器（无第三方跟踪库依赖）实现多目标持续跟踪，支持实时可视化跟踪结果。

## 环境配置
* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件依赖：
  - Python 3.7-3.12（兼容3.7+版本）
  - CARLA Simulator 0.9.10+（需提前安装并启动）
  - PyTorch（YOLO模型依赖，不使用Tensorflow）

### 安装步骤
1. 安装Python 3.7-3.12环境（推荐3.9+）
2. 克隆本仓库：
```shell
git clone https://github.com/haruDT/2d-carla-tracking.git
cd 2d-carla-tracking
```
3. 安装依赖包（使用阿里云镜像加速）：
```shell
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
```
4. 安装CARLA Simulator：
   - 参考[CARLA官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/)下载对应平台的CARLA安装包
   - 解压后启动CARLA服务器（默认端口2000）：
     ```shell
     # Windows
     CarlaUE4.exe -windowed -ResX=800 -ResY=600
     
     # Ubuntu
     ./CarlaUE4.sh -windowed -ResX=800 -ResY=600
     ```

## 快速启动
1. 确保CARLA服务器已启动（端口默认2000，若修改端口需在启动命令中指定）
2. 运行主程序：
```shell
# 默认参数（本地CARLA服务器、30辆NPC车辆）
python main.py

# 自定义参数示例（指定CARLA服务器地址、端口、NPC数量）
python main.py --host=localhost --port=2001 --num_npcs=50
```
3. 查看跟踪结果：
   - 程序启动后会自动生成主车辆、NPC车辆及相机传感器
   - 实时显示跟踪窗口，绿色框为跟踪目标，标注唯一跟踪ID
   - 按`q`键退出程序

## 核心功能
- **CARLA环境交互**：自动生成交通参与者、挂载相机传感器、同步仿真控制
- **目标检测**：集成YOLO系列模型（默认YOLOv5s），精准检测车辆（轿车、公交、卡车）、行人等目标
- **轻量级跟踪**：内置简易SORT跟踪器（基于卡尔曼滤波+IOU匹配），无额外第三方依赖
- **实时可视化**：动态绘制跟踪框与ID，直观展示跟踪效果

## 代码结构
```
2d-carla-tracking/
├── main.py               # 主程序（环境初始化、检测跟踪流程、可视化）
├── requirements.txt      # 依赖包清单
└── README.md             # 项目文档
```

## 贡献指南
准备提交代码之前，请阅读以下规范：
1. 代码风格：遵循[PEP 8 编码规范](https://peps.pythonlang.cn/pep-0008/)，添加清晰注释
2. 功能优化方向：
   - 扩展目标类别（如自行车、摩托车等）
   - 优化跟踪算法（集成DeepSORT、OCSORT等更优跟踪器）
   - 提升检测性能（更换YOLOv8/YOLOv9等模型、量化压缩）
   - 增加数据记录功能（保存检测跟踪结果用于离线分析）
3. 提交要求：
   - 撰写功能说明文档
   - 添加自动化测试用例（参考[GitHub Actions Python测试指南](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)）
   - 确保代码可复现，兼容现有环境依赖

## 参考资源
- [CARLA官方文档](https://carla.readthedocs.io/)
- [YOLO官方仓库](https://github.com/ultralytics/ultralytics)
- [SORT跟踪算法原论文](https://arxiv.org/abs/1602.00763)
- [卡尔曼滤波原理与实现](https://github.com/OpenHUTB/neuro)