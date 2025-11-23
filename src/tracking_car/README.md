<<<<<<< HEAD
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
=======
# 神经网络实现代理

利用神经网络/ROS 实现 Carla（车辆、行人的感知、规划、控制）、AirSim、Mujoco 中人和载具的代理。

## 环境配置

* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（尽量不使用Tensorflow）
* 相关软件下载 [链接](https://pan.baidu.com/s/1IFhCd8X9lI24oeYQm5-Edw?pwd=hutb)


## 贡献指南

准备提交代码之前，请阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md) 。
代码的优化包括：注释、[PEP 8 风格调整](https://peps.pythonlang.cn/pep-0008/) 、将神经网络应用到Carla模拟器中、撰写对应 [文档](https://openhutb.github.io/nn/) 、添加 [源代码对应的自动化测试](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) 等（从Carla场景中获取神经网络所需数据或将神经网络的结果输出到场景中）。

### 约定

* 每个模块位于`src/{模块名}`目录下，`模块名`需要用2-3个单词表示，首字母不需要大写，下划线`_`分隔，不能宽泛，越具体越好
* 每个模块的入口须为`main.`开头，比如：main.py、main.cpp、main.bat、main.sh等，提供的ROS功能以`main.launch`文件作为启动配置文件
* 每次pull request都需要保证能够通过main脚本直接运行整个模块，在提交信息中提供运行动图或截图；Pull Request的标题不能随意，需要概括具体的修改内容；README.md文档中提供运行环境和运行步骤的说明
* 仓库尽量保存文本文件，二进制文件需要慎重，如运行需要示例数据，可以保存少量数据，大量数据可以通过提供网盘链接并说明下载链接和运行说明


### 文档生成

测试生成的文档：
1. 安装python 3.11，并使用以下命令安装`mkdocs`和相关依赖：
```shell
pip install mkdocs -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install -r requirements.txt
```
（可选）安装完成后使用`mkdocs --version`查看是否安装成功。

2. 在命令行中进入`nn`目录下，运行：
```shell
mkdocs build
mkdocs serve
```
然后使用浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，查看文档页面能否正常显示。

## 参考

* [代理模拟器文档](https://openhutb.github.io)
* 已有相关 [无人车](https://openhutb.github.io/doc/used_by/) 、[无人机](https://openhutb.github.io/air_doc/third/used_by/) 、[具身人](https://openhutb.github.io/doc/pedestrian/humanoid/) 的实现
* [神经网络原理](https://github.com/OpenHUTB/neuro)


>>>>>>> e6c85f03b0d39fd629ec0955cfbe4e8ca19e8620
