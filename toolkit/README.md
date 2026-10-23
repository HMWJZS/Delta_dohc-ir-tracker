# DOHC 红外识别独立工具包

功能包0.0.2，内部算法2.0.0-dev.19.4，结果协议dohc.infrared.v1。交付面向Linux ARM64 / RK3588、系统CPython 3.8、MPP/RGA与Mali GPU。SDK不打开相机，不依赖T265、IMU、Dashboard或DOHC录制服务。

本地封装验证不能代替板端安装和硬件验收。本包不承诺固定帧率或绝对精度；当前估计可能保留平面方向歧义。

## 本版升级

从功能包0.0.1 / dev.19.2升级到0.0.2 / dev.19.4，接入已有的有界运动覆盖ROI：严格预测因外推上限或历史不足而不可用时，允许由两个新鲜已确认观测提供独立覆盖线索。覆盖面积不超过原ROI的2倍且不超过全图的12.5%，合并后面积同样受限；线索有效期0.2秒。覆盖只扩大图像搜索区域，不生成用于关联或位姿求解的预测光点，不把未确认观测加入覆盖历史。沿用0.2秒全图恢复冷却。

Python接口、配置和结果协议保持兼容，search.roi_coverage给出覆盖是否应用、原因、原/候选/最终ROI及面积比。旧包及历史测试记录保留。

## 安装与前提

要求已匹配的DOHC Ubuntu 20.04镜像、Python 3.8及venv、libOpenCL.so.1和Mali ICD、MPP、RGA、可访问硬件节点以及可写OpenCL缓存。系统驱动不随包安装或替换。原生codec历史构建记录为MPP 1.3.9、RGA 2.1.0，实际动态链接及设备兼容性由doctor和板端验收确认。

解压后在DOHC上执行（不要覆盖现有DOHC环境）：

```bash
python3 verify_bundle.py
bash install.sh /home/root/venvs/dohc-ir-0.0.2
/home/root/venvs/dohc-ir-0.0.2/bin/dohc-ir doctor
```

安装器拒绝已有目标目录；从wheelhouse离线安装，使用requirements.lock校验所有wheel哈希，不调用apt，不更改系统服务。失败时保留新目录用于诊断；只删除本次指定的新环境即可撤销安装，无需回滚整机。安装过程不修改相机或录制服务；doctor仅初始化GPU和解码器，不采图。

离线依赖候选：NumPy1.23.5、SciPy1.10.1、OpenCV-headless4.13.0.92、PyOpenCL2019.1.1。PyOpenCL由Ubuntu20.04 ARM64发行包原字节重装为wheel，来源和许可证保留；其np.bool兼容性要求NumPy<1.24。它们不是历史DOHC环境的逐字复制：历史OpenCV为5.0.0/NumPy1.24.4。依赖闭包和平台标签在本地检查，Mali行为和最终数值/性能须板端验收，不把历史测试冒充本包验证。

## 初始化配置

复制config/config.example.json为自己的配置。标定示例对应V1镜头、940nm滤光片、原生3840×2160；patterns示例来自已有P01～P05注册表。确认实际光学配置、图案及物理LED板一致后，将optical_profile_confirmed改为true；示例不是对所有设备有效的标定。

calibration和patterns路径相对配置文件，output_dir可设为null以仅返回内存结果；配置输出目录时每次会话单建目录，写latest.json、results.jsonl、记录版本与输入配置哈希的session.json以及session-summary.json。diagnostics默认false；设true会增加诊断计时开销，且必须配置output_dir。算法门限不接受配置覆盖。

## Python调用

```python
import time
from dohc_ir_tracker import Tracker

tracker = Tracker(config='config/config.json')
try:
    tracker.start()
    # 在已有采集回调中调用，不要为了本工具再次打开同一相机：
    # tracker.submit(jpeg_bytes, sequence, application_frame_arrival_monotonic)
    # 如果没有外部序号和到帧时间，可以 tracker.submit(jpeg_bytes)
    result = tracker.latest_result()
finally:
    tracker.close()
```

start等待worker就绪后自动记录会话起点；前5秒清点身份，之后仅跟踪本会话启动名单。迟到的新身份需stop后start重新清点。start不负责录像。

submit接收原生JPEG字节、可选uint64帧序号和time.monotonic()秒值；省略时由SDK生成编号和接收时间。SDK接收时刻不等于曝光时间，也不包含外部调用前的缓存延迟。有真实相机到帧时间时应显式传入，同一会话序号严格递增。True仅表示接收，pending（最新待处理引用）会被新帧覆盖，不表示成功计算。重复/倒退序号返回False；过期输入不处理；非法输入报错。输入上限略低于32MiB以给IPC元数据留空间。

latest_result每次重新检查600ms新鲜度并返回独立快照。需要按source_sequence去重读取，轮询次数不是识别次数。stop结束当前会话并清空可用结果，之后可start新会话；close永久关闭本实例且可重复调用。硬件异常抛出RuntimeError，不自动CPU回退；调用方应在finally中close。

## 输入、输出和坐标

主路径：原生3840×2160 JPEG → MPP整帧YUV → 预测ROI／全图恢复 → RGA裁剪及BGR转换 → Mali信号及高斯 → 提点和对应 → 鱼眼IPPE双候选与解析G → 时序选解 → 身份及新鲜度检查。

输出保留detections（已确认目标）、observations（含待确认观测）、pattern_id、led_ids、points_raw、pose、pose_candidates、pose_resolution、quality、search和分项耗时。raw_observations只作诊断，不能当作当前可用输出。运行时间基准是同机MONOTONIC。

pose.rvec是Rodrigues旋转向量，弧度；pose.tvec_m是平移，米。T_C_T将标靶坐标变换到相机光学坐标：p_C=R_C_T*p_T+t_C_T。相机X向右、Y向下、Z向前；标靶原点为整板100灯坐标均值，z=0。见examples/pose_matrix.py。

quality.pose_usable表示当前策略接受；quality.orientation_supported才表示单帧或运动假设方向支持，两者均不等于独立真值认证。pose_ambiguous保留平面歧义。position_usable按原双候选位置一致性判定。没有新鲜检测时不应使用历史位姿。

## 命令行

```bash
dohc-ir doctor --config config/config.json
dohc-ir image --config config/config.json --input frame.jpg --output frame-result.json
dohc-ir replay --config config/config.json --input /local/DOHC_session --output /local/20261023_红外标靶_回放
dohc-ir report --results /local/results.jsonl --output /local/20261023_红外标靶_报告
```

image只诊断一张图像并返回候选，不重复送同图伪造两帧身份确认。replay支持DOHC会话（cam0/*.mp4、times/cam0.json、states.jsonl）或JPEG输入清单；保留原源序号和源时间差，以当前MONOTONIC重新定时，原速准入且保留真实预算和覆盖机制。

JPEG清单为JSON对象，entries中每项包括file（相对路径）、source_sequence、relative_seconds（相对会话起点的秒）和sha256；时间必须非负、严格递增，序号唯一且递增。附加calibration字段可供报告投影轴。每张JPEG在回放前核验哈希；输入不得越过清单目录。优先使用原始JPEG；NAS数据先复制完整本地副本并核验，不直接回放NAS。

MP4会话先完整解码为Q100 JPEG并保留原时间差和哈希；这是有损录像恢复输入，不是相机原始JPEG。预处理成本单列。replay输出admission.json、results.jsonl、replay-info.json及report/index.html；未处理的录像帧不能算识别成功。源序号跨会话可重新编号。

可选标注视频：

```bash
dohc-ir report --results /local/results.jsonl --input-manifest /local/input/manifest.json --video --output /local/20261023_红外标靶_视频报告
```

优先使用已有ffmpeg/libx264；未找到时使用随包imageio-ffmpeg的ARM64编码器，不影响主相机录制和算法硬解。HTML无需服务器和网络，可在本机浏览器打开。视频按输入平均FPS预览，原始时间以清单为准；源帧无新鲜检测时不画旧灯点或位姿，橙色轴表示方向未获支持。

整段统计窗口为会话开始到最后一条记录的完成时刻，单列处理FPS、位姿Hz、方向支持Hz、最长无确认间隔及ROI/全图耗时。它不是预设固定时长设备测试。JSON和逐帧CSV保留原值，代表单帧为首个新鲜ROI检出，否则首个完成记录。包含耗时存在嵌套，不能直接相加；没有profile不等于耗时为零。绝对位置/旋转精度无真值时始终N/A。

## 构建和验证

在源码仓库使用Python3.10以上、pip、packaging、dpkg-deb。源码包另含SOURCE.json，用于脱离Git后的溯源。依赖下载阶段需要网络，成品安装不需要：

```bash
python3 sdk/prepare_dependencies.py --output dist/toolkit-dependencies
python3 sdk/build_toolkit.py --native-root /path/to/verified/dev19_4 --dependencies dist/toolkit-dependencies --output dist/20261023_DOHC_dev19_4_工具包_0.0.2
```

native-root包含已核验ARM64库和对应build-info.json。构建核对12个算法/数据文件与冻结提交，核对两原生库架构/ABI/源码哈希；脏源码必须显式加--preview且制品记录source_dirty=true。干净交付不使用该开关。包版本、算法版本和系统版本分别管理，独立包不生成DOHC系统发布。

安装包包含source源码、wheelhouse、原生源码/库及清单、配置、示例、许可证、README和SHA256SUMS。已有源代码中的旧方法属于内部实现依赖，不作为SDK公开方法，也不携带RKNN权重或开放NPU选项。

## 故障排查与验收边界

- 非Linux ARM64或非Python3.8：doctor明确失败；本机仍可查看生成报告，不会伪装硬件执行。
- 缺少libOpenCL/MPP/RGA或权限：检查匹配镜像、设备节点与当前用户权限；不覆盖系统驱动解决。
- OpenCL缓存不可写：为运行用户提供可写XDG_CACHE_HOME。
- 原生库或源码哈希不匹配：重新核验或构建整包，不手工替换单个.so。
- 启动后没有身份：检查前5秒内是否可见、光学配置与图案是否一致；空名单需重开会话。
- 方向歧义：检查质量字段，不能以帧率高或重投影误差低推断绝对方向正确。

本轮不连接DOHC，设备CID、IP和当前版本未读取。板端独立环境安装、真实硬件同输入对照、连续60秒吞吐和恢复均待验。已有dev.19.4整机算法65秒记录属于历史不同载体与依赖环境，不能替代本独立包安装及完整时长验收；正式离线精度验收仍须冻结同10组3192帧及既定指标。
