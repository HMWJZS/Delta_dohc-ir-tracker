# Delta DOHC IR Tracker

面向 RK3588 / DOHC 的硬件加速红外标靶识别、跟踪与位姿估计独立功能包。

当前功能包 **0.0.2**，内部算法 **2.0.0-dev.19.4**，输出协议 `dohc.infrared.v1`。支持 P01～P05 注册图案；SDK 使用已有采集程序提供的原生 JPEG，不打开相机，不依赖 T265、IMU、Dashboard 或 DOHC 录制服务。五标靶同时识别尚未实测验收。

## 获取与安装

仓库携带完整 ARM64 离线包及依赖 wheel。可通过 Git 克隆或 GitHub 的 **Code → Download ZIP** 获取，无需 Git LFS。

```bash
git clone https://github.com/HMWJZS/Delta_dohc-ir-tracker.git
cd Delta_dohc-ir-tracker/toolkit
python3 verify_bundle.py
```

以下安装命令在匹配的 **RK3588、Linux ARM64、Ubuntu 20.04、CPython 3.8** 设备上执行；要求已有 MPP、RGA 和 Mali OpenCL 驱动，以及相应硬件节点权限。

```bash
bash install.sh /home/root/venvs/dohc-ir-0.0.2
source /home/root/venvs/dohc-ir-0.0.2/bin/activate
dohc-ir doctor
cp config/config.example.json config/config.local.json
```

先确认标定、V1 镜头、940nm 滤光片及物理 LED 板匹配，再将 `config/config.local.json` 中的 `optical_profile_confirmed` 改为 `true`。标定示例仅对应已有光学配置，不适用于所有设备。

安装器拒绝覆盖已有环境，校验 wheel 哈希，不下载依赖、不安装系统驱动、不修改服务。SDK 不提供 CPU 运行回退；本机 x86_64 可查看生成的 HTML 报告。

## Python 使用

在上面的 `toolkit` 目录下运行。`source` 是已有采集程序提供的迭代器，产生 JPEG 字节、递增序号与同机 `time.monotonic()` 到帧时间：

```python
from dohc_ir_tracker import Tracker

with Tracker(config="config/config.local.json") as tracker:
    last_sequence = None
    for jpeg_bytes, sequence, arrival_monotonic in source:
        tracker.submit(jpeg_bytes, sequence, arrival_monotonic)
        result = tracker.latest_result()
        if result.get("source_sequence") == last_sequence:
            continue
        last_sequence = result.get("source_sequence")
        for target in result.get("detections", []):
            if target["quality"]["pose_usable"]:
                print(target["pattern_id"], target["pose"],
                      target["quality"]["orientation_supported"])
```

输入为原生 **3840×2160 JPEG 字节**。输出包含身份、光点对应、当前位姿、双候选、质量状态、搜索决策和耗时。平移单位为米，旋转为弧度制 Rodrigues 向量；`T_C_T` 将标靶坐标变换至相机光学坐标。

会话前 5 秒清点身份，之后跟踪启动名单；新身份需重开会话。仅保留一个运行任务和一个可覆盖的最新待处理帧，准入成功不等于完成识别。读取结果时按 `source_sequence` 去重，并检查新鲜度和质量字段。

## 本版算法

原生 JPEG → MPP 整帧 YUV 解码 → 预测 ROI／全图恢复 → RGA 局部裁剪与 BGR 转换 → Mali 信号构造及高斯平滑 → 光点提取与对应 → 鱼眼 IPPE 双候选及 G 精修 → 时序选解 → 身份与新鲜度检查。

dev.19.4 在严格预测失效时使用近期两个已确认观测形成有界运动覆盖 ROI。单目标面积不超过原 ROI 的 2 倍、全图的 12.5%，合并面积也受限；线索有效期 0.2 秒。覆盖只改变搜索范围，不生成位姿光点；保留 dev.19.2 的 0.2 秒恢复冷却。

## 文件与溯源

| 路径 | 内容 |
|---|---|
| [toolkit/README.md](toolkit/README.md) | 原始离线包使用说明 |
| [toolkit/source/](toolkit/source/) | 完整算法、SDK、原生源码和测试快照 |
| [toolkit/wheelhouse/](toolkit/wheelhouse/) | 自有 wheel 及 13 个 ARM64 依赖 wheel |
| [toolkit/config/](toolkit/config/) | 标定、图案和运行配置示例 |
| [toolkit/examples/](toolkit/examples/) | 采集回调和坐标变换示例 |
| [toolkit/build-manifest.json](toolkit/build-manifest.json) | 原构建来源、版本及核心哈希 |
| [toolkit/SHA256SUMS](toolkit/SHA256SUMS) | 离线包文件校验清单 |
| [CHANGELOG.md](CHANGELOG.md) | 功能包版本变更 |
| [docs/README.md](docs/README.md) | 使用说明、验证报告及导入记录索引 |

`toolkit/` 原样导入已经验证的 0.0.2 交付包，不含临时构建目录。原算法提交为 `75b097ee1fcf2371f154d5f30f5609704462e6e4`，原封装源码提交为 `aa28bd5773978afd9b804fad5b39b2440c170cd3`，两者来自 `DOHC_workspace`。构建清单和 `SOURCE.json` 中的提交指向原来源，不是本仓库的 Git 提交；导入不改写它们。

从本仓库重建时，在仓库根目录使用 Python 3.10 以上及 `packaging`：

```bash
python3 toolkit/source/sdk/build_toolkit.py \
  --native-root toolkit/source \
  --dependencies toolkit/wheelhouse \
  --output dist/20261023_DOHC_dev19_4_重建包
```

这是冻结源码快照的重建，保留原源码身份。修改算法或 SDK 后应重新确定版本、源码身份和验收记录，不能继续宣称与本冻结交付完全一致。原包说明中的 `sdk/…` 构建路径在本仓库对应 `toolkit/source/sdk/…`。

## 验证与限制

已有本地相关测试 **365 项通过、8 项跳过**；五段录像 **10,440 帧**与原 dev.19.4 实现等价回归通过，灯点、位姿和搜索决策零差异。最终 wheel 的仓库外安装测试、完整性校验和脱离 Git 重建通过。这些是主机封装验证，不是板端性能或绝对精度验收。

独立包的 ARM64 依赖组合、MPP/RGA/Mali 行为和连续吞吐仍需设备验证。`pose_usable` 不等于方向正确，`orientation_supported` 也不等于独立真值认证；平面方向歧义仍可能存在。无外部真值时，绝对位置及旋转精度均为 N/A。使用新 venv 安装，保留旧包和旧环境供回退。

本次仓库导入未连接 DOHC；设备 CID、IP 和当前版本均未读取。原报告中历史整机帧率不作为本独立包成绩。

第三方依赖许可证随 wheel 保留，来源见 [THIRD_PARTY.md](toolkit/THIRD_PARTY.md)。当前项目未指定独立开源许可证，本次导入不新增授权条款。
