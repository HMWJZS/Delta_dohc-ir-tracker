# 第三方来源

算法来源为DOHC_workspace冻结dev.19.4（75b097ee1fcf2371f154d5f30f5609704462e6e4），包含几何、IPPE/G、RPG式ROI预测、有界运动覆盖及恢复调度；上游provenance.json与来源记录保留于核心。

MPP/RGA桥接和原生分配库随源码及build-info交付，动态依赖的系统MPP/RGA/Mali库不重分发。Python依赖的许可证保留在原wheel；PyOpenCL由Ubuntu20.04 ARM64原发行包重装为wheel，原包及copyright随交付提供。

源码仓库未给本项目指定独立开源许可证；本地功能包交付不新增对外授权条款，也不宣称厂商驱动可自由重分发。

imageio-ffmpeg随wheel携带FFmpeg二进制及许可证/构建说明，作为无系统ffmpeg时的视频报告编码工具；不参与识别。
