# nvjmi
封装Jetson Multimedia API的编解码库，基于 https://github.com/jocover/jetson-ffmpeg 和 https://github.com/dusty-nv/jetson-utils 基础进行的修改，未集成于ffmpeg，可单独使用。功能如下。
1. 支持H.264解码。
2. 支持解码后直接硬件完成缩放操作。
3. 支持解码后直接硬件完成颜色空间转换操作。
4. 支持Jetpack 4.3、4.4。
5. 对于Jetpack 4.5需要使用对应的multimedia api，即使用Jetpack 4.5中/usr/src/jetson_multimedia_api更新include/和common/中的文件。

当前仅完成解码器的修改，还未完成编码器的修改。

关于解码API的使用，详见nvjmi.h接口说明。



