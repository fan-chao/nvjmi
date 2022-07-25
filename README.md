# nvjmi:palm_tree:
## 1. 简介
  封装Jetson Multimedia API的编解码库，基于 https://github.com/jocover/jetson-ffmpeg 和 https://github.com/dusty-nv/jetson-utils 基础进行的修改，未集成于ffmpeg，可单独使用。功能如下。
  1. 支持H.264解码。
  2. 支持解码后直接硬件完成缩放操作。
  3. 支持解码后直接硬件完成颜色空间转换操作。
  4. 支持Jetpack 4.3、4.4。
  5. 对于Jetpack 4.5需要使用对应的multimedia api，即使用Jetpack 4.5中/usr/src/jetson_multimedia_api更新include/和common/中的文件。

  当前仅完成解码器的修改，还未完成编码器的修改。

  关于解码API的使用，详见nvjmi.h接口说明。

## 2. 使用说明  
  1. 编译  
  直接使用make编译nvjmi动态库。
  
  2. 示例  
  nvjmi接口使用示例如下:  
  ```cpp
  if(jmi_ctx_ == nullptr) {
    jmi::nvJmiCtxParam jmi_ctx_param{};

    if(rsz_w > 0 && rsz_h > 0){
        jmi_ctx_param.resize_width = rsz_w;
        jmi_ctx_param.resize_height = rsz_h;
    }

    if ("H264" == m_pRtspClient->GetCodeName()) {
        jmi_ctx_param.coding_type =jmi::NV_VIDEO_CodingH264;
    }
    else if ("H265" == m_pRtspClient->GetCodeName()) {
        jmi_ctx_param.coding_type = jmi::NV_VIDEO_CodingHEVC;
    }
    string dec_name = "dec-" + session_id();
    jmi_ctx_ = jmi::nvjmi_create_decoder(dec_name.data(), &jmi_ctx_param);
 }

 //基于jetson nvdec解码
 jmi::nvPacket nvpacket;

 nvpacket.payload_size = dataLen;
 nvpacket.payload = data;

 int ret{};
 ret = jmi::nvjmi_decoder_put_packet(jmi_ctx_, &nvpacket);
 if(ret == jmi::NVJMI_ERROR_STOP) {
    LOG_INFO(VDO_RTSP_LOG, "[{}] frameCallback: nvjmi decode error, frame callback EOF!", m_ip);
 }

 while (ret >= 0) {
    jmi::nvFrameMeta nvframe_meta;
    ret = jmi::nvjmi_decoder_get_frame_meta(jmi_ctx_, &nvframe_meta);
    if (ret < 0) break;

    Buffer buf;
    buf.allocate(nvframe_meta.width, nvframe_meta.height, 3, nvframe_meta.payload_size / nvframe_meta.height);
    jmi::nvjmi_decoder_retrieve_frame_data(jmi_ctx_, &nvframe_meta, (void*)buf.getData());     
 }
 ```
## 3. 常见问题
  1. **Q:** 出现错误nvbuf_utils: Could not get EGL display connection，并且eglGetDisplay(EGL_DEFAULT_DISPLAY)返回NULL？  
     **A:** 1>在ssh终端输入unset DISPLAY，然后再运行程序即可。  
            2>vim /etc/profile，添加unset DISPLAY，然后souce /etc/profile生效，然后重启机器reboot。
  
  




