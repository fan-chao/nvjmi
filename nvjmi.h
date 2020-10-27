/**
* @file
* <b>NVIDIA jetson multimedia wrapper interface for video decoding and encoding</b>
*
*/

#ifndef __NVJMI_H__
#define __NVJMI_H__

#ifdef __GNUC__
#define JMI_API  extern
#endif

#include <stdlib.h>
#include <stdbool.h>

namespace jmi {
    typedef struct nvJmiCtx nvJmiCtx;

    typedef enum{
        NVJMI_OK = 0,
        NVJMI_ERROR_UNKNOWN = -1,
        NVJMI_ERROR_EOS = -2,
        NVJMI_ERROR_STOP = -3,
        NVJMI_ERROR_DEC_INTERNAL = -4,
        NVJMI_ERROR_GET_FRAME_META = -5,
        NVJMI_ERROR_FRAMES_EMPTY = -6,
        NVJMI_ERROR_OUTPUT_PLANE_STOP = -10,
        NVJMI_ERROR_OUTPUT_PLANE_DQBUF = -11,
        NVJMI_ERROR_OUTPUT_PLANE_QBUF = -12,
        NVJMI_ERROR_CAPTURE_PLANE_DQEVENT = -13,
        NVJMI_ERROR_CAPTURE_PLANE_STOP = -20,
    } nvErrorCode;

    typedef enum {
        NV_PIX_NV12 = 0,
        NV_PIX_YUV420,
        NV_PIX_ABGR32
    } nvPixFormat;

    typedef struct _NVENCPARAM{
        unsigned int width;
        unsigned int height;
        unsigned int profile;
        unsigned int level;
        unsigned int bitrate;
        unsigned int peak_bitrate;
        char enableLossless;
        char mode_vbr;
        char insert_spspps_idr;
        unsigned int iframe_interval;
        unsigned int idr_interval;
        unsigned int fps_n;
        unsigned int fps_d;
        int capture_num;
        unsigned int max_b_frames;
        unsigned int refs;
        unsigned int qmax;
        unsigned int qmin;
        unsigned int hw_preset_type;
    } nvEncParam;

    typedef struct _NVPACKET {
        unsigned long flags;
        unsigned long payload_size;
        unsigned char *payload;
        unsigned long  pts;
    } nvPacket;

    typedef struct _NVFRAME {
        unsigned long flags;
        unsigned long payload_size[3];
        unsigned char *payload[3];
        unsigned int linesize[3];
        nvPixFormat type;
        unsigned int width;
        unsigned int height;
        time_t timestamp;
    } nvFrame;

    typedef struct _NVFRAMEMETA {
        unsigned long payload_size;
        unsigned int coded_width;
        unsigned int coded_height;
        unsigned int width;
        unsigned int height;
        time_t timestamp;
        int frame_index;
        char got_data;
    } nvFrameMeta;

    typedef enum {
        NV_VIDEO_CodingUnused,
        NV_VIDEO_CodingH264,             /**< H.264 */
        NV_VIDEO_CodingMPEG4,              /**< MPEG-4 */
        NV_VIDEO_CodingMPEG2,              /**< MPEG-2 */
        NV_VIDEO_CodingVP8,                /**< VP8 */
        NV_VIDEO_CodingVP9,                /**< VP9 */
        NV_VIDEO_CodingHEVC,               /**< H.265/HEVC */
    } nvCodingType;

    typedef struct NVJMICTXPARAM {
        nvCodingType coding_type; //帧编码类型
        int resize_width; //缩放后的frame width
        int resize_height; //缩放后的frame_height
    } nvJmiCtxParam;

    /*
    *  Decoder functions
    */

    /**
    * 创建nvjmi解码器
    *
    * @使用nvhost-nvdec硬件解码.
    *
    * @参数[输入] dec_name 解码器名称，任意字符串
    * @参数[输入] param 创建nvjmi上下文所需要的参数.
    *
    * @返回 nvJmiCtx* 指针, 如果创建成功, 指针非空; 否则为空指针
    * @ingroup nvjmi
    */
    JMI_API nvJmiCtx* nvjmi_create_decoder(char const* dec_name, nvJmiCtxParam* param);

    /**
    * 关闭vjmi解码器
    *
    * @用于关闭之前创建的nvjmi解码器.
    *
    * @参数[输入] nvJmiCtx* nvjmi解码器上下文指针
    *
    * @返回 int , 如果关闭成功返回0; 失败返回非0
    * @ingroup nvjmi
    */
    JMI_API int nvjmi_decoder_close(nvJmiCtx* ctx);

    /**
    * 释放解码器资源
    *
    * @用于释放解码器绑定的资源.
    *
    * @参数[输入] nvJmiCtx** nvjmi解码器上下文指针地址
    *
    * @返回 int , 如果释放成功返回0; 失败返回非0
    * @ingroup nvjmi
    */
    JMI_API int nvjmi_decoder_free_context(nvJmiCtx** ctx);

    /**
    * 解码图像数据帧
    *
    * @将待解码的帧数据包出入nvjmi解码器, 等待解码.
    *
    * @参数[输入] nvJmiCtx* nvjmi解码器上下文指针
    * @参数[输入] nvPacket* 待解码的帧数据包
    *
    * @返回 int , 如果传入成功返回0; 失败返回非0
    * @ingroup nvjmi
    */
    JMI_API int nvjmi_decoder_put_packet(nvJmiCtx* ctx, nvPacket* packet);

    /**
    * 解码图像数据帧 - 获得帧元数据
    *
    * @获取nvjmi解码器成功解码的帧元数据
    * @note 此API仅获取解码后帧的元数据, 不包含具体的帧图像数据
    * @帧数据使用AP Invjmi_decoder_retrieve_frame_data提取
    *
    * @参数[输入] nvJmiCtx* nvjmi解码器上下文指针
    * @参数[输出] frame_meta* 一帧解码后的帧元数据
    *
    * @返回 int , 如果传入成功返回0; 失败返回非0
    * @ingroup nvjmi
    */
    JMI_API int nvjmi_decoder_get_frame_meta(nvJmiCtx* ctx, nvFrameMeta* frame_meta);

    /**
    * 解码图像数据帧 - 提取帧图像数据
    *
    * @获取nvjmi解码器成功解码的帧元数据
    * @note 此API仅当前nvjmi_decoder_get_frame_meta返回成功时调用
    *
    * @参数[输入] nvJmiCtx* nvjmi解码器上下文指针
    * @参数[输入] frame_meta* 要提取的帧图像数据的帧元数据
    * @参数[输出] frame_data* 提取的帧元数据
    *
    * @返回 int , 如果传入成功返回0; 失败返回非0
    * @ingroup nvjmi
    */
    JMI_API int nvjmi_decoder_retrieve_frame_data(nvJmiCtx* ctx, nvFrameMeta* frame_meta, void* frame_data);

    /*
    *  Encoder functions
    */
    JMI_API nvJmiCtx* nvjmi_create_encoder(nvCodingType coding_type, nvEncParam* param);
    JMI_API int nvjmi_encoder_close(nvJmiCtx* ctx);
    JMI_API int nvjmi_endocder_free_context(nvJmiCtx** ctx);

    JMI_API int nvjmi_encoder_put_frame(nvJmiCtx* ctx, nvFrame* frame);
    JMI_API int nvjmi_encoder_get_packet(nvJmiCtx* ctx, nvPacket* packet);
}
#endif // !__NVJMI_H__
