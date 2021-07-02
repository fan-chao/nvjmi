#include "nvjmi.h"

#include "cudaMappedMemory.h"
#include "logging.h"

#include "vic_onverter.h"
#include "cuda_converter.h"
#include "fd_egl_frame_map.h"
#include "NvVideoDecoder.h"
#include "nvbuf_utils.h"

#include <tbb/concurrent_queue.h>

#include <vector>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <queue>
#include <atomic>

using namespace std;

namespace jmi {

#define CHUNK_SIZE 1<<22
#define MAX_BUFFERS 32

 /**
 * LOG_NVJMI_DECODER logging prefix
 * @ingroup codec
  */
#define LOG_NVJMI_DECODER "[nvjmi-decoder] "

#define TEST_ERROR(condition, message, errorCode)    \
    if (condition)  {                              \
          LogError(LOG_NVJMI_DECODER "%s - %d\n", message, errorCode);     \
     }

    struct nvJmiCtx {
        NvVideoDecoder *dec{};
        atomic<bool> eos{}; //流接入结束
        atomic<bool> output_plane_stop {}; //停止output_plane
        atomic<bool> capture_plane_stop{}; //停止capture_plane
        atomic<int> capture_plane_error_code{}; //capture plane错误
        bool got_res_event{};
        int index{};
        unsigned int coded_width{}; //编码的帧图像宽度
        unsigned int coded_height{}; //编码的帧图像高度
        unsigned int resize_width{};
        unsigned int resize_height{};
        unsigned int frame_size{};
        int dst_dma_fd{-1};
        int numberCaptureBuffers{};
        int dmaBufferFileDescriptor[MAX_BUFFERS]{};
        unsigned int decoder_pixfmt{};

        std::thread * dec_capture_thread{};
        std::thread * output_plane_stop_thread{};

        unsigned char * frame_buffer[MAX_BUFFERS]{};
        tbb::concurrent_bounded_queue<int> * frame_pools{};
        tbb::concurrent_bounded_queue<int> * frames{};

        unsigned long long timestamp[MAX_BUFFERS]{};

        //for converter
        VICConverter * vic_converter{};
        CUDAConverter * cuda_converter{};
        FdEglFrameMap * fd_egl_frame_map{};

        cudaStream_t cuda_stream{};
    };

    void respondToResolutionEvent(v4l2_format &format, v4l2_crop &crop, nvJmiCtx* ctx){
        int32_t minimumDecoderCaptureBuffers{};
        int ret{};

        // Get capture plane format from the decoder. This may change after an resolution change event
        ret = ctx->dec->capture_plane.getFormat(format);
        TEST_ERROR(ret < 0, "Error: Could not get format from decoder capture plane", ret);
      
        // Get the video resolution from the decoder
        ret = ctx->dec->capture_plane.getCrop(crop);
        TEST_ERROR(ret < 0, "Error: Could not get crop from decoder capture plane", ret);

        ctx->coded_width = crop.c.width;
        ctx->coded_height = crop.c.height;

        if (ctx->resize_width == 0 || ctx->resize_height == 0){
            ctx->resize_width = ctx->coded_width;
            ctx->resize_height = ctx->coded_height;
        }

        //destroy pre-created capture_plane, dma buffer, egl_frame
        if (ctx->dst_dma_fd != -1) {
            NvBufferDestroy(ctx->dst_dma_fd);
            ctx->dst_dma_fd = -1;
        }

        ctx->dec->capture_plane.deinitPlane();
        for (int idx = 0; idx < ctx->numberCaptureBuffers; idx++) {
            if (ctx->dmaBufferFileDescriptor[idx] != 0) {
                ret = NvBufferDestroy(ctx->dmaBufferFileDescriptor[idx]);
                TEST_ERROR(ret < 0, "Failed to Destroy NvBuffer", ret);
            }
        }

        ctx->vic_converter->exit();
        ctx->fd_egl_frame_map->exit();

        ctx->frame_pools->clear();
        ctx->frames->clear();
        //end destroy

        ret = ctx->dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);
        TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", ret);

        ctx->dec->getMinimumCapturePlaneBuffers(minimumDecoderCaptureBuffers);
        TEST_ERROR(ret < 0, "Error while getting value of minimum capture plane buffers", ret);

        ctx->numberCaptureBuffers = minimumDecoderCaptureBuffers + 5;

        NvBufferCreateParams capture_params{};
        switch (format.fmt.pix_mp.colorspace) {
        case V4L2_COLORSPACE_SMPTE170M:
        {
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                // "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)"
                capture_params.colorFormat = NvBufferColorFormat_NV12;
            }
            else {
                //"Decoder colorspace ITU-R BT.601 with extended range luma (0-255)";
                capture_params.colorFormat = NvBufferColorFormat_NV12_ER;
            }
        } break;
        case V4L2_COLORSPACE_REC709:
        {
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                //"Decoder colorspace ITU-R BT.709 with standard range luma (16-235)";
                capture_params.colorFormat = NvBufferColorFormat_NV12_709;
            }
            else {
                //"Decoder colorspace ITU-R BT.709 with extended range luma (0-255)";
                capture_params.colorFormat = NvBufferColorFormat_NV12_709_ER;
            }
        } break;
        case V4L2_COLORSPACE_BT2020:
        {
            //"Decoder colorspace ITU-R BT.2020";
            capture_params.colorFormat = NvBufferColorFormat_NV12_2020;
        } break;
        default:
        {
            if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT) {
                //"Decoder colorspace ITU-R BT.601 with standard range luma (16-235)";
                capture_params.colorFormat = NvBufferColorFormat_NV12;
            }
            else {
                //"Decoder colorspace ITU-R BT.601 with extended range luma (0-255)";
                capture_params.colorFormat = NvBufferColorFormat_NV12_ER;
            }
        } break;
        }

        //create decoded framed dma fd
        for (int idx = 0; idx < ctx->numberCaptureBuffers; idx++) {
            capture_params.width = crop.c.width;
            capture_params.height = crop.c.height;
            capture_params.layout = NvBufferLayout_BlockLinear;
            capture_params.payloadType = NvBufferPayload_SurfArray;
            capture_params.nvbuf_tag = NvBufferTag_VIDEO_DEC;

            ret = NvBufferCreateEx(&ctx->dmaBufferFileDescriptor[idx], &capture_params);
            TEST_ERROR(ret < 0, "Failed to create buffers", ret);
        }

        ctx->frame_pools->set_capacity(ctx->numberCaptureBuffers);
        ctx->frames->set_capacity(ctx->numberCaptureBuffers);
        for (int i = 0; i < ctx->numberCaptureBuffers; ++i){
            ctx->frame_pools->push(i);
        }

        NvBufferRect src_rect;
        src_rect.top = 0; //top和left均为0，则为scale；否则为crop
        src_rect.left = 0;
        src_rect.width = crop.c.width;
        src_rect.height = crop.c.height;

        NvBufferRect dst_rect;
        dst_rect.top = 0;
        dst_rect.left = 0;
        dst_rect.width = crop.c.width;
        dst_rect.height = crop.c.height;

        //create transform dst dma fd
        NvBufferCreateParams transform_dst_params{};
        transform_dst_params.payloadType = NvBufferPayload_SurfArray;
        transform_dst_params.width = crop.c.width;
        transform_dst_params.height = crop.c.height;
        if (ctx->resize_width > 0 && ctx->resize_height > 0) {
            transform_dst_params.width = ctx->resize_width;
            transform_dst_params.height = ctx->resize_height;

            dst_rect.width = ctx->resize_width;
            dst_rect.height = ctx->resize_height;
        }
        transform_dst_params.layout = NvBufferLayout_Pitch;
        transform_dst_params.colorFormat = NvBufferColorFormat_ABGR32;
        transform_dst_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;
        ret = NvBufferCreateEx(&ctx->dst_dma_fd, &transform_dst_params);
        TEST_ERROR(ret == -1, "create dst_dmabuf failed", ret);

        ctx->vic_converter->init(src_rect, dst_rect);
        ctx->fd_egl_frame_map->init();

        ctx->dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF, ctx->numberCaptureBuffers);
        TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", ret);

        ctx->dec->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", ret);

        for (uint32_t i = 0; i < ctx->dec->capture_plane.getNumBuffers(); i++) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));

            v4l2_buf.index = i;
            v4l2_buf.m.planes = planes;
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            v4l2_buf.m.planes[0].m.fd = ctx->dmaBufferFileDescriptor[i];

            ret = ctx->dec->capture_plane.qBuffer(v4l2_buf, NULL);
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", ret);
        }

        ctx->got_res_event = true;
    }

    void *dec_capture_loop_fcn(void *arg){
        nvJmiCtx* ctx = (nvJmiCtx*)arg;

        struct v4l2_format v4l2Format;
        struct v4l2_crop v4l2Crop;
        struct v4l2_event v4l2Event;
        int ret{};
        int wait_count{};

        while (!(ctx->dec->isInError() || ctx->capture_plane_stop)) {
            NvBuffer *dec_buffer{};

            if (!ctx->got_res_event) {
                ret = ctx->dec->dqEvent(v4l2Event, 1000);
                if (ret == 0) {
                    switch (v4l2Event.type) {
                    case V4L2_EVENT_RESOLUTION_CHANGE:
                        respondToResolutionEvent(v4l2Format, v4l2Crop, ctx);
                        continue;
                    }
                }
                else{
                    ++wait_count;
                    if (wait_count > 10) {
                        ctx->capture_plane_error_code = NVJMI_ERROR_CAPTURE_PLANE_DQEVENT;
                        LogInfo(LOG_NVJMI_DECODER "dqEvent error: capture plane set stopped\n");
                        break;
                    }
                    continue;
                }
            }

            while (!ctx->capture_plane_stop) {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                v4l2_buf.m.planes = planes;

                if (ctx->dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)){
                    if (errno == EAGAIN) {
                        if (ctx->output_plane_stop) {
                            ctx->capture_plane_stop = true;
                            LogInfo(LOG_NVJMI_DECODER "capture plane set stopped\n");
                        }
                        usleep(1000);
                    }
                    else {
                        TEST_ERROR(errno != 0, "Error while calling dequeue at capture plane", errno);
                        ctx->capture_plane_stop = true;
                    }
                    break;
                }

                dec_buffer->planes[0].fd = ctx->dmaBufferFileDescriptor[v4l2_buf.index];

                // do vic conversion conversion: color map convert (NV12@res#1 --> RGBA packed) and scale
                ret = ctx->vic_converter->convert(dec_buffer->planes[0].fd, ctx->dst_dma_fd);        
                TEST_ERROR(ret == -1, "Transform failed", ret);

                //get cuda pointer frm dma fd
                cudaEglFrame egl_frame = ctx->fd_egl_frame_map->get(ctx->dst_dma_fd);

                int buf_index{ -1 };
                while (!ctx->capture_plane_stop && !ctx->frame_pools->try_pop(buf_index)) {
                    std::this_thread::yield();
                }

                if (!ctx->frame_size){
                    ctx->frame_size = ctx->resize_width*ctx->resize_height * 3 * sizeof(unsigned char);
                }

                if (!ctx->capture_plane_stop && (buf_index < MAX_BUFFERS && buf_index >= 0)) {
                    if (ctx->frame_buffer[buf_index] == nullptr){
                        if (!cudaAllocMapped((void**)&ctx->frame_buffer[buf_index], ctx->resize_width, ctx->resize_height, imageFormat::IMAGE_BGR8)) {
                            break;
                        }
                    }

                    // do CUDA conversion: RGBA packed@res#2 --> BGR planar@res#2
                    ctx->cuda_converter->convert(egl_frame,
                        ctx->resize_width,
                        ctx->resize_height,
                        COLOR_FORMAT_BGR,
                        (void *)ctx->frame_buffer[buf_index],
                        ctx->cuda_stream);

                    cudaStreamSynchronize(ctx->cuda_stream);

                    ctx->timestamp[buf_index] = v4l2_buf.timestamp.tv_usec;
                    while (!ctx->capture_plane_stop && !ctx->frames->try_push(buf_index)) {
                        std::this_thread::yield();
                    }
                }
                else{
                    break;
                }

                v4l2_buf.m.planes[0].m.fd = ctx->dmaBufferFileDescriptor[v4l2_buf.index];
                if (ctx->dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0){
                    ERROR_MSG("Error while queueing buffer at decoder capture plane");
                }
            }
        }
		
		ctx->eos = true;
        ctx->output_plane_stop = true;
        ctx->capture_plane_stop = true;
        ctx->dec->capture_plane.setStreamStatus(false);
        LogInfo(LOG_NVJMI_DECODER "capture plane thread stopped\n");
    }

    void *output_plane_stop_fcn(void *arg){
        nvJmiCtx* ctx = (nvJmiCtx*)arg;

        int ret{};
        while (!ctx->output_plane_stop && ctx->dec->output_plane.getNumQueuedBuffers() > 0 && !ctx->dec->isInError()) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));

            v4l2_buf.m.planes = planes;
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
            if (ret < 0) {
                TEST_ERROR(ret < 0, "Eos handling Error: DQing buffer at output plane", ret);
                break;
            }
        }

        ctx->output_plane_stop = true;

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = 0;

        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", ret);
        }
        LogInfo(LOG_NVJMI_DECODER "capture plane stopping ...\n");
    }

    /*
    * NVJMI API
    */
    JMI_API nvJmiCtx* nvjmi_create_decoder(char const* dec_name, nvJmiCtxParam* param) {
        int ret{};
        log_level = DEFAULT_LOG_LEVEL;

        //create nvjmi context
        nvJmiCtx* ctx = new nvJmiCtx;

        ctx->resize_width = param->resize_width;
        ctx->resize_height = param->resize_height;

        //create decoder with specified name
        ctx->dec = NvVideoDecoder::createVideoDecoder(dec_name);
        TEST_ERROR(!ctx->dec, "Could not create decoder", ret);

        // Subscribe to Resolution change event, capture thread wait for this event before getFormat()
        ret = ctx->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
        TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE", ret);

        switch (param->coding_type) {
        case NV_VIDEO_CodingH264:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H264;
            break;
        case NV_VIDEO_CodingHEVC:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H265;
            break;
        case NV_VIDEO_CodingMPEG4:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_MPEG4;
            break;
        case NV_VIDEO_CodingMPEG2:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_MPEG2;
            break;
        case NV_VIDEO_CodingVP8:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_VP8;
            break;
        case NV_VIDEO_CodingVP9:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_VP9;
            break;
        default:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H264;
            break;
        }

        ret = ctx->dec->setOutputPlaneFormat(ctx->decoder_pixfmt, CHUNK_SIZE);
        TEST_ERROR(ret < 0, "Could not set output plane format", ret);

        ret = ctx->dec->setFrameInputMode(0);
        TEST_ERROR(ret < 0, "Error in decoder setFrameInputMode for NALU", ret);

        ret = ctx->dec->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
        TEST_ERROR(ret < 0, "Error while setting up output plane", ret);

        ctx->dec->output_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in output plane stream on", ret);

        ctx->dec_capture_thread = new thread(dec_capture_loop_fcn, ctx);

        ctx->frame_pools = new tbb::concurrent_bounded_queue<int>;
        ctx->frames = new tbb::concurrent_bounded_queue<int>;
        ctx->numberCaptureBuffers = 0;
        ctx->vic_converter = new VICConverter;
        ctx->cuda_converter = new CUDAConverter;
        ctx->fd_egl_frame_map = new FdEglFrameMap;

        //create cuda stream for cuda converter
        auto err = cudaStreamCreateWithFlags(&ctx->cuda_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            LogError(LOG_NVJMI_DECODER "cudaStreamCreateWithFlags: CUDA Runtime API error: %d - %s\n", (int)err, cudaGetErrorString(err));
            return nullptr;
        }

        //create frame buffer pools
        ctx->frame_pools->set_capacity(MAX_BUFFERS);
        ctx->frames->set_capacity(MAX_BUFFERS);
        ctx->frame_pools->clear();
        ctx->frames->clear();
        for (int i = 0; i < MAX_BUFFERS; ++i){
            ctx->frame_pools->push(i);
        }

        return ctx;
    }

    JMI_API int nvjmi_decoder_put_packet(nvJmiCtx* ctx, nvPacket* packet){
        if (ctx->eos){
            if (ctx->capture_plane_stop) {
                return NVJMI_ERROR_STOP;			
			}           
			
			return NVJMI_ERROR_EOS;
        }

        int ret{};

        if (packet->payload_size == 0){
            ctx->eos = true;
            LogInfo(LOG_NVJMI_DECODER "Input file read complete\n");

            ctx->output_plane_stop_thread = new thread(output_plane_stop_fcn, ctx);
            return NVJMI_OK;
        }

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *nvBuffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        if (ctx->index < (int)ctx->dec->output_plane.getNumBuffers()) {
            nvBuffer = ctx->dec->output_plane.getNthBuffer(ctx->index);
        }
        else {
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, &nvBuffer, NULL, -1);
            if (ret < 0) {
                TEST_ERROR(ret < 0, "Error DQing buffer at output plane", ret);
                return NVJMI_ERROR_OUTPUT_PLANE_DQBUF;
            }
        }

        memcpy(nvBuffer->planes[0].data, packet->payload, packet->payload_size);
        nvBuffer->planes[0].bytesused = packet->payload_size;

        if (ctx->index < ctx->dec->output_plane.getNumBuffers()) {
            v4l2_buf.index = ctx->index;
            v4l2_buf.m.planes = planes;
        }

        v4l2_buf.m.planes[0].bytesused = nvBuffer->planes[0].bytesused;

        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        v4l2_buf.timestamp.tv_usec = packet->pts;// - (v4l2_buf.timestamp.tv_sec * (time_t)1000000);

        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", ret);
            return NVJMI_ERROR_OUTPUT_PLANE_QBUF;
        }

        if (ctx->index < ctx->dec->output_plane.getNumBuffers())
            ctx->index++;

        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_get_frame_meta(nvJmiCtx* ctx, nvFrameMeta* frame_meta) {
        int ret{};
        int frame_index{-1};

        if (ctx->dec->isInError()){
            return NVJMI_ERROR_DEC_INTERNAL;
        }

        if (ctx->capture_plane_error_code != NVJMI_OK){
            return ctx->capture_plane_error_code;
        }

        while (ctx->frames->try_pop(frame_index)){
            if (frame_index == -1) {
                return NVJMI_ERROR_GET_FRAME_META;
            }

            frame_meta->coded_width = ctx->coded_width;
            frame_meta->coded_height = ctx->coded_height;
            frame_meta->width = ctx->resize_width;
            frame_meta->height = ctx->resize_height;
            frame_meta->payload_size = ctx->frame_size;
            frame_meta->timestamp = ctx->timestamp[frame_index];
            frame_meta->frame_index = frame_index;
            frame_meta->got_data = 0;

            return ctx->frames->size();
        }

        if (ctx->capture_plane_stop) {
            return NVJMI_ERROR_STOP;
        }

        if (ctx->eos){
            return NVJMI_ERROR_EOS;
        }

        return NVJMI_ERROR_FRAMES_EMPTY;
    }

    JMI_API int nvjmi_decoder_retrieve_frame_data(nvJmiCtx* ctx, nvFrameMeta* frame_meta, void* frame_data){
        if (frame_data){
            memcpy((unsigned char*)frame_data, ctx->frame_buffer[frame_meta->frame_index], frame_meta->payload_size);
            frame_meta->got_data = 1;
        }

        while (!ctx->frame_pools->try_push(frame_meta->frame_index) && !ctx->capture_plane_stop){
            std::this_thread::yield();
        }
        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_close(nvJmiCtx* ctx){
        ctx->eos = true;
        ctx->output_plane_stop = true;
        ctx->capture_plane_stop = true;

        ctx->dec->abort();

        if (ctx->dec_capture_thread && ctx->dec_capture_thread->joinable()) {
            ctx->dec_capture_thread->join();
        }

        if (ctx->output_plane_stop_thread &&ctx->output_plane_stop_thread->joinable()) {
            ctx->output_plane_stop_thread->join();
        }

        LogInfo(LOG_NVJMI_DECODER "------>nvjmi_decoder_close\n");
        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_free_context(nvJmiCtx** ctx) {
        auto& pctx = *ctx;

        if (pctx->dec_capture_thread) {
            delete pctx->dec_capture_thread;
            pctx->dec_capture_thread = nullptr;
        }

        if (pctx->output_plane_stop_thread) {
            if (pctx->output_plane_stop_thread->joinable()) {
                pctx->output_plane_stop_thread->join();
            }          
	    delete pctx->output_plane_stop_thread;
            pctx->output_plane_stop_thread = nullptr;
        }

        delete pctx->dec; pctx->dec = nullptr;

        if (pctx->dst_dma_fd != -1) {
            NvBufferDestroy(pctx->dst_dma_fd);
            pctx->dst_dma_fd = -1;
        }

        for (int idx = 0; idx < pctx->numberCaptureBuffers; idx++) {
            if (pctx->dmaBufferFileDescriptor[idx] != 0) {
                int ret = NvBufferDestroy(pctx->dmaBufferFileDescriptor[idx]);
                TEST_ERROR(ret < 0, "Failed to Destroy NvBuffer", ret);
            }
        }

        pctx->vic_converter->exit();
        pctx->fd_egl_frame_map->exit();

        for (int idx = 0; idx < MAX_BUFFERS; idx++){
            if (pctx->frame_buffer[idx]){
                cudaFreeHost(pctx->frame_buffer[idx]);
                pctx->frame_buffer[idx] = nullptr;
            }
        }

        delete pctx->frame_pools; pctx->frame_pools = nullptr;
        delete pctx->frames; pctx->frames = nullptr;

        delete pctx->vic_converter; pctx->vic_converter = nullptr;
        delete pctx->cuda_converter; pctx->cuda_converter = nullptr;
        delete pctx->fd_egl_frame_map; pctx->fd_egl_frame_map = nullptr;

        delete pctx; pctx = nullptr;

        LogInfo(LOG_NVJMI_DECODER "------>nvjmi_decoder_free_context!!!\n");

        return NVJMI_OK;
    }
}
