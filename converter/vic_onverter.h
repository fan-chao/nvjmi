//!
//! \brief Class VICConverter uses VIC to do YUV to target resolution ABGR32 packed conversion.
//!

#ifndef __VIC_CONVERTER_H__
#define __VIC_CONVERTER_H__

#include "nvbuf_utils.h"
#include <memory.h>

class VICConverter {
public:
    void init(NvBufferRect srcRect, NvBufferRect destRect) {
        memset(&mConvertParams, 0, sizeof(mConvertParams));
        mConvertParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
        mConvertParams.transform_flip = NvBufferTransform_None;
        mConvertParams.transform_filter = NvBufferTransform_Filter_Smart;
        mConvertParams.src_rect = srcRect;
        mConvertParams.dst_rect = destRect;
    }
    void exit() {}
    int convert(int inFd, int outFd) {
        return NvBufferTransform(inFd, outFd, &mConvertParams);
    }

private:
    NvBufferTransformParams mConvertParams;
};

#endif // !__VIC_CONVERTER_H__

