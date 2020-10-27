#ifndef __CUDA_CONVERTER_H__
#define __CUDA_CONVERTER_H__

#include <cuda_runtime.h>
#include "cuda_egl_interop.h"

enum RGBColorFormat {
    COLOR_FORMAT_RGB,
    COLOR_FORMAT_BGR,
};

//!
//! \brief Class CUDAConverter uses CUDA to do ABGR32 packed(int) to BGR/RGB planar(float) conversion.
//!
class CUDAConverter {
public:
    void convert(cudaEglFrame eglFrame
        , int width
        , int height
        , RGBColorFormat colorFormat
        , void* cudaBuf
        , void* offsets
        , void* scales
        , cudaStream_t stream);

    void convert(cudaEglFrame eglFrame
        , int width
        , int height
        , RGBColorFormat colorFormat
        , void* cudaBuf
        , cudaStream_t stream);
};

#endif // !__CUDA_CONVERTER_H__