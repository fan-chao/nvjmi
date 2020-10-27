/*
* Copyright 1993-2019 NVIDIA Corporation. All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee. Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE. IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users. These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item. Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include "cuda_converter.h"
#include "cudaUtility.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

static __global__ void convertIntRGBAPackedToFloatBGRPlanar(void *pDevPtr
    , int width
    , int height
    , void* cudaBuf
    , int pitch
    , void* meanDataInfer
    , void* scalesInfer) {
    float *pData = (float *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;
    int *meanData = (int *)meanDataInfer;
    float *scales = (float *)scalesInfer;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
        for (int k = 0; k < 3; k++) {
            pData[width * height * k + row * width + col] =
                (float)(*(pSrcData + row * pitch + col * 4 + (3 - 1 - k)) - meanData[k]) * scales[k];
        }
    }
}

static __global__ void convertIntRGBAPackedToFloatRGBPlanar(void *pDevPtr
    , int width
    , int height
    , void* cudaBuf
    , int pitch
    , void* meanDataInfer
    , void* scalesInfer) {
    float *pData = (float *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;
    int *meanData = (int *)meanDataInfer;
    float *scales = (float *)scalesInfer;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
        for (int k = 0; k < 3; k++) {
            pData[width * height * k + row * width + col] =
                (float)(*(pSrcData + row * pitch + col * 4 + k) - meanData[k]) * scales[k];
        }
    }
}

static int convertIntPackedToFloatPlanar(void *pDevPtr,
    int width,
    int height,
    int pitch,
    RGBColorFormat colorFormat,
    void* meanData,
    void* scales,
    void* cudaBuf, void* pStream) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    cudaStream_t stream;

    if (pStream != nullptr) {
        stream = *(cudaStream_t*)pStream;
    }
    else {
        fprintf(stderr, "better not to run on default CUDA stream!\n");
        stream = 0;
    }

    if (colorFormat == COLOR_FORMAT_RGB) {
        convertIntRGBAPackedToFloatRGBPlanar << <blocks, threadsPerBlock, 0, stream >> >(pDevPtr, width,
            height, cudaBuf, pitch, meanData, scales);
    }
    else if (colorFormat == COLOR_FORMAT_BGR) {
        convertIntRGBAPackedToFloatBGRPlanar << <blocks, threadsPerBlock, 0, stream >> >(pDevPtr, width,
            height, cudaBuf, pitch, meanData, scales);
    }

    return 0;
}

//conver RGBA to BGR uchar
static __global__ void convertIntBGRAPackedToBGRPlanar(void *pDevPtr
    , int width
    , int height
    , void* cudaBuf
    , int pitch) {

    unsigned char *pData = (unsigned char *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
    for (int k = 0; k < 3; ++k){
        pData[y*width * 3 + x * 3 + k] = pSrcData[y * pitch + x * 4 + (3 - 1 - k)];
    }
}

//conver RGBA to RGB uchar
static __global__ void convertIntBGRAPackedToRGBPlanar(void *pDevPtr
    , int width
    , int height
    , void* cudaBuf
    , int pitch) {

    unsigned char *pData = (unsigned char *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
    for (int k = 0; k < 3; ++k){
        pData[y*width * 3 + x * 3 + k] = pSrcData[y * pitch + x * 4 + k];
    }
}

static int convertIntPackedToPlanar(void *pDevPtr,
    int width,
    int height,
    int pitch,
    RGBColorFormat colorFormat,
    void* cudaBuf, void* pStream) {

    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));

    cudaStream_t stream;

    if (pStream != nullptr) {
        stream = *(cudaStream_t*)pStream;
    }
    else {
        fprintf(stderr, "better not to run on default CUDA stream!\n");
        stream = 0;
    }

    if (colorFormat == COLOR_FORMAT_RGB) {
        convertIntBGRAPackedToRGBPlanar << <gridDim, blockDim, 0, stream >> >(pDevPtr, width,
            height, cudaBuf, pitch);
    }
    else if (colorFormat == COLOR_FORMAT_BGR) {
        convertIntBGRAPackedToBGRPlanar << <gridDim, blockDim, 0, stream >> >(pDevPtr, width,
            height, cudaBuf, pitch);
    }

    return 0;
}


//!
//! \details call the CUDA kernel to convert one BGRA packed frame to
//!          RGB or BGR planar frame
//!
//! \param eglFrame eglImage that is mapping to the BGRA packed frame
//!
//! \param width width of the frame
//!
//! \param height height of the frame
//!
//! \param colorFormat format of output frame, i.e. RGB or BGR
//!
//! \param cudaBuf CUDA buffer for the output frame
//!
//! \param offsets mean value from inference
//!
//! \param scales scale the float for following inference
//!
void CUDAConverter::convert(cudaEglFrame eglFrame
    , int width
    , int height
    , RGBColorFormat colorFormat
    , void* cudaBuf
    , void* meanData
    , void* scales
    , cudaStream_t stream) {
    if (eglFrame.frameType == cudaEglFrameTypePitch) {
        convertIntPackedToFloatPlanar((void *)eglFrame.frame.pPitch[0].ptr,
            width,
            height,
            eglFrame.frame.pPitch[0].pitch,
            colorFormat,
            meanData,
            scales,
            cudaBuf,
            &stream);
    }
}

void CUDAConverter::convert(cudaEglFrame eglFrame
    , int width
    , int height
    , RGBColorFormat colorFormat
    , void* cudaBuf
    , cudaStream_t stream) {
    if (eglFrame.frameType == cudaEglFrameTypePitch) {
        convertIntPackedToPlanar((void *)eglFrame.frame.pPitch[0].ptr,
            width,
            height,
            eglFrame.frame.pPitch[0].pitch,
            colorFormat,
            cudaBuf,
            &stream);
    }
}