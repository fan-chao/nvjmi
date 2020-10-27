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

#include "fd_egl_frame_map.h"

#include "cudaEGL.h"
#include "nvbuf_utils.h"

#include <iostream>

using namespace std;

int FdEglFrameMap::init() {
    mEglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (mEglDisplay == EGL_NO_DISPLAY) {
        cerr << "Error while get EGL display connection" << std::endl;
        return false;
    }

    if (!eglInitialize(mEglDisplay, nullptr, nullptr)) {
        cerr << "Erro while initialize EGL display connection" << std::endl;
        return false;
    }
    return true;
}

cudaEglFrame FdEglFrameMap::get(int fd) {
    auto iter = mEglFrameMap.find(fd);

    if (iter == mEglFrameMap.end()) {
        return createMap(fd);
    }
    return iter->second;
}

cudaEglFrame FdEglFrameMap::createMap(int fd) {
    cudaEglFrame eglFrame;
    EGLImageKHR eglImage;
    cudaGraphicsResource_t resource;
    cudaError_t status;

    eglImage = NvEGLImageFromFd(nullptr, fd);
    if (eglImage == nullptr) {
        cerr << "Error while mapping dmabuf fd (" << fd << ") to EGLImage" << endl;
        return eglFrame;
    }

    status = cudaGraphicsEGLRegisterImage(&resource, eglImage,
                                          cudaGraphicsRegisterFlagsNone);
    if (status != cudaSuccess) {
        cerr << "cuGraphicsEGLRegisterImage failed: " << status
             << " cuda process stop" << endl;
        return eglFrame;
    }

    status = cudaGraphicsResourceGetMappedEglFrame(&eglFrame, resource, 0, 0);
    if (status != cudaSuccess) {
        cerr << "cuGraphicsSubResourceGetMappedArray failed" << endl;
        return eglFrame;
    }
    mEglFrameMap.insert(make_pair(fd, eglFrame));
    mEglImageKHRVector.push_back(eglImage);
    mCudaGraphicsResourceVector.push_back(resource);

    return eglFrame;
}

void FdEglFrameMap::exit() {
    cudaError_t status;

    for (auto &eglImage : mEglImageKHRVector) {
        NvDestroyEGLImage(mEglDisplay, eglImage);
    }
    for (auto &resource : mCudaGraphicsResourceVector) {
        status = cudaGraphicsUnregisterResource(resource);
        if (status != cudaSuccess) {
            cerr << "cuGraphicsEGLUnRegisterResource failed: " << status << endl;
        }
    }
    if (mEglDisplay) {
        if (!eglTerminate(mEglDisplay)) {
            cerr << "Error while terminate EGL display connection" << endl;
        }
    }
}
