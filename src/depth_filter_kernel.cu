/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* This sample demonstrates how to use texture fetches from layered 2D textures in CUDA C
*
* This sample first generates a 3D input data array for the layered texture
* and the expected output. Then it starts CUDA C kernels, one for each layer,
* which fetch their layer's texture data (using normalized texture coordinates)
* transform it to the expected output, and write it to a 3D output data array.
*/

// includes, system
// includes, kernels
#include <cuda_runtime.h>
#include <boost/log/trivial.hpp>
#include "Options.hpp"
// includes, project
#include "helper_cuda.h"
#include "depth_filter_kernel.cuh"


////////////////////////////////////////////////////////////////////////////////
//! Transform a layer of a layered 2D texture using texture lookups
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
transformKernel(  int valid_pixel_num,
                  int *valid_pixel,
                  int width,
                  int height,
                  int pair_num,

                  cudaTextureObject_t ref_image,
                  float* ref_depth,
                  float* ref_prob,
                  float* R_inv,
                  float* T_inv,

                  cudaTextureObject_t src_images,
                  cudaTextureObject_t src_depths,
                  float* src_Rs,
                  float* src_Ts,
                  float* src_R_invs,
                  float* src_T_invs,

                  float disparity_threshold,
                  int *quality,
                  float *points_3d)
{
  // calculate this thread's data point
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= valid_pixel_num){
    return;
  }
  int h = valid_pixel[idx * 2];
  int w = valid_pixel[idx * 2 + 1];

  float ref_d = ref_depth[h * width + w];
//    float p = ref_prob[h * width + w];
  float h_centor = float(h) + 0.5;
  float w_centor = float(w) + 0.5;
  float h_proj = h_centor * ref_d;
  float w_proj = w_centor * ref_d;

  for(int i = 0; i < pair_num; i++){
    float src_w = src_Rs[i * 9 + 0] * w_proj + src_Rs[i * 9 + 3] * h_proj + src_Rs[i * 9 + 6] * ref_d + src_Ts[i * 3 + 0];
    float src_h = src_Rs[i * 9 + 1] * w_proj + src_Rs[i * 9 + 4] * h_proj + src_Rs[i * 9 + 7] * ref_d + src_Ts[i * 3 + 1];
    float src_d = src_Rs[i * 9 + 2] * w_proj + src_Rs[i * 9 + 5] * h_proj + src_Rs[i * 9 + 8] * ref_d + src_Ts[i * 3 + 2];
    if(src_d <= 0){
      continue;
    }
    src_h = src_h/src_d;
    src_w = src_w/src_d;
    if((src_h > 0) && (src_h < height) && (src_w > 0) && (src_w < width)){
      auto d = tex2DLayered<float>(src_depths, src_w/width, src_h/height, i);
      src_h = src_h * d;
      src_w = src_w * d;
      float w_src_proj = src_R_invs[i * 9 + 0] * src_w + src_R_invs[i * 9 + 3] * src_h + src_R_invs[i * 9 + 6] * d + src_T_invs[i * 3 + 0];
      float h_src_proj = src_R_invs[i * 9 + 1] * src_w + src_R_invs[i * 9 + 4] * src_h + src_R_invs[i * 9 + 7] * d + src_T_invs[i * 3 + 1];
      float d_src_proj = src_R_invs[i * 9 + 2] * src_w + src_R_invs[i * 9 + 5] * src_h + src_R_invs[i * 9 + 8] * d + src_T_invs[i * 3 + 2];
      if(d_src_proj > 0){
        w_src_proj = w_src_proj/d_src_proj;
        h_src_proj = h_src_proj/d_src_proj;
        float delta_w = w_src_proj - w_centor;
        float delta_h = h_src_proj - h_centor;
        if(sqrt(delta_w*delta_w + delta_h * delta_h) < disparity_threshold){
          quality[h * width + w] += 1;
        }
      }
    }
  }

  points_3d[(h * width + w)*3 + 0] = R_inv[0] * w_proj + R_inv[3] * h_proj + R_inv[6] * ref_d - T_inv[0];
  points_3d[(h * width + w)*3 + 1] = R_inv[1] * w_proj + R_inv[4] * h_proj + R_inv[7] * ref_d - T_inv[1];
  points_3d[(h * width + w)*3 + 2] = R_inv[2] * w_proj + R_inv[5] * h_proj + R_inv[8] * ref_d - T_inv[2];
}

void dump_gpu_info()
{
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  int devID = findCudaDevice();

  // get number of SMs on this GPU
  cudaDeviceProp deviceProps;

  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  BOOST_LOG_TRIVIAL(info) <<"CUDA device [" << deviceProps.name << "] has "<< deviceProps.multiProcessorCount << " Multi-Processors ";
  BOOST_LOG_TRIVIAL(info) <<"SM " << deviceProps.major << "." << deviceProps.minor;
}

inline void *upload_buffer(const void *data, size_t size)
{
  void *d;
  checkCudaErrors(cudaMalloc((void **)&d, size));
  checkCudaErrors(cudaMemcpy(d, data, size, cudaMemcpyHostToDevice));
  return d;
}

void depth_filter_kernel_gpu(const ProblemDescGpu &pdg, int *quality, float *points_3d)
{
  auto pair_num = pdg.pair_num;
  auto height = pdg.height;
  auto width = pdg.width;

  unsigned int image_size = width * height * sizeof(float);
  unsigned int size = image_size*pair_num;

  int *d_valid_pixel;
  checkCudaErrors(cudaMalloc((void **)&d_valid_pixel, pdg.valid_pixel_num * 2 * sizeof(int)));
  checkCudaErrors(cudaMemcpy(d_valid_pixel, pdg.valid_pixel, pdg.valid_pixel_num * 2 * sizeof(int), cudaMemcpyHostToDevice));

  cudaTextureObject_t         d_ref_tex;
  // Allocate array and copy image data
//  cudaChannelFormatDesc ref_channelDesc =
//      cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindFloat);
//  unsigned char *ref_image_rgba = static_cast<unsigned char *>(malloc(width * height * 4));
//
//  cudaArray *d_ref_array;
//  checkCudaErrors(cudaMallocArray(&d_ref_array,
//                                  &ref_channelDesc,
//                                  width,
//                                  height));
//  checkCudaErrors(cudaMemcpyToArray(d_ref_array,
//                                    0,
//                                    0,
//                                    pdg.ref_image,
//                                    width * height * 4,
//                                    cudaMemcpyHostToDevice));
//
//  cudaResourceDesc            d_ref_texRes;
//  memset(&d_ref_texRes,0,sizeof(cudaResourceDesc));
//
//  d_ref_texRes.resType            = cudaResourceTypeArray;
//  d_ref_texRes.res.array.array    = d_ref_array;
//
//  cudaTextureDesc             d_ref_texDescr;
//  memset(&d_ref_texDescr,0,sizeof(cudaTextureDesc));
//
//  d_ref_texDescr.normalizedCoords = true;
//  d_ref_texDescr.filterMode       = cudaFilterModeLinear;
//  d_ref_texDescr.addressMode[0] = cudaAddressModeWrap;
//  d_ref_texDescr.addressMode[1] = cudaAddressModeWrap;
//  d_ref_texDescr.readMode = cudaReadModeElementType;
//
//  checkCudaErrors(cudaCreateTextureObject(&d_ref_tex, &d_ref_texRes, &d_ref_texDescr, NULL));

  float* d_ref_depth;
  checkCudaErrors(cudaMalloc((void **)&d_ref_depth, width * height * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_ref_depth, pdg.ref_depth, width * height * sizeof(float), cudaMemcpyHostToDevice));

  float* d_ref_prob;
  checkCudaErrors(cudaMalloc((void **)&d_ref_prob, width * height * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_ref_prob, pdg.ref_prob, width * height * sizeof(float), cudaMemcpyHostToDevice));

  float* d_R_inv;
  checkCudaErrors(cudaMalloc((void **)&d_R_inv, 9 * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_R_inv, pdg.R_inv, 9 * sizeof(float), cudaMemcpyHostToDevice));

  float* d_T_inv;
  checkCudaErrors(cudaMalloc((void **)&d_T_inv, 3 * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_T_inv, pdg.T_inv, 3 * sizeof(float), cudaMemcpyHostToDevice));

  cudaTextureObject_t d_src_texs;

  cudaTextureObject_t d_src_depths_tex;
  cudaArray_t d_src_depths_array;
  // allocate array and copy image data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  checkCudaErrors(cudaMalloc3DArray(&d_src_depths_array, &channelDesc, make_cudaExtent(width, height, pair_num), cudaArrayLayered));

  cudaMemcpy3DParms myparms = {0};
  myparms.srcPos = make_cudaPos(0,0,0);
  myparms.dstPos = make_cudaPos(0,0,0);
  myparms.srcPtr = make_cudaPitchedPtr(pdg.src_depths, width * sizeof(float), width, height);
  myparms.dstArray = d_src_depths_array;
  myparms.extent = make_cudaExtent(width, height, pair_num);
  myparms.kind = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&myparms));

  cudaResourceDesc            d_src_depths_texRes;
  memset(&d_src_depths_texRes,0,sizeof(cudaResourceDesc));

  d_src_depths_texRes.resType            = cudaResourceTypeArray;
  d_src_depths_texRes.res.array.array    = d_src_depths_array;

  cudaTextureDesc             d_src_depths_texDescr;
  memset(&d_src_depths_texDescr,0,sizeof(cudaTextureDesc));

  d_src_depths_texDescr.normalizedCoords = true;
  d_src_depths_texDescr.filterMode       = cudaFilterModeLinear;
  d_src_depths_texDescr.addressMode[0] = cudaAddressModeWrap;
  d_src_depths_texDescr.addressMode[1] = cudaAddressModeWrap;
  d_src_depths_texDescr.readMode = cudaReadModeElementType;

  checkCudaErrors(cudaCreateTextureObject(&d_src_depths_tex, &d_src_depths_texRes, &d_src_depths_texDescr, NULL));

  float* d_src_Rs;
  checkCudaErrors(cudaMalloc((void **)&d_src_Rs, 9 * sizeof(float) * pair_num));
  checkCudaErrors(cudaMemcpy(d_src_Rs, pdg.src_Rs, 9 * sizeof(float) * pair_num, cudaMemcpyHostToDevice));

  float* d_src_Ts;
  checkCudaErrors(cudaMalloc((void **)&d_src_Ts, 3 * sizeof(float) * pair_num));
  checkCudaErrors(cudaMemcpy(d_src_Ts, pdg.src_Ts, 3 * sizeof(float) * pair_num, cudaMemcpyHostToDevice));

  float* d_src_R_invs;
  checkCudaErrors(cudaMalloc((void **)&d_src_R_invs, 9 * sizeof(float) * pair_num));
  checkCudaErrors(cudaMemcpy(d_src_R_invs, pdg.src_R_invs, 9 * sizeof(float) * pair_num, cudaMemcpyHostToDevice));

  float* d_src_T_invs;
  checkCudaErrors(cudaMalloc((void **)&d_src_T_invs, 3 * sizeof(float) * pair_num));
  checkCudaErrors(cudaMemcpy(d_src_T_invs, pdg.src_T_invs, 3 * sizeof(float) * pair_num, cudaMemcpyHostToDevice));

  int *d_quality;
  checkCudaErrors(cudaMalloc((void **) &d_quality, image_size));
  checkCudaErrors(cudaMemset(d_quality, 0, image_size));

  float *d_points_3d;
  checkCudaErrors(cudaMalloc((void **) &d_points_3d, width * height * 3 * sizeof(float)));
//  checkCudaErrors(cudaMemset(d_points_3d, 0.0f, width * height * 3 * sizeof(int)));

  int blockSize;
  int minGridSize;
  size_t dynamicSMemUsage = 0;

  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize,
      (void*)transformKernel,
      dynamicSMemUsage,
      pdg.valid_pixel_num));
  auto gridSize = (pdg.valid_pixel_num + blockSize - 1) / blockSize;
//  BOOST_LOG_TRIVIAL(info) << gridSize << " " << blockSize;
  transformKernel<<< gridSize, blockSize, 0 >>>(pdg.valid_pixel_num,
                    d_valid_pixel,
                    width,
                    height,
                    pair_num,

                    d_ref_tex,
                    d_ref_depth,
                    d_ref_prob,
                    d_R_inv,
                    d_T_inv,

                    d_src_texs,
                    d_src_depths_tex,
                    d_src_Rs,
                    d_src_Ts,
                    d_src_R_invs,
                    d_src_T_invs,

                    options.disparity_threshold,
                    d_quality,
                    d_points_3d);  // warmup (for better timing)

  // check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());

  // copy result from device to host
  checkCudaErrors(cudaMemcpy(quality, d_quality, width * height * sizeof(int ), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(points_3d, d_points_3d, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_valid_pixel));
//  checkCudaErrors(cudaDestroyTextureObject(d_ref_tex));
//  checkCudaErrors(cudaFreeArray(d_ref_array));
  checkCudaErrors(cudaFree(d_ref_depth));
  checkCudaErrors(cudaFree(d_ref_prob));
  checkCudaErrors(cudaFree(d_R_inv));
  checkCudaErrors(cudaFree(d_T_inv));

  d_src_texs;
  checkCudaErrors(cudaDestroyTextureObject(d_src_depths_tex));
  checkCudaErrors(cudaFreeArray(d_src_depths_array));
  checkCudaErrors(cudaFree(d_src_Rs));
  checkCudaErrors(cudaFree(d_src_Ts));
  checkCudaErrors(cudaFree(d_src_R_invs));
  checkCudaErrors(cudaFree(d_src_T_invs));

//  exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
