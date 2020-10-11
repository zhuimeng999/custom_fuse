//
// Created by lucius on 10/10/20.
//

#ifndef CUSTOM_FUSE_DEPTH_FILTER_KERNEL_CUH
#define CUSTOM_FUSE_DEPTH_FILTER_KERNEL_CUH

#include <auto_ptr.h>
struct ProblemDescGpu {
  int64_t valid_pixel_num;
  int *valid_pixel;

  int width;
  int height;
  int pair_num;

  unsigned char* ref_image;
  float* ref_depth;
  float* ref_prob;
  float* R_inv;
  float* T_inv;

  unsigned char* src_images;
  float* src_depths;
  float* src_Rs;
  float* src_Ts;
  float* src_R_invs;
  float* src_T_invs;
};

void dump_gpu_info();
void depth_filter_kernel_gpu(const ProblemDescGpu &pdg, int *quality, float *points_3d);

#endif //CUSTOM_FUSE_DEPTH_FILTER_KERNEL_CUH
