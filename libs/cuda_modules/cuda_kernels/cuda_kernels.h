
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#pragma once

#include <cstdint>

#include <driver_types.h>
#include <texture_types.h>
#include <vector_types.h>

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"

#define PATCH_DIM 9
#define PATCH_HALF 4.5f

#define PYRAMID_LEVELS 10

namespace cuvslam::cuda {

struct Size {
  int width, height;
};

cudaError_t init_conv_kernels();
cudaError_t init_box_prefilter_kernels();
cudaError_t init_gauss_coeffs();
cudaError_t init_gauss_square_kernel();
cudaError_t init_scaler();
cudaError_t init_st_tracker();

cudaError_t conv_grad_x(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s);

cudaError_t conv_grad_y(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s);

cudaError_t box_blur_x(cudaTextureObject_t src, uint2 srcSize, float* buffer, size_t buffer_pitch, cudaStream_t s);

cudaError_t box_blur_y(cudaTextureObject_t src, uint2 srcSize, float* buffer, size_t buffer_pitch, cudaStream_t s);

cudaError_t copy_border(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, size_t border_size,
                        cudaStream_t s);

cudaError_t gaussian_scaling(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, uint2 dstSize,
                             cudaStream_t stream);

struct Pyramid {
  int levels;
  cudaTextureObject_t level_tex[PYRAMID_LEVELS];  // max 10 levels
  Size level_sizes[PYRAMID_LEVELS];
};

template <typename T, int Dim>
struct PointCacheData {
  T patch[Dim * Dim * PYRAMID_LEVELS];
  T patch_sums[PYRAMID_LEVELS];
  uint32_t level_mask = 0;
};

using PointCacheDataT = PointCacheData<float, PATCH_DIM>;

struct TrackData {
  float2 track;
  float2 offset;
  float info[4];
  bool track_status;
  int cache_index = -1;

  float4 initial_guess_map = {1, 0, 0, 1};
  float search_radius_px = 2048.f;
  float ncc_threshold = 0.8f;
  float ncc;
};

cudaError_t lk_track(Pyramid prevFrameGradXPyramid, Pyramid prevFrameGradYPyramid, Pyramid prevFrameImagePyramid,
                     Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks, cudaStream_t stream);

cudaError_t lk_track_horizontal(Pyramid prevFrameGradXPyramid, Pyramid prevFrameImagePyramid,
                                Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks,
                                cudaStream_t stream);

cudaError_t st_track(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid prevImagePyramid,
                     Pyramid currentImagePyramid, TrackData* track_data, int num_tracks,
                     unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations, cudaStream_t stream);

cudaError_t st_build_cache(Pyramid previous_image, TrackData* tracks, PointCacheDataT* cache_data, size_t num_tracks,
                           cudaStream_t stream);

cudaError_t st_track_with_cache(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid currentImagePyramid,
                                PointCacheDataT* cache_data, TrackData* track_data, int num_tracks,
                                unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations,
                                cudaStream_t stream);

cudaError_t gftt_values(cudaTextureObject_t gradX, cudaTextureObject_t gradY, float* values, size_t v_pitch,
                        uint2 image_size, cudaStream_t stream);

struct Keypoint {
  float x, y;
  float strength;
};

cudaError_t downsample_gftt_x8(float* in, size_t in_pitch, float* out, uint2 out_size, size_t out_pitch,
                               uint2* out_indices, size_t out_indices_pitch, cudaStream_t s);

cudaError_t non_max_suppression(cudaTextureObject_t gftt, uint2 size, float* measure, size_t measure_pitch,
                                cudaStream_t s);

cudaError_t filter_maximums(float* measure, uint2 size, size_t measure_pitch, const Keypoint* kp, size_t kpCount,
                            cudaStream_t s);

cudaError_t select_features(float* gftt, uint2 gftt_size, size_t gftt_pitch, float* measure, uint2 size,
                            size_t measure_pitch, uint2* indices, size_t indices_pitch, Keypoint* kp, size_t kpCapacity,
                            int* kpCount, int* kpIndex, cudaStream_t s);

cudaError_t accumulateGFTT(cudaTextureObject_t gftt, uint2 size, uint2 bin_size, uint2 num_bins, float* gtff_accum,
                           cudaStream_t s);

cudaError_t cast_image(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s);

cudaError_t cast_depth_u16(const uint16_t* src, size_t spitch, float scale, float* dst, size_t dpitch, uint2 size,
                           cudaStream_t s);

cudaError_t burn_depth_mask(float* dst, size_t dpitch, uint8_t* mask, size_t mpitch, const uint2& size, cudaStream_t s);

cudaError_t cast_image_rgb(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s);

cudaError_t resize_mask(const uint8_t* src, uint2 src_size, size_t spitch, uint8_t* dst, uint2 dst_size, size_t dpitch,
                        cudaStream_t s);

void sort_keypoints(Keypoint* kp, size_t size, void* temp_buffer, size_t temp_buffer_size, cudaStream_t s);

size_t sort_keypoints_get_temp_buffer_size(size_t size);

struct Inctinsics {
  float focal_x;
  float focal_y;

  float principal_x;
  float principal_y;

  int size_x;
  int size_y;
};

struct Extrinsics {
  Pose cam_from_world;
  Pose world_from_cam;
};

struct ImgTextures {
  cudaTextureObject_t curr_depth;   // tex with linear interpolation!
  cudaTextureObject_t curr_image;   // tex with linear interpolation!
  cudaTextureObject_t curr_grad_x;  // tex with linear interpolation!
  cudaTextureObject_t curr_grad_y;  // tex with linear interpolation!
};

struct Track {
  float2 obs_xy;  // observation in image frame
  float3 lm_xyz;  // landmark in world frame
};

cudaError_t photometric(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                        const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid, float* rhs,
                        float* Hessian, cudaStream_t stream);

cudaError_t point_to_point(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                           const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid,
                           float* rhs, float* Hessian, cudaStream_t stream);

cudaError_t lift(cudaTextureObject_t curr_depth, const Inctinsics& intrinsics, const Track* tracks, size_t num_tracks,
                 float3* landmarks, cudaStream_t stream);

}  // namespace cuvslam::cuda
