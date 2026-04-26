#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ultra_ep::kernels {

using WeightScalar = __nv_bfloat16;
using GradScalar = float;

inline constexpr int kMaxNvlDomainSize = 72;
inline constexpr int kMaxGradReduceTaskCount = 256;
inline constexpr int kMaxWeightSyncTaskCount = 256;

inline constexpr std::size_t kNvshmemAlignment = 16;

inline constexpr int kWeightElementBytes = sizeof(WeightScalar);
inline constexpr int kGradElementBytes = sizeof(GradScalar);

inline constexpr int kWeightSyncTileSizeBytes = 32 * 1024;
inline constexpr int kWeightSyncTileElements = kWeightSyncTileSizeBytes / kWeightElementBytes;
inline constexpr int kWeightSyncThreadsPerBlock = 256;
inline constexpr int kWeightSyncPipelineStages = 2;
inline constexpr int kWeightSyncRelayChunkTiles = 8;

inline constexpr int kGradReduceTileSizeBytes = 64 * 1024;
inline constexpr int kGradReduceTileElements = kGradReduceTileSizeBytes / kGradElementBytes;
inline constexpr int kGradReducePipelineStages = 2;
inline constexpr int kGradReduceThreadsPerBlock = 256;

inline constexpr int kDenseRerouteTileTokens = 128;
inline constexpr int kDenseRerouteWarpsPerBlock = 8;
inline constexpr int kDenseRerouteThreadsPerBlock = kDenseRerouteWarpsPerBlock * 32;
inline constexpr int kDenseRerouteBackwardRowsPerBlock = 4;

}  // namespace ultra_ep::kernels