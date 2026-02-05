#pragma once

#include <cuda/barrier>

namespace ultra_ep::kernels::ptx {

using mbarrier = cuda::barrier<cuda::thread_scope_block>;
using arrival_phase = uint32_t;

static constexpr int kNumTMAAlignBytes = 16;

// Thread layout
__forceinline__ __device__ int get_warp_idx() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__forceinline__ __device__ int get_lane_idx() {
    int lane_idx;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lane_idx));
    return lane_idx;
}

// Election
__forceinline__ __device__ int elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    int pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_idx() == 0;
#endif
}

// mbarrier
__forceinline__ __device__ mbarrier* create_mbarrier() {
    __shared__ __align__(8) uint64_t mbarrier_storage;
    return reinterpret_cast<mbarrier*>(&mbarrier_storage);
}

template <int kNumBuffers>
__forceinline__ __device__ mbarrier* create_mbarriers() {
    __shared__ __align__(8) uint64_t mbarrier_storage[kNumBuffers];
    return reinterpret_cast<mbarrier*>(mbarrier_storage);
}

__forceinline__ __device__ void mbarrier_init(mbarrier* ptr, const int& arrive_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_invalidate(mbarrier* ptr) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_arrive(mbarrier* ptr) {
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_arrive_and_set_tx(mbarrier* ptr, const int& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_wait_and_flip_phase(mbarrier* ptr, arrival_phase& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase),
        "r"(0x989680));
    phase ^= 1;
}

// TMA primitives
__forceinline__ __device__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(kNumRemainingWaits) : "memory");
}

enum TMACacheHint : int64_t { kEvictFirst = 0x12f0000000000000ll, kEvictNormal = 0x1000000000000000ll };

__forceinline__ __device__ void tma_load_1d(const void* dst_ptr,
                                            const void* src_ptr,
                                            mbarrier* ptr,
                                            const int& num_bytes,
                                            const TMACacheHint& hint = TMACacheHint::kEvictFirst) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "l"(hint)
        : "memory");
}

__forceinline__ __device__ void tma_store_1d(const void* dst_ptr,
                                             const void* src_ptr,
                                             const int& num_bytes,
                                             const TMACacheHint& hint = TMACacheHint::kEvictFirst) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

__forceinline__ __device__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

// Streaming store with EvictFirst hint
__forceinline__ __device__ void st_global_v4_u32_streaming(
    const void* dst_ptr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(dst_ptr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
                 : "memory");
}

}  // namespace ultra_ep::kernels::ptx