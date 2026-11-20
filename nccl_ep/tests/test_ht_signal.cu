/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HT expert-major dispatch and combine signal tests. Require NCCL_EP_HT_PULL_PUSH=1 (skipped otherwise).
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include "device/device_primitives.cuh"
#include <cuda/ptx>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <numeric>
#include <tuple>
#include <vector>

// HT group config of the test groups, for tokens of up to max_token_bytes.
static ncclEpGroupConfig_t ep_group_config(size_t max_token_bytes) {
    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank = kNumTokens;
    gcfg.max_token_bytes = max_token_bytes;
    gcfg.rdma_buffer_size = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank = NCCL_EP_AUTO;
    gcfg.num_channels = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
    return gcfg;
}

static constexpr size_t kAlign = 4;     // dispatch per-expert zone padding
static constexpr int kTileSize = 2;     // dispatch rows per tile; divides kAlign
static constexpr int kScaleBytes = 16;  // forwarded scale row
static constexpr int kSigHidden = 4096; // combine row: four push tiles of 1024 elements
static constexpr const char* kChunkEnv = "NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE";
static constexpr int kDropBudget = 4;
static constexpr int kLateUs = 20000;   // a late peer's producer delay, far beyond a producer's and a combine's runtime
static constexpr int kPadTile = 8;      // alignment and tile of the drop and eager dispatch tests; > kDropBudget

#define REQUIRE_PULL_PUSH()                                                             \
    do {                                                                                \
        if (!ht_pull_push_active()) {                                                   \
            GTEST_SKIP() << "signals require NCCL_EP_HT_PULL_PUSH=1";                   \
        }                                                                               \
        if (env_flag_active("NCCL_EP_HT_UNFUSED_SYNC")) {                               \
            GTEST_SKIP() << "signals require fused sync (not NCCL_EP_HT_UNFUSED_SYNC)"; \
        }                                                                               \
    } while (0)

// Sets an env var for its scope and restores the prior value on destruction.
struct ScopedEnv {
    std::string name, prev;
    bool had_prev = false;
    ScopedEnv(const char* n, const char* v) : name(n) {
        if (const char* p = std::getenv(n)) {
            had_prev = true;
            prev = p;
        }
        setenv(n, v, 1);
    }
    ~ScopedEnv() {
        if (had_prev) {
            setenv(name.c_str(), prev.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

// Launches kernel on stream as the programmatic dependent of the work before it.
template <typename... Params, typename... Args>
static cudaError_t launch_programmatic(void (*kernel)(Params...), dim3 block, cudaStream_t stream, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = block;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <typename T>
static std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

// ---------------------------------------------------------------- dispatch

// The dispatch's programmatic dependent: acquires tiles in order and copies their rows, weights and scales.
__global__ void consumeTilesKernel(const int32_t* signal, int tiles, const int4* rows, const float* weights,
                                   const int4* scales, int row_int4, int4* rows_seen, float* weights_seen,
                                   int4* scales_seen) {
    for (int t = 0; t < tiles; ++t) {
        if (threadIdx.x == 0) {
            while (nccl_ep::ld_acquire_global(&signal[t]) != t) {
            }
        }
        __syncthreads();
        const int first = t * kTileSize;
        for (int i = threadIdx.x; i < kTileSize * row_int4; i += blockDim.x) {
            rows_seen[first * row_int4 + i] = rows[first * row_int4 + i];
        }
        for (int i = threadIdx.x; i < kTileSize; i += blockDim.x) {
            weights_seen[first + i] = weights[first + i];
            scales_seen[first + i] = scales[first + i];
        }
    }
    nccl_ep::grid_dependency_wait();
}

// Lets its programmatic dependent start, then doubles n weights: only the dependent's wait orders the writes first.
__global__ void doubleWeightsKernel(float* w, int n) {
    nccl_ep::launch_dependents();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        w[i] *= 2.0f;
    }
}

// Device buffers of one forward EM dispatch with forwarded scales and a dispatch_signal.
struct DispatchBuffers {
    static constexpr int kRowHidden = kHidden / 2;  // a 16-byte token row beside a 16-byte scale row
    int4 *d_tok = nullptr, *d_scales = nullptr, *d_recv = nullptr, *d_recv_scales = nullptr;
    float *d_w = nullptr, *d_recv_w = nullptr;
    int32_t* d_signal = nullptr;
    ncclEpTensor_t *t_tok = nullptr, *t_w = nullptr, *t_scales = nullptr, *t_recv = nullptr, *t_recv_w = nullptr,
                   *t_recv_scales = nullptr;
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;

    void create() {
        std::vector<nv_bfloat16> tok(kNumTokens * kRowHidden);
        std::vector<float> w(kNumTokens * kTopK);
        for (size_t i = 0; i < tok.size(); ++i) {
            tok[i] = __float2bfloat16(static_cast<float>(g_rank * 1000 + i + 1));
        }
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = 0.01f * static_cast<float>(g_rank * kNumTokens + i + 1);
        }
        CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * sizeof(int4)));
        CUDA_ASSERT(cudaMalloc(&d_scales, kNumTokens * sizeof(int4)));
        CUDA_ASSERT(cudaMalloc(&d_w, w.size() * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv, kMaxRecvSlots * sizeof(int4)));
        CUDA_ASSERT(cudaMalloc(&d_recv_scales, kMaxRecvSlots * sizeof(int4)));
        CUDA_ASSERT(cudaMalloc(&d_recv_w, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_signal, kMaxRecvSlots / kTileSize * sizeof(int32_t)));
        CUDA_ASSERT(cudaMemcpy(d_tok, tok.data(), tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemset(d_scales, g_rank + 1, kNumTokens * sizeof(int4)));
        NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kRowHidden));
        NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_w, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_scales, 2, ncclUint8, d_scales, kNumTokens, kScaleBytes));
        NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kRowHidden));
        NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots));
        NCCL_ASSERT(epTensorCreate(&t_recv_scales, 2, ncclUint8, d_recv_scales, kMaxRecvSlots, kScaleBytes));
        NCCL_ASSERT(epTensorCreate(&sig.signal, 1, ncclInt32, d_signal, kMaxRecvSlots / kTileSize));
        sig.granularity = kTileSize;
        in.tokens = t_tok;
        in.topk_weights = t_w;
        in.scales = t_scales;
        out.tokens = t_recv;
        out.topk_weights = t_recv_w;
        out.scales = t_recv_scales;
        out.dispatch_signal = &sig;
        cfg.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    }

    // Resets the flags to -1 and poisons the outputs, on g_stream.
    void reset() {
        CUDA_ASSERT(cudaMemsetAsync(d_signal, 0xFF, kMaxRecvSlots / kTileSize * sizeof(int32_t), g_stream));
        CUDA_ASSERT(cudaMemsetAsync(d_recv, 0xAB, kMaxRecvSlots * sizeof(int4), g_stream));
        CUDA_ASSERT(cudaMemsetAsync(d_recv_scales, 0xAB, kMaxRecvSlots * sizeof(int4), g_stream));
        CUDA_ASSERT(cudaMemsetAsync(d_recv_w, 0xAB, kMaxRecvSlots * sizeof(float), g_stream));
    }

    void destroy() {
        for (ncclEpTensor_t* t : {sig.signal, t_recv_scales, t_recv_w, t_recv, t_scales, t_w, t_tok}) {
            ncclEpTensorDestroy(t);
        }
        for (void* p : {static_cast<void*>(d_signal), static_cast<void*>(d_recv_w), static_cast<void*>(d_recv_scales),
                        static_cast<void*>(d_recv), static_cast<void*>(d_w), static_cast<void*>(d_scales),
                        static_cast<void*>(d_tok)}) {
            cudaFree(p);
        }
    }
};

class DispatchSignalTest : public EpTestBase {
protected:
    ncclEpHandle_t h_ = nullptr;
    DispatchBuffers b_;

    void SetUp() override {
        REQUIRE_PULL_PUSH();
        EpTestBase::SetUp();
        ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
        cfg.dispatch_output_per_expert_alignment = kAlign;
        h_ = make_handle_em(&cfg);
        ASSERT_NE(h_, nullptr);
        ASSERT_NO_FATAL_FAILURE(b_.create());
    }

    void TearDown() override {
        if (h_ != nullptr) {
            b_.destroy();
            ncclEpHandleDestroy(h_);
        }
        EpTestBase::TearDown();
    }

    // Flags of the padded output: signal[t] == t below `tiles`, -1 above.
    void expect_flags(int tiles) {
        const std::vector<int32_t> flags = to_host(b_.d_signal, kMaxRecvSlots / kTileSize);
        for (int t = 0; t < static_cast<int>(flags.size()); ++t) {
            EXPECT_EQ(flags[t], t < tiles ? t : -1) << "tile " << t;
        }
    }

    // Unquantized forward dispatch of handle h, signaled or plain, then its received row count.
    unsigned int dispatch_forward(ncclEpHandle_t h, bool signaled) {
        b_.cfg.quant_recipe = NCCL_EP_DISP_QUANT_NONE;
        b_.in.scales = nullptr;
        b_.out.scales = nullptr;
        b_.out.dispatch_signal = signaled ? &b_.sig : nullptr;
        b_.reset();
        unsigned int rows = 0;
        EXPECT_EQ(ncclEpDispatch(h, &b_.in, &b_.out, nullptr, &b_.cfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        EXPECT_EQ(ncclEpHandle_test_getNumRecvTokens(h, &rows), ncclSuccess);
        b_.out.dispatch_signal = &b_.sig;
        return rows;
    }

    // Routing that gives every rank a token count other than the fixture's: token 0 to expert 0, the rest to the next
    // rank's first expert. Free the tensor and *d_topk after use.
    ncclEpTensor_t* shifted_topk(int64_t** d_topk) {
        std::vector<int64_t> topk(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i) {
            topk[i] = i == 0 ? 0 : ((g_rank + 1) % g_nranks) * (kNumExperts / g_nranks);
        }
        ncclEpTensor_t* topk_idx = nullptr;
        EXPECT_EQ(cudaMalloc(d_topk, topk.size() * sizeof(int64_t)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(*d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        EXPECT_EQ(epTensorCreate(&topk_idx, 2, ncclInt64, *d_topk, kNumTokens, kTopK), ncclSuccess);
        return topk_idx;
    }

    // Signaled backward dispatch of handle h: every tile is flagged and the rows equal the plain backward's.
    void expect_backward_signaled(ncclEpHandle_t h, unsigned int rows) {
        int4* d_ref = nullptr;
        ncclEpTensor_t* t_ref = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_ref, kMaxRecvSlots * sizeof(int4)));
        NCCL_ASSERT(epTensorCreate(&t_ref, 2, ncclBfloat16, d_ref, kMaxRecvSlots, DispatchBuffers::kRowHidden));
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        in.tokens = b_.t_tok;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        out.tokens = b_.t_recv;
        out.dispatch_signal = &b_.sig;
        ncclEpDispatchOutputs_t ref = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ref.tokens = t_ref;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.pass_direction = NCCL_EP_BWD_PASS;
        b_.reset();
        NCCL_ASSERT(ncclEpDispatch(h, &in, &out, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(d_ref, 0xCD, kMaxRecvSlots * sizeof(int4), g_stream));
        NCCL_ASSERT(ncclEpDispatch(h, &in, &ref, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        expect_flags(static_cast<int>(rows) / kTileSize);
        const std::vector<int4> got = to_host(b_.d_recv, rows), want = to_host(d_ref, rows);
        EXPECT_EQ(std::memcmp(got.data(), want.data(), rows * sizeof(int4)), 0);
        ncclEpTensorDestroy(t_ref);
        cudaFree(d_ref);
    }

    // Capture body on g_stream into an instantiated graph.
    template <class Body>
    cudaGraphExec_t capture_graph(Body body) {
        cudaGraph_t graph = nullptr;
        EXPECT_EQ(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        body();
        EXPECT_EQ(cudaStreamEndCapture(g_stream, &graph), cudaSuccess);
        cudaGraphExec_t exec = nullptr;
        EXPECT_EQ(cudaGraphInstantiateWithFlags(&exec, graph, 0), cudaSuccess);
        cudaGraphDestroy(graph);
        return exec;
    }

    // Routing that permutes each rank's token-to-expert map by swapping adjacent tokens, so every rank's receive count
    // is unchanged while the source permutation differs. Free the tensor and *d_topk after use.
    ncclEpTensor_t* swapped_topk(int64_t** d_topk) {
        std::vector<int64_t> topk(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i) {
            topk[i] = (g_rank * kNumTokens + (i ^ 1)) % kNumExperts;
        }
        ncclEpTensor_t* topk_idx = nullptr;
        EXPECT_EQ(cudaMalloc(d_topk, topk.size() * sizeof(int64_t)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(*d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        EXPECT_EQ(epTensorCreate(&topk_idx, 2, ncclInt64, *d_topk, kNumTokens, kTopK), ncclSuccess);
        return topk_idx;
    }
};

// Every tile of the padded output is flagged once complete, and a consumer launched as the dispatch's programmatic
// dependent sees the final rows, weights and scales of each tile it acquires. Invalid tile sizes and too few flags
// are rejected.
TEST_F(DispatchSignalTest, ProgrammaticConsumerSeesTiles) {
    b_.reset();
    NCCL_ASSERT(ncclEpDispatch(h_, &b_.in, &b_.out, nullptr, &b_.cfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    unsigned int rows = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h_, &rows));
    ASSERT_GT(rows, 0u);
    const int tiles = static_cast<int>(rows) / kTileSize;
    expect_flags(tiles);

    int4 *d_rows_seen = nullptr, *d_scales_seen = nullptr;
    float* d_w_seen = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_rows_seen, rows * sizeof(int4)));
    CUDA_ASSERT(cudaMalloc(&d_scales_seen, rows * sizeof(int4)));
    CUDA_ASSERT(cudaMalloc(&d_w_seen, rows * sizeof(float)));
    b_.reset();
    NCCL_ASSERT(ncclEpDispatch(h_, &b_.in, &b_.out, nullptr, &b_.cfg, g_stream));
    CUDA_ASSERT(launch_programmatic(consumeTilesKernel, dim3(64), g_stream, static_cast<const int32_t*>(b_.d_signal),
                                    tiles, static_cast<const int4*>(b_.d_recv), static_cast<const float*>(b_.d_recv_w),
                                    static_cast<const int4*>(b_.d_recv_scales), 1, d_rows_seen, d_w_seen,
                                    d_scales_seen));
    NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(tiles);
    const auto bytes = [](const int4* d, size_t n) {
        std::vector<int4> v = to_host(d, n);
        return std::vector<uint8_t>(reinterpret_cast<uint8_t*>(v.data()), reinterpret_cast<uint8_t*>(v.data() + n));
    };
    const std::vector<uint8_t> recv = bytes(b_.d_recv, rows);
    EXPECT_EQ(bytes(d_rows_seen, rows), recv);
    EXPECT_EQ(bytes(d_scales_seen, rows), bytes(b_.d_recv_scales, rows));
    EXPECT_EQ(to_host(d_w_seen, rows), to_host(b_.d_recv_w, rows));
    EXPECT_EQ(std::count(recv.begin(), recv.end(), 0xAB), 0) << "a row was not written";
    cudaFree(d_w_seen);
    cudaFree(d_scales_seen);
    cudaFree(d_rows_seen);
    for (int tile : {0, 3, 8, 65536}) {
        b_.sig.granularity = tile;
        EXPECT_EQ(ncclEpDispatch(h_, &b_.in, &b_.out, nullptr, &b_.cfg, g_stream), ncclInvalidArgument) << tile;
    }
    b_.sig.granularity = 1;  // needs kMaxRecvSlots flags
    EXPECT_EQ(ncclEpDispatch(h_, &b_.in, &b_.out, nullptr, &b_.cfg, g_stream), ncclInvalidArgument);
}

// The signaled backward flags every tile and writes the plain backward's rows after a signaled forward, after a plain
// forward, with two interleaved handles of different receive counts, and after a plain forward under new routing
// replaced the maps of the signaled one.
TEST_F(DispatchSignalTest, BackwardSignaled) {
    expect_backward_signaled(h_, dispatch_forward(h_, true));
    expect_backward_signaled(h_, dispatch_forward(h_, false));

    int64_t* d_topk = nullptr;
    ncclEpTensor_t* topk_idx = shifted_topk(&d_topk);
    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;
    ncclEpHandle_t h2 = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&h2, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx, nullptr, &cfg,
                                   g_stream));
    unsigned int rows = dispatch_forward(h_, true);
    const unsigned int rows2 = dispatch_forward(h2, true);
    expect_backward_signaled(h_, rows);
    expect_backward_signaled(h2, rows2);
    NCCL_ASSERT(ncclEpHandleDestroy(h2));

    NCCL_ASSERT(ncclEpUpdateHandle(h_, topk_idx, nullptr, g_stream));
    rows = dispatch_forward(h_, false);
    expect_backward_signaled(h_, rows);
    ncclEpTensorDestroy(topk_idx);
    cudaFree(d_topk);
}

// Weights a kernel writes directly before a forward dispatch reach the peers: an EM dispatch after a signaled one,
// which the group then launches programmatically, and a flat dispatch. Each layout matches a dispatch of the same
// weights after a sync, and differs from the one before the write.
TEST_F(DispatchSignalTest, PredecessorWeightsReachPeers) {
    ncclEpHandle_t hf = make_handle(nullptr);
    ASSERT_NE(hf, nullptr);
    int64_t* d_recv_idx = nullptr;
    ncclEpTensor_t *t_recv_w = nullptr, *t_recv_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 2, ncclFloat32, b_.d_recv_w, kMaxRecvSlots, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64, d_recv_idx, kMaxRecvSlots, kTopK));
    ncclEpDispatchOutputs_t flat_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    flat_out.tokens = b_.t_recv;
    flat_out.topk_weights = t_recv_w;
    flat_out.topk_idx = t_recv_idx;

    dispatch_forward(h_, true);  // later EM forwards of the group launch programmatically
    // Received weights of one dispatch, sorted, optionally after a kernel that doubles the source weights.
    const auto received = [&](ncclEpHandle_t h, ncclEpDispatchOutputs_t* out, bool write_first) {
        b_.reset();
        if (write_first) {
            doubleWeightsKernel<<<1, 64, 0, g_stream>>>(b_.d_w, kNumTokens * kTopK);
            EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        }
        EXPECT_EQ(ncclEpDispatch(h, &b_.in, out, nullptr, &b_.cfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(h, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        unsigned int rows = 0;
        EXPECT_EQ(ncclEpHandle_test_getNumRecvTokens(h, &rows), ncclSuccess);
        EXPECT_GT(rows, 0u);
        const size_t n = out == &flat_out ? rows * kTopK : rows;
        std::vector<float> w = to_host(b_.d_recv_w, n);
        std::sort(w.begin(), w.end());
        return w;
    };
    for (ncclEpHandle_t h : {h_, hf}) {
        ncclEpDispatchOutputs_t* out = h == h_ ? &b_.out : &flat_out;
        const std::vector<float> before = received(h, out, false);
        const std::vector<float> after = received(h, out, true);
        EXPECT_NE(after, before) << (h == h_ ? "EM" : "flat");
        EXPECT_EQ(after, received(h, out, false)) << (h == h_ ? "EM" : "flat");
    }
    ncclEpTensorDestroy(t_recv_idx);
    ncclEpTensorDestroy(t_recv_w);
    cudaFree(d_recv_idx);
    NCCL_ASSERT(ncclEpHandleDestroy(hf));
}

// A signaled forward at tile size 1 leaves counts in the scratch words that a backward at tile size 16 must not read as
// row map entries. Ranks 0 and 1 receive 3 and 5 tokens in their local experts, each zone padded to 16 rows, so zone 0
// has padding rows below the rows whose scratch words the forward counted.
TEST_F(DispatchSignalTest, TileSizeChangeKeepsRowMap) {
    constexpr int kWideTile = 16;
    if (g_nranks != 4 || kNumExperts != 8 || kNumTokens != 4 || kMaxRecvSlots != 32) {
        GTEST_SKIP() << "needs 4 ranks, 8 experts, 4 tokens and 32 receive slots";
    }
    std::vector<int64_t> topk(kNumTokens);
    for (int i = 0; i < kNumTokens; ++i) {
        const int g = g_rank * kNumTokens + i;  // global token: 0-2 to expert 0, 3-7 to 1, 8-10 to 2, 11-15 to 3
        topk[i] = g < 3 ? 0 : g < 8 ? 1 : g < 11 ? 2 : 3;
    }
    int64_t* d_topk = nullptr;
    ncclEpTensor_t* topk_idx = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk, topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    NCCL_ASSERT(epTensorCreate(&topk_idx, 2, ncclInt64, d_topk, kNumTokens, kTopK));
    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kWideTile;
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&h, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx, nullptr, &cfg, g_stream));

    int32_t* d_flags = nullptr;
    ncclEpTensor_t* t_flags = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_flags, kMaxRecvSlots * sizeof(int32_t)));
    NCCL_ASSERT(epTensorCreate(&t_flags, 1, ncclInt32, d_flags, kMaxRecvSlots));
    ncclEpTensor_t* const own_flags = b_.sig.signal;
    b_.sig.signal = t_flags;
    b_.sig.granularity = 1;
    CUDA_ASSERT(cudaMemsetAsync(d_flags, 0xFF, kMaxRecvSlots * sizeof(int32_t), g_stream));
    const unsigned int rows = dispatch_forward(h, true);

    b_.sig.granularity = kWideTile;
    int4* d_ref = nullptr;
    ncclEpTensor_t* t_ref = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_ref, kMaxRecvSlots * sizeof(int4)));
    NCCL_ASSERT(epTensorCreate(&t_ref, 2, ncclBfloat16, d_ref, kMaxRecvSlots, DispatchBuffers::kRowHidden));
    ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
    in.tokens = b_.t_tok;
    ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    out.tokens = b_.t_recv;
    out.dispatch_signal = &b_.sig;
    ncclEpDispatchOutputs_t ref = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ref.tokens = t_ref;
    ncclEpDispatchConfig_t bwd = NCCL_EP_DISPATCH_CONFIG_INIT;
    bwd.pass_direction = NCCL_EP_BWD_PASS;
    b_.reset();
    CUDA_ASSERT(cudaMemsetAsync(d_flags, 0xFF, kMaxRecvSlots * sizeof(int32_t), g_stream));
    NCCL_ASSERT(ncclEpDispatch(h, &in, &out, nullptr, &bwd, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaMemsetAsync(d_ref, 0xCD, kMaxRecvSlots * sizeof(int4), g_stream));
    NCCL_ASSERT(ncclEpDispatch(h, &in, &ref, nullptr, &bwd, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    const std::vector<int32_t> flags = to_host(d_flags, kMaxRecvSlots);
    for (int t = 0; t < kMaxRecvSlots; ++t) {
        EXPECT_EQ(flags[t], t < static_cast<int>(rows) / kWideTile ? t : -1) << "tile " << t;
    }
    const std::vector<int4> got = to_host(b_.d_recv, rows), want = to_host(d_ref, rows);
    EXPECT_EQ(std::memcmp(got.data(), want.data(), rows * sizeof(int4)), 0);

    b_.sig.signal = own_flags;
    b_.sig.granularity = kTileSize;
    ncclEpTensorDestroy(t_ref);
    ncclEpTensorDestroy(t_flags);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
    ncclEpTensorDestroy(topk_idx);
    cudaFree(d_ref);
    cudaFree(d_flags);
    cudaFree(d_topk);
}

// Each update rebuilds the handle's send counts: after two more updates with the same routing, the dispatch receives
// the same rows.
TEST_F(DispatchSignalTest, RepeatedUpdateKeepsCounts) {
    const unsigned int rows = dispatch_forward(h_, true);
    ASSERT_GT(rows, 0u);
    NCCL_ASSERT(ncclEpUpdateHandle(h_, topk_idx_em_, nullptr, g_stream));
    NCCL_ASSERT(ncclEpUpdateHandle(h_, topk_idx_em_, nullptr, g_stream));
    EXPECT_EQ(dispatch_forward(h_, true), rows);
    expect_flags(static_cast<int>(rows) / kTileSize);
}

// A handle's first update is captured into a graph that never runs; an eager update after it still counts from zero,
// on stale handle memory.
TEST_F(DispatchSignalTest, EagerUpdateAfterCapturedFirstUpdate) {
    const unsigned int rows = dispatch_forward(h_, true);
    ASSERT_GT(rows, 0u);
    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;
    size_t bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, &bytes, kTopK));
    void* d_mem = nullptr;
    ncclEpTensor_t* t_mem = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_mem, bytes));
    CUDA_ASSERT(cudaMemset(d_mem, 0xFF, bytes));
    NCCL_ASSERT(epTensorCreate(&t_mem, 1, ncclUint8, d_mem, static_cast<unsigned int>(bytes)));
    ncclEpHandle_t h2 = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h2, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, kTopK, t_mem));
    cudaGraph_t graph = nullptr;
    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
    NCCL_ASSERT(ncclEpUpdateHandle(h2, topk_idx_em_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
    NCCL_ASSERT(ncclEpUpdateHandle(h2, topk_idx_em_, nullptr, g_stream));
    EXPECT_EQ(dispatch_forward(h2, true), rows);
    expect_flags(static_cast<int>(rows) / kTileSize);
    CUDA_ASSERT(cudaGraphDestroy(graph));
    NCCL_ASSERT(ncclEpHandleDestroy(h2));
    ncclEpTensorDestroy(t_mem);
    CUDA_ASSERT(cudaFree(d_mem));
}

// With a handle imported per op from each layer's handle memory, the signaled backwards of two layers, run after both
// forwards and in reverse order, each follow their own layer's maps; after a plain forward under new routing, the
// signaled backward still matches the plain one. Every rank receives as many tokens under both routings, but one fills
// a single expert zone and the other two, so the other routing's maps would miss rows. A last plain forward
// receives one token, so the old maps would also pull stale slots into the padding rows.
TEST_F(DispatchSignalTest, ImportedBackwardSignaled) {
    const int local_experts = kNumExperts / g_nranks;
    if (local_experts < 2) {
        GTEST_SKIP() << "needs two local experts per rank";
    }
    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;
    size_t mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, &mem_bytes, kTopK));
    // Routing s: token i of rank r goes to rank r + i, into its first local expert (s = 0) or expert i % 2 (s = 1);
    // routing 2 sends only token 0, to the first local expert of rank r + 1.
    int64_t* d_topk[3] = {};
    ncclEpTensor_t* topk[3] = {};
    void* d_mem[2] = {};
    ncclEpTensor_t* mem[2] = {};
    for (int s = 0; s < 3; ++s) {
        std::vector<int64_t> h(kNumTokens, -1);
        for (int i = 0; i < kNumTokens; ++i) {
            if (s < 2 || i == 0) {
                h[i] = (g_rank + i + s / 2) % g_nranks * local_experts + s % 2 * (i % 2);
            }
        }
        CUDA_ASSERT(cudaMalloc(&d_topk[s], h.size() * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(d_topk[s], h.data(), h.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk[s], 2, ncclInt64, d_topk[s], kNumTokens, kTopK));
    }
    for (int layer = 0; layer < 2; ++layer) {
        CUDA_ASSERT(cudaMalloc(&d_mem[layer], mem_bytes));
        NCCL_ASSERT(epTensorCreate(&mem[layer], 1, ncclUint8, d_mem[layer], mem_bytes));
    }
    const auto prepare = [&](int layer, ncclEpTensor_t* routing) {
        ncclEpHandle_t h = nullptr;
        NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, kTopK, mem[layer]));
        NCCL_ASSERT(ncclEpUpdateHandle(h, routing, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        NCCL_ASSERT(ncclEpHandleDestroy(h));
    };
    const auto import_handle = [&](int layer) {
        ncclEpHandleState_t state = NCCL_EP_HANDLE_STATE_INIT;
        state.num_tokens = kNumTokens;
        state.num_topk = kTopK;
        state.topk_idx_datatype = ncclInt64;
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpImportHandle(&h, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, &state, mem[layer]),
                  ncclSuccess);
        return h;
    };
    const auto forward = [&](int layer, bool signaled) {
        ncclEpHandle_t h = import_handle(layer);
        const unsigned int rows = dispatch_forward(h, signaled);
        EXPECT_EQ(ncclEpHandleDestroy(h), ncclSuccess);
        return rows;
    };
    const auto backward = [&](int layer, unsigned int rows) {
        ncclEpHandle_t h = import_handle(layer);
        expect_backward_signaled(h, rows);
        EXPECT_EQ(ncclEpHandleDestroy(h), ncclSuccess);
    };

    prepare(0, topk[0]);
    const unsigned int rows0 = forward(0, true);
    prepare(1, topk[1]);
    const unsigned int rows1 = forward(1, true);
    EXPECT_LT(rows0, rows1);
    backward(1, rows1);
    backward(0, rows0);

    prepare(1, topk[0]);
    EXPECT_EQ(forward(1, false), rows0);
    backward(1, rows0);

    // Fewer tokens than the map's: its padding rows keep the old map's slots unless the rebuild clears them.
    prepare(0, topk[2]);
    const unsigned int rows2 = forward(0, false);
    EXPECT_EQ(rows2, static_cast<unsigned int>(kAlign));
    backward(0, rows2);

    for (int layer = 0; layer < 2; ++layer) {
        ncclEpTensorDestroy(mem[layer]);
        cudaFree(d_mem[layer]);
    }
    for (int s = 0; s < 3; ++s) {
        ncclEpTensorDestroy(topk[s]);
        cudaFree(d_topk[s]);
    }
}

// Captured signaled backwards replay correctly. Both share the group's fixed scratch, so a capture keeps its
// address across handles. One follows a signaled forward and the other a plain forward, after which a routing change
// that keeps every rank's receive count must still be reflected.
TEST_F(DispatchSignalTest, GraphReplay) {
    {
        const unsigned int rows = dispatch_forward(h_, true);
        ASSERT_GT(rows, 0u);
        const int tiles = static_cast<int>(rows) / kTileSize;

        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        in.tokens = b_.t_tok;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        out.tokens = b_.t_recv;
        out.dispatch_signal = &b_.sig;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.pass_direction = NCCL_EP_BWD_PASS;

        // Reference: a plain backward, which equals the signaled backward bitwise.
        int4* d_ref = nullptr;
        ncclEpTensor_t* t_ref = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_ref, kMaxRecvSlots * sizeof(int4)));
        NCCL_ASSERT(epTensorCreate(&t_ref, 2, ncclBfloat16, d_ref, kMaxRecvSlots, DispatchBuffers::kRowHidden));
        ncclEpDispatchOutputs_t ref = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ref.tokens = t_ref;
        CUDA_ASSERT(cudaMemsetAsync(d_ref, 0xCD, kMaxRecvSlots * sizeof(int4), g_stream));
        NCCL_ASSERT(ncclEpDispatch(h_, &in, &ref, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        const std::vector<int4> want = to_host(d_ref, rows);

        cudaGraphExec_t g = capture_graph([&] {
            b_.reset();
            NCCL_ASSERT(ncclEpDispatch(h_, &in, &out, nullptr, &cfg, g_stream));
            NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
        });
        CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        expect_flags(tiles);
        EXPECT_EQ(std::memcmp(to_host(b_.d_recv, rows).data(), want.data(), rows * sizeof(int4)), 0)
            << "replay before growth";

        // A backward on another handle reuses the scratch the captured backward replays on.
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        hcfg.dispatch_output_per_expert_alignment = kAlign;
        ncclEpHandle_t h2 = nullptr;
        NCCL_ASSERT(ncclEpCreateHandle(&h2, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_em_, nullptr, &hcfg,
                                       g_stream));
        const unsigned int rows2 = dispatch_forward(h2, false);
        expect_backward_signaled(h2, rows2);

        CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        expect_flags(tiles);
        EXPECT_EQ(std::memcmp(to_host(b_.d_recv, rows).data(), want.data(), rows * sizeof(int4)), 0)
            << "replay after growth";

        CUDA_ASSERT(cudaGraphExecDestroy(g));
        NCCL_ASSERT(ncclEpHandleDestroy(h2));
        ncclEpTensorDestroy(t_ref);
        cudaFree(d_ref);
    }
    {
        dispatch_forward(h_, false);

        int4* d_ref = nullptr;
        ncclEpTensor_t* t_ref = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_ref, kMaxRecvSlots * sizeof(int4)));
        NCCL_ASSERT(epTensorCreate(&t_ref, 2, ncclBfloat16, d_ref, kMaxRecvSlots, DispatchBuffers::kRowHidden));
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        in.tokens = b_.t_tok;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        out.tokens = b_.t_recv;
        out.dispatch_signal = &b_.sig;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.pass_direction = NCCL_EP_BWD_PASS;

        cudaGraphExec_t g = capture_graph([&] {
            b_.reset();
            NCCL_ASSERT(ncclEpDispatch(h_, &in, &out, nullptr, &cfg, g_stream));
            NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
        });

        int64_t* d_topk = nullptr;
        ncclEpTensor_t* topk_idx = swapped_topk(&d_topk);
        NCCL_ASSERT(ncclEpUpdateHandle(h_, topk_idx, nullptr, g_stream));
        const unsigned int rows = dispatch_forward(h_, false);

        // Reference: a plain backward under the new routing.
        ncclEpDispatchOutputs_t ref = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ref.tokens = t_ref;
        CUDA_ASSERT(cudaMemsetAsync(d_ref, 0xCD, kMaxRecvSlots * sizeof(int4), g_stream));
        NCCL_ASSERT(ncclEpDispatch(h_, &in, &ref, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        expect_flags(static_cast<int>(rows) / kTileSize);
        const std::vector<int4> got = to_host(b_.d_recv, rows), want = to_host(d_ref, rows);
        EXPECT_EQ(std::memcmp(got.data(), want.data(), rows * sizeof(int4)), 0);

        CUDA_ASSERT(cudaGraphExecDestroy(g));
        ncclEpTensorDestroy(t_ref);
        cudaFree(d_ref);
        ncclEpTensorDestroy(topk_idx);
        cudaFree(d_topk);
    }
}

// Group, handle and device buffers shared by the pull-dispatch signal fixtures below. A subclass builds the group
// and handle in SetUp; this base frees the handle, group and every buffer() allocation.
class SignalPullGroupTest : public ::testing::Test {
protected:
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    std::vector<ncclEpTensor_t*> tensors_;
    std::vector<void*> buffers_;

    void TearDown() override {
        if (handle_ != nullptr) {
            ncclEpHandleDestroy(handle_);
        }
        if (group_ != nullptr) {
            ncclEpGroupDestroy(group_);
        }
        for (ncclEpTensor_t* t : tensors_) {
            ncclEpTensorDestroy(t);
        }
        for (void* p : buffers_) {
            cudaFree(p);
        }
    }

    // A device tensor of rows x cols elements of elem_bytes each (1-D when cols is 0), every byte set to fill.
    ncclEpTensor_t* buffer(ncclDataType_t type, size_t elem_bytes, size_t rows, size_t cols, int fill) {
        void* d = nullptr;
        ncclEpTensor_t* t = nullptr;
        const size_t bytes = rows * std::max<size_t>(cols, 1) * elem_bytes;
        EXPECT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemset(d, fill, bytes), cudaSuccess);
        if (cols == 0) {
            EXPECT_EQ(epTensorCreate(&t, 1, type, d, rows), ncclSuccess);
        } else {
            EXPECT_EQ(epTensorCreate(&t, 2, type, d, rows, cols), ncclSuccess);
        }
        buffers_.push_back(d);
        tensors_.push_back(t);
        return t;
    }

    // A rows x cols tensor of the given type holding the bytes of h.
    template <typename T>
    ncclEpTensor_t* upload(const std::vector<T>& h, ncclDataType_t type, size_t rows, size_t cols) {
        ncclEpTensor_t* t = buffer(type, h.size() * sizeof(T) / (rows * cols), rows, cols, 0);
        EXPECT_EQ(cudaMemcpy(t->data, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        return t;
    }

    static std::vector<uint8_t> bytes(const void* d, size_t n) {
        return to_host(static_cast<const uint8_t*>(d), n);
    }

    // signal[t] == t below `tiles`, -1 above.
    void expect_flags(const ncclEpSignal_t& sig, int tiles) {
        std::vector<int32_t> expected(sig.signal->sizes[0], -1);
        std::iota(expected.begin(), expected.begin() + tiles, 0);
        EXPECT_TRUE(to_host(static_cast<const int32_t*>(sig.signal->data), expected.size()) == expected);
    }
};

// A one-SM signaled pull of many tokens, each routed to both experts of its destination rank, so every pull warp
// counts several full batches of rows. The parameter is the row hidden: a narrow row runs two CTAs per SM, a wide row one.
class DispatchSignalBatchTest : public SignalPullGroupTest, public ::testing::WithParamInterface<int> {
protected:
    static constexpr int kTokens = 2048;  // per rank
    static constexpr int kBatchTopK = 2;
    static constexpr int kLocalExperts = 2;
    const int row_hidden_ = GetParam();
    int slots_ = 0;  // receive rows per rank

    void SetUp() override {
        REQUIRE_PULL_PUSH();
        slots_ = kTokens * kBatchTopK + kLocalExperts * static_cast<int>(kAlign);
        ncclEpGroupConfig_t gcfg = ep_group_config(row_hidden_ * sizeof(nv_bfloat16) + kScaleBytes);
        gcfg.num_experts = kLocalExperts * g_nranks;
        gcfg.max_dispatch_tokens_per_rank = kTokens;
        gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(slots_);
        gcfg.dispatch_num_sms = 1;
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
        std::vector<int64_t> topk(kTokens * kBatchTopK);
        for (size_t i = 0; i < topk.size(); ++i) {
            topk[i] = (i / kBatchTopK % g_nranks) * kLocalExperts + i % kBatchTopK;
        }
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        hcfg.dispatch_output_per_expert_alignment = kAlign;
        NCCL_ASSERT(ncclEpCreateHandle(&handle_, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                       upload(topk, ncclInt64, kTokens, kBatchTopK), nullptr, &hcfg, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Each row of a forward output as its token, weight and scale bytes, sorted within each expert's zone: the forward
    // places an expert's rows in arrival order. Every token has a row in each local expert, so the zones are equal.
    std::vector<std::vector<uint8_t>> sorted_rows(const ncclEpDispatchOutputs_t& o, unsigned int rows) const {
        const size_t row_bytes = row_hidden_ * sizeof(nv_bfloat16);
        const std::vector<uint8_t> tok = bytes(o.tokens->data, rows * row_bytes);
        const std::vector<uint8_t> w = bytes(o.topk_weights->data, rows * sizeof(float));
        const std::vector<uint8_t> scales = bytes(o.scales->data, rows * kScaleBytes);
        std::vector<std::vector<uint8_t>> sorted(rows);
        for (size_t r = 0; r < rows; ++r) {
            sorted[r].assign(tok.begin() + r * row_bytes, tok.begin() + (r + 1) * row_bytes);
            sorted[r].insert(sorted[r].end(), w.begin() + r * sizeof(float), w.begin() + (r + 1) * sizeof(float));
            sorted[r].insert(sorted[r].end(), scales.begin() + r * kScaleBytes, scales.begin() + (r + 1) * kScaleBytes);
        }
        const unsigned int zone = rows / kLocalExperts;
        for (int e = 0; e < kLocalExperts; ++e) {
            std::sort(sorted.begin() + e * zone, sorted.begin() + (e + 1) * zone);
        }
        return sorted;
    }
};

// The signaled forward with 16-byte scales receives the plain forward's rows, weights and scales, the signaled
// backward equals the plain backward bitwise, both flag every tile, and a programmatic consumer behind the forward
// sees the final rows, weights and scales of each tile.
TEST_P(DispatchSignalBatchTest, OneSmMatchesPlainBitwise) {
    std::vector<uint32_t> tok(kTokens * row_hidden_ / 2), scales(kTokens * kScaleBytes / 4);
    std::vector<float> w(kTokens * kBatchTopK);
    for (size_t i = 0; i < tok.size(); ++i) {
        tok[i] = static_cast<uint32_t>(g_rank * tok.size() + i);
    }
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = static_cast<uint32_t>(~(g_rank * scales.size() + i));
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.5f * static_cast<float>(g_rank * w.size() + i + 1);
    }
    ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
    in.tokens = upload(tok, ncclBfloat16, kTokens, row_hidden_);
    in.topk_weights = upload(w, ncclFloat32, kTokens, kBatchTopK);
    in.scales = upload(scales, ncclUint8, kTokens, kScaleBytes);
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    sig.granularity = kTileSize;
    sig.signal = buffer(ncclInt32, sizeof(int32_t), slots_ / kTileSize, 0, 0xFF);
    const size_t row_bytes = row_hidden_ * sizeof(nv_bfloat16);

    // Forward: [0] signaled, [1] plain.
    ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    cfg.quant_recipe = NCCL_EP_DISP_QUANT_FWD;
    ncclEpDispatchOutputs_t out[2] = {NCCL_EP_DISPATCH_OUTPUTS_INIT, NCCL_EP_DISPATCH_OUTPUTS_INIT};
    for (ncclEpDispatchOutputs_t& o : out) {
        o.tokens = buffer(ncclBfloat16, sizeof(nv_bfloat16), slots_, row_hidden_, 0xAB);
        o.topk_weights = buffer(ncclFloat32, sizeof(float), slots_, 0, 0xAB);
        o.scales = buffer(ncclUint8, 1, slots_, kScaleBytes, 0xAB);
    }
    out[0].dispatch_signal = &sig;
    NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out[1], nullptr, &cfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    unsigned int rows = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(handle_, &rows));
    ASSERT_GE(rows, static_cast<unsigned int>(kTokens * kBatchTopK));
    const int tiles = static_cast<int>(rows) / kTileSize;
    ncclEpTensor_t* seen_tok = buffer(ncclBfloat16, sizeof(nv_bfloat16), rows, row_hidden_, 0);
    ncclEpTensor_t* seen_w = buffer(ncclFloat32, sizeof(float), rows, 0, 0);
    ncclEpTensor_t* seen_scales = buffer(ncclUint8, 1, rows, kScaleBytes, 0);
    NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out[0], nullptr, &cfg, g_stream));
    CUDA_ASSERT(launch_programmatic(
        consumeTilesKernel, dim3(64), g_stream, static_cast<const int32_t*>(sig.signal->data), tiles,
        static_cast<const int4*>(out[0].tokens->data), static_cast<const float*>(out[0].topk_weights->data),
        static_cast<const int4*>(out[0].scales->data), static_cast<int>(row_bytes / sizeof(int4)),
        static_cast<int4*>(seen_tok->data), static_cast<float*>(seen_w->data), static_cast<int4*>(seen_scales->data)));
    NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(sig, tiles);
    EXPECT_TRUE(sorted_rows(out[0], rows) == sorted_rows(out[1], rows));
    EXPECT_TRUE(bytes(seen_tok->data, rows * row_bytes) == bytes(out[0].tokens->data, rows * row_bytes));
    EXPECT_TRUE(bytes(seen_w->data, rows * sizeof(float)) == bytes(out[0].topk_weights->data, rows * sizeof(float)));
    EXPECT_TRUE(bytes(seen_scales->data, rows * kScaleBytes) == bytes(out[0].scales->data, rows * kScaleBytes));

    // Backward: [0] signaled, [1] plain.
    cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    cfg.pass_direction = NCCL_EP_BWD_PASS;
    in.topk_weights = nullptr;
    in.scales = nullptr;
    ncclEpDispatchOutputs_t bwd[2] = {NCCL_EP_DISPATCH_OUTPUTS_INIT, NCCL_EP_DISPATCH_OUTPUTS_INIT};
    for (ncclEpDispatchOutputs_t& o : bwd) {
        o.tokens = buffer(ncclBfloat16, sizeof(nv_bfloat16), slots_, row_hidden_, 0xAB);
    }
    bwd[0].dispatch_signal = &sig;
    CUDA_ASSERT(cudaMemsetAsync(sig.signal->data, 0xFF, sig.signal->sizes[0] * sizeof(int32_t), g_stream));
    for (ncclEpDispatchOutputs_t& o : bwd) {
        NCCL_ASSERT(ncclEpDispatch(handle_, &in, &o, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    }
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(sig, tiles);
    EXPECT_TRUE(bytes(bwd[0].tokens->data, rows * row_bytes) == bytes(bwd[1].tokens->data, rows * row_bytes));
}

INSTANTIATE_TEST_SUITE_P(RowHidden, DispatchSignalBatchTest, ::testing::Values(64, 2304));

// A full-grid signaled pull whose routing alternates the lowest local expert every two lanes of a source chunk, so
// each four-byte bucket word holds two distinct buckets, each on several lanes. This drives concurrent chunk splits
// (pull_pool_split_chunk) against cross-bucket reads of the shared word in dispatch_pull_drain_pool. Each rank sends
// all its tokens to one destination, so every destination builds chunks of the alternating pattern.
class DispatchSignalCrossBucketTest : public SignalPullGroupTest {
protected:
    static constexpr int kTokens = 256;      // per rank, all to one destination
    static constexpr int kLocalExperts = 2;  // buckets 0 and 1
    static constexpr int kRowHidden = 512;   // bf16 elements per row
    int slots_ = 0;

    void SetUp() override {
        REQUIRE_PULL_PUSH();
        slots_ = kTokens + kLocalExperts * static_cast<int>(kAlign);
        ncclEpGroupConfig_t gcfg = ep_group_config(kRowHidden * sizeof(nv_bfloat16));
        gcfg.num_experts = kLocalExperts * g_nranks;
        gcfg.max_dispatch_tokens_per_rank = kTokens;
        gcfg.max_recv_tokens_per_rank = static_cast<unsigned int>(slots_);
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
        std::vector<int64_t> topk(kTokens);
        const int dest = (g_rank + 1) % g_nranks;
        for (int i = 0; i < kTokens; ++i) {
            topk[i] = dest * kLocalExperts + ((i / 2) & 1);
        }
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        hcfg.dispatch_output_per_expert_alignment = kAlign;
        NCCL_ASSERT(ncclEpCreateHandle(&handle_, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                       upload(topk, ncclInt64, kTokens, 1), nullptr, &hcfg, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Forward token and weight bytes of each row, sorted within each expert's equal-size zone (top-k 1, so both
    // experts receive kTokens/2 rows): the pull order within a zone differs between the ordered and plain paths.
    std::vector<std::vector<uint8_t>> sorted_rows(const ncclEpTensor_t* tokens, const ncclEpTensor_t* weights,
                                                  unsigned int rows) const {
        const size_t row_bytes = kRowHidden * sizeof(nv_bfloat16);
        const std::vector<uint8_t> tok = bytes(tokens->data, rows * row_bytes);
        const std::vector<uint8_t> w = bytes(weights->data, rows * sizeof(float));
        std::vector<std::vector<uint8_t>> sorted(rows);
        for (size_t r = 0; r < rows; ++r) {
            sorted[r].assign(tok.begin() + r * row_bytes, tok.begin() + (r + 1) * row_bytes);
            sorted[r].insert(sorted[r].end(), w.begin() + r * sizeof(float), w.begin() + (r + 1) * sizeof(float));
        }
        const unsigned int zone = rows / kLocalExperts;
        for (int e = 0; e < kLocalExperts; ++e) {
            std::sort(sorted.begin() + e * zone, sorted.begin() + (e + 1) * zone);
        }
        return sorted;
    }
};

// The signaled forward and backward, whose ordered pull splits and republishes chunks across buckets sharing a word,
// place every token and weight exactly as the plain pull does.
TEST_F(DispatchSignalCrossBucketTest, SplitBucketsMatchPlain) {
    std::vector<uint32_t> tok(kTokens * kRowHidden / 2);
    std::vector<float> w(kTokens);
    for (size_t i = 0; i < tok.size(); ++i) {
        tok[i] = static_cast<uint32_t>(g_rank * tok.size() + i + 1);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.25f * static_cast<float>(g_rank * kTokens + i + 1);
    }
    ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
    in.tokens = upload(tok, ncclBfloat16, kTokens, kRowHidden);
    in.topk_weights = upload(w, ncclFloat32, kTokens, 1);
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    sig.granularity = kTileSize;
    sig.signal = buffer(ncclInt32, sizeof(int32_t), slots_ / kTileSize, 0, 0xFF);
    const size_t row_bytes = kRowHidden * sizeof(nv_bfloat16);

    // Forward: [0] signaled, [1] plain.
    ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    ncclEpDispatchOutputs_t out[2] = {NCCL_EP_DISPATCH_OUTPUTS_INIT, NCCL_EP_DISPATCH_OUTPUTS_INIT};
    for (ncclEpDispatchOutputs_t& o : out) {
        o.tokens = buffer(ncclBfloat16, sizeof(nv_bfloat16), slots_, kRowHidden, 0xAB);
        o.topk_weights = buffer(ncclFloat32, sizeof(float), slots_, 0, 0xAB);
    }
    out[0].dispatch_signal = &sig;
    NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out[1], nullptr, &cfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    unsigned int rows = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(handle_, &rows));
    ASSERT_EQ(rows, static_cast<unsigned int>(kTokens));
    NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out[0], nullptr, &cfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(sig, static_cast<int>(rows) / kTileSize);
    EXPECT_TRUE(sorted_rows(out[0].tokens, out[0].topk_weights, rows) ==
                sorted_rows(out[1].tokens, out[1].topk_weights, rows));

    // Backward follows the forward's maps, so it fills the same alternating buckets: [0] signaled, [1] plain.
    cfg.pass_direction = NCCL_EP_BWD_PASS;
    in.topk_weights = nullptr;
    ncclEpDispatchOutputs_t bwd[2] = {NCCL_EP_DISPATCH_OUTPUTS_INIT, NCCL_EP_DISPATCH_OUTPUTS_INIT};
    for (ncclEpDispatchOutputs_t& o : bwd) {
        o.tokens = buffer(ncclBfloat16, sizeof(nv_bfloat16), slots_, kRowHidden, 0xAB);
    }
    bwd[0].dispatch_signal = &sig;
    CUDA_ASSERT(cudaMemsetAsync(sig.signal->data, 0xFF, sig.signal->sizes[0] * sizeof(int32_t), g_stream));
    for (ncclEpDispatchOutputs_t& o : bwd) {
        NCCL_ASSERT(ncclEpDispatch(handle_, &in, &o, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
    }
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(sig, static_cast<int>(rows) / kTileSize);
    EXPECT_TRUE(bytes(bwd[0].tokens->data, rows * row_bytes) == bytes(bwd[1].tokens->data, rows * row_bytes));
}

// ---------------------------------------------------------------- combine

// Receive sizing of a combine test group: the fixed budget, a small budget with overflow drop, or eager.
enum class RecvMode { Fixed, Drop, Eager };

// Drop and Eager route around the last rank's experts, so that rank receives no token.
static int skewed_expert_for_token(int i) {
    return (g_rank * kNumTokens + i) % (kNumExperts - kNumExperts / g_nranks);
}

// The FC2 stand-in, the kernel before the combine: copies golden into live chunk by chunk, release-stores
// ready[s] = s after chunk s, and stores finished = 1 last. It waits late_us before chunk late_chunk (-1: never).
__global__ void producerKernel(const nv_bfloat16* golden, nv_bfloat16* live, int rows, int hidden, int chunk,
                               int chunks, int32_t* ready, int32_t* finished, int late_chunk, int late_us) {
    nccl_ep::launch_dependents();
    for (int s = 0; s < chunks; ++s) {
        if (s == late_chunk && threadIdx.x == 0) {
            const uint64_t until = cuda::ptx::get_sreg_globaltimer() + 1000ull * late_us;
            while (cuda::ptx::get_sreg_globaltimer() < until) {
            }
        }
        __syncthreads();
        const int width = min(chunk, hidden - s * chunk);  // the last chunk of a row can be partial
        for (int i = threadIdx.x; i < rows * width; i += blockDim.x) {
            const size_t at = static_cast<size_t>(i / width) * hidden + s * chunk + i % width;
            live[at] = golden[at];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            nccl_ep::st_release_gpu_global(&ready[s], s);
            for (int d = 0; d < 200; ++d) {
                __nanosleep(1000);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *finished = 1;
    }
}

// The combine's programmatic dependent: after its wait, records whether the kernel before the combine has finished.
__global__ void afterCombineKernel(const int32_t* finished, int32_t* seen) {
    nccl_ep::grid_dependency_wait();
    *seen = nccl_ep::ld_acquire_global(finished);
}

class CombineSignalTest : public ::testing::Test {
protected:
    RecvMode mode_ = RecvMode::Fixed;
    size_t align_ = 0;               // handle per-expert alignment; 0 is the default
    int hidden_ = kSigHidden;        // combine row elements
    unsigned int send_only_ = 0;     // combine config send_only
    ncclEpCombQuant_t quant_ = NCCL_EP_COMB_QUANT_NONE;  // combine config quant_recipe
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    int64_t* d_topk_ = nullptr;
    nv_bfloat16 *d_tok_ = nullptr, *d_golden_ = nullptr, *d_live_ = nullptr;  // golden: the dispatch output
    float *d_w_ = nullptr, *d_recv_w_ = nullptr;
    int32_t* d_ready_ = nullptr;
    int32_t* d_finished_ = nullptr;  // [0] the producer's finished word, [1] what the combine's dependent saw
    ncclEpTensor_t *topk_idx_ = nullptr, *t_tok_ = nullptr, *t_w_ = nullptr, *t_golden_ = nullptr,
                   *t_live_ = nullptr, *t_recv_w_ = nullptr;
    int rows_ = 0;  // rows the dispatch received, which the producer writes
    bool producer_done_ = false;  // the producer completes before the combine: every chunk is ready at the push's entry
    int late_chunk_ = -1;         // on the last rank only, the producer waits kLateUs before this chunk (-1: never)

    void SetUp() override {
        REQUIRE_PULL_PUSH();
        ncclEpGroupConfig_t gcfg = ep_group_config(hidden_ * sizeof(nv_bfloat16));
        if (mode_ == RecvMode::Drop) {
            gcfg.max_recv_tokens_per_rank = kDropBudget;
            gcfg.overflow_policy = NCCL_EP_OVERFLOW_DROP;
        } else if (mode_ == RecvMode::Eager) {
            gcfg.max_recv_tokens_per_rank = NCCL_EP_AUTO;
            gcfg.num_topk = kTopK;
        }
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
        std::vector<int64_t> topk(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i) {
            topk[i] = mode_ == RecvMode::Fixed ? expert_for_token(i) : skewed_expert_for_token(i);
        }
        CUDA_ASSERT(cudaMalloc(&d_topk_, topk.size() * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(d_topk_, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx_, 2, ncclInt64, d_topk_, kNumTokens, kTopK));
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        hcfg.dispatch_output_per_expert_alignment = align_;
        NCCL_ASSERT(ncclEpCreateHandle(&handle_, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, topk_idx_, nullptr, &hcfg,
                                       g_stream));
        ASSERT_NO_FATAL_FAILURE(dispatch_golden());
        CUDA_ASSERT(cudaMalloc(&d_live_, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16)));
        NCCL_ASSERT(epTensorCreate(&t_live_, 2, ncclBfloat16, d_live_, t_golden_->sizes[0], hidden_));
        CUDA_ASSERT(cudaMalloc(&d_ready_, hidden_ * sizeof(int32_t)));
        CUDA_ASSERT(cudaMalloc(&d_finished_, 2 * sizeof(int32_t)));
    }

    void TearDown() override {
        for (ncclEpTensor_t* t : {t_live_, t_recv_w_, t_golden_, t_w_, t_tok_, topk_idx_}) {
            ncclEpTensorDestroy(t);
        }
        for (void* p : {static_cast<void*>(d_ready_), static_cast<void*>(d_live_), static_cast<void*>(d_recv_w_),
                        static_cast<void*>(d_golden_), static_cast<void*>(d_w_), static_cast<void*>(d_tok_),
                        static_cast<void*>(d_topk_), static_cast<void*>(d_finished_)}) {
            cudaFree(p);
        }
        if (handle_ != nullptr) {
            ncclEpHandleDestroy(handle_);
        }
        if (group_ != nullptr) {
            ncclEpGroupDestroy(group_);
        }
    }

    // Dispatches tokens whose value depends on source and column, so a misplaced chunk is visible.
    void dispatch_golden() {
        std::vector<nv_bfloat16> tok(kNumTokens * hidden_);
        std::vector<float> w(kNumTokens * kTopK);
        for (size_t i = 0; i < tok.size(); ++i) {
            tok[i] = __float2bfloat16(static_cast<float>(g_rank * kNumTokens + i / hidden_ + 1) +
                                      static_cast<float>(i % hidden_) * 0.125f);
        }
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = 0.125f * static_cast<float>(g_rank * 10 + i + 1);
        }
        CUDA_ASSERT(cudaMalloc(&d_tok_, tok.size() * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_w_, w.size() * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_golden_, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv_w_, kMaxRecvSlots * sizeof(float)));
        CUDA_ASSERT(cudaMemcpy(d_tok_, tok.data(), tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_w_, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&t_tok_, 2, ncclBfloat16, d_tok_, kNumTokens, hidden_));
        NCCL_ASSERT(epTensorCreate(&t_w_, 2, ncclFloat32, d_w_, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_golden_, 2, ncclBfloat16, d_golden_, kMaxRecvSlots, hidden_));
        NCCL_ASSERT(epTensorCreate(&t_recv_w_, 1, ncclFloat32, d_recv_w_, kMaxRecvSlots));
        ASSERT_NO_FATAL_FAILURE(dispatch_tokens());
    }

    // Dispatches this rank's tokens into golden and records the rows received.
    void dispatch_tokens() {
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        in.tokens = t_tok_;
        in.topk_weights = t_w_;
        out.tokens = t_golden_;
        out.topk_weights = t_recv_w_;
        NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out, nullptr, nullptr, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        unsigned int rows = 0;
        NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(handle_, &rows));
        rows_ = static_cast<int>(rows);
        if (mode_ == RecvMode::Drop && g_rank == 0) {
            EXPECT_EQ(rows_, kDropBudget) << "rank 0 must fill its budget";
        }
        if (mode_ == RecvMode::Eager) {
            t_golden_->sizes[0] = rows_;
            t_recv_w_->sizes[0] = rows_;
        }
    }

    // Combines into fresh outputs and returns their bytes: golden (or plain_tokens) plain, or live signaled behind its
    // producer. chunk_elems is the push chunk the producer publishes at: 0 for the whole row. A programmatic kernel
    // behind a signaled combine checks that the combine completed after its producer.
    std::vector<uint8_t> combine(ncclEpSignal_t* signal, bool backward, ncclResult_t* rc,
                                 ncclEpTensor_t* plain_tokens, int chunk_elems) {
        const std::string chunk_env = std::to_string(chunk_elems);
        ScopedEnv push_chunk(kChunkEnv, chunk_env.c_str());
        nv_bfloat16* d_x = nullptr;
        float* d_w = nullptr;
        ncclEpTensor_t *t_x = nullptr, *t_w = nullptr;
        EXPECT_EQ(cudaMalloc(&d_x, kNumTokens * hidden_ * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_w, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_x, 0xAB, kNumTokens * hidden_ * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_w, 0, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        const size_t tokens = topk_idx_->sizes[0];  // the handle's tokens (set_tokens)
        EXPECT_EQ(epTensorCreate(&t_x, 2, ncclBfloat16, d_x, tokens, hidden_), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w, 2, ncclFloat32, d_w, tokens, kTopK), ncclSuccess);
        ncclEpCombineInputs_t in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
        cfg.send_only = send_only_;
        cfg.quant_recipe = quant_;
        in.tokens = plain_tokens != nullptr ? plain_tokens : t_golden_;
        out.tokens = t_x;
        if (backward) {
            in.topk_weights = t_recv_w_;
            out.topk_weights = t_w;
            cfg.pass_direction = NCCL_EP_BWD_PASS;
        }
        if (signal != nullptr) {
            in.tokens = t_live_;
            in.combine_signal = signal;
        }
        if (signal != nullptr) {
            const int chunk = chunk_elems > 0 ? chunk_elems : hidden_;
            const int chunks = rows_ > 0 ? nccl_ep::ceil_div(hidden_, chunk) : 0;
            EXPECT_EQ(cudaMemsetAsync(d_live_, 0xFF, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16), g_stream),
                      cudaSuccess);
            EXPECT_EQ(cudaMemsetAsync(d_ready_, 0xFF, hidden_ * sizeof(int32_t), g_stream), cudaSuccess);
            EXPECT_EQ(cudaMemsetAsync(d_finished_, 0, 2 * sizeof(int32_t), g_stream), cudaSuccess);
            // A normal launch: the kernel before the combine triggers only after earlier work completes.
            const int late_chunk = g_rank == g_nranks - 1 ? late_chunk_ : -1;
            producerKernel<<<1, 1024, 0, g_stream>>>(d_golden_, d_live_, rows_, hidden_, chunk, chunks, d_ready_,
                                                     d_finished_, late_chunk, kLateUs);
            EXPECT_EQ(cudaGetLastError(), cudaSuccess);
            if (producer_done_) {
                EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
            }
        }
        *rc = ncclEpCombine(handle_, &in, &out, &cfg, g_stream);
        const bool check_finished = signal != nullptr && *rc == ncclSuccess;
        if (check_finished) {
            const int32_t* finished = d_finished_;
            EXPECT_EQ(launch_programmatic(afterCombineKernel, dim3(1), g_stream, finished, d_finished_ + 1),
                      cudaSuccess);
        }
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        if (check_finished) {
            EXPECT_EQ(to_host(d_finished_ + 1, 1)[0], 1) << "the combine completed before the kernel before it";
        }
        std::vector<uint8_t> result;
        if (*rc == ncclSuccess) {
            const std::vector<nv_bfloat16> x = to_host(d_x, kNumTokens * hidden_);
            const std::vector<float> w = to_host(d_w, kNumTokens * kTopK);
            result.assign(reinterpret_cast<const uint8_t*>(x.data()),
                          reinterpret_cast<const uint8_t*>(x.data() + x.size()));
            result.insert(result.end(), reinterpret_cast<const uint8_t*>(w.data()),
                          reinterpret_cast<const uint8_t*>(w.data() + w.size()));
        }
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_x);
        cudaFree(d_w);
        cudaFree(d_x);
        return result;
    }

    // Gives this rank n of its tokens and dispatches them again.
    void set_tokens(int n) {
        for (ncclEpTensor_t* t : {topk_idx_, t_tok_, t_w_}) {
            t->sizes[0] = n;
        }
        NCCL_ASSERT(ncclEpUpdateHandle(handle_, topk_idx_, nullptr, g_stream));
        ASSERT_NO_FATAL_FAILURE(dispatch_tokens());
    }

    ncclEpSignal_t make_signal(size_t capacity) {
        ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
        EXPECT_EQ(epTensorCreate(&sig.signal, 1, ncclInt32, d_ready_, capacity), ncclSuccess);
        return sig;
    }

    // The signaled combine of 1024-element chunks, with the last rank's producer late on the last chunk, then on
    // chunk 1, equals the plain combine bitwise, forward and backward. The first case also compiles the kernels, so
    // the second has no compile delay.
    void expect_late_peer_matches() {
        constexpr int kChunk = 1024;
        const int chunks = hidden_ / kChunk;
        ncclEpSignal_t sig = make_signal(chunks);
        sig.granularity = kChunk;
        for (int late : {chunks - 1, 1}) {
            for (bool backward : {false, true}) {
                ncclResult_t rc = ncclSuccess;
                const std::vector<uint8_t> base = combine(nullptr, backward, &rc, nullptr, kChunk);
                ASSERT_EQ(rc, ncclSuccess);
                CUDA_ASSERT(cudaMemsetAsync(d_live_, 0, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16), g_stream));
                combine(nullptr, backward, &rc, t_live_, kChunk);
                ASSERT_EQ(rc, ncclSuccess);
                late_chunk_ = late;
                EXPECT_EQ(combine(&sig, backward, &rc, nullptr, kChunk), base)
                    << "late chunk " << late << " backward " << backward;
                late_chunk_ = -1;
                ASSERT_EQ(rc, ncclSuccess);
            }
        }
        ncclEpTensorDestroy(sig.signal);
    }
};

// Mode and hidden of a combine signal case.
class CombineSignalModeTest : public CombineSignalTest, public ::testing::WithParamInterface<std::tuple<RecvMode, int>> {
protected:
    CombineSignalModeTest() {
        mode_ = std::get<0>(GetParam());
        hidden_ = std::get<1>(GetParam());
    }
};

// The signaled combine behind its producer equals the plain combine bitwise, forward and backward, for multiple push
// chunks and the whole row on a four-tile row, with overflow drop and eager sizing (the last two with an empty rank),
// and for a row that ends in a partial push tile, as one whole-row chunk and as three chunks of which the last holds
// the final 832 elements, with the chunks published during the push or before it. The fixed four-tile row also runs
// its 1024-element chunks through the TMA store, which splits the combine at every team size. A plain combine of
// zeroed rows runs first, so a column the signaled push skips keeps a wrong value.
TEST_P(CombineSignalModeTest, MatchesUnsignaledBitwise) {
    struct Pass {
        int chunk;
        const char* tma_store;
    };
    std::vector<Pass> passes = {{1024, "0"}, {0, "0"}};
    if (hidden_ == kSigHidden && mode_ != RecvMode::Fixed) {
        passes = {{mode_ == RecvMode::Drop ? 1024 : 2048, "0"}};
    }
    if (hidden_ == kSigHidden && mode_ == RecvMode::Fixed) {
        passes.push_back({1024, "1"});
    }
    for (const Pass& pass : passes) {
        const int chunk = pass.chunk;
        ScopedEnv store("NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE", pass.tma_store);
        const int chunks = chunk > 0 ? nccl_ep::ceil_div(hidden_, chunk) : 1;
        ncclEpSignal_t sig = make_signal(chunks);
        sig.granularity = chunk;  // 0 = whole row; exercises the struct granularity override
        for (bool backward : {false, true}) {
            ncclResult_t rc = ncclSuccess;
            const std::vector<uint8_t> base = combine(nullptr, backward, &rc, nullptr, chunk);
            ASSERT_EQ(rc, ncclSuccess);
            // The producer completes first in the first call, which also compiles the kernels, so the call behind a
            // running producer has no compile delay.
            for (bool done : {true, false}) {
                producer_done_ = done;
                CUDA_ASSERT(cudaMemsetAsync(d_live_, 0, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16), g_stream));
                combine(nullptr, backward, &rc, t_live_, chunk);
                ASSERT_EQ(rc, ncclSuccess);
                EXPECT_EQ(combine(&sig, backward, &rc, nullptr, chunk), base)
                    << "chunk " << chunk << " TMA store " << pass.tma_store << " backward " << backward
                    << " producer done " << done;
                ASSERT_EQ(rc, ncclSuccess);
            }
            producer_done_ = false;
        }
        ncclEpTensorDestroy(sig.signal);
    }
}

INSTANTIATE_TEST_SUITE_P(Modes, CombineSignalModeTest,
                         ::testing::Values(std::make_tuple(RecvMode::Fixed, kSigHidden),
                                           std::make_tuple(RecvMode::Drop, kSigHidden),
                                           std::make_tuple(RecvMode::Eager, kSigHidden),
                                           std::make_tuple(RecvMode::Fixed, 2880)));

// A rank with no tokens still counts its whole-row signaled combine: the last rank combines none, then every rank
// combines its tokens, which equals the plain combine bitwise. The plain combine of zeroed rows before it leaves
// staging that a reduce passing its wait a call early would read.
TEST_F(CombineSignalTest, EmptyRankKeepsCallsInStep) {
    ncclEpSignal_t sig = make_signal(1);
    for (bool backward : {false, true}) {
        ncclResult_t rc = ncclSuccess;
        combine(&sig, backward, &rc, nullptr, 0);  // compiles the kernels on every rank first
        ASSERT_EQ(rc, ncclSuccess);
        ASSERT_NO_FATAL_FAILURE(set_tokens(g_rank == g_nranks - 1 ? 0 : kNumTokens));
        combine(&sig, backward, &rc, nullptr, 0);
        ASSERT_EQ(rc, ncclSuccess);
        ASSERT_NO_FATAL_FAILURE(set_tokens(kNumTokens));
        const std::vector<uint8_t> base = combine(nullptr, backward, &rc, nullptr, 0);
        ASSERT_EQ(rc, ncclSuccess);
        CUDA_ASSERT(cudaMemsetAsync(d_live_, 0, kMaxRecvSlots * hidden_ * sizeof(nv_bfloat16), g_stream));
        combine(nullptr, backward, &rc, t_live_, 0);
        ASSERT_EQ(rc, ncclSuccess);
        EXPECT_EQ(combine(&sig, backward, &rc, nullptr, 0), base) << "backward " << backward;
        ASSERT_EQ(rc, ncclSuccess);
    }
    ncclEpTensorDestroy(sig.signal);
}

// One late peer: the last rank's producer publishes a chunk well after the other ranks have pushed theirs, so every
// reduce sees some peers' rows of that chunk arrive before the late peer's. A plain combine of zeroed rows runs
// first, so a reduce that reads the late peer's staging before it arrives returns a wrong value.
TEST_F(CombineSignalTest, LatePeerMatchesUnsignaledBitwise) {
    expect_late_peer_matches();
}

// Dispatch signal in the drop and eager groups, whose handles pad each expert to one tile: kPadTile rows, or (wide)
// kMaxRecvSlots split among the local experts.
class DispatchSignalModeTest : public CombineSignalTest,
                               public ::testing::WithParamInterface<std::tuple<RecvMode, bool>> {
protected:
    DispatchSignalModeTest() {
        mode_ = std::get<0>(GetParam());
        align_ = std::get<1>(GetParam()) ? kMaxRecvSlots / std::max(1, kNumExperts / g_nranks) : kPadTile;
    }
};

// Drop: the budget ends inside the first tile, which is published once its rows are complete. Eager: the outputs
// hold exactly the received rows, so the last rank, which receives none, passes an empty, unbound flag tensor; with
// the wide padding, the padded rows exceed the derived budget (4 ranks), which bounds only a dropping group.
TEST_P(DispatchSignalModeTest, PublishesEveryTile) {
    const int tile = static_cast<int>(align_);
    const size_t capacity = nccl_ep::ceil_div<size_t>(t_golden_->sizes[0], tile);
    if (mode_ == RecvMode::Eager && g_rank == g_nranks - 1) {
        EXPECT_EQ(capacity, 0u);
    }
    int32_t* d_flags = nullptr;
    if (capacity > 0) {
        CUDA_ASSERT(cudaMalloc(&d_flags, capacity * sizeof(int32_t)));
    }
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    sig.granularity = tile;
    NCCL_ASSERT(epTensorCreate(&sig.signal, 1, ncclInt32, d_flags, capacity));
    ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    in.tokens = t_tok_;
    out.tokens = t_golden_;
    out.dispatch_signal = &sig;
    // Backward rebuilds its row order from the forward's maps.
    for (ncclEpPassDir_t dir : {NCCL_EP_FWD_PASS, NCCL_EP_BWD_PASS}) {
        const bool fwd = dir == NCCL_EP_FWD_PASS;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.pass_direction = dir;
        in.topk_weights = fwd ? t_w_ : nullptr;
        out.topk_weights = fwd ? t_recv_w_ : nullptr;
        if (capacity > 0) {
            CUDA_ASSERT(cudaMemset(d_flags, 0xFF, capacity * sizeof(int32_t)));
        }
        NCCL_ASSERT(ncclEpDispatch(handle_, &in, &out, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle_, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        if (capacity > 0) {
            std::vector<int32_t> expected(capacity, -1);
            std::iota(expected.begin(), expected.begin() + nccl_ep::ceil_div<int>(rows_, tile), 0);
            EXPECT_EQ(to_host(d_flags, capacity), expected) << "rows " << rows_ << " forward " << fwd;
        }
    }
    ncclEpTensorDestroy(sig.signal);
    cudaFree(d_flags);
}

INSTANTIATE_TEST_SUITE_P(Modes, DispatchSignalModeTest,
                         ::testing::Values(std::make_tuple(RecvMode::Drop, false),
                                           std::make_tuple(RecvMode::Eager, true)));

// A signal with too few flags, a push combine with send_only set, and a signaled MXFP8 combine (the push quantizes
// whole rows; the plain one runs) are rejected.
TEST_F(CombineSignalTest, InvalidSignalRejected) {
    constexpr int kChunk = 1024;
    ncclResult_t rc = ncclSuccess;
    ncclEpSignal_t sig = make_signal(kSigHidden / kChunk - 1);
    combine(&sig, false, &rc, nullptr, kChunk);
    EXPECT_EQ(rc, ncclInvalidArgument) << "too few flags";
    ncclEpTensorDestroy(sig.signal);

    sig = make_signal(kSigHidden / kChunk);
    send_only_ = 1;
    combine(&sig, false, &rc, nullptr, kChunk);
    EXPECT_EQ(rc, ncclInvalidArgument) << "send_only";
    send_only_ = 0;

    quant_ = NCCL_EP_COMB_QUANT_MXFP8;
    combine(nullptr, false, &rc, nullptr, kChunk);
    ASSERT_EQ(rc, ncclSuccess) << "plain MXFP8 push combine";
    combine(&sig, false, &rc, nullptr, kChunk);
    EXPECT_EQ(rc, ncclInvalidArgument) << "MXFP8";
    ncclEpTensorDestroy(sig.signal);
}

// What a signaled dispatch and a signaled combine return on a handle of the given layout of a group made from gcfg.
// Every rejection comes before any kernel, so the buffers are never written.
static void signaled_call_results(const ncclEpGroupConfig_t& gcfg, ncclEpLayout_t layout, ncclResult_t* dispatch_rc,
                                  ncclResult_t* combine_rc) {
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &gcfg));
    int64_t* d_topk = nullptr;
    void* d_x = nullptr;
    ncclEpTensor_t *topk_idx = nullptr, *t_x = nullptr, *t_ready = nullptr;
    std::vector<int64_t> topk(kNumTokens);
    for (int i = 0; i < kNumTokens; ++i) {
        topk[i] = expert_for_token(i);
    }
    CUDA_ASSERT(cudaMalloc(&d_topk, topk.size() * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    NCCL_ASSERT(epTensorCreate(&topk_idx, 2, ncclInt64, d_topk, kNumTokens, kTopK));
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&handle, group, layout, topk_idx, nullptr, nullptr, g_stream));
    CUDA_ASSERT(cudaMalloc(&d_x, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    NCCL_ASSERT(epTensorCreate(&t_x, 2, ncclBfloat16, d_x, kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_ready, 1, ncclInt32, d_x, kMaxRecvSlots));
    ncclEpSignal_t dsig = NCCL_EP_SIGNAL_INIT;
    dsig.granularity = 1;
    dsig.signal = t_ready;
    ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_out.tokens = t_x;
    d_out.dispatch_signal = &dsig;
    *dispatch_rc = ncclEpDispatch(handle, &d_in, &d_out, nullptr, nullptr, g_stream);
    ncclEpSignal_t csig = NCCL_EP_SIGNAL_INIT;
    csig.signal = t_ready;
    ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens = t_x;
    c_in.combine_signal = &csig;
    c_out.tokens = t_x;
    *combine_rc = ncclEpCombine(handle, &c_in, &c_out, nullptr, g_stream);
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    ncclEpTensorDestroy(t_ready);
    ncclEpTensorDestroy(t_x);
    cudaFree(d_x);
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    ncclEpTensorDestroy(topk_idx);
    cudaFree(d_topk);
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

// Several LSA teams: a signaled dispatch or combine returns ncclInvalidUsage before any kernel. Not in run_tests.sh:
// two LSA teams need a GIN-capable network; run it by hand with NCCL_LSA_TEAM_SIZE below the world size.
TEST(SignalLsaTeamTest, MultipleLsaTeamsRejected) {
    if (ncclTeamLsa(g_comm).nRanks == g_nranks) {
        GTEST_SKIP() << "requires more than one LSA team";
    }
    ncclEpGroupConfig_t gcfg = ep_group_config(kHidden * sizeof(nv_bfloat16));
    gcfg.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
    ncclResult_t dispatch_rc = ncclSuccess, combine_rc = ncclSuccess;
    signaled_call_results(gcfg, NCCL_EP_LAYOUT_EXPERT_MAJOR, &dispatch_rc, &combine_rc);
    EXPECT_EQ(dispatch_rc, ncclInvalidUsage);
    EXPECT_EQ(combine_rc, ncclInvalidUsage);
}

// One LSA team: a signaled dispatch or combine returns ncclInvalidArgument before any kernel outside the pull-push
// flow, on a FLAT handle and under NCCL_EP_HT_UNFUSED_SYNC. The flows are explicit, so this runs in every row.
TEST(SignalSupportTest, UnsupportedFlowLayoutAndSyncRejected) {
    if (ncclTeamLsa(g_comm).nRanks != g_nranks) {
        GTEST_SKIP() << "requires one LSA team";
    }
    ncclEpGroupConfig_t gcfg = ep_group_config(kHidden * sizeof(nv_bfloat16));
    ncclResult_t dispatch_rc = ncclSuccess, combine_rc = ncclSuccess;
    {
        ScopedEnv fused("NCCL_EP_HT_UNFUSED_SYNC", "0");
        gcfg.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
        signaled_call_results(gcfg, NCCL_EP_LAYOUT_EXPERT_MAJOR, &dispatch_rc, &combine_rc);
        EXPECT_EQ(dispatch_rc, ncclInvalidArgument) << "push-pull dispatch";
        EXPECT_EQ(combine_rc, ncclInvalidArgument) << "push-pull combine";
        gcfg.ht_flow = NCCL_EP_HT_FLOW_PULL_PUSH;
        signaled_call_results(gcfg, NCCL_EP_LAYOUT_FLAT, &dispatch_rc, &combine_rc);
        EXPECT_EQ(dispatch_rc, ncclInvalidArgument) << "FLAT dispatch";
        EXPECT_EQ(combine_rc, ncclInvalidArgument) << "FLAT combine";
    }
    ScopedEnv unfused("NCCL_EP_HT_UNFUSED_SYNC", "1");
    signaled_call_results(gcfg, NCCL_EP_LAYOUT_EXPERT_MAJOR, &dispatch_rc, &combine_rc);
    EXPECT_EQ(dispatch_rc, ncclInvalidArgument) << "unfused-sync dispatch";
    EXPECT_EQ(combine_rc, ncclInvalidArgument) << "unfused-sync combine";
}

// Two microbatches in a 1F1B order on one fixed-budget group, queued on one stream with a single sync: A's signaled
// forward dispatch and combine, then B's signaled forward dispatch, A's signaled backward dispatch, B's signaled forward
// combine and A's signaled backward combine. The microbatches differ in routing, receive counts, values and flags.
// Every received row and combined token and gradient is checked against values computed from the routing, and the
// same ordered sequence is captured in a graph and replayed with fresh payloads.
class TwoMicrobatchSignalTest : public ::testing::Test {
protected:
    static constexpr int kHid = kSigHidden;
    static constexpr int kChunkElems = 2048;
    static constexpr int kChunks = kHid / kChunkElems;
    static constexpr int kTiles = kMaxRecvSlots / kTileSize;

    struct Microbatch {
        ncclEpHandle_t handle = nullptr;
        unsigned int rows = 0;
        nv_bfloat16 *tok, *gtok, *recv, *recv_g, *live, *live_g, *out, *out_g;
        float *w, *recv_w, *out_w;
        int32_t *flags_f, *flags_g, *ready_f, *ready_g, *finished;
        ncclEpTensor_t *t_topk, *t_tok, *t_gtok, *t_recv, *t_recv_g, *t_live, *t_live_g, *t_out, *t_out_g, *t_w,
            *t_recv_w, *t_out_w;
        ncclEpSignal_t sig_df = NCCL_EP_SIGNAL_INIT, sig_dg = NCCL_EP_SIGNAL_INIT, sig_cf = NCCL_EP_SIGNAL_INIT,
                       sig_cg = NCCL_EP_SIGNAL_INIT;
    };

    ncclEpGroup_t group_ = nullptr;
    Microbatch mb_[2];  // [0] is A, [1] is B
    std::vector<void*> allocs_;
    std::vector<ncclEpTensor_t*> tensors_;
    cudaGraphExec_t graph_ = nullptr;

    template <class T>
    T* alloc(size_t n) {
        void* p = nullptr;
        EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
        allocs_.push_back(p);
        return static_cast<T*>(p);
    }

    template <class... Dims>
    ncclEpTensor_t* tensor(int ndim, ncclDataType_t type, void* data, Dims... dims) {
        ncclEpTensor_t* t = nullptr;
        EXPECT_EQ(epTensorCreate(&t, ndim, type, data, dims...), ncclSuccess);
        tensors_.push_back(t);
        return t;
    }

    // Expert of token i of a rank: A spreads the tokens over all experts, B sends all but the last to the next rank
    // and the last to expert 0, so its receive counts differ from A's.
    static int64_t expert_of(int mb, int rank, int i) {
        const int per_rank = kNumExperts / g_nranks;
        if (mb == 0) {
            return (rank * kNumTokens + i) % kNumExperts;
        }
        return i < kNumTokens - 1 ? ((rank + 1) % g_nranks) * per_rank + i % per_rank : 0;
    }
    static int owner_of(int mb, int rank, int i) { return static_cast<int>(expert_of(mb, rank, i)) / (kNumExperts / g_nranks); }
    // Values stay integers below 256, exact in BF16; the second chunk of a row differs from the first.
    static float token_value(int mb, int rank, int i, int chunk, int shift, bool grad) {
        return (grad ? 128 : mb == 0 ? 1 : 20) + rank * kNumTokens + i + 64 * chunk + shift;
    }
    static float weight_value(int mb, int rank, int i, int shift) {
        return 0.125f * ((mb + 1) * 1000 + (rank + 1) * 100 + (i + 1) * 10) + shift;
    }

    void SetUp() override {
        REQUIRE_PULL_PUSH();
        ncclEpGroupConfig_t gcfg = ep_group_config(kHid * sizeof(nv_bfloat16));
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        hcfg.dispatch_output_per_expert_alignment = kAlign;
        for (int m = 0; m < 2; ++m) {
            Microbatch& b = mb_[m];
            std::vector<int64_t> topk(kNumTokens);
            for (int i = 0; i < kNumTokens; ++i) {
                topk[i] = expert_of(m, g_rank, i);
            }
            int64_t* d_topk = alloc<int64_t>(kNumTokens * kTopK);
            CUDA_ASSERT(cudaMemcpy(d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
            b.t_topk = tensor(2, ncclInt64, d_topk, kNumTokens, kTopK);
            b.tok = alloc<nv_bfloat16>(kNumTokens * kHid);
            b.gtok = alloc<nv_bfloat16>(kNumTokens * kHid);
            b.recv = alloc<nv_bfloat16>(kMaxRecvSlots * kHid);
            b.recv_g = alloc<nv_bfloat16>(kMaxRecvSlots * kHid);
            b.live = alloc<nv_bfloat16>(kMaxRecvSlots * kHid);
            b.live_g = alloc<nv_bfloat16>(kMaxRecvSlots * kHid);
            b.out = alloc<nv_bfloat16>(kNumTokens * kHid);
            b.out_g = alloc<nv_bfloat16>(kNumTokens * kHid);
            b.w = alloc<float>(kNumTokens * kTopK);
            b.recv_w = alloc<float>(kMaxRecvSlots);
            b.out_w = alloc<float>(kNumTokens * kTopK);
            b.flags_f = alloc<int32_t>(kTiles);
            b.flags_g = alloc<int32_t>(kTiles);
            b.ready_f = alloc<int32_t>(kChunks);
            b.ready_g = alloc<int32_t>(kChunks);
            b.finished = alloc<int32_t>(2);
            b.t_tok = tensor(2, ncclBfloat16, b.tok, kNumTokens, kHid);
            b.t_gtok = tensor(2, ncclBfloat16, b.gtok, kNumTokens, kHid);
            b.t_recv = tensor(2, ncclBfloat16, b.recv, kMaxRecvSlots, kHid);
            b.t_recv_g = tensor(2, ncclBfloat16, b.recv_g, kMaxRecvSlots, kHid);
            b.t_live = tensor(2, ncclBfloat16, b.live, kMaxRecvSlots, kHid);
            b.t_live_g = tensor(2, ncclBfloat16, b.live_g, kMaxRecvSlots, kHid);
            b.t_out = tensor(2, ncclBfloat16, b.out, kNumTokens, kHid);
            b.t_out_g = tensor(2, ncclBfloat16, b.out_g, kNumTokens, kHid);
            b.t_w = tensor(2, ncclFloat32, b.w, kNumTokens, kTopK);
            b.t_recv_w = tensor(1, ncclFloat32, b.recv_w, kMaxRecvSlots);
            b.t_out_w = tensor(2, ncclFloat32, b.out_w, kNumTokens, kTopK);
            b.sig_df.signal = tensor(1, ncclInt32, b.flags_f, kTiles);
            b.sig_dg.signal = tensor(1, ncclInt32, b.flags_g, kTiles);
            b.sig_cf.signal = tensor(1, ncclInt32, b.ready_f, kChunks);
            b.sig_cg.signal = tensor(1, ncclInt32, b.ready_g, kChunks);
            b.sig_df.granularity = b.sig_dg.granularity = kTileSize;
            b.sig_cf.granularity = b.sig_cg.granularity = kChunkElems;
            NCCL_ASSERT(ncclEpCreateHandle(&b.handle, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, b.t_topk, nullptr, &hcfg,
                                           g_stream));
        }
    }

    void TearDown() override {
        if (graph_ != nullptr) {
            cudaGraphExecDestroy(graph_);
        }
        for (Microbatch& b : mb_) {
            if (b.handle != nullptr) {
                ncclEpHandleDestroy(b.handle);
            }
        }
        for (ncclEpTensor_t* t : tensors_) {
            ncclEpTensorDestroy(t);
        }
        for (void* p : allocs_) {
            cudaFree(p);
        }
        if (group_ != nullptr) {
            ncclEpGroupDestroy(group_);
        }
    }

    // Writes the payloads of this rank's tokens, forward and gradient, and weights, offset by shift.
    void fill_payload(int shift) {
        for (int m = 0; m < 2; ++m) {
            std::vector<nv_bfloat16> tok(kNumTokens * kHid), gtok(kNumTokens * kHid);
            std::vector<float> w(kNumTokens * kTopK);
            for (int i = 0; i < kNumTokens; ++i) {
                for (int c = 0; c < kHid; ++c) {
                    tok[i * kHid + c] = __float2bfloat16(token_value(m, g_rank, i, c / kChunkElems, shift, false));
                    gtok[i * kHid + c] = __float2bfloat16(token_value(m, g_rank, i, c / kChunkElems, shift, true));
                }
                w[i] = weight_value(m, g_rank, i, shift);
            }
            CUDA_ASSERT(cudaMemcpyAsync(mb_[m].tok, tok.data(), tok.size() * 2, cudaMemcpyHostToDevice, g_stream));
            CUDA_ASSERT(cudaMemcpyAsync(mb_[m].gtok, gtok.data(), gtok.size() * 2, cudaMemcpyHostToDevice, g_stream));
            CUDA_ASSERT(cudaMemcpyAsync(mb_[m].w, w.data(), w.size() * 4, cudaMemcpyHostToDevice, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        }
    }

    // A signaled dispatch of the microbatch's tokens (forward) or gradients (backward) into poisoned rows and flags.
    void dispatch(Microbatch& b, bool backward) {
        ncclEpSignal_t* sig = backward ? &b.sig_dg : &b.sig_df;
        CUDA_ASSERT(cudaMemsetAsync(backward ? b.flags_g : b.flags_f, 0xFF, kTiles * sizeof(int32_t), g_stream));
        CUDA_ASSERT(cudaMemsetAsync(backward ? b.recv_g : b.recv, 0xAB, kMaxRecvSlots * kHid * 2, g_stream));
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        in.tokens = backward ? b.t_gtok : b.t_tok;
        out.tokens = backward ? b.t_recv_g : b.t_recv;
        if (!backward) {
            in.topk_weights = b.t_w;
            out.topk_weights = b.t_recv_w;
        }
        out.dispatch_signal = sig;
        if (backward) {
            cfg.pass_direction = NCCL_EP_BWD_PASS;
        }
        NCCL_ASSERT(ncclEpDispatch(b.handle, &in, &out, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(b.handle, nullptr, g_stream));
    }

    // A signaled combine of the dispatched rows, copied by a producer kernel that publishes each chunk, into poisoned
    // outputs. The backward combine returns the gradients and the received weights.
    void combine(Microbatch& b, bool backward) {
        nv_bfloat16* live = backward ? b.live_g : b.live;
        int32_t* ready = backward ? b.ready_g : b.ready_f;
        CUDA_ASSERT(cudaMemsetAsync(live, 0xFF, kMaxRecvSlots * kHid * 2, g_stream));
        CUDA_ASSERT(cudaMemsetAsync(ready, 0xFF, kChunks * sizeof(int32_t), g_stream));
        CUDA_ASSERT(cudaMemsetAsync(backward ? b.out_g : b.out, 0xFF, kNumTokens * kHid * 2, g_stream));
        if (backward) {
            CUDA_ASSERT(cudaMemsetAsync(b.out_w, 0xFF, kNumTokens * kTopK * sizeof(float), g_stream));
        }
        producerKernel<<<1, 1024, 0, g_stream>>>(backward ? b.recv_g : b.recv, live, b.rows, kHid, kChunkElems, kChunks,
                                                 ready, b.finished, -1, 0);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        ncclEpCombineInputs_t in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
        in.tokens = backward ? b.t_live_g : b.t_live;
        in.combine_signal = backward ? &b.sig_cg : &b.sig_cf;
        out.tokens = backward ? b.t_out_g : b.t_out;
        if (backward) {
            in.topk_weights = b.t_recv_w;
            out.topk_weights = b.t_out_w;
            cfg.pass_direction = NCCL_EP_BWD_PASS;
        }
        ScopedEnv push_chunk(kChunkEnv, std::to_string(kChunkElems).c_str());
        NCCL_ASSERT(ncclEpCombine(b.handle, &in, &out, &cfg, g_stream));
    }

    // The ordered sequence, with no host synchronization.
    void queue() {
        Microbatch &a = mb_[0], &b = mb_[1];
        ASSERT_NO_FATAL_FAILURE(dispatch(a, false));
        ASSERT_NO_FATAL_FAILURE(combine(a, false));
        ASSERT_NO_FATAL_FAILURE(dispatch(b, false));
        ASSERT_NO_FATAL_FAILURE(dispatch(a, true));
        ASSERT_NO_FATAL_FAILURE(combine(b, false));
        ASSERT_NO_FATAL_FAILURE(combine(a, true));
    }

    // The data rows of a received buffer, as (first chunk, second chunk) values: the zero pad rows are dropped and
    // every row must be written.
    static std::vector<std::pair<float, float>> data_rows(const std::vector<nv_bfloat16>& recv, int rows) {
        std::vector<std::pair<float, float>> out;
        for (int r = 0; r < rows; ++r) {
            const float v0 = __bfloat162float(recv[r * kHid]), v1 = __bfloat162float(recv[r * kHid + kHid - 1]);
            EXPECT_NE(__bfloat16_as_ushort(recv[r * kHid]), 0xABABu) << "row " << r << " was not written";
            if (v0 >= 1.0f) {
                EXPECT_EQ(__bfloat162float(recv[r * kHid + kChunkElems - 1]), v0) << "row " << r;
                EXPECT_EQ(__bfloat162float(recv[r * kHid + kChunkElems]), v1) << "row " << r;
                out.emplace_back(v0, v1);
            } else {
                EXPECT_EQ(v0, 0.0f) << "pad row " << r;
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    void expect_flags(const int32_t* d_flags, unsigned int rows, const char* what) {
        const std::vector<int32_t> flags = to_host(d_flags, kTiles);
        for (int t = 0; t < kTiles; ++t) {
            EXPECT_EQ(flags[t], t < static_cast<int>(rows) / kTileSize ? t : -1) << what << " tile " << t;
        }
    }

    void expect_tokens(const nv_bfloat16* d_out, int mb, int shift, bool grad, const char* what) {
        const std::vector<nv_bfloat16> out = to_host(d_out, kNumTokens * kHid);
        for (int i = 0; i < kNumTokens; ++i) {
            for (int c = 0; c < kHid; ++c) {
                ASSERT_EQ(__bfloat162float(out[i * kHid + c]),
                          token_value(mb, g_rank, i, c / kChunkElems, shift, grad))
                    << what << " token " << i << " column " << c;
            }
        }
    }

    // Received rows, combined tokens, gradients, weights and flags of both microbatches, after one sync.
    void check(int shift) {
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        for (int m = 0; m < 2; ++m) {
            Microbatch& b = mb_[m];
            for (bool grad : {false, true}) {
                if (grad && m == 1) {
                    continue;  // B has no backward
                }
                std::vector<std::pair<float, float>> want;
                for (int r = 0; r < g_nranks; ++r) {
                    for (int i = 0; i < kNumTokens; ++i) {
                        if (owner_of(m, r, i) == g_rank) {
                            want.emplace_back(token_value(m, r, i, 0, shift, grad), token_value(m, r, i, 1, shift, grad));
                        }
                    }
                }
                std::sort(want.begin(), want.end());
                const std::vector<nv_bfloat16> recv = to_host(grad ? b.recv_g : b.recv, kMaxRecvSlots * kHid);
                EXPECT_EQ(data_rows(recv, static_cast<int>(b.rows)), want)
                    << "microbatch " << m << (grad ? " backward" : " forward") << " received rows";
                expect_flags(grad ? b.flags_g : b.flags_f, b.rows, grad ? "backward" : "forward");
            }
        }
        expect_tokens(mb_[0].out, 0, shift, false, "A forward combine");
        expect_tokens(mb_[1].out, 1, shift, false, "B forward combine");
        expect_tokens(mb_[0].out_g, 0, shift, true, "A backward combine");
        const std::vector<float> w = to_host(mb_[0].out_w, kNumTokens * kTopK);
        for (int i = 0; i < kNumTokens; ++i) {
            EXPECT_EQ(w[i], weight_value(0, g_rank, i, shift)) << "A backward combine weight " << i;
        }
    }
};

TEST_F(TwoMicrobatchSignalTest, QueuedSequenceAndGraphReplay) {
    ASSERT_NO_FATAL_FAILURE(fill_payload(0));
    // A plain pass learns each handle's receive count and compiles the kernels; the queue below is then the first
    // ordered run.
    for (Microbatch& b : mb_) {
        ASSERT_NO_FATAL_FAILURE(dispatch(b, false));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(b.handle, &b.rows));
    }
    if (g_rank == 0) {
        EXPECT_NE(mb_[0].rows, mb_[1].rows) << "rank 0 receives more tokens under B's routing";
    }
    ASSERT_NO_FATAL_FAILURE(queue());
    ASSERT_NO_FATAL_FAILURE(check(0));

    cudaGraph_t graph = nullptr;
    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    queue();
    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
    ASSERT_NE(graph, nullptr);
    CUDA_ASSERT(cudaGraphInstantiateWithFlags(&graph_, graph, 0));
    cudaGraphDestroy(graph);
    for (int shift : {3, 7}) {
        ASSERT_NO_FATAL_FAILURE(fill_payload(shift));
        CUDA_ASSERT(cudaGraphLaunch(graph_, g_stream));
        ASSERT_NO_FATAL_FAILURE(check(shift));
    }
}

// A token the receive budget drops entirely returns zero, not what an earlier combine left in its staging row. The
// first microbatch keeps every token and routes the token t of rank t to rank 0, which fills its budget; the second
// sends all tokens to rank 0, so most are dropped, among them tokens whose row from rank 0 still holds the first
// microbatch's value. The second combine is signaled in two chunks, both published before it. Every output row is
// zero or the token itself, exactly kDropBudget rows come back overall, and a token of the first microbatch's stale
// kind is dropped on at least one rank.
TEST(DropStaleStagingTest, DroppedTokenReturnsZero) {
    REQUIRE_PULL_PUSH();
    constexpr int kHidden = 2048;
    constexpr int kChunks = 2;
    ncclEpGroupConfig_t gcfg = ep_group_config(kHidden * sizeof(nv_bfloat16));
    gcfg.max_recv_tokens_per_rank = kDropBudget;
    gcfg.overflow_policy = NCCL_EP_OVERFLOW_DROP;
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &gcfg));
    const int experts_per_rank = kNumExperts / g_nranks;

    nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr, *d_out = nullptr;
    float *d_w = nullptr, *d_recv_w = nullptr;
    int64_t* d_topk = nullptr;
    int32_t* d_count = nullptr;
    int32_t* d_ready = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w, kNumTokens * kTopK * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w, kMaxRecvSlots * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_topk, kNumTokens * kTopK * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_count, 2 * sizeof(int32_t)));
    CUDA_ASSERT(cudaMalloc(&d_ready, kChunks * sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_w, 0, kNumTokens * kTopK * sizeof(float)));
    const int32_t ready[kChunks] = {0, 1};  // flag c == c publishes chunk c
    CUDA_ASSERT(cudaMemcpy(d_ready, ready, sizeof(ready), cudaMemcpyHostToDevice));
    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_out = nullptr, *t_w = nullptr, *t_recv_w = nullptr,
                   *t_topk = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w, 2, ncclFloat32, d_w, kNumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots));
    NCCL_ASSERT(epTensorCreate(&t_topk, 2, ncclInt64, d_topk, kNumTokens, kTopK));
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    NCCL_ASSERT(epTensorCreate(&sig.signal, 1, ncclInt32, d_ready, kChunks));
    sig.granularity = kHidden / kChunks;

    // One microbatch: dispatches tokens of the given base value to experts of the given rank, combines them back
    // unchanged (signaled when signal is set) and returns the rows that came back.
    using Rows = std::vector<nv_bfloat16>;
    auto run = [&](float base, auto rank_of_token, ncclEpSignal_t* signal, Rows* sent, Rows* back) -> void {
        std::vector<int64_t> topk(kNumTokens);
        std::vector<nv_bfloat16> tok(kNumTokens * kHidden);
        for (int t = 0; t < kNumTokens; ++t) {
            topk[t] = static_cast<int64_t>(rank_of_token(t)) * experts_per_rank;
            for (int c = 0; c < kHidden; ++c) {
                tok[t * kHidden + c] = __float2bfloat16(base + g_rank * kNumTokens + t + c * 0.125f);
            }
        }
        CUDA_ASSERT(cudaMemcpy(d_topk, topk.data(), topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_tok, tok.data(), tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        ncclEpHandle_t handle = nullptr;
        ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
        NCCL_ASSERT(ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_EXPERT_MAJOR, t_topk, nullptr, &hcfg, g_stream));
        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t dispatch_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = t_tok;
        d_in.topk_weights = t_w;
        dispatch_out.tokens = t_recv;
        dispatch_out.topk_weights = t_recv_w;
        NCCL_ASSERT(ncclEpDispatch(handle, &d_in, &dispatch_out, nullptr, nullptr, g_stream));
        NCCL_ASSERT(ncclEpComplete(handle, nullptr, g_stream));
        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = t_recv;
        c_in.combine_signal = signal;
        c_out.tokens = t_out;
        NCCL_ASSERT(ncclEpCombine(handle, &c_in, &c_out, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        NCCL_ASSERT(ncclEpHandleDestroy(handle));
        *sent = tok;
        *back = to_host(d_out, kNumTokens * kHidden);
    };

    // Rank (t - r) mod n gets token t of rank r: every rank receives one token from each rank, exactly its budget.
    static_assert(kDropBudget == 4 && kNumTokens == 4, "the first microbatch fills each budget exactly");
    Rows first_sent, first_back, second_sent, second_back;
    run(1.0f, [&](int t) { return (t - g_rank + g_nranks) % g_nranks; }, nullptr, &first_sent, &first_back);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(first_back, first_sent) << "first microbatch round trip";
    run(64.0f, [](int) { return 0; }, &sig, &second_sent, &second_back);
    ASSERT_FALSE(HasFatalFailure());

    int32_t counts[2] = {0, 0};  // rows back, tokens whose stale row came from the first microbatch and are dropped
    for (int t = 0; t < kNumTokens; ++t) {
        const auto row = second_back.begin() + t * kHidden;
        const auto want = second_sent.begin() + t * kHidden;
        const bool zero = std::all_of(row, row + kHidden, [](nv_bfloat16 v) { return static_cast<float>(v) == 0.0f; });
        const bool whole = std::equal(row, row + kHidden, want, [](nv_bfloat16 a, nv_bfloat16 b) {
            return static_cast<float>(a) == static_cast<float>(b);
        });
        EXPECT_TRUE(zero || whole) << "rank " << g_rank << " token " << t << " is neither dropped nor whole";
        counts[0] += whole ? 1 : 0;
        counts[1] += (zero && t == g_rank) ? 1 : 0;
    }
    CUDA_ASSERT(cudaMemcpy(d_count, counts, sizeof(counts), cudaMemcpyHostToDevice));
    NCCL_ASSERT(ncclAllReduce(d_count, d_count, 2, ncclInt32, ncclSum, g_comm, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    const std::vector<int32_t> total = to_host(d_count, 2);
    EXPECT_EQ(total[0], kDropBudget) << "rows back";
    EXPECT_GE(total[1], 1) << "no stale-row token was dropped; the test does not exercise the case";

    for (ncclEpTensor_t* t : {sig.signal, t_topk, t_recv_w, t_w, t_out, t_recv, t_tok}) {
        ncclEpTensorDestroy(t);
    }
    for (void* p : {static_cast<void*>(d_ready), static_cast<void*>(d_count), static_cast<void*>(d_topk),
                    static_cast<void*>(d_recv_w), static_cast<void*>(d_w), static_cast<void*>(d_out),
                    static_cast<void*>(d_recv), static_cast<void*>(d_tok)}) {
        cudaFree(p);
    }
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_signal_uid")) {
        return 0;
    }
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
