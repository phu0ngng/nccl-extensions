/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HT expert-major dispatch and combine signal tests. Require NCCL_EP_HT_PULL_PUSH=1 (skipped otherwise).
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include "device/device_primitives.cuh"
#include "device/ht_ep.cuh"
#include <cuda/ptx>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

static constexpr int kLocalExperts = 2;
static constexpr int kScaleBytes = 16;            // forwarded scale row
static constexpr int kPushTileElems = 1024;       // BF16 elements of one 2048-byte push tile
static constexpr int kSigHidden = 4096;           // four push tiles
static constexpr int kDropBudget = 4;             // receive budget of the drop case
static constexpr int kLateUs = 20000;             // a late peer's delay, far beyond a combine's runtime
static constexpr int kPublishGapUs = 200;         // the producer's pause after each published chunk
static constexpr int kPredecessorDelayUs = 1000;  // the delay before a predecessor kernel's writes
static constexpr uint64_t kConsumerTimeoutNs = 30ull * 1000 * 1000 * 1000;
static constexpr const char* kChunkEnv = "NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE";

// HT config of a small test group, for tokens of up to max_token_bytes.
static ncclEpGroupConfig_t group_config(size_t max_token_bytes) {
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

// Launches kernel on g_stream as the programmatic dependent of the work before it.
template <typename... Params, typename... Args>
static cudaError_t launch_programmatic(void (*kernel)(Params...), dim3 block, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(1);
    cfg.blockDim = block;
    cfg.stream = g_stream;
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

static int device_sm_count() {
    int dev = 0, sms = 0;
    EXPECT_EQ(cudaGetDevice(&dev), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), cudaSuccess);
    return sms;
}

static size_t tensor_bytes(const ncclEpTensor_t* t) {
    size_t n = nccl_ep::ncclTypeSize(t->datatype);
    for (unsigned int d = 0; d < t->ndim; ++d) {
        n *= t->sizes[d];
    }
    return n;
}

// The first n bytes of t (SIZE_MAX: all).
static std::vector<uint8_t> bytes(const ncclEpTensor_t* t, size_t n) {
    n = std::min(n, tensor_bytes(t));
    return n > 0 ? to_host(static_cast<const uint8_t*>(t->data), n) : std::vector<uint8_t>();
}

// Sets every byte of t to fill, on g_stream.
static void poison(const ncclEpTensor_t* t, int fill) {
    EXPECT_EQ(cudaMemsetAsync(t->data, fill, tensor_bytes(t), g_stream), cudaSuccess);
}

// Device buffers and their tensors, freed on destruction.
class DeviceBuffers {
public:
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    ~DeviceBuffers() {
        for (ncclEpTensor_t* t : tensors_) {
            ncclEpTensorDestroy(t);
        }
        for (void* p : allocs_) {
            cudaFree(p);
        }
    }

    // n elements of T, every byte set to fill.
    template <typename T>
    T* alloc(size_t n, int fill) {
        void* p = nullptr;
        EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemset(p, fill, n * sizeof(T)), cudaSuccess);
        allocs_.push_back(p);
        return static_cast<T*>(p);
    }

    // A tensor over data.
    template <typename... Dims>
    ncclEpTensor_t* view(ncclDataType_t type, void* data, Dims... dims) {
        ncclEpTensor_t* t = nullptr;
        EXPECT_EQ(epTensorCreate(&t, sizeof...(Dims), type, data, dims...), ncclSuccess);
        tensors_.push_back(t);
        return t;
    }

    // A rows x cols tensor (1-D when cols is 0), every byte set to fill.
    ncclEpTensor_t* buffer(ncclDataType_t type, size_t rows, size_t cols, int fill) {
        void* d = alloc<uint8_t>(rows * std::max<size_t>(cols, 1) * nccl_ep::ncclTypeSize(type), fill);
        return cols == 0 ? view(type, d, rows) : view(type, d, rows, cols);
    }

    // A zeroed rows x cols tensor holding h.
    template <typename T>
    ncclEpTensor_t* upload(ncclDataType_t type, size_t rows, size_t cols, const std::vector<T>& h) {
        ncclEpTensor_t* t = buffer(type, rows, cols, 0);
        EXPECT_EQ(cudaMemcpy(t->data, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        return t;
    }

    // A tokens x topk routing whose flat entry i is expert_of(i).
    template <typename F>
    ncclEpTensor_t* upload_topk(int tokens, int topk, F expert_of) {
        std::vector<int64_t> h(static_cast<size_t>(tokens) * topk);
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = expert_of(static_cast<int>(i));
        }
        return upload(ncclInt64, tokens, topk, h);
    }

private:
    std::vector<ncclEpTensor_t*> tensors_;
    std::vector<void*> allocs_;
};

// Flags signal[t] == t below tiles, -1 from there on.
static void expect_flags(const ncclEpSignal_t& sig, int tiles, const std::string& what) {
    const size_t n = sig.signal->sizes[0];
    const std::vector<int32_t> got =
        n > 0 ? to_host(static_cast<const int32_t*>(sig.signal->data), n) : std::vector<int32_t>{};
    for (size_t t = 0; t < n; ++t) {
        const int32_t want = static_cast<int>(t) < tiles ? static_cast<int32_t>(t) : -1;
        if (got[t] != want) {
            ADD_FAILURE() << what << ": flag " << t << " is " << got[t] << ", want " << want;
            return;
        }
    }
}

// Equal byte strings of row_bytes rows; reports the first mismatching row and byte.
static void expect_same_bytes(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, size_t row_bytes,
                              const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    const auto miss = std::mismatch(got.begin(), got.end(), want.begin());
    if (miss.first != got.end()) {
        const size_t at = miss.first - got.begin();
        ADD_FAILURE() << what << ": row " << at / row_bytes << " byte " << at % row_bytes << " is "
                      << static_cast<int>(*miss.first) << ", want " << static_cast<int>(*miss.second);
    }
}

// The first rows rows of two dispatch outputs, as entries of token, weight and scale bytes, are equal after sorting
// within each of zones equal zones (0: in place): a forward places an expert's rows in arrival order.
static void expect_same_rows(const ncclEpDispatchOutputs_t& got, const ncclEpDispatchOutputs_t& want,
                             unsigned int rows, int zones, const std::string& what) {
    std::vector<std::vector<uint8_t>> entries[2];
    for (int k = 0; k < 2; ++k) {
        const ncclEpDispatchOutputs_t& o = k == 0 ? got : want;
        entries[k].resize(rows);
        for (const ncclEpTensor_t* t : {o.tokens, o.topk_weights, o.scales}) {
            if (t == nullptr || rows == 0) {
                continue;
            }
            const size_t n = tensor_bytes(t) / t->sizes[0];
            const std::vector<uint8_t> b = bytes(t, rows * n);
            for (size_t r = 0; r < rows; ++r) {
                entries[k][r].insert(entries[k][r].end(), b.begin() + r * n, b.begin() + (r + 1) * n);
            }
        }
        for (int z = 0; z < zones; ++z) {
            std::sort(entries[k].begin() + z * (rows / zones), entries[k].begin() + (z + 1) * (rows / zones));
        }
    }
    const auto miss = std::mismatch(entries[0].begin(), entries[0].end(), entries[1].begin());
    EXPECT_TRUE(miss.first == entries[0].end()) << what << ": row " << miss.first - entries[0].begin() << " of "
                                                 << rows << " differs (" << zones << " sorted zones)";
}

// ---------------------------------------------------------------- kernels

// Spins thread 0 for us microseconds.
__device__ void spin_us(int us) {
    const uint64_t until = cuda::ptx::get_sreg_globaltimer() + 1000ull * us;
    while (cuda::ptx::get_sreg_globaltimer() < until) {
    }
}

// The dispatch's programmatic dependent: acquires tiles of tile rows in order, each within the timeout, and copies
// their rows, weights and scales. With layout (num_experts offsets, then the receive total, reset to -1), it first
// polls each word until it is set, copies them to layout_seen, and reads the tiles of min(total, max_rows) rows instead
// of tiles. *late is 1 + the tile it timed out on (0: none).
__global__ void consumeTilesKernel(const int32_t* flags, int tiles, int tile, const int32_t* layout, int num_experts,
                                   int max_rows, int32_t* layout_seen, const int4* rows, const float* weights,
                                   const int4* scales, int row_int4, int4* rows_seen, float* weights_seen,
                                   int4* scales_seen, int32_t* late) {
    const uint64_t deadline = cuda::ptx::get_sreg_globaltimer() + kConsumerTimeoutNs;
    __shared__ int s_tiles;
    if (threadIdx.x == 0) {
        s_tiles = tiles;
        for (int i = 0; layout != nullptr && i <= num_experts; ++i) {
            int value = nccl_ep::ld_acquire_global(&layout[i]);
            while (value == -1 && cuda::ptx::get_sreg_globaltimer() <= deadline) {
                value = nccl_ep::ld_acquire_global(&layout[i]);
            }
            layout_seen[i] = value;
        }
        if (layout != nullptr) {
            s_tiles = nccl_ep::ceil_div(min(layout_seen[num_experts], max_rows), tile);
        }
    }
    __syncthreads();
    for (int t = 0; t < s_tiles; ++t) {
        bool missed = false;
        while (threadIdx.x == 0 && !missed && nccl_ep::ld_acquire_global(&flags[t]) != t) {
            missed = cuda::ptx::get_sreg_globaltimer() > deadline;
        }
        if (__syncthreads_or(missed)) {
            if (threadIdx.x == 0) {
                *late = t + 1;
            }
            break;
        }
        const size_t first = static_cast<size_t>(t) * tile;
        for (int i = threadIdx.x; i < tile * row_int4; i += blockDim.x) {
            rows_seen[first * row_int4 + i] = rows[first * row_int4 + i];
        }
        for (int i = threadIdx.x; i < tile; i += blockDim.x) {
            weights_seen[first + i] = weights[first + i];
            scales_seen[first + i] = scales[first + i];
        }
    }
    nccl_ep::grid_dependency_wait();
}

// Lets its programmatic dependent start, waits delay_us, then doubles n weights: only the dependent's wait orders the
// writes first.
__global__ void doubleWeightsKernel(float* w, int n, int delay_us) {
    nccl_ep::launch_dependents();
    if (threadIdx.x == 0) {
        spin_us(delay_us);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        w[i] *= 2.0f;
    }
}

// The FC2 stand-in, the kernel before the combine: copies golden into live chunk by chunk, release-stores
// ready[s] = s after chunk s, and stores finished = 1 last. It waits late_us before chunk late_chunk (-1: never).
__global__ void producerKernel(const nv_bfloat16* golden, nv_bfloat16* live, int rows, int hidden, int chunk,
                               int chunks, int32_t* ready, int32_t* finished, int late_chunk, int late_us) {
    nccl_ep::launch_dependents();
    for (int s = 0; s < chunks; ++s) {
        if (s == late_chunk && threadIdx.x == 0) {
            spin_us(late_us);
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
            for (int d = 0; d < kPublishGapUs; ++d) {
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

// ---------------------------------------------------------------- shared fixture

// Receive sizing of a group: the fixed budget, a small budget with overflow drop, or eager.
enum class Sizing { Fixed, Drop, Eager };

// Routing of flat top-k entry i of a rank: Spread over all experts; Skewed around the last rank's experts, which
// receive none; Pull, token i / topk to topk local experts of one rank; CrossBucket, to the next rank, the local
// expert alternating every two tokens.
enum class Route { Spread, Skewed, Pull, CrossBucket };

struct Case {
    const char* name;
    Sizing sizing;
    Route route;
    int hidden;         // BF16 elements per row
    int tokens;         // per rank
    int topk;
    int align;          // per-expert zone padding; 0: the receive slots split among the local experts
    int tile;           // dispatch rows per flag; 0: align
    int chunk;          // combine granularity; 0: one push tile and the whole row; -1: no combine
    int num_sms;        // dispatch SMs; 0: the default
    bool scales;        // forward scales, and a programmatic consumer behind the signaled forward
    bool empty_sender;  // the last rank sends no token
};

static const Case kBase{"Base", Sizing::Fixed, Route::Spread, kSigHidden, kNumTokens, 1, 4, 2, 0, 0, false, false};

// A group, routing and handle of one case, with forward outputs plain (ref_) and signaled (out_), backward outputs,
// and combine buffers that combine the plain forward's rows.
class SignalTest : public ::testing::Test {
protected:
    Case c_ = kBase;
    ncclEpGroup_t group_ = nullptr;
    ncclEpHandle_t handle_ = nullptr;
    std::vector<ncclEpHandle_t> handles_;
    DeviceBuffers mem_;
    int align_ = 0, tile_ = 0;
    unsigned int slots_ = 0;  // receive rows of an output
    unsigned int rows_ = 0;   // rows the combine's producer writes
    ncclEpTensor_t *topk_ = nullptr, *bwd_ref_ = nullptr, *bwd_out_ = nullptr, *live_ = nullptr, *x_ = nullptr,
                   *xw_ = nullptr;
    ncclEpDispatchInputs_t in_ = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t ref_ = NCCL_EP_DISPATCH_OUTPUTS_INIT, out_ = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpSignal_t sig_ = NCCL_EP_SIGNAL_INIT;  // dispatch flags of tile_ rows, enough for tile size 1
    const ncclEpLayoutInfo_t* layout_ = nullptr;  // layout_info of forward dispatches
    int32_t* ready_ = nullptr;                  // combine flags
    int32_t* finished_ = nullptr;  // [0] the producer's finished word, [1] what the combine's dependent saw
    bool producer_done_ = false;   // the producer completes before the combine: every chunk is ready at its entry
    int late_chunk_ = -1;          // on the last rank only, the producer waits kLateUs before this chunk (-1: never)

    void SetUp() override {
        if (!ht_pull_push_active() || env_flag_active("NCCL_EP_HT_UNFUSED_SYNC")) {
            GTEST_SKIP() << "signals require NCCL_EP_HT_PULL_PUSH=1 and fused sync";
        }
        align_ = c_.align > 0 ? c_.align : kMaxRecvSlots / kLocalExperts;
        tile_ = c_.tile > 0 ? c_.tile : align_;
        slots_ = std::max(kMaxRecvSlots, c_.tokens * c_.topk + kLocalExperts * align_);
        ncclEpGroupConfig_t gcfg = group_config(c_.hidden * sizeof(nv_bfloat16) + (c_.scales ? kScaleBytes : 0));
        gcfg.num_experts = kLocalExperts * g_nranks;
        gcfg.max_dispatch_tokens_per_rank = c_.tokens;
        gcfg.max_recv_tokens_per_rank = slots_;
        gcfg.dispatch_num_sms = c_.num_sms > 0 ? c_.num_sms : gcfg.dispatch_num_sms;
        if (c_.sizing == Sizing::Drop) {
            gcfg.max_recv_tokens_per_rank = kDropBudget;
            gcfg.overflow_policy = NCCL_EP_OVERFLOW_DROP;
        } else if (c_.sizing == Sizing::Eager) {
            gcfg.max_recv_tokens_per_rank = NCCL_EP_AUTO;
            gcfg.num_topk = c_.topk;
        }
        NCCL_ASSERT(ncclEpCreateGroup(&group_, g_comm, &gcfg));
        topk_ = mem_.upload_topk(c_.tokens, c_.topk, [&](int i) { return expert_of(i); });
        handle_ = make_handle(topk_, align_, NCCL_EP_LAYOUT_EXPERT_MAJOR);

        std::vector<nv_bfloat16> tok(static_cast<size_t>(c_.tokens) * c_.hidden);
        std::vector<float> w(static_cast<size_t>(c_.tokens) * c_.topk);
        for (size_t i = 0; i < tok.size(); ++i) {
            tok[i] = __float2bfloat16(g_rank * c_.tokens + i / c_.hidden + 1 + (i % 64) * 0.125f);
        }
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = 0.125f * static_cast<float>(g_rank * w.size() + i + 1);
        }
        in_.tokens = mem_.upload(ncclBfloat16, c_.tokens, c_.hidden, tok);
        in_.topk_weights = mem_.upload(ncclFloat32, c_.tokens, c_.topk, w);
        in_.scales = c_.scales ? mem_.buffer(ncclUint8, c_.tokens, kScaleBytes, g_rank + 1) : nullptr;
        for (ncclEpDispatchOutputs_t* o : {&ref_, &out_}) {
            o->tokens = mem_.buffer(ncclBfloat16, slots_, c_.hidden, 0);
            o->topk_weights = mem_.buffer(ncclFloat32, slots_, 0, 0);
            o->scales = c_.scales ? mem_.buffer(ncclUint8, slots_, kScaleBytes, 0) : nullptr;
        }
        bwd_ref_ = mem_.buffer(ncclBfloat16, slots_, c_.hidden, 0);
        bwd_out_ = mem_.buffer(ncclBfloat16, slots_, c_.hidden, 0);
        live_ = mem_.buffer(ncclBfloat16, slots_, c_.hidden, 0);
        sig_.granularity = tile_;
        sig_.signal = mem_.buffer(ncclInt32, slots_, 0, 0xFF);
        if (c_.empty_sender) {
            for (ncclEpTensor_t* t : {topk_, in_.tokens, in_.topk_weights}) {
                t->sizes[0] = g_rank == g_nranks - 1 ? 0 : t->sizes[0];
            }
            NCCL_ASSERT(ncclEpUpdateHandle(handle_, topk_, nullptr, g_stream));
        }
        x_ = mem_.buffer(ncclBfloat16, topk_->sizes[0], c_.hidden, 0);
        xw_ = mem_.buffer(ncclFloat32, topk_->sizes[0], c_.topk, 0);
        ready_ = mem_.alloc<int32_t>(c_.hidden, 0xFF);
        finished_ = mem_.alloc<int32_t>(2, 0);
    }

    void TearDown() override {
        for (ncclEpHandle_t h : handles_) {
            ncclEpHandleDestroy(h);
        }
        if (group_ != nullptr) {
            ncclEpGroupDestroy(group_);
        }
    }

    int64_t expert_of(int i) const {
        const int experts = kLocalExperts * g_nranks;
        switch (c_.route) {
            case Route::Skewed:
                return (g_rank * c_.tokens + i) % (experts - kLocalExperts);
            case Route::Pull:
                return (i / c_.topk % g_nranks) * kLocalExperts + i % c_.topk;
            case Route::CrossBucket:
                return (g_rank + 1) % g_nranks * kLocalExperts + (i / 2) % 2;
            default:
                return (g_rank * c_.tokens + i) % experts;
        }
    }

    static int tiles(unsigned int rows, int tile) { return nccl_ep::ceil_div(static_cast<int>(rows), tile); }

    ncclEpHandle_t make_handle(ncclEpTensor_t* topk, int align, ncclEpLayout_t layout) {
        ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
        cfg.dispatch_output_per_expert_alignment = align;
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, group_, layout, topk, nullptr, &cfg, g_stream), ncclSuccess);
        handles_.push_back(h);
        return h;
    }

    // Queues a forward dispatch of handle h into the poisoned outputs o, signaled when sig is set: behind a kernel that
    // writes the source weights after predecessor_us (-1: none), and ahead of dependent.
    void queue_forward(ncclEpHandle_t h, ncclEpDispatchOutputs_t& o, ncclEpSignal_t* sig, int predecessor_us,
                       const std::function<cudaError_t()>& dependent) {
        for (const ncclEpTensor_t* t : {o.tokens, o.topk_weights, o.scales}) {
            if (t != nullptr) {
                poison(t, 0xAB);
            }
        }
        if (sig != nullptr) {
            poison(sig->signal, 0xFF);
        }
        o.dispatch_signal = sig;
        if (predecessor_us >= 0) {
            doubleWeightsKernel<<<1, 64, 0, g_stream>>>(static_cast<float*>(in_.topk_weights->data),
                                                        c_.tokens * c_.topk, predecessor_us);
        }
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.quant_recipe = c_.scales ? NCCL_EP_DISP_QUANT_FWD : NCCL_EP_DISP_QUANT_NONE;
        NCCL_ASSERT(ncclEpDispatch(h, &in_, &o, layout_, &cfg, g_stream));
        CUDA_ASSERT(dependent ? dependent() : cudaGetLastError());
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    }

    // A forward dispatch as queue_forward with no kernel before it, then the rows it received.
    unsigned int forward(ncclEpHandle_t h, ncclEpDispatchOutputs_t& o, ncclEpSignal_t* sig) {
        queue_forward(h, o, sig, -1, nullptr);
        return received_rows(h);
    }

    // The rows handle h received, once the stream is idle.
    unsigned int received_rows(ncclEpHandle_t h) {
        unsigned int rows = 0;
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        EXPECT_EQ(ncclEpHandle_test_getNumRecvTokens(h, &rows), ncclSuccess);
        return rows;
    }

    // Queues a backward dispatch of handle h into bwd_out_ signaled, or into bwd_ref_ plain; capturable.
    void queue_backward(ncclEpHandle_t h, bool signaled) {
        ncclEpDispatchInputs_t in = NCCL_EP_DISPATCH_INPUTS_INIT;
        in.tokens = in_.tokens;
        ncclEpDispatchOutputs_t o = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        o.tokens = signaled ? bwd_out_ : bwd_ref_;
        o.dispatch_signal = signaled ? &sig_ : nullptr;
        ncclEpDispatchConfig_t cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        cfg.pass_direction = NCCL_EP_BWD_PASS;
        poison(o.tokens, signaled ? 0xAB : 0xCD);
        if (signaled) {
            poison(sig_.signal, 0xFF);
        }
        NCCL_ASSERT(ncclEpDispatch(h, &in, &o, nullptr, &cfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    }

    // After the queued signaled backward: it flagged every tile of rows rows and wrote want.
    void expect_backward_rows(unsigned int rows, const std::vector<uint8_t>& want, const std::string& what) {
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        const size_t row = c_.hidden * sizeof(nv_bfloat16);
        expect_flags(sig_, tiles(rows, sig_.granularity), what);
        expect_same_bytes(bytes(bwd_out_, rows * row), want, row, what);
    }

    // The signaled backward of handle h flags every tile and writes the plain backward's rows.
    void expect_backward(ncclEpHandle_t h, unsigned int rows, const std::string& what) {
        ASSERT_NO_FATAL_FAILURE(queue_backward(h, false));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
        const std::vector<uint8_t> want = bytes(bwd_ref_, rows * c_.hidden * sizeof(nv_bfloat16));
        ASSERT_NO_FATAL_FAILURE(queue_backward(h, true));
        expect_backward_rows(rows, want, what);
    }

    // Queues a combine of the rows of ref_ into poisoned outputs: plain, or signaled from live_ behind a producer
    // that copies the rows into it and publishes chunks of the signal's granularity (published: from ref_, with the
    // flags already set).
    ncclResult_t queue_combine(const ncclEpSignal_t* sig, bool backward, ncclEpCombineConfig_t cfg, bool published) {
        ncclEpCombineInputs_t in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        const bool produce = sig != nullptr && !published;
        in.tokens = produce ? live_ : ref_.tokens;
        in.combine_signal = const_cast<ncclEpSignal_t*>(sig);
        out.tokens = x_;
        poison(x_, 0xAB);
        poison(xw_, 0xAB);
        if (backward) {
            in.topk_weights = ref_.topk_weights;
            out.topk_weights = xw_;
            cfg.pass_direction = NCCL_EP_BWD_PASS;
        }
        if (produce) {
            const int chunks = rows_ > 0 ? nccl_ep::ceil_div(c_.hidden, sig->granularity) : 0;
            poison(live_, 0xFF);
            poison(sig->signal, 0xFF);
            EXPECT_EQ(cudaMemsetAsync(finished_, 0, 2 * sizeof(int32_t), g_stream), cudaSuccess);
            producerKernel<<<1, 1024, 0, g_stream>>>(
                static_cast<const nv_bfloat16*>(ref_.tokens->data), static_cast<nv_bfloat16*>(live_->data), rows_,
                c_.hidden, sig->granularity, chunks, static_cast<int32_t*>(sig->signal->data), finished_,
                g_rank == g_nranks - 1 ? late_chunk_ : -1, kLateUs);
            if (producer_done_) {
                EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
            }
        }
        return ncclEpCombine(handle_, &in, &out, &cfg, g_stream);
    }

    // Combines and returns the combined tokens and weights; a programmatic kernel behind a signaled combine checks
    // that the combine completed after its producer.
    std::vector<uint8_t> combine(const ncclEpSignal_t* sig, bool backward) {
        EXPECT_EQ(queue_combine(sig, backward, NCCL_EP_COMBINE_CONFIG_INIT, false), ncclSuccess);
        if (sig != nullptr) {
            EXPECT_EQ(launch_programmatic(afterCombineKernel, dim3(1), static_cast<const int32_t*>(finished_),
                                          finished_ + 1),
                      cudaSuccess);
        }
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        if (sig != nullptr) {
            EXPECT_EQ(to_host(finished_ + 1, 1)[0], 1) << "the combine completed before the kernel before it";
        }
        std::vector<uint8_t> result = bytes(x_, SIZE_MAX), w = bytes(xw_, SIZE_MAX);
        result.insert(result.end(), w.begin(), w.end());
        return result;
    }

    // A plain combine of zero rows, which leaves zero staging, then the signaled combine, which equals base.
    void expect_signaled_matches(const ncclEpSignal_t& sig, bool backward, const std::vector<uint8_t>& base,
                                 const std::string& what) {
        ncclEpTensor_t* const rows = ref_.tokens;
        ref_.tokens = live_;
        CUDA_ASSERT(cudaMemsetAsync(live_->data, 0, tensor_bytes(live_), g_stream));
        combine(nullptr, backward);
        ref_.tokens = rows;
        expect_same_bytes(combine(&sig, backward), base, c_.hidden * sizeof(nv_bfloat16), what);
    }

    // n combine flags over ready_.
    ncclEpSignal_t make_signal(size_t n, int granularity) {
        ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
        sig.signal = mem_.view(ncclInt32, ready_, n);
        sig.granularity = granularity;
        return sig;
    }

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
};

class SignalCaseTest : public SignalTest, public ::testing::WithParamInterface<Case> {
protected:
    SignalCaseTest() { c_ = GetParam(); }
};

// The signaled forward dispatch writes the plain one's rows, weights and scales into poisoned outputs and flags every
// tile, and a programmatic consumer sees each tile's final rows once flagged; the signaled backward equals the plain
// one bitwise. The signaled combine, forward and backward, equals the plain one bitwise in chunks of the granularity,
// published before the push or during it with the last rank late, while the env names another chunk. One-tile chunks
// of a fixed row also run through the TMA store, which splits the combine at every team size, and each reduce's call
// word then equals its push's done word. Only a row of several chunks pushed by TMA splits. A fixed group also replays
// a graph of the signaled forward dispatch and combine.
TEST_P(SignalCaseTest, MatchesPlain) {
    if (c_.sizing == Sizing::Drop) {
        // Every staging row first holds a token of the spread routing, which drops none: a token the skewed routing
        // drops must still return zero.
        static_assert(kDropBudget == kNumTokens, "the spread routing fills each budget exactly");
        NCCL_ASSERT(ncclEpUpdateHandle(handle_, mem_.upload_topk(c_.tokens, 1, [](int i) -> int64_t {
            return (g_rank * kNumTokens + i) % (kLocalExperts * g_nranks);
        }), nullptr, g_stream));
        rows_ = forward(handle_, ref_, nullptr);
        combine(nullptr, false);
        NCCL_ASSERT(ncclEpUpdateHandle(handle_, topk_, nullptr, g_stream));
    }
    // In count mode the consumer bounds its tiles by the receive total the signaled forward writes, polled from -1 as
    // TE does; it and the expert offsets must equal the plain forward's. Words: plain offsets and total, then signaled.
    const bool poll = c_.scales && ht_count_mode_active(handle_);
    constexpr int kLayoutWords = kLocalExperts + 1;
    int32_t* layout_words = mem_.alloc<int32_t>(2 * kLayoutWords, 0xFF);
    ncclEpLayoutInfo_t layouts[2] = {NCCL_EP_LAYOUT_INFO_INIT, NCCL_EP_LAYOUT_INFO_INIT};
    for (int i = 0; i < 2; ++i) {
        layouts[i].expert_offsets = mem_.view(ncclInt32, layout_words + i * kLayoutWords, kLocalExperts);
        layouts[i].recv_total_counter = mem_.view(ncclInt32, layout_words + i * kLayoutWords + kLocalExperts, 1);
    }
    layout_ = poll ? &layouts[0] : nullptr;
    const unsigned int rows = forward(handle_, ref_, nullptr);
    if (c_.sizing == Sizing::Eager) {
        // The outputs hold exactly the received rows: a rank that receives none passes an empty, unbound flag tensor.
        for (ncclEpTensor_t* t : {ref_.tokens, ref_.topk_weights, out_.tokens, out_.topk_weights, bwd_ref_, bwd_out_,
                                  live_}) {
            t->sizes[0] = rows;
        }
        sig_.signal = mem_.view(ncclInt32, rows > 0 ? sig_.signal->data : nullptr, tiles(rows, tile_));
    }
    rows_ = rows;
    // Sorted zones: every expert receives as many rows unless the routing is skewed or a rank sends nothing.
    const int zones = c_.route == Route::Skewed || c_.empty_sender ? 1 : kLocalExperts;

    ncclEpDispatchOutputs_t seen = NCCL_EP_DISPATCH_OUTPUTS_INIT;  // what the consumer copied
    int32_t* late = mem_.alloc<int32_t>(1, 0);
    int32_t* layout_seen = mem_.alloc<int32_t>(kLayoutWords, 0);
    std::function<cudaError_t()> consumer;
    if (c_.scales) {
        seen.tokens = mem_.buffer(ncclBfloat16, slots_, c_.hidden, 0);
        seen.topk_weights = mem_.buffer(ncclFloat32, slots_, 0, 0);
        seen.scales = mem_.buffer(ncclUint8, slots_, kScaleBytes, 0);
        consumer = [&] {
            return launch_programmatic(
                consumeTilesKernel, dim3(64), static_cast<const int32_t*>(sig_.signal->data), tiles(rows, tile_),
                tile_, poll ? static_cast<const int32_t*>(layout_words + kLayoutWords) : nullptr, kLocalExperts,
                static_cast<int>(out_.tokens->sizes[0]), layout_seen, static_cast<const int4*>(out_.tokens->data),
                static_cast<const float*>(out_.topk_weights->data), static_cast<const int4*>(out_.scales->data),
                static_cast<int>(c_.hidden * sizeof(nv_bfloat16) / sizeof(int4)), static_cast<int4*>(seen.tokens->data),
                static_cast<float*>(seen.topk_weights->data), static_cast<int4*>(seen.scales->data), late);
        };
    }
    layout_ = poll ? &layouts[1] : nullptr;
    CUDA_ASSERT(cudaMemsetAsync(layout_words + kLayoutWords, 0xFF, kLayoutWords * sizeof(int32_t), g_stream));
    ASSERT_NO_FATAL_FAILURE(queue_forward(handle_, out_, &sig_, -1, consumer));
    layout_ = nullptr;
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    expect_flags(sig_, tiles(rows, tile_), "forward");
    if (poll) {
        const std::vector<int32_t> plain = to_host(layout_words, kLayoutWords);
        EXPECT_EQ(plain[kLocalExperts], static_cast<int32_t>(rows)) << "the plain forward's receive total";
        EXPECT_EQ(to_host(layout_seen, kLayoutWords), plain) << "offsets and total the consumer polled";
        EXPECT_EQ(to_host(layout_words + kLayoutWords, kLayoutWords), plain) << "the signaled forward's layout";
    }
    expect_same_rows(out_, ref_, rows, zones, "forward rows");
    if (c_.scales) {
        EXPECT_EQ(to_host(late, 1)[0], 0) << "the consumer timed out on tile " << to_host(late, 1)[0] - 1;
        expect_same_rows(seen, out_, rows, 0, "rows the consumer saw");
    }
    ASSERT_NO_FATAL_FAILURE(expect_backward(handle_, rows, "backward"));
    if (c_.chunk < 0) {
        return;
    }

    struct Pass {
        int granularity;
        const char* tma_store;  // NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE; null: unset
    };
    std::vector<Pass> passes = {{c_.chunk, nullptr}};
    if (c_.chunk == 0) {
        passes = {{kPushTileElems, nullptr},
                  {nccl_ep::align(c_.hidden, kPushTileElems), nullptr},
                  {kPushTileElems, "1"}};
    }
    for (const Pass& pass : passes) {
        const int granularity = pass.granularity;
        ScopedEnvVar push_chunk(kChunkEnv, granularity == kPushTileElems ? "0" : "1024");
        ScopedEnvVar tma_store("NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE", pass.tma_store);
        const int chunks = nccl_ep::ceil_div(c_.hidden, granularity);
        const ncclEpSignal_t sig = make_signal(chunks, granularity);
        const std::string store = pass.tma_store != nullptr ? ", TMA store" : "";
        for (bool backward : {false, true}) {
            uint32_t before[4] = {};
            NCCL_ASSERT(ncclEpGroup_test_getCombinePushDone(group_, before));
            const std::vector<uint8_t> base = combine(nullptr, backward);
            // The first call, behind a completed producer, also compiles the kernels.
            for (bool done : {true, false}) {
                producer_done_ = done;
                late_chunk_ = done ? -1 : chunks - 1;
                ASSERT_NO_FATAL_FAILURE(expect_signaled_matches(
                    sig, backward, base,
                    "granularity " + std::to_string(granularity) + store + (backward ? " backward" : " forward") +
                        (done ? ", published before" : ", published during")));
            }
            producer_done_ = false;
            late_chunk_ = -1;
            bool direct = false;
            ncclEpGroup_test_getLastCombinePushLaunch(group_, nullptr, nullptr, nullptr, &direct);
            EXPECT_EQ(direct, pass.tma_store == nullptr &&
                                  (chunks == 1 || g_nranks < ht_ep::kCombinePushTmaStoreMinTeam))
                << "granularity " << granularity << store;
            uint32_t words[4] = {};
            NCCL_ASSERT(ncclEpGroup_test_getCombinePushDone(group_, words));
            EXPECT_EQ(words[1], words[0]) << "granularity " << granularity << store;
            EXPECT_EQ(words[3], words[2]) << "granularity " << granularity << store;
            // Only a row of several chunks pushed by TMA splits, and never under overflow drop.
            const bool splits = chunks > 1 && c_.sizing != Sizing::Drop &&
                                (pass.tma_store != nullptr || g_nranks >= ht_ep::kCombinePushTmaStoreMinTeam);
            EXPECT_EQ(words[2] != before[2], splits) << "split, granularity " << granularity << store;
            if (c_.sizing != Sizing::Fixed || backward) {
                continue;
            }
            cudaGraphExec_t g = capture_graph([&] {
                queue_forward(handle_, out_, &sig_, -1, nullptr);
                EXPECT_EQ(queue_combine(&sig, false, NCCL_EP_COMBINE_CONFIG_INIT, false), ncclSuccess);
            });
            CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
            CUDA_ASSERT(cudaStreamSynchronize(g_stream));
            CUDA_ASSERT(cudaGraphExecDestroy(g));
            expect_flags(sig_, tiles(rows, tile_), "replayed forward");
            expect_same_rows(out_, ref_, rows, zones, "replayed forward rows");
            // The replayed forward may place rows in another order: compare with a plain combine on its maps.
            const std::vector<uint8_t> got = bytes(x_, SIZE_MAX), want = combine(nullptr, false);
            expect_same_bytes(got, std::vector<uint8_t>(want.begin(), want.begin() + got.size()),
                              c_.hidden * sizeof(nv_bfloat16), "replayed combine");
        }
    }
}

// NCCL_EP_HT_COMBINE_PUSH_LAST_CHUNK_SMS and NCCL_EP_HT_COMBINE_REDUCE_LAST_CHUNK_SMS, read at group creation: one SM
// each; all but one SM and every SM (the default reduce); different values on even and odd ranks; and out-of-range
// values, which fall back to the default. The base row in one-tile chunks, pushed by TMA at every team size, still
// splits, forward and backward, with the last rank late, and matches the plain combine bitwise.
class SignalLastChunkSmsTest : public SignalTest, public ::testing::WithParamInterface<int> {
protected:
    void SetUp() override {
        const int sms = device_sm_count();
        const bool even = g_rank % 2 == 0;
        const int values[][2] = {{1, 1}, {sms - 1, sms}, {even ? sms - 1 : 1, even ? 1 : sms}, {0, sms + 1}};
        const std::string push = std::to_string(values[GetParam()][0]);
        const std::string reduce = std::to_string(values[GetParam()][1]);
        ScopedEnvVar push_env("NCCL_EP_HT_COMBINE_PUSH_LAST_CHUNK_SMS", push.c_str());
        ScopedEnvVar reduce_env("NCCL_EP_HT_COMBINE_REDUCE_LAST_CHUNK_SMS", reduce.c_str());
        SignalTest::SetUp();
    }
};

TEST_P(SignalLastChunkSmsTest, MatchesPlain) {
    rows_ = forward(handle_, ref_, nullptr);
    ScopedEnvVar push_chunk(kChunkEnv, "0");
    ScopedEnvVar tma_store("NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE", "1");
    const int chunks = nccl_ep::ceil_div(c_.hidden, kPushTileElems);
    const ncclEpSignal_t sig = make_signal(chunks, kPushTileElems);
    for (bool backward : {false, true}) {
        uint32_t before[4] = {};
        NCCL_ASSERT(ncclEpGroup_test_getCombinePushDone(group_, before));
        const std::vector<uint8_t> base = combine(nullptr, backward);
        late_chunk_ = chunks - 1;
        ASSERT_NO_FATAL_FAILURE(expect_signaled_matches(sig, backward, base, backward ? "backward" : "forward"));
        late_chunk_ = -1;
        uint32_t words[4] = {};
        NCCL_ASSERT(ncclEpGroup_test_getCombinePushDone(group_, words));
        EXPECT_EQ(words[1], words[0]);
        EXPECT_EQ(words[3], words[2]);
        EXPECT_NE(words[2], before[2]) << "split";
    }
}

INSTANTIATE_TEST_SUITE_P(Values, SignalLastChunkSmsTest, ::testing::Range(0, 4));

// Base: four push tiles, every rank sending and receiving, with a consumer. PartialTile: a row that ends in a partial
// push tile, the last rank sending nothing. Drop: the budget ends inside the first tile. Eager: exact outputs with the
// receive slots split among the experts, beyond the derived budget. One SM, every token to both experts of its
// destination: every pull warp counts several full batches of rows, two CTAs per SM (narrow row) or one (wide).
// CrossBucket: the full grid, each bucket word holding two buckets on several lanes, so chunks split while other lanes
// read the shared word.
INSTANTIATE_TEST_SUITE_P(
    Cases, SignalCaseTest,
    ::testing::Values(
        Case{"Base", Sizing::Fixed, Route::Spread, kSigHidden, kNumTokens, 1, 4, 2, 0, 0, true, false},
        Case{"PartialTile", Sizing::Fixed, Route::Spread, 2880, kNumTokens, 1, 4, 2, 0, 0, false, true},
        Case{"Drop", Sizing::Drop, Route::Skewed, kSigHidden, kNumTokens, 1, 8, 0, kPushTileElems, 0, false, false},
        Case{"Eager", Sizing::Eager, Route::Skewed, kSigHidden, kNumTokens, 1, 0, 0, 2048, 0, false, false},
        Case{"OneSmNarrowRow", Sizing::Fixed, Route::Pull, 64, 2048, 2, 4, 2, -1, 1, true, false},
        Case{"OneSmWideRow", Sizing::Fixed, Route::Pull, 2304, 2048, 2, 4, 2, -1, 1, true, false},
        Case{"CrossBucket", Sizing::Fixed, Route::CrossBucket, 512, 256, 1, 4, 2, -1, 0, false, false}),
    [](const ::testing::TestParamInfo<Case>& info) { return std::string(info.param.name); });

// A granularity past the row is one whole-row chunk, also when its bytes overflow int: FP32 rows (the BF16 rows
// read as FP32), where the overflowed width would be one push tile. The combine matches the plain one bitwise.
TEST_F(SignalTest, HugeGranularityIsOneChunk) {
    rows_ = forward(handle_, ref_, nullptr);
    for (ncclEpTensor_t** t : {&ref_.tokens, &live_, &x_}) {
        *t = mem_.view(ncclFloat32, (*t)->data, (*t)->sizes[0], c_.hidden / 2);
    }
    const std::vector<uint8_t> base = combine(nullptr, false);
    // (2^30 + one FP32 push tile) * 4 bytes wraps to one push tile.
    const ncclEpSignal_t sig = make_signal(1, (1 << 30) + kPushTileElems / 2);
    ASSERT_NO_FATAL_FAILURE(expect_signaled_matches(sig, false, base, "huge granularity"));
    int chunk_int4 = 0;
    ncclEpGroup_test_getLastCombinePushLaunch(group_, &chunk_int4, nullptr, nullptr, nullptr);
    EXPECT_EQ(chunk_int4, static_cast<int>(c_.hidden * sizeof(nv_bfloat16) / 16)) << "one whole-row chunk";
}

// Weights a kernel writes, after a delay, directly before a forward dispatch reach the peers: a signaled EM dispatch,
// a plain EM dispatch after it, and a flat dispatch. Each matches a dispatch of the same weights after a sync, and
// differs from the one before the write.
TEST_F(SignalTest, PredecessorWeightsReachPeers) {
    ncclEpDispatchOutputs_t flat_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    flat_out.tokens = out_.tokens;
    flat_out.topk_weights = mem_.view(ncclFloat32, out_.topk_weights->data, slots_, c_.topk);
    flat_out.topk_idx = mem_.buffer(ncclInt64, slots_, c_.topk, 0);
    const struct {
        ncclEpHandle_t h;
        ncclEpDispatchOutputs_t* o;
        ncclEpSignal_t* sig;
        const char* what;
    } cases[] = {{handle_, &out_, &sig_, "signaled EM"},
                 {handle_, &out_, nullptr, "plain EM after a signaled one"},
                 {make_handle(topk_, 0, NCCL_EP_LAYOUT_FLAT), &flat_out, nullptr, "flat"}};
    for (const auto& k : cases) {
        // The sorted received weights, the source weights doubled first when delay_us >= 0.
        const auto received = [&](int delay_us) {
            queue_forward(k.h, *k.o, k.sig, delay_us, nullptr);
            const unsigned int rows = received_rows(k.h);
            EXPECT_GT(rows, 0u);
            std::vector<float> w = to_host(static_cast<const float*>(k.o->topk_weights->data), rows * c_.topk);
            std::sort(w.begin(), w.end());
            return w;
        };
        const std::vector<float> before = received(-1), after = received(kPredecessorDelayUs);
        EXPECT_NE(after, before) << k.what;
        EXPECT_EQ(after, received(-1)) << k.what;
    }
}

// Each update rebuilds the send counts: two more updates with the same routing, and an eager update after a first
// update that was captured into a graph that never runs (on stale handle memory), each give the same rows.
TEST_F(SignalTest, UpdatesKeepCounts) {
    const unsigned int rows = forward(handle_, out_, &sig_);
    ASSERT_GT(rows, 0u);
    NCCL_ASSERT(ncclEpUpdateHandle(handle_, topk_, nullptr, g_stream));
    NCCL_ASSERT(ncclEpUpdateHandle(handle_, topk_, nullptr, g_stream));
    EXPECT_EQ(forward(handle_, out_, &sig_), rows);
    expect_flags(sig_, tiles(rows, tile_), "after two updates");

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = align_;
    size_t mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, &mem_bytes, c_.topk));
    ncclEpTensor_t* t_mem = mem_.view(ncclUint8, mem_.alloc<uint8_t>(mem_bytes, 0xFF), mem_bytes);
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &cfg, c_.topk, t_mem));
    handles_.push_back(h);
    cudaGraph_t graph = nullptr;
    CUDA_ASSERT(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeRelaxed));
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamEndCapture(g_stream, &graph));
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_, nullptr, g_stream));
    EXPECT_EQ(forward(h, out_, &sig_), rows);
    expect_flags(sig_, tiles(rows, tile_), "after the eager update");
    CUDA_ASSERT(cudaGraphDestroy(graph));
}

// The signaled backward follows the maps of its handle's last forward: after a plain forward, with two interleaved
// handles of other receive counts, after a routing of one token whose padding rows held real rows before, after a
// forward at another tile size, and replayed from a graph captured before another handle's backward and a routing
// change.
TEST_F(SignalTest, BackwardFollowsForwardMaps) {
    // Token 0 to expert 0, the rest to the next rank's first expert: every rank receives another count.
    ncclEpTensor_t* other = mem_.upload_topk(kNumTokens, 1, [](int i) -> int64_t {
        return i == 0 ? 0 : (g_rank + 1) % g_nranks * kLocalExperts;
    });
    // Adjacent tokens swapped: every rank receives as many rows, in another source order.
    ncclEpTensor_t* swapped = mem_.upload_topk(kNumTokens, 1, [](int i) -> int64_t {
        return (g_rank * kNumTokens + (i ^ 1)) % (kLocalExperts * g_nranks);
    });
    const size_t row = c_.hidden * sizeof(nv_bfloat16);

    const unsigned int rows = forward(handle_, ref_, nullptr);
    ASSERT_NO_FATAL_FAILURE(expect_backward(handle_, rows, "after a plain forward"));
    cudaGraphExec_t g = capture_graph([&] { queue_backward(handle_, true); });

    ncclEpHandle_t h2 = make_handle(other, align_, NCCL_EP_LAYOUT_EXPERT_MAJOR);
    const unsigned int rows1 = forward(handle_, out_, &sig_), rows2 = forward(h2, out_, &sig_);
    ASSERT_NO_FATAL_FAILURE(expect_backward(handle_, rows1, "first of two handles"));
    const std::vector<uint8_t> want = bytes(bwd_ref_, rows1 * row);  // on the maps of handle_'s last forward
    ASSERT_NO_FATAL_FAILURE(expect_backward(h2, rows2, "second of two handles"));
    CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
    expect_backward_rows(rows1, want, "replay after another handle's backward");

    NCCL_ASSERT(ncclEpUpdateHandle(handle_, swapped, nullptr, g_stream));
    const unsigned int rows3 = forward(handle_, ref_, nullptr);
    ASSERT_NO_FATAL_FAILURE(queue_backward(handle_, false));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    CUDA_ASSERT(cudaGraphLaunch(g, g_stream));
    expect_backward_rows(rows3, bytes(bwd_ref_, rows3 * row), "replay after a routing change");
    CUDA_ASSERT(cudaGraphExecDestroy(g));
    // Only token 0, to the next rank: a padding row of the new map held a real row of the old one.
    NCCL_ASSERT(ncclEpUpdateHandle(handle_, mem_.upload_topk(kNumTokens, 1, [](int i) -> int64_t {
        return i == 0 ? (g_rank + 1) % g_nranks * kLocalExperts : -1;
    }), nullptr, g_stream));
    const unsigned int one = forward(handle_, ref_, nullptr);
    EXPECT_EQ(one, static_cast<unsigned int>(align_));
    ASSERT_NO_FATAL_FAILURE(expect_backward(handle_, one, "after a routing of one token"));

    // A forward at tile size 1 leaves counts in scratch words that a backward at tile size 16 must not read as row map
    // entries: ranks 0 and 1 receive 3 and 5 tokens per local expert, each zone padded to 16 rows.
    static_assert(kNumTokens == 4 && kMaxRecvSlots == 32, "the routing below assumes these");
    if (g_nranks == 4) {
        ncclEpHandle_t h = make_handle(mem_.upload_topk(kNumTokens, 1, [](int i) -> int64_t {
            const int t = g_rank * kNumTokens + i;  // 0-2 to expert 0, 3-7 to 1, 8-10 to 2, 11-15 to 3
            return (t >= 3) + (t >= 8) + (t >= 11);
        }), 16, NCCL_EP_LAYOUT_EXPERT_MAJOR);
        sig_.granularity = 1;
        const unsigned int wide_rows = forward(h, out_, &sig_);
        sig_.granularity = 16;
        ASSERT_NO_FATAL_FAILURE(expect_backward(h, wide_rows, "backward at tile size 16"));
        sig_.granularity = tile_;
    }
}

// A signaled dispatch and combine on a handle of layout in a group of gcfg each return want before any kernel.
static void expect_group_rejects(const ncclEpGroupConfig_t& gcfg, ncclEpLayout_t layout, bool combine,
                                 ncclResult_t want, const std::string& what) {
    ncclEpGroup_t group = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&group, g_comm, &gcfg));
    DeviceBuffers mem;
    ncclEpTensor_t* topk = mem.upload_topk(kNumTokens, kTopK, [&](int i) -> int64_t {
        return (g_rank * kNumTokens + i) % gcfg.num_experts;
    });
    ncclEpHandle_t handle = nullptr;
    NCCL_ASSERT(ncclEpCreateHandle(&handle, group, layout, topk, nullptr, nullptr, g_stream));
    ncclEpSignal_t sig = NCCL_EP_SIGNAL_INIT;
    sig.granularity = 1;
    sig.signal = mem.buffer(ncclInt32, kMaxRecvSlots, 0, 0);
    ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens = mem.buffer(ncclBfloat16, kNumTokens, kHidden, 0);
    d_out.tokens = mem.buffer(ncclBfloat16, kMaxRecvSlots, kHidden, 0);
    d_out.dispatch_signal = &sig;
    EXPECT_EQ(ncclEpDispatch(handle, &d_in, &d_out, nullptr, nullptr, g_stream), want) << what << " dispatch";
    if (combine) {
        sig.granularity = kPushTileElems;
        ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens = d_out.tokens;
        c_in.combine_signal = &sig;
        c_out.tokens = d_in.tokens;
        EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out, nullptr, g_stream), want) << what << " combine";
    }
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    NCCL_ASSERT(ncclEpHandleDestroy(handle));
    NCCL_ASSERT(ncclEpGroupDestroy(group));
}

// Every documented rejection returns ncclInvalidArgument before any kernel, one row per case; a combine wrongly
// accepted finds its flags set and completes. The handle then still dispatches and combines.
TEST_F(SignalTest, InvalidSignalRejected) {
    rows_ = forward(handle_, ref_, nullptr);
    const int flags = kSigHidden / kPushTileElems;
    void* d = sig_.signal->data;
    const auto signal = [](ncclEpTensor_t* t, int granularity, int unit) {
        ncclEpSignal_t s = NCCL_EP_SIGNAL_INIT;
        s.signal = t;
        s.granularity = granularity;
        s.unit = static_cast<ncclEpSignalUnit_t>(unit);
        return s;
    };
    std::vector<int32_t> published(c_.hidden);
    std::iota(published.begin(), published.end(), 0);
    CUDA_ASSERT(cudaMemcpy(ready_, published.data(), c_.hidden * sizeof(int32_t), cudaMemcpyHostToDevice));
    ncclEpTensor_t* ready = make_signal(flags, kPushTileElems).signal;
    const struct {
        const char* name;
        bool combine;
        ncclEpSignal_t sig;
        unsigned int send_only = 0;
        ncclEpCombQuant_t quant = NCCL_EP_COMB_QUANT_NONE;
    } call_rows[] = {
        {"granularity 0", false, signal(sig_.signal, 0, 0)},
        {"negative granularity", false, signal(sig_.signal, -tile_, 0)},
        {"granularity not dividing the alignment", false, signal(sig_.signal, 2 * align_, 0)},
        {"too few flags", false, signal(mem_.view(ncclInt32, d, tiles(slots_, tile_) - 1), tile_, 0)},
        {"unit COLUMNS", false, signal(sig_.signal, tile_, NCCL_EP_SIGNAL_UNIT_COLUMNS)},
        {"unit 3", false, signal(sig_.signal, tile_, 3)},
        {"float flags", false, signal(mem_.view(ncclFloat32, d, slots_), tile_, 0)},
        {"2D flags", false, signal(mem_.view(ncclInt32, d, slots_, 1), tile_, 0)},
        {"misaligned flags", false, signal(mem_.view(ncclInt32, static_cast<char*>(d) + 2, slots_ - 1), tile_, 0)},
        {"too few flags", true, signal(make_signal(flags - 1, 0).signal, kPushTileElems, 0)},
        {"granularity 0", true, signal(ready, 0, 0)},
        {"negative granularity", true, signal(ready, -kPushTileElems, 0)},
        {"granularity not whole push tiles", true, signal(ready, 1500, 0)},
        {"unit ROWS", true, signal(ready, kPushTileElems, NCCL_EP_SIGNAL_UNIT_ROWS)},
        {"unit 3", true, signal(ready, kPushTileElems, 3)},
        {"send_only", true, signal(ready, kPushTileElems, 0), 1},
        {"MXFP8", true, signal(ready, kPushTileElems, 0), 0, NCCL_EP_COMB_QUANT_MXFP8},
    };
    for (const auto& r : call_rows) {
        ncclEpDispatchOutputs_t o = out_;
        o.dispatch_signal = const_cast<ncclEpSignal_t*>(&r.sig);
        ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
        cfg.send_only = r.send_only;
        cfg.quant_recipe = r.quant;
        const ncclResult_t rc = r.combine ? queue_combine(&r.sig, false, cfg, true)
                                          : ncclEpDispatch(handle_, &in_, &o, nullptr, nullptr, g_stream);
        EXPECT_EQ(rc, ncclInvalidArgument) << (r.combine ? "combine " : "dispatch ") << r.name;
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    }

    // Groups of their own; NCCL_EP_HT_UNFUSED_SYNC is read at group creation.
    ncclEpGroupConfig_t push_pull = group_config(kHidden * sizeof(nv_bfloat16)), pull_push = push_pull, ll = push_pull;
    push_pull.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
    pull_push.ht_flow = NCCL_EP_HT_FLOW_PULL_PUSH;
    ll.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    ll.max_recv_tokens_per_rank = kNumTokens;
    ll.num_qp_per_rank = kNumExperts / g_nranks;
    // Over 2^17 chunks of 32 tokens overflow the 16 KB ordered-pull pool with one bucket; only pull-count builds it.
    ncclEpGroupConfig_t pool = pull_push;
    pool.max_dispatch_tokens_per_rank = ((1 << 17) / g_nranks + 1) * 32;
    pool.max_recv_tokens_per_rank = pool.max_dispatch_tokens_per_rank;
    const struct {
        const char* name;
        ncclEpGroupConfig_t gcfg;
        ncclEpLayout_t layout;
        const char* unfused;
        bool combine, skip;
    } group_rows[] = {
        {"push-pull flow", push_pull, NCCL_EP_LAYOUT_EXPERT_MAJOR, "0", true, false},
        {"FLAT layout", pull_push, NCCL_EP_LAYOUT_FLAT, "0", true, false},
        {"NCCL_EP_HT_UNFUSED_SYNC", pull_push, NCCL_EP_LAYOUT_EXPERT_MAJOR, "1", true, false},
        {"LL group", ll, NCCL_EP_LAYOUT_EXPERT_MAJOR, "0", true, false},
        {"ordered-pull pool over its cap", pool, NCCL_EP_LAYOUT_EXPERT_MAJOR, "0", false,
         ht_ag_scan_mode_active() || g_nranks < 4},
    };
    for (const auto& r : group_rows) {
        if (!r.skip) {
            ScopedEnvVar unfused("NCCL_EP_HT_UNFUSED_SYNC", r.unfused);
            expect_group_rejects(r.gcfg, r.layout, r.combine, ncclInvalidArgument, r.name);
        }
    }

    EXPECT_EQ(forward(handle_, out_, &sig_), rows_);
    expect_flags(sig_, tiles(rows_, tile_), "dispatch after the rejections");
    const std::vector<uint8_t> base = combine(nullptr, false);
    expect_signaled_matches(make_signal(flags, kPushTileElems), false, base, "combine after the rejections");
}

// Several LSA teams: a signaled dispatch or combine returns ncclInvalidUsage before any kernel. Runs only with
// NCCL_LSA_TEAM_SIZE below the world size, which needs a GIN-capable network.
TEST(SignalLsaTeamTest, MultipleLsaTeamsRejected) {
    if (ncclTeamLsa(g_comm).nRanks == g_nranks) {
        GTEST_SKIP() << "requires more than one LSA team";
    }
    ncclEpGroupConfig_t gcfg = group_config(kHidden * sizeof(nv_bfloat16));
    gcfg.ht_flow = NCCL_EP_HT_FLOW_PUSH_PULL;
    expect_group_rejects(gcfg, NCCL_EP_LAYOUT_EXPERT_MAJOR, true, ncclInvalidUsage, "several LSA teams");
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_signal_uid")) {
        return 0;
    }
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
