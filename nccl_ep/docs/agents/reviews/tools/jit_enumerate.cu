/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Review tool: generate the LL JIT kernel sources for a fixed matrix of
// configurations without a GPU, in the layout the library's JIT cache uses
// (<out>/<variant>/kernel.cu plus metadata.json), so jit_variants.py can
// compile them with ptxas -v. Built and run by jit_enumerate.sh against one
// source tree. This file holds only the matrix; the generators' and selectors'
// argument lists and the kernel enumerators come from jit_enumerate_gen.py,
// which reads them from that tree's headers. Variant names are stable across
// trees so the base and head tables join.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>

#include "nccl_ep.h"
#include "device/jit/ll_dispatch_jit.cuh"
#include "device/jit/ll_combine_jit.cuh"

#ifndef REVIEW_ARCH_NUM
#define REVIEW_ARCH_NUM 90
#endif

using namespace nccl_ep::ll;
using namespace nccl_ep::ll::jit;
namespace fs = std::filesystem;

namespace {

// Every knob a generator or selector may take; jit_enumerate_gen.py binds
// parameter names to these fields.
struct Cfg {
    int hidden;
    int topk;
    ncclEpLayout_t layout;
    bool nvlinkOnly;
    NcclEpDispatchLsaMode lsaMode;
    LlDispatchAlgo algo;
    bool cftEnabled;
    int tmaStages;
    bool cftZeroCopy;
    bool cftCounted;
    LlCombineKernel combineKernel;
    const char* recipeLiteral;
    ncclDataType_t weightDtype;
};

#include "jit_enumerate_gen.inc"

const char* layoutName(ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_RANK_MAJOR ? "rm" : "em";
}

const char* weightName(ncclDataType_t dtype) {
    return dtype == ncclBfloat16 ? "wbf16" : "wfp32";
}

struct Emitter {
    fs::path root;
    std::set<std::size_t> seen;
    int written = 0;

    void emit(const std::string& name, const std::string& source) {
        if (!seen.insert(std::hash<std::string>{}(source)).second) {
            return;
        }
        const fs::path dir = root / name;
        fs::create_directories(dir);
        std::ofstream(dir / "kernel.cu") << source;
        std::ofstream(dir / "metadata.json") << "{\"variant_name\": \"" << name << "\"}\n";
        ++written;
    }
};

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <out-dir>\n", argv[0]);
        return 2;
    }
    Emitter out{fs::path(argv[1]), {}, 0};
    fs::create_directories(out.root);

    std::vector<ncclDataType_t> weights = {ncclFloat32};
#if REVIEW_HAS_WEIGHT_DTYPE
    weights.push_back(ncclBfloat16);
#endif
    const int topk = 8;
    const ncclEpLayout_t layouts[] = {NCCL_EP_LAYOUT_RANK_MAJOR, NCCL_EP_LAYOUT_EXPERT_MAJOR};
    const int hiddens[] = {7168, 256};

    nccl_ep::DispatchKernelSpec spec{};
    if (nccl_ep::resolveDispatchKernelSpec(NCCL_EP_DISP_QUANT_NONE, ncclBfloat16, ncclFloat32, &spec) !=
        ncclSuccess) {
        std::fprintf(stderr, "resolveDispatchKernelSpec failed\n");
        return 1;
    }

    std::set<LlDispatchAlgo> reachedAlgos;
    std::set<LlCombineKernel> reachedKernels;
    for (ncclEpLayout_t layout : layouts) {
        for (int hidden : hiddens) {
            const std::string tail = "_h" + std::to_string(hidden) + "_k" + std::to_string(topk) + "_tbf16_";
            Cfg c{};
            c.hidden = hidden;
            c.topk = topk;
            c.layout = layout;
            c.recipeLiteral = "";
            for (bool nvlinkOnly : {true, false}) {
                for (NcclEpDispatchLsaMode mode : kAllNcclEpDispatchLsaMode) {
                    c.nvlinkOnly = nvlinkOnly;
                    c.lsaMode = mode;
                    c.cftEnabled = false;
                    c.algo = selectAlgo(c, spec);
                    // The TMA kernel needs its stage count; other kernels ignore it.
                    c.tmaStages = c.algo == LlDispatchAlgo::k2SidedRmLsaTma ? 2 : 0;
                    reachedAlgos.insert(c.algo);
                    for (ncclDataType_t w : weights) {
                        c.weightDtype = w;
                        const std::string name = std::string("disp_") + layoutName(layout) + "_" +
                                                 LlDispatchAlgoTag(c.algo) + (nvlinkOnly ? "_nvl" : "_net") +
                                                 tail + weightName(w);
                        out.emit(name, dispatchSource(c, spec));
                    }
                }
            }
            // The CFT kernels compile only where device/cft.cuh enables them: CUDA 13.3+ for
            // sm_100 and newer. REVIEW_ARCH_NUM is the target the sources will be compiled for.
            const bool cftCompiles = CUDA_VERSION >= 13030 && REVIEW_ARCH_NUM >= 100;
            if (layout == NCCL_EP_LAYOUT_RANK_MAJOR && hidden == hiddens[0] && !cftCompiles) {
                std::printf("CFT variants skipped: need CUDA 13.3+ and sm_100+ (have CUDA %d, sm_%d)\n",
                            CUDA_VERSION, REVIEW_ARCH_NUM);
            }
            if (layout == NCCL_EP_LAYOUT_RANK_MAJOR && cftCompiles) {
                struct CftMode { const char* name; bool zeroCopy; bool counted; };
                const CftMode cftModes[] = {{"counted", false, true}, {"uncounted", false, false},
                                            {"zerocopy", true, false}};
                c.nvlinkOnly = true;
                c.lsaMode = NcclEpDispatchLsaMode::kTma;
                c.cftEnabled = true;
                c.algo = selectAlgo(c, spec);
                c.tmaStages = 0;
                reachedAlgos.insert(c.algo);
                for (const CftMode& m : cftModes) {
                    c.cftZeroCopy = m.zeroCopy;
                    c.cftCounted = m.counted;
                    for (ncclDataType_t w : weights) {
                        c.weightDtype = w;
                        const std::string name = std::string("disp_rm_cft_") + m.name + "_nvl" + tail + weightName(w);
                        out.emit(name, dispatchSource(c, spec));
                    }
                }
                c.cftZeroCopy = false;
                c.cftCounted = false;
            }
            // The LSA combine kernels exist only for shapes the selector accepts; a source for any
            // other shape fails a static_assert and says nothing about the change.
            c.nvlinkOnly = true;
            c.lsaMode = NcclEpDispatchLsaMode::kTma;
            c.cftEnabled = false;
            const bool combineLsa = lsaEligible(c, spec);
            for (LlCombineKernel kernel : kAllLlCombineKernel) {
                const bool isDefault = kernel == LlCombineKernel::kDefault;
                if (!isDefault && !(layout == NCCL_EP_LAYOUT_RANK_MAJOR && combineLsa)) {
                    continue;
                }
                c.combineKernel = kernel;
                c.recipeLiteral = isDefault ? ::nccl_ep::jit::combine_recipe_literal(NCCL_EP_COMB_QUANT_NONE) : "";
                reachedKernels.insert(kernel);
                for (ncclDataType_t w : weights) {
                    c.weightDtype = w;
                    const std::string name = std::string("comb_") + layoutName(layout) + "_" +
                                             LlCombineKernelTag(kernel) + tail + weightName(w);
                    out.emit(name, combineSource(c, spec));
                }
            }
        }
    }
    // A kernel the library defines but this matrix never builds is a gap in the matrix.
    for (LlDispatchAlgo a : kAllLlDispatchAlgo) {
        if (!reachedAlgos.count(a)) {
            std::printf("NOT COVERED: dispatch algo %s (CFT kernels need CUDA 13.3+ and sm_100+)\n",
                        LlDispatchAlgoTag(a));
        }
    }
    for (LlCombineKernel k : kAllLlCombineKernel) {
        if (!reachedKernels.count(k)) {
            std::printf("NOT COVERED: combine kernel %s\n", LlCombineKernelTag(k));
        }
    }
    std::printf("%d distinct JIT sources written to %s\n", out.written, out.root.c_str());
    return 0;
}
