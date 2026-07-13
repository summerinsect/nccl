// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDACHECK(cmd)                                                                               \
  do {                                                                                               \
    cudaError_t err__ = (cmd);                                                                       \
    if (err__ != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd,                  \
                   cudaGetErrorString(err__));                                                       \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

#define NCCLCHECK(cmd)                                                                               \
  do {                                                                                               \
    ncclResult_t err__ = (cmd);                                                                      \
    if (err__ != ncclSuccess) {                                                                      \
      std::fprintf(stderr, "NCCL error %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd,                  \
                   ncclGetErrorString(err__));                                                       \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

__global__ void fillBf16Kernel(__nv_bfloat16* dst, size_t n, int rank) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long x = static_cast<unsigned long long>(i) * 1315423911ull +
                         static_cast<unsigned long long>(rank + 1) * 2654435761ull;
  int centered = static_cast<int>(x & 2047ull) - 1024;
  float value = centered * (1.0f / 1024.0f) + static_cast<float>(rank) * 0.125f;
  dst[i] = __float2bfloat16(value);
}

__global__ void bf16ToFloatKernel(const __nv_bfloat16* src, float* dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

struct RankBuffers {
  cudaStream_t stream = nullptr;
  __nv_bfloat16* sendBf16 = nullptr;
  float* sendF32 = nullptr;
  float* recvGold = nullptr;
  float* recvMixed = nullptr;
  __nv_bfloat16* recvBf16 = nullptr;
  float* recvBf16F32 = nullptr;
};

struct ErrorStats {
  double maxAbs = 0.0;
  double maxRel = 0.0;
  int rank = 0;
  size_t index = 0;
};

static size_t parseSize(const char* s) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(s, &end, 0);
  if (end == s || *end != '\0') {
    std::fprintf(stderr, "Invalid integer argument: %s\n", s);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(value);
}

static void syncAll(const std::vector<int>& devices, const std::vector<RankBuffers>& bufs) {
  for (size_t r = 0; r < bufs.size(); ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    CUDACHECK(cudaStreamSynchronize(bufs[r].stream));
  }
}

enum class Path {
  GoldF32,
  Mixed,
  NativeBf16,
};

static void runCollective(Path path, const std::vector<int>& devices, const std::vector<ncclComm_t>& comms,
                          std::vector<RankBuffers>& bufs, size_t recvcount) {
  NCCLCHECK(ncclGroupStart());
  for (size_t r = 0; r < bufs.size(); ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    if (path == Path::GoldF32) {
      NCCLCHECK(ncclReduceScatter(bufs[r].sendF32, bufs[r].recvGold, recvcount, ncclFloat32, ncclSum, comms[r],
                                  bufs[r].stream));
    } else if (path == Path::Mixed) {
      NCCLCHECK(ncclMixedPrecisionReduceScatter(bufs[r].sendBf16, bufs[r].recvMixed, recvcount, ncclFloat32,
                                                ncclBfloat16, ncclSum, comms[r], bufs[r].stream));
    } else {
      NCCLCHECK(ncclReduceScatter(bufs[r].sendBf16, bufs[r].recvBf16, recvcount, ncclBfloat16, ncclSum, comms[r],
                                  bufs[r].stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
}

static double timeCollective(Path path, const std::vector<int>& devices, const std::vector<ncclComm_t>& comms,
                             std::vector<RankBuffers>& bufs, size_t recvcount, int iters) {
  constexpr int warmup = 5;
  for (int i = 0; i < warmup; ++i) runCollective(path, devices, comms, bufs, recvcount);
  syncAll(devices, bufs);

  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) runCollective(path, devices, comms, bufs, recvcount);
  syncAll(devices, bufs);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count() / static_cast<double>(iters);
}

static ErrorStats compareToGold(const std::vector<int>& devices, const std::vector<RankBuffers>& bufs,
                                size_t recvcount, bool compareMixed) {
  ErrorStats stats;
  std::vector<float> gold(recvcount);
  std::vector<float> got(recvcount);

  for (size_t r = 0; r < bufs.size(); ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    CUDACHECK(cudaMemcpy(gold.data(), bufs[r].recvGold, recvcount * sizeof(float), cudaMemcpyDeviceToHost));
    const float* src = compareMixed ? bufs[r].recvMixed : bufs[r].recvBf16F32;
    CUDACHECK(cudaMemcpy(got.data(), src, recvcount * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < recvcount; ++i) {
      double absErr = std::abs(static_cast<double>(got[i]) - static_cast<double>(gold[i]));
      double denom = std::max(std::abs(static_cast<double>(gold[i])), 1.0e-20);
      double relErr = absErr / denom;
      if (absErr > stats.maxAbs) {
        stats.maxAbs = absErr;
        stats.rank = static_cast<int>(r);
        stats.index = i;
      }
      stats.maxRel = std::max(stats.maxRel, relErr);
    }
  }
  return stats;
}

static void dumpSmall(const std::vector<int>& devices, const std::vector<RankBuffers>& bufs, size_t recvcount) {
  size_t n = std::min<size_t>(recvcount, 16);
  std::vector<float> gold(n);
  std::vector<float> mixed(n);
  std::vector<float> bf16(n);

  for (size_t r = 0; r < bufs.size(); ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    CUDACHECK(cudaMemcpy(gold.data(), bufs[r].recvGold, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(mixed.data(), bufs[r].recvMixed, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(bf16.data(), bufs[r].recvBf16F32, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("rank %zu first %zu values:\n", r, n);
    for (size_t i = 0; i < n; ++i) {
      std::printf("  [%zu] gold=% .9g mixed=% .9g bf16=% .9g\n", i, gold[i], mixed[i], bf16[i]);
    }
  }
}

int main(int argc, char** argv) {
  int deviceCount = 0;
  CUDACHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::fprintf(stderr, "mixed RS PoC requires at least 2 CUDA devices; found %d\n", deviceCount);
    return EXIT_FAILURE;
  }

  int nranks = argc > 1 ? static_cast<int>(parseSize(argv[1])) : deviceCount;
  size_t recvcount = argc > 2 ? parseSize(argv[2]) : (1ull << 20);
  int iters = argc > 3 ? static_cast<int>(parseSize(argv[3])) : 20;
  if (nranks < 2 || nranks > deviceCount) {
    std::fprintf(stderr, "Usage: %s [nranks<=%d] [recvcount] [iters]\n", argv[0], deviceCount);
    return EXIT_FAILURE;
  }
  if (iters <= 0 || recvcount == 0) {
    std::fprintf(stderr, "recvcount and iters must be positive\n");
    return EXIT_FAILURE;
  }

  setenv("NCCL_ALGO", "Ring", 1);
  setenv("NCCL_PROTO", "Simple", 1);

  std::vector<int> devices(nranks);
  std::iota(devices.begin(), devices.end(), 0);
  std::vector<ncclComm_t> comms(nranks);
  std::vector<RankBuffers> bufs(nranks);
  size_t sendcount = recvcount * static_cast<size_t>(nranks);

  NCCLCHECK(ncclCommInitAll(comms.data(), nranks, devices.data()));

  for (int r = 0; r < nranks; ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    CUDACHECK(cudaStreamCreate(&bufs[r].stream));
    CUDACHECK(cudaMalloc(&bufs[r].sendBf16, sendcount * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&bufs[r].sendF32, sendcount * sizeof(float)));
    CUDACHECK(cudaMalloc(&bufs[r].recvGold, recvcount * sizeof(float)));
    CUDACHECK(cudaMalloc(&bufs[r].recvMixed, recvcount * sizeof(float)));
    CUDACHECK(cudaMalloc(&bufs[r].recvBf16, recvcount * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&bufs[r].recvBf16F32, recvcount * sizeof(float)));

    int threads = 256;
    int blocksSend = static_cast<int>((sendcount + threads - 1) / threads);
    fillBf16Kernel<<<blocksSend, threads, 0, bufs[r].stream>>>(bufs[r].sendBf16, sendcount, r);
    bf16ToFloatKernel<<<blocksSend, threads, 0, bufs[r].stream>>>(bufs[r].sendBf16, bufs[r].sendF32, sendcount);
    CUDACHECK(cudaGetLastError());
  }
  syncAll(devices, bufs);

  runCollective(Path::GoldF32, devices, comms, bufs, recvcount);
  runCollective(Path::Mixed, devices, comms, bufs, recvcount);
  runCollective(Path::NativeBf16, devices, comms, bufs, recvcount);
  syncAll(devices, bufs);

  for (int r = 0; r < nranks; ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    int threads = 256;
    int blocksRecv = static_cast<int>((recvcount + threads - 1) / threads);
    bf16ToFloatKernel<<<blocksRecv, threads, 0, bufs[r].stream>>>(bufs[r].recvBf16, bufs[r].recvBf16F32, recvcount);
    CUDACHECK(cudaGetLastError());
  }
  syncAll(devices, bufs);

  ErrorStats mixedErr = compareToGold(devices, bufs, recvcount, true);
  ErrorStats bf16Err = compareToGold(devices, bufs, recvcount, false);
  if (std::getenv("MIXED_RS_DUMP") != nullptr) dumpSmall(devices, bufs, recvcount);

  double goldMs = timeCollective(Path::GoldF32, devices, comms, bufs, recvcount, iters);
  double mixedMs = timeCollective(Path::Mixed, devices, comms, bufs, recvcount, iters);
  double bf16Ms = timeCollective(Path::NativeBf16, devices, comms, bufs, recvcount, iters);

  double shardMiB = recvcount * sizeof(float) / (1024.0 * 1024.0);
  double fullBf16MiB = sendcount * sizeof(__nv_bfloat16) / (1024.0 * 1024.0);
  double fullF32MiB = sendcount * sizeof(float) / (1024.0 * 1024.0);
  std::printf("nranks=%d recvcount=%zu shard_fp32=%.3f MiB full_bf16_send=%.3f MiB full_fp32_gold=%.3f MiB\n",
              nranks, recvcount, shardMiB, fullBf16MiB, fullF32MiB);
  std::printf("mixed_vs_gold: max_abs=%.9g max_rel=%.9g at rank=%d index=%zu\n", mixedErr.maxAbs, mixedErr.maxRel,
              mixedErr.rank, mixedErr.index);
  std::printf("bf16_vs_gold:  max_abs=%.9g max_rel=%.9g at rank=%d index=%zu\n", bf16Err.maxAbs, bf16Err.maxRel,
              bf16Err.rank, bf16Err.index);
  std::printf("avg_ms fp32_gold_ring_simple=%.3f mixed_ring_simple=%.3f native_bf16_ring_simple=%.3f\n", goldMs,
              mixedMs, bf16Ms);

  for (int r = 0; r < nranks; ++r) {
    CUDACHECK(cudaSetDevice(devices[r]));
    cudaFree(bufs[r].sendBf16);
    cudaFree(bufs[r].sendF32);
    cudaFree(bufs[r].recvGold);
    cudaFree(bufs[r].recvMixed);
    cudaFree(bufs[r].recvBf16);
    cudaFree(bufs[r].recvBf16F32);
    cudaStreamDestroy(bufs[r].stream);
    ncclCommDestroy(comms[r]);
  }

  return EXIT_SUCCESS;
}
