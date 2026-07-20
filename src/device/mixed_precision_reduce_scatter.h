/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_MIXED_PRECISION_REDUCE_SCATTER_H_
#define NCCL_DEVICE_MIXED_PRECISION_REDUCE_SCATTER_H_

#include "device.h"
#include "collectives.h"
#include "primitives.h"

#if CUDART_VERSION >= 11000

namespace {

template <int Recv, int Unroll>
__device__ __forceinline__ void mixedReduceCopy(int tid, int nworkers, void* src0, void* src1, void* dst, int nelem) {
  // Use the fp32 wire/output width as the unit of work. Four bf16 values occupy
  // eight input bytes and expand to one 16-byte fp32 pack. Keeping Unroll fp32
  // packs live matches the native reduceCopy pipeline without doubling its
  // accumulator register footprint.
  constexpr int EltPerPack = 4;
  constexpr int EltPerHunk = Unroll * WARP_SIZE * EltPerPack;
  __nv_bfloat16 const* local = static_cast<__nv_bfloat16 const*>(src0);
  float const* peer = static_cast<float const*>(src1);
  float* out = static_cast<float*>(dst);

  uintptr_t localAddr = cvta_to_global(local);
  uintptr_t peerAddr = Recv ? cvta_to_global(peer) : 0;
  uintptr_t outAddr = cvta_to_global(out);
  bool aligned = (localAddr & 0x7) == 0 && ((peerAddr | outAddr) & 0xf) == 0;
  int packedElem = aligned ? nelem / EltPerPack * EltPerPack : 0;
  int fullHunkElem = packedElem / EltPerHunk * EltPerHunk;
  int nwarps = nworkers / WARP_SIZE;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  // Assign one contiguous hunk to each warp and keep multiple independent
  // loads in flight per thread, as native NCCL reduceCopyPacks does.
  for (int hunk = warp * EltPerHunk; hunk < fullHunkElem; hunk += nwarps * EltPerHunk) {
    BytePack<16> acc[Unroll];

    NVCC_PRAGMA_UNROLL(Unroll)
    for (int u = 0; u < Unroll; u++) {
      int i = hunk + (u * WARP_SIZE + lane) * EltPerPack;
      BytePack<8> localPack = ld_volatile_global<8>(localAddr + i * sizeof(__nv_bfloat16));
      NVCC_PRAGMA_UNROLL(EltPerPack)
      for (int j = 0; j < EltPerPack; j++) acc[u].u32[j] = uint32_t(localPack.u16[j]) << 16;
    }

    if (Recv) {
      NVCC_PRAGMA_UNROLL(Unroll)
      for (int u = 0; u < Unroll; u++) {
        int i = hunk + (u * WARP_SIZE + lane) * EltPerPack;
        BytePack<16> peerPack = ld_volatile_global<16>(peerAddr + i * sizeof(float));
        NVCC_PRAGMA_UNROLL(EltPerPack)
        for (int j = 0; j < EltPerPack; j++) {
          float sum = __uint_as_float(acc[u].u32[j]) + __uint_as_float(peerPack.u32[j]);
          acc[u].u32[j] = __float_as_uint(sum);
        }
      }
    }

    NVCC_PRAGMA_UNROLL(Unroll)
    for (int u = 0; u < Unroll; u++) {
      int i = hunk + (u * WARP_SIZE + lane) * EltPerPack;
      st_global<16>(outAddr + i * sizeof(float), acc[u]);
    }
  }

  // Handle complete packs left after the last full warp hunk.
  for (int i = fullHunkElem + tid * EltPerPack; i < packedElem; i += nworkers * EltPerPack) {
    BytePack<8> localPack = ld_volatile_global<8>(localAddr + i * sizeof(__nv_bfloat16));
    BytePack<16> acc;
    NVCC_PRAGMA_UNROLL(EltPerPack)
    for (int j = 0; j < EltPerPack; j++) acc.u32[j] = uint32_t(localPack.u16[j]) << 16;
    if (Recv) {
      BytePack<16> peerPack = ld_volatile_global<16>(peerAddr + i * sizeof(float));
      NVCC_PRAGMA_UNROLL(EltPerPack)
      for (int j = 0; j < EltPerPack; j++) {
        float sum = __uint_as_float(acc.u32[j]) + __uint_as_float(peerPack.u32[j]);
        acc.u32[j] = __float_as_uint(sum);
      }
    }
    st_global<16>(outAddr + i * sizeof(float), acc);
  }

  for (int i = packedElem + tid; i < nelem; i += nworkers) {
    float acc = __bfloat162float(local[i]);
    if (Recv) acc += ((volatile float const*)peer)[i];
    out[i] = acc;
  }
}

template <typename RedOp>
class MixedPrecisionReduceScatterPrims {
  static constexpr int SlicePerChunk = REDUCESCATTER_CHUNKSTEPS / REDUCESCATTER_SLICESTEPS;
  static constexpr int StepPerSlice = REDUCESCATTER_SLICESTEPS;

  static constexpr int RoleWaitRecv = 0x01;
  static constexpr int RoleWaitSend = 0x02;
  static constexpr int RolePostRecv = 0x04;
  static constexpr int RolePostSend = 0x08;
  static constexpr int Aborted = 0x10;

 public:
  __device__ MixedPrecisionReduceScatterPrims(int tid, int nthreads, void const* inputBuf, void* outputBuf,
                                              ncclRing* ring)
      : tid(tid),
        nthreads(nthreads),
        nworkers(nthreads - (nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0)),
        input(static_cast<__nv_bfloat16 const*>(inputBuf)),
        output(static_cast<float*>(outputBuf)),
        ringRanks(ring->userRanks),
        stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / int(sizeof(float))) {
    flags = 0;
    conn = nullptr;
    connFifo = nullptr;
    connEltsFifo = nullptr;
    connStepPtr = nullptr;
    connStepCache = 0;
    connStepSize = 0;
    step = 0;

    if (tid == 0) {
      flags |= RoleWaitRecv;
    } else if (tid == 1) {
      flags |= RoleWaitSend;
    } else if (tid == nthreads - 2) {
      flags |= RolePostRecv;
    } else if (tid == nthreads - 1) {
      flags |= RolePostSend;
    }

    loadConn(ring->prev, /*isRecv=*/true);
    loadConn(ring->next, /*isRecv=*/false);
  }

  __device__ ~MixedPrecisionReduceScatterPrims() {
    barrier();
    if (flags & (RolePostRecv | RolePostSend)) conn->step = step;
    barrier();
  }

  template <int Send, int Recv>
  __device__ __forceinline__ void doRound(size_t inputOffset, size_t outputOffset, int nelem) {
    int sliceSize = stepSize * StepPerSlice;
    sliceSize = max(divUp(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem) {
      NVCC_PRAGMA_UNROLL_DISABLED
      do {
        sliceSize = min(sliceSize, nelem - offset);
        if (tid == 0) {
          ncclShmem.groups[0].srcs[0] = (void*)(input + inputOffset + offset);
          if (!Send) ncclShmem.groups[0].dsts[0] = output + outputOffset + offset;
        }
        waitPeer<Send, Recv>(sliceSize);
        subBarrier();

        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (workSize > 0) {
          mixedReduceCopy<Recv, ncclCollUnroll()>(tid, nworkers, ncclShmem.groups[0].srcs[0],
                                                 ncclShmem.groups[0].srcs[1], ncclShmem.groups[0].dsts[0], workSize);
        }

        barrier();
        postPeer<Send, Recv>(workSize > 0);
        offset += sliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    NVCC_PRAGMA_UNROLL_DISABLED
    while (slice < SlicePerChunk) {
      sliceSize = min(sliceSize, nelem - offset);
      waitPeer<Send, Recv>(0);
      barrier();
      postPeer<Send, Recv>(sliceSize > 0 && !ncclShmem.aborted);
      offset += sliceSize;
      slice += 1;
    }
  }

 private:
  __device__ __forceinline__ void barrier() {
    barrier_sync(15, nthreads);
  }

  __device__ __forceinline__ void subBarrier() {
    barrier_sync(14, nworkers);
  }

  template <int Send, int Recv>
  __device__ __forceinline__ void waitPeer(int nelts) {
    bool isSendNotRecv = (Send && Recv) ? ((flags & RoleWaitSend) != 0) : (Send != 0);
    if (flags & (Recv * RoleWaitRecv | Send * RoleWaitSend)) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = ld_volatile_global(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }

      if (isSendNotRecv && connFifo != nullptr) connFifo[step % NCCL_STEPS].size = nelts * int(sizeof(float));

      int curStep = step % NCCL_STEPS;
      int delta = (connFifo != nullptr && connFifo[curStep].mode == NCCL_MODE_OFFSET) ?
                    loadInt(&connFifo[curStep].offset) / int(sizeof(float)) :
                    curStep * connStepSize;
      float* buff = connEltsFifo + delta;
      if (isSendNotRecv) {
        ncclShmem.groups[0].dsts[0] = buff;
      } else {
        ncclShmem.groups[0].srcs[1] = buff;
      }
      step += StepPerSlice;
    }
  }

  template <int Send, int Recv>
  __device__ __forceinline__ void postPeer(bool dataStored) {
    if (flags & (Recv * RolePostRecv | Send * RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored || connFifo != nullptr)) fence_acq_rel_sys();
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  __device__ __forceinline__ void loadConn(int peer, bool isRecv) {
    bool postRole = isRecv ? (flags & RolePostRecv) : (flags & RolePostSend);
    bool waitRole = isRecv ? (flags & RoleWaitRecv) : (flags & RoleWaitSend);
    if (!postRole && !waitRole) return;

    conn = isRecv ? &ncclShmem.channel.peers[peer]->recv[0] : &ncclShmem.channel.peers[peer]->send[0];
    step = roundUp(conn->step, SlicePerChunk * StepPerSlice);
    if (postRole) {
      connStepPtr = isRecv ? conn->head : conn->tail;
      if (isRecv) *connStepPtr = step;
    }
    if (waitRole) {
      connStepPtr = isRecv ? conn->tail : conn->head;
      connStepCache = ld_volatile_global(connStepPtr);
      connStepSize = conn->stepSize / int(sizeof(float));
      connEltsFifo = (float*)conn->buffs[NCCL_PROTO_SIMPLE];
      connFifo = conn->connFifo;
      if (isRecv) {
        ncclShmem.groups[0].recvConns[0] = conn;
      } else {
        ncclShmem.groups[0].sendConns[0] = conn;
      }
    }
  }

  int tid;
  int nthreads;
  int nworkers;
  int flags;
  __nv_bfloat16 const* input;
  float* output;
  int const* ringRanks;
  int stepSize;
  uint64_t step;
  ncclConnInfo* conn;
  ncclConnFifo* connFifo;
  float* connEltsFifo;
  uint64_t* connStepPtr;
  uint64_t connStepCache;
  int connStepSize;
};

} // namespace

template <typename RedOp>
struct RunWorkColl<ncclFuncMixedPrecisionReduceScatter, float, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing* ring = &ncclShmem.channel.ring;
    int const* ringRanks = ring->userRanks;
    int nranks = ncclShmem.comm.nRanks;

    size_t count;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(float), &count, &gridOffset, &channelCount,
                    &chunkCount);

    MixedPrecisionReduceScatterPrims<RedOp> prims(tid, nthreads, work->sendbuff, work->recvbuff, ring);
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      int nelem = min(chunkCount, channelCount - elemOffset);
      size_t dataOffset = gridOffset + elemOffset;

      int rankDest = ringRanks[nranks - 1];
      size_t inputOffset = dataOffset + rankDest * count;
      prims.template doRound<1, 0>(inputOffset, dataOffset, nelem);

      for (int j = 2; j < nranks; ++j) {
        rankDest = ringRanks[nranks - j];
        inputOffset = dataOffset + rankDest * count;
        prims.template doRound<1, 1>(inputOffset, dataOffset, nelem);
      }

      rankDest = ringRanks[0];
      inputOffset = dataOffset + rankDest * count;
      prims.template doRound<0, 1>(inputOffset, dataOffset, nelem);
    }
  }
};

#endif // CUDART_VERSION >= 11000

#endif
