# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= ../build
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCC_GENCODE ?= -gencode=arch=compute_100,code=sm_100

TARGET ?= ../build/bin/mixed_rs_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): mixed_rs.cu
	@mkdir -p $(dir $@)
	$(NVCC) -std=c++17 -O2 $(NVCC_GENCODE) -I$(NCCL_HOME)/include $< \
	  -L$(NCCL_HOME)/lib -lnccl -lcudart \
	  -Xlinker --disable-new-dtags -Xlinker -rpath -Xlinker $(abspath $(NCCL_HOME))/lib -o $@

clean:
	rm -f $(TARGET)
