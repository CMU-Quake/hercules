/* -*- C -*- */

/* @copyright_notice_start
 *
 * This file is part of the CMU Hercules ground motion simulator developed
 * by the CMU Quake project.
 *
 * Copyright (C) Carnegie Mellon University. All rights reserved.
 *
 * This program is covered by the terms described in the 'LICENSE.txt' file
 * included with this software package.
 *
 * This program comes WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * 'LICENSE.txt' file for more details.
 *
 *  @copyright_notice_end
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include "psolve.h"
#include "quake_util.h"

#include <cuda_runtime.h>


/* Kernel identifiers for FLOP counts */
#define FLOP_MAX_KERNEL        3
#define FLOP_STIFFNESS_KERNEL  0
#define FLOP_CALCCONV_KERNEL   1
#define FLOP_DAMPING_KERNEL    2


int64_t kernel_flops_per_thread(int kid);

int dumpRegisterCounts();

int32_t gpu_get_blocksize(gpu_spec_t *gpuSpecs, char* kernel, int32_t memPerThread);

__global__  void kernelStiffnessCalcLocal(gpu_data_t* gpuDataDevice,
					  int32_t     myLinearElementsCount,
					  int32_t*    myLinearElementsMapperDevice);

__global__  void kernelDampingCalcConv(gpu_data_t* gpuDataDevice,
				       double rmax);

__global__  void kernelDampingCalcLocal(gpu_data_t* gpuDataDevice,
					double rmax);

__global__  void kernelDispCalc(gpu_data_t* gpuDataDevice,
                                noyesflag_t printAccel);

__host__ __device__ int vector_is_zero( const fvector_t* v );
__host__ __device__ void aTransposeU( fvector_t* un, double* atu );
__host__ __device__ void firstVector( const double* atu, double* finalVector, double a, double c, double b );
__host__ __device__ void au( fvector_t* resVec, const double* u );
__host__ __device__ void reformF( const double* u, double* newU );
__host__ __device__ void reformU( const double* u, double* newU );
__host__ __device__ void firstVector_kappa( const double* atu, double* finalVector, double kappa);
__host__ __device__ void firstVector_mu( const double* atu, double* finalVector, double b);

__device__ double atomicAdd(double* address, double val);

#endif /* KERNEL_H_ */
