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

#include <cuda_runtime.h>


int32_t gpu_get_blocksize(gpu_spec_t *gpuSpecs, char* kernel);

__global__  void kernelInitReverseLookup(gpu_data_t* gpuDataDevice);

__global__  void kernelInitLinearLookup(gpu_data_t* gpuDataDevice,
					int32_t     myLinearElementsCount,
					int32_t*    myLinearElementsMapperDevice,
					rev_entry_t* reverseLookupLinearDevice);

__global__  void kernelStiffnessCalcLocal(gpu_data_t* gpuDataDevice,
					  int32_t     myLinearElementsCount,
					  int32_t*    myLinearElementsMapperDevice);

__global__  void kernelAddLocalForces(gpu_data_t* gpuDataDevice);

__global__  void kernelDampingCalcConv(gpu_data_t* gpuDataDevice,
				       double rmax);

__global__  void kernelDampingCalcLocal(gpu_data_t* gpuDataDevice,
					double rmax);


__host__ __device__ int vector_is_zero( const fvector_t* v );
__host__ __device__ void aTransposeU( fvector_t* un, double* atu );
__host__ __device__ void firstVector( const double* atu, double* finalVector, double a, double c, double b );
__host__ __device__ void au( fvector_t* resVec, const double* u );
__host__ __device__ void reformF( const double* u, double* newU );
__host__ __device__ void reformU( const double* u, double* newU );
__host__ __device__ void firstVector_kappa( const double* atu, double* finalVector, double kappa);
__host__ __device__ void firstVector_mu( const double* atu, double* finalVector, double b);

#endif /* KERNEL_H_ */
