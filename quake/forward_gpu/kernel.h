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


__global__  void kernelStiffnessInitLookup(int32_t nharbored,
				int32_t    myLinearElementsCount,
				int32_t*   myLinearElementsMapperDevice,
				elem_t*    elemTableDevice,
				rev_entry_t* reverseLookupDevice);

__global__  void kernelStiffnessCalcLocal(int32_t    myLinearElementsCount,
				int32_t*   myLinearElementsMapperDevice,
				elem_t*    elemTableDevice,
				e_t*       eTableDevice,
				fvector_t* tm1Device,
				fvector_t* localForceDevice);

__global__  void kernelStiffnessAddLocal(int32_t nharbored,
					 rev_entry_t* reverseLookupDevice,
					 fvector_t* localForceDevice,
					 fvector_t* forceDevice);

__host__ __device__ int vector_is_zero( const fvector_t* v );
__host__ __device__ void aTransposeU( fvector_t* un, double* atu );
__host__ __device__ void firstVector( const double* atu, double* finalVector, double a, double c, double b );
__host__ __device__ void au( fvector_t* resVec, const double* u );
__host__ __device__ void reformF( const double* u, double* newU );
__host__ __device__ void reformU( const double* u, double* newU );

#endif /* KERNEL_H_ */
