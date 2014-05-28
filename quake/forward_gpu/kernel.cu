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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "psolve.h"
#include "kernel.h"
#include "quake_util.h"
#include <cuda.h>
#include <cuda_runtime.h>


/**
 * For regular simulations use 1e-20.  For performance and scalability,
 * uncomment the immediate return in vector_is_zero() and vector_is_all_zero().
 * Alternatives with each platform underflow precision limit for 'double'
 * may also work, though this have not been thoroughly tested.
 *
 * Known underflow limits:
 * - NICS' Kraken: Limit is ~2.2e-308 --> use 1e-200
 *
 */
#define UNDERFLOW_CAP_STIFFNESS 1e-20


/* Constant memory symbols */
__constant__ fmatrix_t theK1Device[8][8];
__constant__ fmatrix_t theK2Device[8][8];


/* Flag denoting if kernel FLOP count cache initialized */
static int kernel_cached_flag = 0;

/* Operation type array indices */
#define FLOP_MULT_INDEX  0
#define FLOP_ADD_INDEX   1
#define FLOP_TRANS_INDEX 2
#define FLOP_MEM_INDEX   3

/* Operations counts for each kernel by thread (MULT/DIV, ADD/SUB, TRANS, MEM). 
   These values were manually tabulated by code review. */
static int64_t kernel_ops[3][4] = {52, 373, 0, 416,
				   63, 54, 4, 716,
				   85, 547, 0, 592};

/* Operation coefficents (MULT/DIV, ADD/SUB, TRANS, MEM). */
static int64_t kernel_coef[4] = {1, 1, 1, 1};

/* Cached FLOP/memory storage */
static int64_t kernel_ops_cached[3];
static int64_t kernel_mem_cached[3];


/* Initialize the kernel FLOP count cache */
int kernel_init_cache()
{
  int i, j;

  for (i = 0; i < FLOP_MAX_KERNEL; i++) {
    kernel_ops_cached[i] = 0;
    for (j = 0; j < 3; j++) {
      kernel_ops_cached[i] += kernel_coef[j] * kernel_ops[i][j];
    }
    kernel_mem_cached[i] = kernel_ops[i][3];
  }

  return (0);
}


/* Return FLOP count for the specified kernel */
int64_t kernel_flops_per_thread(int kid)
{
  if ((kid < 0) || (kid >= FLOP_MAX_KERNEL)) {
    return (0);
  }

  /* Initialize the kernel FLOP count cache */
  if (kernel_cached_flag == 0) {
    kernel_init_cache();
    kernel_cached_flag = 1;
  }

  return(kernel_ops_cached[kid]);
}


/* Return memory count for the specified kernel */
int64_t kernel_mem_per_thread(int kid)
{
  if ((kid < 0) || (kid >= FLOP_MAX_KERNEL)) {
    return (0);
  }

  /* Initialize the kernel FLOP count cache */
  if (kernel_cached_flag == 0) {
    kernel_init_cache();
    kernel_cached_flag = 1;
  }

  return(kernel_mem_cached[kid]);
}



/* Get the register count for the specified kernel function handle */
int32_t get_reg_count(char* kernel)
{
    /* Get kernel attributes */
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernel);

    return(attributes.numRegs);
}


/* Dump register counts for all kernels to stdout */
int dumpRegisterCounts()
{
    int count;

    printf("Kernel Register Counts:\n");
    count =  get_reg_count((char *)kernelStiffnessCalcLocal);
    printf("\tkernelStiffnessCalcLocal  : %d\n", count);

    count =  get_reg_count((char *)kernelDampingCalcConv);
    printf("\tkernelDampingCalcConv     : %d\n", count);

    count =  get_reg_count((char*)kernelDampingCalcLocal);
    printf("\tkernelDampingCalcLocal    : %d\n", count);

    return(0);
}


/* Get the recommended block size for a kernel given the GPU specifications
   and shared memory per thread */
int32_t gpu_get_blocksize(gpu_spec_t *gpuSpecs, 
			  char* kernel, 
			  int32_t memPerThread)
{
    int32_t regs_per_warp;
    int32_t max_threads;
    int32_t computed;

    /* Get kernel attributes */
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernel);

    /* Maximum theads possible based on register use */
    if (attributes.numRegs > 0) {
      //computed = gpuSpecs->regs_per_block / attributes.numRegs;
      //computed = 1 << (int)floor(log(computed)/log(2));
      regs_per_warp = (int)ceil((attributes.numRegs * gpuSpecs->warp_size) 
				/ (float)gpuSpecs->register_allocation_size) 
	* gpuSpecs->register_allocation_size;
      computed = (int)floor(gpuSpecs->regs_per_block 
			    / (float)(regs_per_warp * 
				      gpuSpecs->warp_allocation_size)) * 
	gpuSpecs->warp_allocation_size * gpuSpecs->warp_size;
      max_threads = imin(computed, gpuSpecs->max_threads);
    } else {
      max_threads = gpuSpecs->max_threads;
    }

    /* Maximum threads possible based on shared memory use */
    if (memPerThread > 0) {
      computed = gpuSpecs->shared_per_block / memPerThread;
      computed = 1 << (int)floor(log(computed)/log(2));
      max_threads = imin(computed, max_threads);
    }

    return(max_threads);
}


int gpu_copy_constant_symbols(int myID, fmatrix_t (*theK1)[8], 
			      fmatrix_t (*theK2)[8])
{
  cudaError_t cerror;
  
  cudaMemcpyToSymbol(theK1Device, &(theK1[0][0]), 
		     8 * 8 * sizeof(fmatrix_t));
  cudaMemcpyToSymbol(theK2Device, &(theK2[0][0]), 
		     8 * 8 * sizeof(fmatrix_t));

  cerror = cudaGetLastError();
  if (cerror != cudaSuccess) {
    fprintf(stderr, "Thread %d: Failed to copy symbols - %s\n", myID, 
	    cudaGetErrorString(cerror));
    return(1);
  }

  return(0);
}


/* Stiffness Calc-Force Kernel 
   FLOPs: 4M + 25A + ((147A) + (48M + 33A) + (168A)) = 52M + 373A
   MEM  : 4 + 4 + 8*(24 + 1) + 2*8 + 8 * (24) = 416
*/
__global__ 
//__launch_bounds__(768, 1)
void kernelStiffnessCalcLocal(int32_t lenum,
			      gpu_data_t *gpuData,
			      int32_t   myLinearElementsCount,
			      int32_t*  myLinearElementsMapperDevice)
{
    int       i;
    int32_t   eindex;
    int32_t   lin_eindex = (blockIdx.x * blockDim.x) + threadIdx.x; 
    fvector_t localForce[8];

    /* Extra threads and threads for non-linear elements exit here */
    if (lin_eindex >= myLinearElementsCount) {
      return;
    }

    eindex = myLinearElementsMapperDevice[lin_eindex];
    int32_t *lnid = &(gpuData->lnidArrayDevice[eindex]);

    for (i = 0; i < 8; i++) {
      const __restrict__ fvector_t *tm1Disp = gpuData->tm1Device + *lnid;

      localForce[i].f[0] = tm1Disp->f[0];
      localForce[i].f[1] = tm1Disp->f[1];
      localForce[i].f[2] = tm1Disp->f[2];

      lnid += lenum;
    }

    /* Coefficients for new stiffness matrix calculation */
    if (vector_is_zero( localForce ) != 0) {
      double first_coeff  = -0.5625 * (gpuData->c2ArrayDevice[eindex] + 
				       2 * gpuData->c1ArrayDevice[eindex]);
      double second_coeff = -0.5625 * (gpuData->c2ArrayDevice[eindex]);
      double third_coeff  = -0.5625 * (gpuData->c1ArrayDevice[eindex]);

      double atu[24];
      double firstVec[24];
      
      aTransposeU( localForce, atu );
      firstVector( atu, firstVec, first_coeff, second_coeff, third_coeff );
      au( localForce, firstVec );
    } else {
      /* Returning saves us from memset of localForce buffer */
      return;
    }

    /* Sum the nodal forces */
    lnid = &(gpuData->lnidArrayDevice[eindex]);
    for (i = 0; i < 8; i++) {
      fvector_t *nodalForce = gpuData->forceDevice + *lnid;

#ifdef  SINGLE_PRECISION_SOLVER  
      atomicAdd((float *)&(nodalForce->f[0]), (float)localForce[i].f[0]);
      atomicAdd((float *)&(nodalForce->f[1]), (float)localForce[i].f[1]);
      atomicAdd((float *)&(nodalForce->f[2]), (float)localForce[i].f[2]);
#else
      atomicAdd((double *)&(nodalForce->f[0]), (double)localForce[i].f[0]);
      atomicAdd((double *)&(nodalForce->f[1]), (double)localForce[i].f[1]);
      atomicAdd((double *)&(nodalForce->f[2]), (double)localForce[i].f[2]);
#endif

      lnid += lenum;
    }

    return;
}


/* Damping Calc-Conv Kernel 
   FLOPs: 63M + 54A + 4T 
   MEM  : 4 + 8*(48 + 1) + 2*(8*5 + 8*(6*2 + 3)) = 716
*/
__global__  void kernelDampingCalcConv(int32_t lenum,
				       gpu_data_t *gpuData,
				       double rmax) 
{
  int i;
  int32_t   index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t   eindex;
  int32_t   numcomp;

  /* Since number of elements may not be exactly divisible by block size,
     check that we are not off the end of the element array */
  if (index >= lenum) {
    return;
  }

  eindex = index; // Element number
  numcomp = lenum * 8;

  double g0, g1;
  double coef_1, coef_2, coef_3, coef_4;
  double exp_coef_0, exp_coef_1;
#ifdef  SINGLE_PRECISION_SOLVER  
  float   *f0_tm1[3], *f1_tm1[3], *convVec[3];
#else
  double   *f0_tm1[3], *f1_tm1[3], *convVec[3];
#endif
  fvector_t   tm1Disp[8], tm2Disp[8];

  double coef, a0, a1, b;
  int do_conv_update, do_damping_calc;

  int32_t *lnid = &gpuData->lnidArrayDevice[eindex];
  for (i = 0; i < 8; i++) {
    const __restrict__ fvector_t *disp = gpuData->tm1Device + *lnid;
    tm1Disp[i].f[0] = disp->f[0];
    tm1Disp[i].f[1] = disp->f[1];
    tm1Disp[i].f[2] = disp->f[2];

    disp = gpuData->tm2Device + *lnid;
    tm2Disp[i].f[0] = disp->f[0];
    tm2Disp[i].f[1] = disp->f[1];
    tm2Disp[i].f[2] = disp->f[2];

    lnid += lenum;
  }

  // SHEAR RELATED CONVOLUTION  
  g0 = gpuData->g0_shearArrayDevice[eindex];
  g1 = gpuData->g1_shearArrayDevice[eindex];
  coef = gpuData->b_shearArrayDevice[eindex];
  a0 = gpuData->a0_shearArrayDevice[eindex];
  a1 = gpuData->a1_shearArrayDevice[eindex];
  b = coef * rmax;

  coef_1 = g0 / 2.;
  coef_2 = coef_1 * ( 1. - g0 );
  
  coef_3 = g1 / 2.;
  coef_4 = coef_3 * ( 1. - g1 );
  
  exp_coef_0 = exp( -g0 );
  exp_coef_1 = exp( -g1 );
  
  do_conv_update = ((g0 != 0) && (g1 != 0));
  do_damping_calc = ((a0 + a1 + b) != 0);

#ifdef  SINGLE_PRECISION_SOLVER  
  f0_tm1[0] = (float *)gpuData->conv_shear_1Device + eindex;
  f1_tm1[0] = (float *)gpuData->conv_shear_2Device + eindex;
  convVec[0] = (float *)gpuData->shearVectorDevice + eindex;
#else
  f0_tm1[0] = (double *)gpuData->conv_shear_1Device + eindex;
  f1_tm1[0] = (double *)gpuData->conv_shear_2Device + eindex;
  convVec[0] = (double *)gpuData->shearVectorDevice + eindex;
#endif
  for (i = 0; i < 8; i++) {
  
    f0_tm1[1] = f0_tm1[0] + numcomp;
    f0_tm1[2] = f0_tm1[1] + numcomp;
    
    f1_tm1[1] = f1_tm1[0] + numcomp;
    f1_tm1[2] = f1_tm1[1] + numcomp;
    
    /* Update the convolution */
    if (do_conv_update) {
      
      *f0_tm1[0] = coef_2 * tm1Disp[i].f[0] + coef_1 * tm2Disp[i].f[0] + exp_coef_0 * (*f0_tm1[0]);
      *f1_tm1[0] = coef_4 * tm1Disp[i].f[0] + coef_3 * tm2Disp[i].f[0] + exp_coef_1 * (*f1_tm1[0]);
      
      *f0_tm1[1] = coef_2 * tm1Disp[i].f[1] + coef_1 * tm2Disp[i].f[1] + exp_coef_0 * (*f0_tm1[1]);
      *f1_tm1[1] = coef_4 * tm1Disp[i].f[1] + coef_3 * tm2Disp[i].f[1] + exp_coef_1 * (*f1_tm1[1]);
      
      *f0_tm1[2] = coef_2 * tm1Disp[i].f[2] + coef_1 * tm2Disp[i].f[2] + exp_coef_0 * (*f0_tm1[2]);
      *f1_tm1[2] = coef_4 * tm1Disp[i].f[2] + coef_3 * tm2Disp[i].f[2] + exp_coef_1 * (*f1_tm1[2]);
      
    }

    convVec[1] = convVec[0] + numcomp;
    convVec[2] = convVec[1] + numcomp;
    
    /* Construct the damping vector */
    if (do_damping_calc ) {
      
      *convVec[0] = coef * (tm1Disp[i].f[0] - tm2Disp[i].f[0])
	- (a0 * (*f0_tm1[0]) + a1 * (*f1_tm1[0])) + tm1Disp[i].f[0];
      
      *convVec[1] = coef * (tm1Disp[i].f[1] - tm2Disp[i].f[1])
	- (a0 * (*f0_tm1[1]) + a1 * (*f1_tm1[1])) + tm1Disp[i].f[1];
      
      *convVec[2] = coef * (tm1Disp[i].f[2] - tm2Disp[i].f[2])
	- (a0 * (*f0_tm1[2]) + a1 * (*f1_tm1[2])) + tm1Disp[i].f[2];
      
    } else {
      
      *convVec[0] = tm1Disp[i].f[0];
      *convVec[1] = tm1Disp[i].f[1];
      *convVec[2] = tm1Disp[i].f[2];
      
    }

    f0_tm1[0] += lenum;
    f1_tm1[0] += lenum;
    convVec[0] += lenum;
  }

  // DILATION RELATED CONVOLUTION
  g0 = gpuData->g0_kappaArrayDevice[eindex];
  g1 = gpuData->g1_kappaArrayDevice[eindex];
  coef = gpuData->b_kappaArrayDevice[eindex];
  a0 = gpuData->a0_kappaArrayDevice[eindex];
  a1 = gpuData->a1_kappaArrayDevice[eindex];
  b = coef * rmax;

  coef_1 = g0 / 2.;
  coef_2 = coef_1 * ( 1. - g0 );
  
  coef_3 = g1 / 2.;
  coef_4 = coef_3 * ( 1. - g1 );
  
  exp_coef_0 = exp( -g0 );
  exp_coef_1 = exp( -g1 );

  do_conv_update = ((g0 != 0) && (g1 != 0));
  do_damping_calc = ((a0 + a1 + b) != 0);

  f0_tm1[0] = (double *)gpuData->conv_kappa_1Device + eindex;
  f1_tm1[0] = (double *)gpuData->conv_kappa_2Device + eindex;
  convVec[0] = (double *)gpuData->kappaVectorDevice + eindex;
  for (i = 0; i < 8; i++) {
  
    f0_tm1[1] = f0_tm1[0] + numcomp;
    f0_tm1[2] = f0_tm1[1] + numcomp;
    
    f1_tm1[1] = f1_tm1[0] + numcomp;
    f1_tm1[2] = f1_tm1[1] + numcomp;
    
    /* Update the convolution */
    if (do_conv_update) {
      
      *f0_tm1[0] = coef_2 * tm1Disp[i].f[0] + coef_1 * tm2Disp[i].f[0] + exp_coef_0 * (*f0_tm1[0]);
      *f1_tm1[0] = coef_4 * tm1Disp[i].f[0] + coef_3 * tm2Disp[i].f[0] + exp_coef_1 * (*f1_tm1[0]);
      
      *f0_tm1[1] = coef_2 * tm1Disp[i].f[1] + coef_1 * tm2Disp[i].f[1] + exp_coef_0 * (*f0_tm1[1]);
      *f1_tm1[1] = coef_4 * tm1Disp[i].f[1] + coef_3 * tm2Disp[i].f[1] + exp_coef_1 * (*f1_tm1[1]);
      
      *f0_tm1[2] = coef_2 * tm1Disp[i].f[2] + coef_1 * tm2Disp[i].f[2] + exp_coef_0 * (*f0_tm1[2]);
      *f1_tm1[2] = coef_4 * tm1Disp[i].f[2] + coef_3 * tm2Disp[i].f[2] + exp_coef_1 * (*f1_tm1[2]);
      
    }
    
    convVec[1] = convVec[0] + numcomp;
    convVec[2] = convVec[1] + numcomp;
    
    /* Construct the damping vector */
    if (do_damping_calc) {
      
      *convVec[0] = coef * (tm1Disp[i].f[0] - tm2Disp[i].f[0])
	- (a0 * (*f0_tm1[0]) + a1 * (*f1_tm1[0])) + tm1Disp[i].f[0];
      
      *convVec[1] = coef * (tm1Disp[i].f[1] - tm2Disp[i].f[1])
	- (a0 * (*f0_tm1[1]) + a1 * (*f1_tm1[1])) + tm1Disp[i].f[1];
      
      *convVec[2] = coef * (tm1Disp[i].f[2] - tm2Disp[i].f[2])
	- (a0 * (*f0_tm1[2]) + a1 * (*f1_tm1[2])) + tm1Disp[i].f[2];
      
    } else {
      
      *convVec[0] = tm1Disp[i].f[0];
      *convVec[1] = tm1Disp[i].f[1];
      *convVec[2] = tm1Disp[i].f[2];
      
    }

    f0_tm1[0] += lenum;
    f1_tm1[0] += lenum;
    convVec[0] += lenum;
  }

  return;
}


/* Damping Calc-force Kerne
   FLOPs: (4M + 1A) + 2*(147A) + (60M + 48A) + (21M + 36A) + 168A = 85M + 547A
   MEM  : 2*(8 * 24 + 8) + 8(24 + 1) = 592
*/
__global__
//__launch_bounds__(512, 1)
void kernelDampingCalcLocal(int32_t lenum,
			    gpu_data_t *gpuData,
			    double rmax) 
{
  int       i, j;
  int32_t   eindex = (blockIdx.x * blockDim.x) + threadIdx.x;
  fvector_t damping_vector[8];

  /* Since number of elements may not be exactly divisible by block size,
     check that we are not off the end of the element array */
  if (eindex >= lenum) {
    return;
  }

  int         numcomp = lenum * 8;
  int32_t     *lnid;
  int         do_write = 0;
  double      atu[24], firstVec[24];
  double      mu, kappa;
#ifdef  SINGLE_PRECISION_SOLVER
  float *cVec = (float *)gpuData->shearVectorDevice + eindex;
#else  
  double *cVec = (double *)gpuData->shearVectorDevice + eindex;
#endif
  for (i = 0; i < 8; i++) {
    damping_vector[i].f[j] = *cVec;
    damping_vector[i].f[1] = *(cVec + numcomp);
    damping_vector[i].f[2] = *(cVec + 2*numcomp);
    firstVec[i] = 0.0;
    firstVec[i+8] = 0.0;
    firstVec[i+16] = 0.0;
    cVec += lenum;
  }

  /* SHEAR CONTRIBUTION */
  mu = -0.5625 * gpuData->c1ArrayDevice[eindex];  

  if(vector_is_zero( damping_vector ) != 0) {
    aTransposeU( damping_vector, atu );
    firstVector_mu( atu, firstVec, mu);
    do_write = 1;
  }

  /* DILATION CONTRIBUTION */
#ifdef  SINGLE_PRECISION_SOLVER
  cVec = (float *)gpuData->kappaVectorDevice + eindex;
#else
  cVec = (double *)gpuData->kappaVectorDevice + eindex;
#endif
  for (i = 0; i < 8; i++) {
    damping_vector[i].f[0] = *cVec;
    damping_vector[i].f[1] = *(cVec + numcomp);
    damping_vector[i].f[2] = *(cVec + 2*numcomp);
    cVec += lenum;
  }

  kappa = -0.5625 * (gpuData->c2ArrayDevice[eindex] + 
		     2. / 3. * gpuData->c1ArrayDevice[eindex]);

  if(vector_is_zero( damping_vector ) != 0) {
    aTransposeU( damping_vector, atu );
    firstVector_kappa( atu, firstVec, kappa);
    do_write = 1;
  }

  if (!do_write) {
    return;
  }

  /* Re-use the damping vector for the local force */
  au( damping_vector, firstVec );    
  
  /* Sum the nodal forces */
  lnid = &(gpuData->lnidArrayDevice[eindex]);
  for (i = 0; i < 8; i++) {
    fvector_t *nodalForce = gpuData->forceDevice + *lnid;

#ifdef  SINGLE_PRECISION_SOLVER  
    atomicAdd((float *)&(nodalForce->f[0]), (float)damping_vector[i].f[0]);
    atomicAdd((float *)&(nodalForce->f[1]), (float)damping_vector[i].f[1]);
    atomicAdd((float *)&(nodalForce->f[2]), (float)damping_vector[i].f[2]);
#else
    atomicAdd((double *)&(nodalForce->f[0]), (double)damping_vector[i].f[0]);
    atomicAdd((double *)&(nodalForce->f[1]), (double)damping_vector[i].f[1]);
    atomicAdd((double *)&(nodalForce->f[2]), (double)damping_vector[i].f[2]);
#endif
    
    lnid += lenum;
  }

  return;
}


__global__  void kernelDispCalc_node(gpu_data_t* gpuDataDevice,
                                noyesflag_t printAccel)
{
    lnid_t index = (blockIdx.x * blockDim.x) + threadIdx.x; 
    lnid_t nindex, i;

    /* Since number of nodes may not be exactly divisible by block size,
       check that we are not off the end of the node array */
    if (index >= gpuDataDevice->nharbored*3) {
      return;
    }

    nindex = index / 3;
    i = index % 3;

    n_t*             np         = &(gpuDataDevice->nTableDevice[nindex]);
    fvector_t*       nodalForce = &(gpuDataDevice->forceDevice[nindex]);
    fvector_t*       tm1Disp    = gpuDataDevice->tm1Device + nindex;
    fvector_t*       tm2Disp    = gpuDataDevice->tm2Device + nindex;

    /* total nodal forces */
    nodalForce->f[i] += np->mass2_minusaM[i] * tm1Disp->f[i]
      - np->mass_minusaM[i]  * tm2Disp->f[i];

    /* Save tm3 for accelerations */
    if ( printAccel == YES ) {
    
        fvector_t* tm3Disp = gpuDataDevice->tm3Device + nindex;
    
    	tm3Disp->f[i] = tm2Disp->f[i];

    }

    /* overwrite tm2 */
    tm2Disp->f[i] = nodalForce->f[i] / np->mass_simple;
   
    return;
}


__global__  void kernelDispCalc(gpu_data_t* gpuDataDevice,
                                noyesflag_t printAccel)
{
    lnid_t nindex = (blockIdx.x * blockDim.x) + threadIdx.x; 

    /* Since number of nodes may not be exactly divisible by block size,
       check that we are not off the end of the node array */
    if (nindex >= gpuDataDevice->nharbored) {
      return;
    }

    n_t*             np         = &(gpuDataDevice->nTableDevice[nindex]);
    fvector_t*       nodalForce = &(gpuDataDevice->forceDevice[nindex]);
    fvector_t*       tm1Disp    = gpuDataDevice->tm1Device + nindex;
    fvector_t*       tm2Disp    = gpuDataDevice->tm2Device + nindex;

    /* total nodal forces */
    nodalForce->f[0] += np->mass2_minusaM[0] * tm1Disp->f[0]
      - np->mass_minusaM[0]  * tm2Disp->f[0];
    nodalForce->f[1] += np->mass2_minusaM[1] * tm1Disp->f[1]
      - np->mass_minusaM[1]  * tm2Disp->f[1];
    nodalForce->f[2] += np->mass2_minusaM[2] * tm1Disp->f[2]
      - np->mass_minusaM[2]  * tm2Disp->f[2];

    /* Save tm3 for accelerations */
    if ( printAccel == YES ) {
    
        fvector_t* tm3Disp = gpuDataDevice->tm3Device + nindex;
    
    	tm3Disp->f[0] = tm2Disp->f[0];
    	tm3Disp->f[1] = tm2Disp->f[1];
    	tm3Disp->f[2] = tm2Disp->f[2];
    }

    /* overwrite tm2 */
    tm2Disp->f[0] = nodalForce->f[0] / np->mass_simple;
    tm2Disp->f[1] = nodalForce->f[1] / np->mass_simple;
    tm2Disp->f[2] = nodalForce->f[2] / np->mass_simple;
   
    return;
}


/* -------------------------------------------------------------------------- */
/*                         Efficient Method Utilities                         */
/* -------------------------------------------------------------------------- */


/**
 * For effective stiffness method:
 *
 * Check whether all components of a 3D vector are close to zero,
 * i.e., less than a small threshold.
 *
 * \return 1 when there is at least one "non-zero" component;
 *         0 when all the components are "zero".
 */
__host__ __device__ int vector_is_zero( const fvector_t* v )
{
    /*
     * For scalability studies, uncomment the immediate return.
     */

    /* return 1; */

    int i;
#ifdef  SINGLE_PRECISION_SOLVER
    const float *tmp = (float *)v;
#else
    const double *tmp = (double *)v;
#endif

    for (i = 0; i < 24; i++) {
      if (fabs( tmp[i] ) > UNDERFLOW_CAP_STIFFNESS) {
	return 1;
      }
    }

    return(0);

}


__host__ __device__ int vector_is_all_zero( const fvector_t* v )
{
    /*
     * For scalability studies, uncomment the immediate return.
     */

    /* return 1; */

  if ((fabs( v->f[0] ) > UNDERFLOW_CAP_STIFFNESS) ||
      (fabs( v->f[1] ) > UNDERFLOW_CAP_STIFFNESS) || 
      (fabs( v->f[2] ) > UNDERFLOW_CAP_STIFFNESS)) {
    return 1;
  }

  //    int i;
  //
  //  for (i = 0; i < 3; i++) {
  //    if (fabs( v->f[i] ) > UNDERFLOW_CAP_STIFFNESS) {
  //        return 1;
  //    }
  //  }

    return 0;
}


/* FLOPs: 147A */
__host__ __device__ void aTransposeU( fvector_t* un, double* atu )
{
#ifdef  SINGLE_PRECISION_SOLVER
    double temp[24];
    double u[24];
    int    i, j;

    /* arrange displacement values in an array */
    for (i=0; i<8; i++) {
        for(j=0; j<3; j++) {
            temp[i*3 + j] = un[i].f[j];     /* u1 u2 u3 .... v1 v2 v3 ... z1 z2 
z3 */
	}
    }

    reformU( temp, u );
#else
     double u[24];

     reformU( (double *)un, u );
#endif

    /* atu[0] = u[0] + u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7]; */
    atu[0]  = 0;
    atu[1]  = -u[0] - u[1] - u[2] - u[3] + u[4] + u[5] + u[6] + u[7];
    atu[2]  = -u[0] - u[1] + u[2] + u[3] - u[4] - u[5] + u[6] + u[7];
    atu[3]  = -u[0] + u[1] - u[2] + u[3] - u[4] + u[5] - u[6] + u[7];
    atu[4]  =  u[0] + u[1] - u[2] - u[3] - u[4] - u[5] + u[6] + u[7];
    atu[5]  =  u[0] - u[1] + u[2] - u[3] - u[4] + u[5] - u[6] + u[7];
    atu[6]  =  u[0] - u[1] - u[2] + u[3] + u[4] - u[5] - u[6] + u[7];
    atu[7]  = -u[0] + u[1] + u[2] - u[3] + u[4] - u[5] - u[6] + u[7];

    /* atu[8] = u[8] + u[9] + u[10] + u[11] + u[12] + u[13] + u[14] + u[15]; */
    atu[8]  = 0;
    atu[9]  = -u[8] - u[9] - u[10] - u[11] + u[12] + u[13] + u[14] + u[15];
    atu[10] = -u[8] - u[9] + u[10] + u[11] - u[12] - u[13] + u[14] + u[15];
    atu[11] = -u[8] + u[9] - u[10] + u[11] - u[12] + u[13] - u[14] + u[15];
    atu[12] =  u[8] + u[9] - u[10] - u[11] - u[12] - u[13] + u[14] + u[15];
    atu[13] =  u[8] - u[9] + u[10] - u[11] - u[12] + u[13] - u[14] + u[15];
    atu[14] =  u[8] - u[9] - u[10] + u[11] + u[12] - u[13] - u[14] + u[15];
    atu[15] = -u[8] + u[9] + u[10] - u[11] + u[12] - u[13] - u[14] + u[15];

    /* atu[16] = u[16] + u[17] + u[18] + u[19] + u[20] + u[21] + u[22] + u[23]; */
    atu[16] = 0;
    atu[17] = -u[16] - u[17] - u[18] - u[19] + u[20] + u[21] + u[22] + u[23];
    atu[18] = -u[16] - u[17] + u[18] + u[19] - u[20] - u[21] + u[22] + u[23];
    atu[19] = -u[16] + u[17] - u[18] + u[19] - u[20] + u[21] - u[22] + u[23];
    atu[20] =  u[16] + u[17] - u[18] - u[19] - u[20] - u[21] + u[22] + u[23];
    atu[21] =  u[16] - u[17] + u[18] - u[19] - u[20] + u[21] - u[22] + u[23];
    atu[22] =  u[16] - u[17] - u[18] + u[19] + u[20] - u[21] - u[22] + u[23];
    atu[23] = -u[16] + u[17] + u[18] - u[19] + u[20] - u[21] - u[22] + u[23];
}


/* FLOPs: 48M + 33A */
__host__ __device__ void firstVector( const double* atu, 
				      double* finalVector, 
				      double a, 
				      double c, 
				      double b )
{
    finalVector[0] = 0;
    finalVector[1] = b * (atu[19] + atu[1]);
    finalVector[2] = b * (atu[11] + atu[2]);
    finalVector[3] = a * atu[3] + c * (atu[10] + atu[17]);
    finalVector[4] = b * (atu[13] + atu[22] + 2. * atu[4]) / 3.;
    finalVector[5] = ( (a + b) * atu[5] + c * atu[12] ) /3.;
    finalVector[6] = ( (a + b) * atu[6] + c * atu[20] ) /3.;
    finalVector[7] = ( (a + 2.*b) * atu[7] ) / 9.;

    finalVector[8] = 0;
    finalVector[9] = b * (atu[18] + atu[9]);
    finalVector[10] = a * atu[10] + c * (atu[3] + atu[17]);
    finalVector[11] = b * (atu[11] + atu[2]);
    finalVector[12] = ( (a + b) * atu[12] + c * atu[5] ) / 3.;
    finalVector[13] = b * (atu[4] + atu[22] + 2. * atu[13]) / 3.;
    finalVector[14] = ( (a + b) * atu[14] + c * atu[21] ) /3.;
    finalVector[15] = (a + 2. * b) * atu[15] / 9.;

    finalVector[16] = 0;
    finalVector[17] = a * atu[17] + c * (atu[3] + atu[10]);
    finalVector[18] = b * (atu[18] + atu[9]);
    finalVector[19] = b * (atu[19] + atu[1]);
    finalVector[20] = ( (a + b) * atu[20] + c * atu[6] ) / 3.;
    finalVector[21] = ( (a + b) * atu[21] + c * atu[14] ) / 3.;
    finalVector[22] = b * ( atu[4] + atu[13] + 2. * atu[22]) / 3.;
    finalVector[23] = (a + 2. * b) * atu[23] / 9.;
}


/* FLOPs: 168A */
__host__ __device__ void au( fvector_t* resVec, const double* u )
{
#ifdef  SINGLE_PRECISION_SOLVER
    int    i, j;
    double finVec[24];
    double temp[24];
#else
    double finVec[24];
#endif

    finVec[0]  = u[0]  - u[1] - u[2] - u[3] + u[4] + u[5] + u[6] - u[7];
    finVec[1]  = u[0]  - u[1] - u[2] + u[3] + u[4] - u[5] - u[6] + u[7];
    finVec[2]  = u[0]  - u[1] + u[2] - u[3] - u[4] + u[5] - u[6] + u[7];
    finVec[3]  = u[0]  - u[1] + u[2] + u[3] - u[4] - u[5] + u[6] - u[7];
    finVec[4]  = u[0]  + u[1] - u[2] - u[3] - u[4] - u[5] + u[6] + u[7];
    finVec[5]  = u[0]  + u[1] - u[2] + u[3] - u[4] + u[5] - u[6] - u[7];
    finVec[6]  = u[0]  + u[1] + u[2] - u[3] + u[4] - u[5] - u[6] - u[7];
    finVec[7]  = u[0]  + u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7];

    finVec[8]  = u[8]  - u[9] - u[10] - u[11] + u[12] + u[13] + u[14] - u[15];
    finVec[9]  = u[8]  - u[9] - u[10] + u[11] + u[12] - u[13] - u[14] + u[15];
    finVec[10] = u[8]  - u[9] + u[10] - u[11] - u[12] + u[13] - u[14] + u[15];
    finVec[11] = u[8]  - u[9] + u[10] + u[11] - u[12] - u[13] + u[14] - u[15];
    finVec[12] = u[8]  + u[9] - u[10] - u[11] - u[12] - u[13] + u[14] + u[15];
    finVec[13] = u[8]  + u[9] - u[10] + u[11] - u[12] + u[13] - u[14] - u[15];
    finVec[14] = u[8]  + u[9] + u[10] - u[11] + u[12] - u[13] - u[14] - u[15];
    finVec[15] = u[8]  + u[9] + u[10] + u[11] + u[12] + u[13] + u[14] + u[15];

    finVec[16] = u[16] - u[17] - u[18] - u[19] + u[20] + u[21] + u[22] - u[23];
    finVec[17] = u[16] - u[17] - u[18] + u[19] + u[20] - u[21] - u[22] + u[23];
    finVec[18] = u[16] - u[17] + u[18] - u[19] - u[20] + u[21] - u[22] + u[23];
    finVec[19] = u[16] - u[17] + u[18] + u[19] - u[20] - u[21] + u[22] - u[23];
    finVec[20] = u[16] + u[17] - u[18] - u[19] - u[20] - u[21] + u[22] + u[23];
    finVec[21] = u[16] + u[17] - u[18] + u[19] - u[20] + u[21] - u[22] - u[23];
    finVec[22] = u[16] + u[17] + u[18] - u[19] + u[20] - u[21] - u[22] - u[23];
    finVec[23] = u[16] + u[17] + u[18] + u[19] + u[20] + u[21] + u[22] + u[23];

#ifdef  SINGLE_PRECISION_SOLVER
    reformF( finVec, temp );

    for (j = 0; j<8; j++)
    {
        for (i = 0; i<3; i++)
        {
            resVec[j].f[i] += temp[j*3 + i];
        }
    }
#else
    reformF( finVec, (double *)resVec );
#endif

}


__host__ __device__ void reformF( const double* u, double* newU )
{
    newU[0]  = u[0];
    newU[1]  = u[8];
    newU[2]  = u[16];
    newU[3]  = u[1];
    newU[4]  = u[9];
    newU[5]  = u[17];
    newU[6]  = u[2];
    newU[7]  = u[10];
    newU[8]  = u[18];
    newU[9]  = u[3];
    newU[10] = u[11];
    newU[11] = u[19];
    newU[12] = u[4];
    newU[13] = u[12];
    newU[14] = u[20];
    newU[15] = u[5];
    newU[16] = u[13];
    newU[17] = u[21];
    newU[18] = u[6];
    newU[19] = u[14];
    newU[20] = u[22];
    newU[21] = u[7];
    newU[22] = u[15];
    newU[23] = u[23];
}


__host__ __device__ void reformU( const double* u, double* newU )
{
    newU[0]  = u[0];
    newU[1]  = u[3];
    newU[2]  = u[6];
    newU[3]  = u[9];
    newU[4]  = u[12];
    newU[5]  = u[15];
    newU[6]  = u[18];
    newU[7]  = u[21];
    newU[8]  = u[1];
    newU[9]  = u[4];
    newU[10] = u[7];
    newU[11] = u[10];
    newU[12] = u[13];
    newU[13] = u[16];
    newU[14] = u[19];
    newU[15] = u[22];
    newU[16] = u[2];
    newU[17] = u[5];
    newU[18] = u[8];
    newU[19] = u[11];
    newU[20] = u[14];
    newU[21] = u[17];
    newU[22] = u[20];
    newU[23] = u[23];
}


/* FLOPs: 21M + 36A */
__host__ __device__ void firstVector_kappa( const double* atu, double* finalVector, double kappa)
{
    finalVector[0] += 0.;
    finalVector[1] += 0.;
    finalVector[2] += 0.;
    finalVector[3] += kappa * (atu[3] + atu[10] + atu[17]);
    finalVector[4] += 0.;
    finalVector[5] += kappa * ( atu[5] + atu[12] ) /3.;
    finalVector[6] += kappa * ( atu[6] + atu[20] ) /3.;
    finalVector[7] += kappa * atu[7] / 9.;

    finalVector[8] += 0.;
    finalVector[9] += 0.;
    finalVector[10] += kappa * ( atu[10] + atu[3] + atu[17]);
    finalVector[11] += 0.;
    finalVector[12] += kappa * ( atu[12] + atu[5] ) / 3.;
    finalVector[13] += 0.;
    finalVector[14] += kappa * ( atu[14] + atu[21] ) /3.;
    finalVector[15] += kappa * atu[15] / 9.;

    finalVector[16] += 0.;
    finalVector[17] += kappa * ( atu[17] + atu[3] + atu[10] );
    finalVector[18] += 0.;
    finalVector[19] += 0.;
    finalVector[20] += kappa * ( atu[20] + atu[6] ) / 3.;
    finalVector[21] += kappa * ( atu[21] + atu[14] ) / 3.;
    finalVector[22] += 0.;
    finalVector[23] += kappa * atu[23] / 9.;
}


/* FLOPs: 60M + 48A */
__host__ __device__ void firstVector_mu( const double* atu, double* finalVector, double b )
{
    finalVector[0] += 0;
    finalVector[1] += b * (atu[19] + atu[1]);
    finalVector[2] += b * (atu[11] + atu[2]);
    finalVector[3] += b * ( 4. * atu[3] - 2. * (atu[10] + atu[17]) ) / 3.;
    finalVector[4] += b * (atu[13] + atu[22] + 2. * atu[4]) / 3.;
    finalVector[5] += b * ( 7. * atu[5] - 2. * atu[12] ) / 9.;
    finalVector[6] += b * ( 7. * atu[6] - 2. * atu[20] ) / 9.;
    finalVector[7] += ( 10. * b * atu[7] ) / 27.;

    finalVector[8]  += 0;
    finalVector[9]  += b * (atu[18] + atu[9]);
    finalVector[10] += b * ( 4. * atu[10] - 2. * (atu[3] + atu[17]) ) / 3.;
    finalVector[11] += b * (atu[11] + atu[2]);
    finalVector[12] += b * ( 7. * atu[12] - 2. * atu[5] ) / 9.;
    finalVector[13] += b * (atu[4] + atu[22] + 2. * atu[13]) / 3.;
    finalVector[14] += b * ( 7. * atu[14] - 2. * atu[21] ) / 9.;
    finalVector[15] += (10. * b * atu[15] ) / 27.;

    finalVector[16] += 0;
    finalVector[17] += b * ( 4. * atu[17] - 2. * (atu[3] + atu[10]) ) / 3.;
    finalVector[18] += b * (atu[18] + atu[9]);
    finalVector[19] += b * (atu[19] + atu[1]);
    finalVector[20] += b * ( 7. * atu[20] - 2. * atu[6] ) / 9.;
    finalVector[21] += b * ( 7. * atu[21] - 2. * atu[14] ) / 9.;
    finalVector[22] += b * ( atu[4] + atu[13] + 2. * atu[22]) / 3.;
    finalVector[23] += (10. * b * atu[23] ) / 27.;
}


__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


/**
 * MultAddMatVec: Multiply a 3 x 3 Matrix (M) with a 3 x 1 vector (V1)
 *                and a constant (c). Then add the result to the same
 *                target vector (V2)
 *
 *  V2 = V2 + M * V1 * c
 *
 */
__host__ __device__ void MultAddMatVecGPU( fmatrix_t* M, fvector_t* V1, 
					   double c, fvector_t* V2 )
{
  if (1) {
  V2->f[0] += c*(M->f[0][0] * V1->f[0] + 
		 M->f[0][1] * V1->f[1] + M->f[0][2] * V1->f[2]);

  V2->f[1] += c*(M->f[1][0] * V1->f[0] + 
		 M->f[1][1] * V1->f[1] + M->f[1][2] * V1->f[2]);

  V2->f[2] += c*(M->f[2][0] * V1->f[0] + 
		 M->f[2][1] * V1->f[1] + M->f[2][2] * V1->f[2]);
  }

  if (0) {
  V2->f[0] += (M->f[0][0] * V1->f[0] + 
		 M->f[0][1] * V1->f[1] + M->f[0][2] * V1->f[2]);

  V2->f[1] += (M->f[1][0] * V1->f[0] + 
		 M->f[1][1] * V1->f[1] + M->f[1][2] * V1->f[2]);

  V2->f[2] += (M->f[2][0] * V1->f[0] + 
		 M->f[2][1] * V1->f[1] + M->f[2][2] * V1->f[2]);
  }

  return;
}
