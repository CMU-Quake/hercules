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
static int64_t kernel_ops[4][4] = {52, 373, 0, 416,
				   63,  54, 4, 716,
				   85, 547, 0, 592,
                                    6,   9, 0, 128};

/* Operation coefficents (MULT/DIV, ADD/SUB, TRANS, MEM). */
static int64_t kernel_coef[4] = {1, 1, 1, 1};

/* Cached FLOP/memory storage */
static int64_t kernel_ops_cached[4];
static int64_t kernel_mem_cached[4];


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
int32_t gpu_get_reg_count(char* kernel)
{
    /* Get kernel attributes */
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernel);

    return(attributes.numRegs);
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
    int       do_contrib = 0;

    /* Extra threads and threads for non-linear elements exit here */
    if (lin_eindex >= myLinearElementsCount) {
      return;
    }

    eindex = myLinearElementsMapperDevice[lin_eindex];
    int32_t *lnid = &(gpuData->lnidArrayDevice[eindex]);

    for (i = 0; i < 8; i++) {
      const __restrict__ fvector_t *tm1Disp = gpuData->tm1Device + *lnid;

      localForce[i].f[0] = tm1Disp->f[0];
      do_contrib = (fabs( localForce[i].f[0] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
      localForce[i].f[1] = tm1Disp->f[1];
      do_contrib = (fabs( localForce[i].f[1] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
      localForce[i].f[2] = tm1Disp->f[2];
      do_contrib = (fabs( localForce[i].f[2] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;

      lnid += lenum;
    }

    if (!do_contrib) {
      return;
    }

    /* Coefficients for new stiffness matrix calculation */
    //if (vector_is_zero( localForce ) != 0) {
    double first_coeff  = -0.5625 * (gpuData->c2ArrayDevice[eindex] + 
				     2 * gpuData->c1ArrayDevice[eindex]);
    double second_coeff = -0.5625 * (gpuData->c2ArrayDevice[eindex]);
    double third_coeff  = -0.5625 * (gpuData->c1ArrayDevice[eindex]);
    
    double atu[24];
    double firstVec[24];
    
    aTransposeU( localForce, atu );
    firstVector( atu, firstVec, first_coeff, second_coeff, third_coeff );
    au( localForce, firstVec );
    //} else {
    //  /* Returning saves us from memset of localForce buffer */
    //  return;
    //}

    /* Sum the nodal forces */
    lnid = &(gpuData->lnidArrayDevice[eindex]);
    for (i = 0; i < 8; i++) {
      fvector_t *nodalForce = gpuData->forceDevice + *lnid;

      atomicAdd((solver_float *)&(nodalForce->f[0]), 
		(solver_float)localForce[i].f[0]);
      atomicAdd((solver_float *)&(nodalForce->f[1]), 
		(solver_float)localForce[i].f[1]);
      atomicAdd((solver_float *)&(nodalForce->f[2]), 
		(solver_float)localForce[i].f[2]);

      lnid += lenum;
    }

    return;
}


/* Damping Calc-Conv Kernel 
   FLOPs: 63M + 54A + 4T 
   MEM  : 4 + 8*(48 + 1) + 2*(8*5 + 8*(6*2 + 3)) = 716
*/
__global__  void kernelDampingCalcConv(int32_t lenum,
				       int32_t numcomp,
				       gpu_data_t *gpuData,
				       double rmax) 
{
  int i;
  int32_t   eindex = (blockIdx.x * blockDim.x) + threadIdx.x; // Element number

  /* Since number of elements may not be exactly divisible by block size,
     check that we are not off the end of the element array */
  if (eindex >= lenum) {
    return;
  }

  double g0, g1;
  double coef_1_shear, coef_2_shear, coef_3_shear, coef_4_shear;
  double exp_coef_0_shear, exp_coef_1_shear;
  double coef_1_kappa, coef_2_kappa, coef_3_kappa, coef_4_kappa;
  double exp_coef_0_kappa, exp_coef_1_kappa;
  fvector_t     tm1Disp, tm2Disp;
  solver_float *f0_tm1[3], *f1_tm1[3], *convVec[3];

  double coef_shear, a0_shear, a1_shear;
  double coef_kappa, a0_kappa, a1_kappa;
  int do_conv_update_shear, do_damping_calc_shear;
  int do_conv_update_kappa, do_damping_calc_kappa;
  int32_t offset0;

  // SHEAR RELATED CONVOLUTION  
  g0 = gpuData->g0_shearArrayDevice[eindex];
  coef_1_shear = g0 / 2.;
  coef_2_shear = coef_1_shear * ( 1. - g0 );
  exp_coef_0_shear = exp( -g0 );

  g1 = gpuData->g1_shearArrayDevice[eindex];
  coef_3_shear = g1 / 2.;
  coef_4_shear = coef_3_shear * ( 1. - g1 );
  exp_coef_1_shear = exp( -g1 );

  a0_shear = gpuData->a0_shearArrayDevice[eindex];
  a1_shear = gpuData->a1_shearArrayDevice[eindex];
  coef_shear = gpuData->b_shearArrayDevice[eindex] / rmax;

  do_conv_update_shear = ((g0 != 0) && (g1 != 0));
  do_damping_calc_shear = ((a0_shear + a1_shear + gpuData->b_shearArrayDevice[eindex]) != 0);

  // DILATION RELATED CONVOLUTION
  g0 = gpuData->g0_kappaArrayDevice[eindex];
  coef_1_kappa = g0 / 2.;
  coef_2_kappa = coef_1_kappa * ( 1. - g0 );
  exp_coef_0_kappa = exp( -g0 );

  g1 = gpuData->g1_kappaArrayDevice[eindex];
  coef_3_kappa = g1 / 2.;
  coef_4_kappa = coef_3_kappa * ( 1. - g1 );
  exp_coef_1_kappa = exp( -g1 );

  a0_kappa = gpuData->a0_kappaArrayDevice[eindex];
  a1_kappa = gpuData->a1_kappaArrayDevice[eindex];
  coef_kappa = gpuData->b_kappaArrayDevice[eindex] / rmax;

  do_conv_update_kappa = ((g0 != 0) && (g1 != 0));
  do_damping_calc_kappa = ((a0_kappa + a1_kappa + gpuData->b_kappaArrayDevice[eindex]) != 0);

  for (i = 0, offset0 = eindex; i < 8; i++, offset0 += lenum) {
    int32_t offset1 = offset0 + numcomp;
    int32_t offset2 = offset1 + numcomp;
    int32_t lnid = *(gpuData->lnidArrayDevice + offset0);

    const __restrict__ fvector_t *disp = gpuData->tm1Device + lnid;
    tm1Disp.f[0] = disp->f[0];
    tm1Disp.f[1] = disp->f[1];
    tm1Disp.f[2] = disp->f[2];

    disp = gpuData->tm2Device + lnid;
    tm2Disp.f[0] = disp->f[0];
    tm2Disp.f[1] = disp->f[1];
    tm2Disp.f[2] = disp->f[2];

    /* Read aligned tm2disp from the shear vector storage */
    //tm2Disp.f[0] = *((solver_float *)gpuData->shearVectorDevice + offset0);
    //tm2Disp.f[1] = *((solver_float *)gpuData->shearVectorDevice + offset1);
    //tm2Disp.f[2] = *((solver_float *)gpuData->shearVectorDevice + offset2);

    f0_tm1[0] = (solver_float *)gpuData->conv_shear_1Device + offset0;
    f0_tm1[1] = (solver_float *)gpuData->conv_shear_1Device + offset1;
    f0_tm1[2] = (solver_float *)gpuData->conv_shear_1Device + offset2;

    f1_tm1[0] = (solver_float *)gpuData->conv_shear_2Device + offset0;
    f1_tm1[1] = (solver_float *)gpuData->conv_shear_2Device + offset1;
    f1_tm1[2] = (solver_float *)gpuData->conv_shear_2Device + offset2;
    
    /* Update the convolution */
    if (do_conv_update_shear) {

      *f0_tm1[0] = coef_2_shear * tm1Disp.f[0] + coef_1_shear * tm2Disp.f[0] + exp_coef_0_shear * (*f0_tm1[0]);
      *f1_tm1[0] = coef_4_shear * tm1Disp.f[0] + coef_3_shear * tm2Disp.f[0] + exp_coef_1_shear * (*f1_tm1[0]);
      
      *f0_tm1[1] = coef_2_shear * tm1Disp.f[1] + coef_1_shear * tm2Disp.f[1] + exp_coef_0_shear * (*f0_tm1[1]);
      *f1_tm1[1] = coef_4_shear * tm1Disp.f[1] + coef_3_shear * tm2Disp.f[1] + exp_coef_1_shear * (*f1_tm1[1]);
      
      *f0_tm1[2] = coef_2_shear * tm1Disp.f[2] + coef_1_shear * tm2Disp.f[2] + exp_coef_0_shear * (*f0_tm1[2]);
      *f1_tm1[2] = coef_4_shear * tm1Disp.f[2] + coef_3_shear * tm2Disp.f[2] + exp_coef_1_shear * (*f1_tm1[2]);
      
    }

    convVec[0] = (solver_float *)gpuData->shearVectorDevice + offset0;
    convVec[1] = (solver_float *)gpuData->shearVectorDevice + offset1;
    convVec[2] = (solver_float *)gpuData->shearVectorDevice + offset2;
    
    /* Construct the damping vector */
    if (do_damping_calc_shear ) {
      
      *convVec[0] = coef_shear * (tm1Disp.f[0] - tm2Disp.f[0])
      	- (a0_shear * (*f0_tm1[0]) + a1_shear * (*f1_tm1[0])) + tm1Disp.f[0];
      
      *convVec[1] = coef_shear * (tm1Disp.f[1] - tm2Disp.f[1])
      	- (a0_shear * (*f0_tm1[1]) + a1_shear * (*f1_tm1[1])) + tm1Disp.f[1];
      
      *convVec[2] = coef_shear * (tm1Disp.f[2] - tm2Disp.f[2])
      	- (a0_shear * (*f0_tm1[2]) + a1_shear * (*f1_tm1[2])) + tm1Disp.f[2];
      
    } else {
      
      *convVec[0] = tm1Disp.f[0];
      *convVec[1] = tm1Disp.f[1];
      *convVec[2] = tm1Disp.f[2];
      
    }

    f0_tm1[0] = (solver_float *)gpuData->conv_kappa_1Device + offset0;
    f0_tm1[1] = (solver_float *)gpuData->conv_kappa_1Device + offset1;
    f0_tm1[2] = (solver_float *)gpuData->conv_kappa_1Device + offset2;

    f1_tm1[0] = (solver_float *)gpuData->conv_kappa_2Device + offset0;
    f1_tm1[1] = (solver_float *)gpuData->conv_kappa_2Device + offset1;
    f1_tm1[2] = (solver_float *)gpuData->conv_kappa_2Device + offset2;

    /* Update the convolution */
    if (do_conv_update_kappa) {

      *f0_tm1[0] = coef_2_kappa * tm1Disp.f[0] + coef_1_kappa * tm2Disp.f[0] + exp_coef_0_kappa * (*f0_tm1[0]);
      *f1_tm1[0] = coef_4_kappa * tm1Disp.f[0] + coef_3_kappa * tm2Disp.f[0] + exp_coef_1_kappa * (*f1_tm1[0]);
      
      *f0_tm1[1] = coef_2_kappa * tm1Disp.f[1] + coef_1_kappa * tm2Disp.f[1] + exp_coef_0_kappa * (*f0_tm1[1]);
      *f1_tm1[1] = coef_4_kappa * tm1Disp.f[1] + coef_3_kappa * tm2Disp.f[1] + exp_coef_1_kappa * (*f1_tm1[1]);
      
      *f0_tm1[2] = coef_2_kappa * tm1Disp.f[2] + coef_1_kappa * tm2Disp.f[2] + exp_coef_0_kappa * (*f0_tm1[2]);
      *f1_tm1[2] = coef_4_kappa * tm1Disp.f[2] + coef_3_kappa * tm2Disp.f[2] + exp_coef_1_kappa * (*f1_tm1[2]);
      
    }

    convVec[0] = (solver_float *)gpuData->kappaVectorDevice + offset0;
    convVec[1] = (solver_float *)gpuData->kappaVectorDevice + offset1;
    convVec[2] = (solver_float *)gpuData->kappaVectorDevice + offset2;
    
    /* Construct the damping vector */
    if (do_damping_calc_kappa) {
      
      *convVec[0] = coef_kappa * (tm1Disp.f[0] - tm2Disp.f[0])
      	- (a0_kappa * (*f0_tm1[0]) + a1_kappa * (*f1_tm1[0])) + tm1Disp.f[0];
      
      *convVec[1] = coef_kappa * (tm1Disp.f[1] - tm2Disp.f[1])
      	- (a0_kappa * (*f0_tm1[1]) + a1_kappa * (*f1_tm1[1])) + tm1Disp.f[1];
      
      *convVec[2] = coef_kappa * (tm1Disp.f[2] - tm2Disp.f[2])
      	- (a0_kappa * (*f0_tm1[2]) + a1_kappa * (*f1_tm1[2])) + tm1Disp.f[2];
      
    } else {

      *convVec[0] = tm1Disp.f[0];
      *convVec[1] = tm1Disp.f[1];
      *convVec[2] = tm1Disp.f[2];
      
    }

  }

  return;
}



/* Damping Calc-force Kernel
   FLOPs: (4M + 1A) + 2*(147A) + (60M + 48A) + (21M + 36A) + 168A = 85M + 547A
   MEM  : 2*(8 * 24 + 8) + 8(24 + 1) = 592
*/
__global__
//__launch_bounds__(512, 1)
void kernelDampingCalcLocal(int32_t lenum,
			    gpu_data_t *gpuData) 
{
  int       i;
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
  solver_float *cVec;
  int         do_contrib;

  /* SHEAR CONTRIBUTION */
  cVec = (solver_float *)gpuData->shearVectorDevice + eindex;
  do_contrib = 0;
  for (i = 0; i < 8; i++) {
    damping_vector[i].f[0] = *cVec;
    do_contrib = (fabs( damping_vector[i].f[0] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    firstVec[i] = 0.0;
    damping_vector[i].f[1] = *(cVec + numcomp);
    do_contrib = (fabs( damping_vector[i].f[1] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    firstVec[i+8] = 0.0;
    damping_vector[i].f[2] = *(cVec + 2*numcomp);
    do_contrib = (fabs( damping_vector[i].f[2] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    firstVec[i+16] = 0.0;
    cVec += lenum;
  }

  mu = -0.5625 * gpuData->c1ArrayDevice[eindex];  
  if (do_contrib) {
    //if(vector_is_zero( damping_vector ) != 0) {
    aTransposeU( damping_vector, atu );
    firstVector_mu( atu, firstVec, mu);
    do_write = 1;
  }

  /* DILATION CONTRIBUTION */
  cVec = (solver_float *)gpuData->kappaVectorDevice + eindex;
  do_contrib = 0;
  for (i = 0; i < 8; i++) {
    damping_vector[i].f[0] = *cVec;
    do_contrib = (fabs( damping_vector[i].f[0] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    damping_vector[i].f[1] = *(cVec + numcomp);
    do_contrib = (fabs( damping_vector[i].f[1] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    damping_vector[i].f[2] = *(cVec + 2*numcomp);
    do_contrib = (fabs( damping_vector[i].f[2] ) > UNDERFLOW_CAP_STIFFNESS) ? 1 : do_contrib;
    cVec += lenum;
  }

  kappa = -0.5625 * (gpuData->c2ArrayDevice[eindex] + 
		     2. / 3. * gpuData->c1ArrayDevice[eindex]);
  if (do_contrib) {
    //if(vector_is_zero( damping_vector ) != 0) {
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

    atomicAdd((solver_float *)&(nodalForce->f[0]), 
	      (solver_float)damping_vector[i].f[0]);
    atomicAdd((solver_float *)&(nodalForce->f[1]), 
	      (solver_float)damping_vector[i].f[1]);
    atomicAdd((solver_float *)&(nodalForce->f[2]), 
	      (solver_float)damping_vector[i].f[2]);
    
    lnid += lenum;
  }

  return;
}


/* Displacement Calculation Kernel
   FLOPs: 3*(2A + 3M) = 6A + 9M 
   MEM  : 8(3*3) + 8 + 8*6 + 8*3 = 128
*/
__global__  void kernelDispCalc(int32_t nharbored,
				gpu_data_t* gpuData,
				noyesflag_t printAccel)
{
    lnid_t nindex = (blockIdx.x * blockDim.x) + threadIdx.x; 

    /* Shared memory for coalesced reads */
    extern __shared__ solver_float disp[];
    solver_float *t1, *t2, *f;
    int32_t i, startnode, stride, offset;

    fvector_t       *tm1Disp, *tm2Disp, *nodalForce;
    solver_float     mass2_minusaM, mass_minusaM, mass_simple;

    startnode = blockIdx.x * blockDim.x;

    /* Since number of nodes may not be exactly divisible by block size,
       check that we are not off the end of the node array */
    if ((nindex >= gpuData->nharbored) || (startnode >= gpuData->nharbored)) {
      return;
    }

    stride = (startnode + blockDim.x >= gpuData->nharbored) ? (gpuData->nharbored - startnode) : blockDim.x;

    offset = threadIdx.x;
    //t1 = &disp[threadIdx.x];
    //t2 = &disp[3*blockDim.x + threadIdx.x];
    //f = &disp[6*blockDim.x + threadIdx.x];
    t1 = &disp[threadIdx.x];
    t2 = t1 + 3*blockDim.x;
    f = t2 + 3*blockDim.x;;
    for (i = 0; i < 3; i++) {
      *t1 = *(((solver_float *)(gpuData->tm1Device + startnode)) + offset);
      *t2 = *(((solver_float *)(gpuData->tm2Device + startnode)) + offset);
      *f = *(((solver_float *)(gpuData->forceDevice + startnode)) + offset);
      
      offset += stride;
      t1 += stride;
      t2 += stride;
      f += stride;
    }

    __syncthreads();

    //tm1Disp = (fvector_t *)&disp[threadIdx.x*3];
    //tm2Disp = (fvector_t *)&disp[3*blockDim.x + threadIdx.x*3];
    //nodalForce = (fvector_t *)&disp[6*blockDim.x + threadIdx.x*3];
    tm1Disp = (fvector_t *)&disp[threadIdx.x*3];
    tm2Disp = tm1Disp + blockDim.x;
    nodalForce = tm2Disp + blockDim.x;
    mass_simple = gpuData->mass_simpleArrayDevice[nindex];

    /* total nodal forces */
    mass2_minusaM = gpuData->mass2_minusaMArrayDevice[0][nindex];
    mass_minusaM = gpuData->mass_minusaMArrayDevice[0][nindex];
    tm2Disp->f[0] = (nodalForce->f[0] + mass2_minusaM * tm1Disp->f[0]
		      - mass_minusaM  * tm2Disp->f[0]) / mass_simple;
    
    mass2_minusaM = gpuData->mass2_minusaMArrayDevice[1][nindex];
    mass_minusaM = gpuData->mass_minusaMArrayDevice[1][nindex];
    tm2Disp->f[1] = (nodalForce->f[1] + mass2_minusaM * tm1Disp->f[1]
		      - mass_minusaM  * tm2Disp->f[1]) / mass_simple;
    
    mass2_minusaM = gpuData->mass2_minusaMArrayDevice[2][nindex];
    mass_minusaM = gpuData->mass_minusaMArrayDevice[2][nindex];
    tm2Disp->f[2] = (nodalForce->f[2] + mass2_minusaM * tm1Disp->f[2]
		      - mass_minusaM  * tm2Disp->f[2]) / mass_simple;

    __syncthreads();

    /* overwrite tm2 */
    offset = threadIdx.x;
    t2 = &disp[3*blockDim.x + threadIdx.x];
    for (i = 0; i < 3; i++) {
      *(((solver_float *)(gpuData->tm2Device + startnode)) + offset) = *t2;

      offset += stride;
      t2 += stride;
    }
   
    return;
}



/* tm2Disp Alignment Kernel 
   FLOPs: 0 (overhead kernel)
   MEM  : 0 (overhead kernel)
*/
__global__  void kernelAlignTm2Disp(int32_t lenum,
				    int32_t numcomp,
				    gpu_data_t *gpuData) 
{

  int i;
  int32_t   eindex = (blockIdx.x * blockDim.x) + threadIdx.x; // Element number
  int offset;

  /* Since number of elements may not be exactly divisible by block size,
     check that we are not off the end of the element array */
  if (eindex >= lenum) {
    return;
  }

  /* Align tm2Disp */
  for (i = 0, offset = eindex; i < 8; i++, offset += lenum) {
    int32_t lnid = *(gpuData->lnidArrayDevice + offset);
    const __restrict__ fvector_t *disp = gpuData->tm1Device + lnid;
    solver_float *dampVec = (solver_float *)gpuData->shearVectorDevice + offset;

    *dampVec = disp->f[0];
    *(dampVec + numcomp) = disp->f[1];
    *(dampVec + 2*numcomp) = disp->f[2];
      
  }

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
    const solver_float *tmp = (solver_float *)v;

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
