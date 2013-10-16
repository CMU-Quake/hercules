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
#include "quake_util.h"
#include "damping.h"
#include "util.h"
#include "kernel.h"


void damping_addforce(mesh_t *myMesh, mysolver_t *mySolver, fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8]){

    fvector_t localForce[8];
    int       i,j;
    int32_t   eindex;

    fvector_t deltaDisp[8];

    /* loop on the number of elements */
    for (eindex = 0; eindex < myMesh->lenum; eindex++)
    {
        elem_t *elemp;
        e_t    *ep;

        elemp = &myMesh->elemTable[eindex];
        ep = &mySolver->eTable[eindex];

        /* Step 1. calculate the force due to the element stiffness */
        memset(localForce, 0, 8 * sizeof(fvector_t));

        /* compute the diff between disp(tm1) and disp(tm2) */

        /* the_E1_timer -= MPI_Wtime();*/
        for (i = 0; i < 8; i++) {
            fvector_t *tm1Disp, *tm2Disp;
            int32_t    lnid;

            lnid = elemp->lnid[i];

            tm1Disp = mySolver->tm1 + lnid;
            tm2Disp = mySolver->tm2 + lnid;

            deltaDisp[i].f[0] = tm1Disp->f[0] - tm2Disp->f[0];
            deltaDisp[i].f[1] = tm1Disp->f[1] - tm2Disp->f[1];
            deltaDisp[i].f[2] = tm1Disp->f[2] - tm2Disp->f[2];
        }

        if(vector_is_zero( deltaDisp ) != 0) {
            /*the_E3_timer += MPI_Wtime();  */

            for (i = 0; i < 8; i++)
            {
                fvector_t* toForce;
                toForce = &localForce[i];

                for (j = 0; j < 8; j++)
                {
                    fvector_t *myDeltaDisp;

                    /* contribution by ( - b * deltaT * Ke * ( Ut - Ut-1 ) ) */
                    /* But if myDeltaDisp is zero avoids multiplications     */
                    myDeltaDisp = &deltaDisp[j];

                    MultAddMatVec(&theK1[i][j], myDeltaDisp, -ep->c3, toForce);
                    MultAddMatVec(&theK2[i][j], myDeltaDisp, -ep->c4, toForce);

                }
            }
        }
        for (i = 0; i < 8; i++) {
            int32_t lnid;
            fvector_t *nodalForce;

            lnid = elemp->lnid[i];

            nodalForce = mySolver->force + lnid;
            nodalForce->f[0] += localForce[i].f[0];
            nodalForce->f[1] += localForce[i].f[1];
            nodalForce->f[2] += localForce[i].f[2];
        }

    } /* for all the elements */

    return;
}

/**
 * Introduce BKT Model: Compute and add the force due to the element
 *              damping.
 */

void calc_conv_gpu(int32_t myID, mesh_t *myMesh, mysolver_t *mySolver, double theFreq, double theDeltaT, double theDeltaTSquared)
{
    double rmax = 2. * M_PI * theFreq * theDeltaT;
    
    int blocksize = gpu_get_blocksize(mySolver->gpu_spec,
				      (char *)kernelDampingCalcConv, 0);
    /* Number of threads is lenum * 2 convolutions/element * 8 values/conv */
    int gridsize = (myMesh->lenum * 16 / blocksize) + 1;
    cudaGetLastError();
    kernelDampingCalcConv<<<gridsize, blocksize>>>(mySolver->gpuDataDevice, 
						   rmax);

    cudaError_t cerror = cudaGetLastError();
    if (cerror != cudaSuccess) {
      fprintf(stderr, "Thread %d: Calc damping conv kernel - %s\n", myID, 
	      cudaGetErrorString(cerror));
      MPI_Abort(MPI_COMM_WORLD, ERROR);
      exit(1);
    }

    mySolver->gpu_spec->numflops += kernel_flops_per_thread(FLOP_CALCCONV_KERNEL) * myMesh->lenum * 16;

    return;
}


/**
 * Introduce BKT Model: Compute and add the force due to the element
 *              damping.
 */

void calc_conv_cpu(mesh_t *myMesh, mysolver_t *mySolver, double theFreq, double theDeltaT, double theDeltaTSquared){

    int32_t eindex;
    int i;
    double rmax = 2. * M_PI * theFreq * theDeltaT;

    for (eindex = 0; eindex < myMesh->lenum; eindex++)
    {
    	elem_t *elemp;
    	edata_t *edata;

    	elemp = &myMesh->elemTable[eindex];
    	edata = (edata_t *)elemp->data;

        // SHEAR RELATED CONVOLUTION

    	if ( (edata->g0_shear != 0) && (edata->g1_shear != 0) ) {

            double c0_shear = edata->g0_shear;
            double c1_shear = edata->g1_shear;

            double g0_shear = c0_shear * rmax;
            double g1_shear = c1_shear * rmax;

            double coef_shear_1 = g0_shear / 2.;
            double coef_shear_2 = coef_shear_1 * ( 1. - g0_shear );

            double coef_shear_3 = g1_shear / 2.;
            double coef_shear_4 = coef_shear_3 * ( 1. - g1_shear );

            double exp_coef_shear_0 = exp( -g0_shear );
            double exp_coef_shear_1 = exp( -g1_shear );

            for(i = 0; i < 8; i++)
            {
                int32_t     lnid, cindex;
                fvector_t   *f0_tm1, *f1_tm1, *tm1Disp, *tm2Disp;

                lnid = elemp->lnid[i];

                /* cindex is the index of the node in the convolution vector */
                cindex = eindex * 8 + i;

                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;

                f0_tm1 = mySolver->conv_shear_1 + cindex;
                f1_tm1 = mySolver->conv_shear_2 + cindex;

                f0_tm1->f[0] = coef_shear_2 * tm1Disp->f[0] + coef_shear_1 * tm2Disp->f[0] + exp_coef_shear_0 * f0_tm1->f[0];
                f0_tm1->f[1] = coef_shear_2 * tm1Disp->f[1] + coef_shear_1 * tm2Disp->f[1] + exp_coef_shear_0 * f0_tm1->f[1];
                f0_tm1->f[2] = coef_shear_2 * tm1Disp->f[2] + coef_shear_1 * tm2Disp->f[2] + exp_coef_shear_0 * f0_tm1->f[2];

                f1_tm1->f[0] = coef_shear_4 * tm1Disp->f[0] + coef_shear_3 * tm2Disp->f[0] + exp_coef_shear_1 * f1_tm1->f[0];
                f1_tm1->f[1] = coef_shear_4 * tm1Disp->f[1] + coef_shear_3 * tm2Disp->f[1] + exp_coef_shear_1 * f1_tm1->f[1];
                f1_tm1->f[2] = coef_shear_4 * tm1Disp->f[2] + coef_shear_3 * tm2Disp->f[2] + exp_coef_shear_1 * f1_tm1->f[2];

            } // For local nodes (0:7)

    	} // end if null coefficients

        // DILATATION RELATED CONVOLUTION

        if ( (edata->g0_kappa != 0) && (edata->g1_kappa != 0) ) {

            double c0_kappa = edata->g0_kappa;
            double c1_kappa = edata->g1_kappa;

            double g0_kappa = c0_kappa * rmax;
            double g1_kappa = c1_kappa * rmax;

            double coef_kappa_1 = g0_kappa / 2.;
            double coef_kappa_2 = coef_kappa_1 * ( 1. - g0_kappa );

            double coef_kappa_3 = g1_kappa / 2.;
            double coef_kappa_4 = coef_kappa_3 * ( 1. - g1_kappa );

            double exp_coef_kappa_0 = exp( -g0_kappa );
            double exp_coef_kappa_1 = exp( -g1_kappa );

            for(i = 0; i < 8; i++)
            {
                int32_t     lnid, cindex;
                fvector_t   *f0_tm1, *f1_tm1, *tm1Disp, *tm2Disp;

                lnid = elemp->lnid[i];

                /* cindex is the index of the node in the convolution vector */
                cindex = eindex * 8 + i;

                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;

                f0_tm1 = mySolver->conv_kappa_1 + cindex;
                f1_tm1 = mySolver->conv_kappa_2 + cindex;

                f0_tm1->f[0] = coef_kappa_2 * tm1Disp->f[0] + coef_kappa_1 * tm2Disp->f[0] + exp_coef_kappa_0 * f0_tm1->f[0];
                f0_tm1->f[1] = coef_kappa_2 * tm1Disp->f[1] + coef_kappa_1 * tm2Disp->f[1] + exp_coef_kappa_0 * f0_tm1->f[1];
                f0_tm1->f[2] = coef_kappa_2 * tm1Disp->f[2] + coef_kappa_1 * tm2Disp->f[2] + exp_coef_kappa_0 * f0_tm1->f[2];

                f1_tm1->f[0] = coef_kappa_4 * tm1Disp->f[0] + coef_kappa_3 * tm2Disp->f[0] + exp_coef_kappa_1 * f1_tm1->f[0];
                f1_tm1->f[1] = coef_kappa_4 * tm1Disp->f[1] + coef_kappa_3 * tm2Disp->f[1] + exp_coef_kappa_1 * f1_tm1->f[1];
                f1_tm1->f[2] = coef_kappa_4 * tm1Disp->f[2] + coef_kappa_3 * tm2Disp->f[2] + exp_coef_kappa_1 * f1_tm1->f[2];

            } // For local nodes (0:7)

        } // end if null coefficients

    } // For all elements

    return;

}


/**
 * new_damping: Compute and add the force due to the element
 *              damping.
 */
void constant_Q_addforce_gpu(int myID, mesh_t *myMesh, mysolver_t *mySolver, double theFreq, double theDeltaT, double theDeltaTSquared)
{
    /* \todo use mu_and_lamda to compute first,second and third coefficients */

    double rmax = 2. * M_PI * theFreq * theDeltaT;

	/* theAddForceETime -= MPI_Wtime(); */

    /* Copy working data to device */
    cudaMemcpy(mySolver->gpuData->forceDevice, mySolver->force, 
    	       myMesh->nharbored * sizeof(fvector_t), cudaMemcpyHostToDevice);
    mySolver->gpu_spec->numbytes += myMesh->nharbored * sizeof(fvector_t);

    /* Each thread saves a shear or kappa vector in shared mem */
    int blocksize = gpu_get_blocksize(mySolver->gpu_spec,
    				      (char *)kernelDampingCalcLocal, 
    				      2 * sizeof(fvector_t));
    /* Number of threads is lenum * 8 components/element */
    int gridsize = (myMesh->lenum * 8 / blocksize) + 1;
    int sharedmem = blocksize * 2 * sizeof(fvector_t);
    cudaGetLastError();
    kernelDampingCalcLocal<<<gridsize, blocksize, sharedmem>>>(mySolver->gpuDataDevice, rmax);

    cudaError_t cerror = cudaGetLastError();
    if (cerror != cudaSuccess) {
      fprintf(stderr, "Thread %d: Calc damping local kernel - %s\n", myID, 
	      cudaGetErrorString(cerror));
      MPI_Abort(MPI_COMM_WORLD, ERROR);
      exit(1);
    }

    mySolver->gpu_spec->numflops += kernel_flops_per_thread(FLOP_DAMPING_KERNEL) * myMesh->lenum * 8;

    /* Copy working data back to host */
    cudaMemcpy(mySolver->force, mySolver->gpuData->forceDevice,
	       myMesh->nharbored * sizeof(fvector_t), cudaMemcpyDeviceToHost);
    mySolver->gpu_spec->numbytes += myMesh->nharbored * sizeof(fvector_t);

    return;
}



/**
 * new_damping: Compute and add the force due to the element
 *              damping.
 */
void constant_Q_addforce_cpu(mesh_t *myMesh, mysolver_t *mySolver, double theFreq, double theDeltaT, double theDeltaTSquared)
{
	/* \todo use mu_and_lamda to compute first,second and third coefficients */

	int i;
	fvector_t localForce[8];
	int32_t   eindex;
	fvector_t damping_vector_shear[8], damping_vector_kappa[8];

	double rmax = 2. * M_PI * theFreq * theDeltaT;

	/* theAddForceETime -= MPI_Wtime(); */

	/* loop on the number of elements */
	for (eindex = 0; eindex < myMesh->lenum; eindex++)
	{
		elem_t *elemp;
		e_t    *ep;
		edata_t *edata;

		double a0_shear, a1_shear, b_shear, a0_kappa, a1_kappa, b_kappa, csum;

		elemp = &myMesh->elemTable[eindex];
		edata = (edata_t *)elemp->data;
		ep = &mySolver->eTable[eindex];

		// SHEAR CONTRIBUTION

        a0_shear = edata->a0_shear;
        a1_shear = edata->a1_shear;
        b_shear  = edata->b_shear;

        csum = a0_shear + a1_shear + b_shear;

		if ( csum != 0 ) {

	        double coef_shear = b_shear / rmax;

            for (i = 0; i < 8; i++) {

                fvector_t *tm1Disp, *tm2Disp, *f0_tm1, *f1_tm1;
                int32_t    lnid, cindex;

                cindex = eindex * 8 + i;

                lnid = elemp->lnid[i];

                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;
                f0_tm1  = mySolver->conv_shear_1 + cindex;
                f1_tm1  = mySolver->conv_shear_2 + cindex;

                damping_vector_shear[i].f[0] = coef_shear * (tm1Disp->f[0] - tm2Disp->f[0])
                                             - (a0_shear * f0_tm1->f[0] + a1_shear * f1_tm1->f[0])
                                             + tm1Disp->f[0];

                damping_vector_shear[i].f[1] = coef_shear * (tm1Disp->f[1] - tm2Disp->f[1])
                                             - (a0_shear * f0_tm1->f[1] + a1_shear * f1_tm1->f[1])
                                             + tm1Disp->f[1];

                damping_vector_shear[i].f[2] = coef_shear * (tm1Disp->f[2] - tm2Disp->f[2])
                                             - (a0_shear * f0_tm1->f[2] + a1_shear * f1_tm1->f[2])
                                             + tm1Disp->f[2];

            } // end for nodes in the element

		} else {

		    for (i = 0; i < 8; i++) {

                fvector_t *tm1Disp, *tm2Disp;
                int32_t    lnid;

                lnid = elemp->lnid[i];
                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;

                damping_vector_shear[i].f[0] = tm1Disp->f[0];
                damping_vector_shear[i].f[1] = tm1Disp->f[1];
                damping_vector_shear[i].f[2] = tm1Disp->f[2];

            } // end for nodes in the element

		} // end if for coefficients

		// DILATATION CONTRIBUTION

        a0_kappa   = edata->a0_kappa;
        a1_kappa   = edata->a1_kappa;
        b_kappa    = edata->b_kappa;

        csum = a0_kappa + a1_kappa + b_kappa;

        if ( csum != 0 ) {

            double coef_kappa = b_kappa / rmax;

            for (i = 0; i < 8; i++) {

                fvector_t *tm1Disp, *tm2Disp, *f0_tm1, *f1_tm1;
                int32_t    lnid, cindex;

                cindex = eindex * 8 + i;

                lnid = elemp->lnid[i];

                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;

                f0_tm1  = mySolver->conv_kappa_1 + cindex;
                f1_tm1  = mySolver->conv_kappa_2 + cindex;

                damping_vector_kappa[i].f[0] = coef_kappa * (tm1Disp->f[0] - tm2Disp->f[0])
                                             - (a0_kappa * f0_tm1->f[0] + a1_kappa * f1_tm1->f[0])
                                             + tm1Disp->f[0];

                damping_vector_kappa[i].f[1] = coef_kappa * (tm1Disp->f[1] - tm2Disp->f[1])
                                             - (a0_kappa * f0_tm1->f[1] + a1_kappa * f1_tm1->f[1])
                                             + tm1Disp->f[1];

                damping_vector_kappa[i].f[2] = coef_kappa * (tm1Disp->f[2] - tm2Disp->f[2])
                                             - (a0_kappa * f0_tm1->f[2] + a1_kappa * f1_tm1->f[2])
                                             + tm1Disp->f[2];

            } // end for nodes in the element

        } else {

            for (i = 0; i < 8; i++) {

                fvector_t *tm1Disp, *tm2Disp;
                int32_t    lnid;

                lnid = elemp->lnid[i];
                tm1Disp = mySolver->tm1 + lnid;
                tm2Disp = mySolver->tm2 + lnid;

                damping_vector_kappa[i].f[0] = tm1Disp->f[0];
                damping_vector_kappa[i].f[1] = tm1Disp->f[1];
                damping_vector_kappa[i].f[2] = tm1Disp->f[2];

            } // end for nodes in the element

        } // end if for coefficients

		double kappa = -0.5625 * (ep->c2 + 2. / 3. * ep->c1);
		double mu = -0.5625 * ep->c1;

		double atu[24];
		double firstVec[24];

		for(i = 0; i<24; i++)
			firstVec[i] = 0.;

		memset(localForce, 0, 8 * sizeof(fvector_t));

        if(vector_is_zero( damping_vector_shear ) != 0) {

            aTransposeU( damping_vector_shear, atu );
            firstVector_mu( atu, firstVec, mu);

        }

		if(vector_is_zero( damping_vector_kappa ) != 0) {

			aTransposeU( damping_vector_kappa, atu );
			firstVector_kappa( atu, firstVec, kappa);

		}

		au( localForce, firstVec );

		for (i = 0; i < 8; i++) {
			int32_t lnid;
			fvector_t *nodalForce;

			lnid = elemp->lnid[i];

			nodalForce = mySolver->force + lnid;
			nodalForce->f[0] += localForce[i].f[0];
			nodalForce->f[1] += localForce[i].f[1];
			nodalForce->f[2] += localForce[i].f[2];
		}

	} /* for all the elements */

	return;
}

