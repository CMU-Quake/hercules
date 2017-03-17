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
#include "nonlinear.h" //NEEDS TO BE HERE FOR NONLINEAR TO RUN
#include "stiffness.h"
#include "quake_util.h"
#include "util.h"
#include "timers.h"
#include "cvm.h"
#include "topography.h"



/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static int32_t  myLinearElementsCount;
static int32_t *myLinearElementsMapper;

/* -------------------------------------------------------------------------- */
/*                      Initialization of parameters for
 *                   Nonlinear and Topography compatibility                   */
/* -------------------------------------------------------------------------- */

/**
 * Counts the number of nonlinear elements in my local mesh
 */
void linear_elements_count(int32_t myID, mesh_t *myMesh) {

    int32_t eindex;
    int32_t count = 0;

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        if ( ( isThisElementNonLinear(myMesh, eindex) == NO ) &&
        	 ( BelongstoTopography   (myMesh, eindex) == NO )  ) {
            count++;
        }
    }

    if ( count > myMesh-> lenum ) {
        fprintf(stderr,"Thread %d: linear_elements_count: "
                "more elements than expected\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    myLinearElementsCount = count;

    return;
}


/**
 * Re-counts and stores the nonlinear element indices to a static local array
 * that will serve as mapping tool to the local mesh elements table.
 */
void linear_elements_mapping(int32_t myID, mesh_t *myMesh) {

    int32_t eindex;
    int32_t count = 0;

    XMALLOC_VAR_N(myLinearElementsMapper, int32_t, myLinearElementsCount);

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {


        if ( ( isThisElementNonLinear(myMesh, eindex) == NO ) &&
        	 ( BelongstoTopography(myMesh, eindex)    == NO ) ) {
            myLinearElementsMapper[count] = eindex;
            count++;
        }
    }

    if ( count != myLinearElementsCount ) {
        fprintf(stderr,"Thread %d: linear_elements_mapping: "
                "more elements than the count\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    return;
}


void stiffness_init(int32_t myID, mesh_t *myMesh) {

    linear_elements_count(myID, myMesh);
    linear_elements_mapping(myID, myMesh);
}


/* -------------------------------------------------------------------------- */
/*                       Stiffness Contribution Methods                       */
/* -------------------------------------------------------------------------- */


/**
 * Compute and add the force due to the element stiffness matrices.
 *
 * \param myMesh   Pointer to the solver mesh structure.
 * \param mySolver Pointer to the solver main data structures.
 * \param theK1    First stiffness matrix (K1).
 * \param theK2    Second stiffness matrix (K2).
 */
void compute_addforce_conventional( mesh_t* myMesh, mysolver_t* mySolver, 
				    fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8] )
{
    fvector_t localForce[8];
    int       i, j;
    int32_t   eindex;
    int32_t   lin_eindex;

    /* loop on the number of elements */
    for (lin_eindex = 0; lin_eindex < myLinearElementsCount; lin_eindex++) {

        elem_t* elemp;
        e_t*    ep;

        eindex = myLinearElementsMapper[lin_eindex];
        elemp  = &myMesh->elemTable[eindex];
        ep     = &mySolver->eTable[eindex];

        /* step 1: calculate the force due to the element stiffness */
        memset( localForce, 0, 8 * sizeof(fvector_t) );

        /* contribution by node j to node i */
        for (i = 0; i < 8; i++)
        {
            fvector_t* toForce = &localForce[i];

            for (j = 0; j < 8; j++)
            {
                int32_t    nodeJ  = elemp->lnid[j];
                fvector_t* myDisp = mySolver->tm1 + nodeJ;

                /*
		 * contributions by the stiffnes/damping matrix
		 * contribution by ( - deltaT_square * Ke * Ut )
		 * But if myDisp is zero avoids multiplications
		 */
                if ( vector_is_all_zero( myDisp ) != 0 ) {
                    MultAddMatVec( &theK1[i][j], myDisp, -ep->c1, toForce );
                    MultAddMatVec( &theK2[i][j], myDisp, -ep->c2, toForce );
                }
            }
        }

        /* step 2: sum up my contribution to my vertex nodes */
        for (i = 0; i < 8; i++) {
            int32_t    lnid       = elemp->lnid[i];
            fvector_t* nodalForce = mySolver->force + lnid;

            nodalForce->f[0] += localForce[i].f[0];
            nodalForce->f[1] += localForce[i].f[1];
            nodalForce->f[2] += localForce[i].f[2];
        }
    } /* for all the elements */
}


/**
 * Compute and add the force due to the element stiffness matrices with the effective method.
 */
void compute_addforce_effective( mesh_t* myMesh, mysolver_t* mySolver )
{
    /* \TODO use mu_and_lamda to compute first,second and third coefficients */

    fvector_t localForce[8];
    fvector_t curDisp[8];
    int       i;
    int32_t   eindex;
    int32_t   lin_eindex;

    /* loop on the number of elements */
    for (lin_eindex = 0; lin_eindex < myLinearElementsCount; lin_eindex++) {

        elem_t* elemp;
        e_t*    ep;

        eindex = myLinearElementsMapper[lin_eindex];
        elemp  = &myMesh->elemTable[eindex];
        ep     = &mySolver->eTable[eindex];

        memset( localForce, 0, 8 * sizeof(fvector_t) );

        double b_over_dt = ep->c3 / ep->c1;

        for (i = 0; i < 8; i++) {
            int32_t    lnid = elemp->lnid[i];
            fvector_t* tm1Disp = mySolver->tm1 + lnid;
   	        fvector_t* tm2Disp = mySolver->tm2 + lnid;

//            curDisp[i].f[0] = tm1Disp->f[0];
//            curDisp[i].f[1] = tm1Disp->f[1];
//            curDisp[i].f[2] = tm1Disp->f[2];

   	        /* Dorian says: this is done to  consider Rayleigh damping simultaneously
   	         * with the stiffness force computation */
            curDisp[i].f[0] = tm1Disp->f[0] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[0];
            curDisp[i].f[1] = tm1Disp->f[1] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[1];
            curDisp[i].f[2] = tm1Disp->f[2] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[2];

        }

        /* Coefficients for new stiffness matrix calculation */
        if (vector_is_zero( curDisp ) != 0) {

            double first_coeff  = -0.5625 * (ep->c2 + 2 * ep->c1);
            double second_coeff = -0.5625 * (ep->c2);
            double third_coeff  = -0.5625 * (ep->c1);

            double atu[24];
            double firstVec[24];

            aTransposeU( curDisp, atu );
            firstVector( atu, firstVec, first_coeff, second_coeff, third_coeff );
            au( localForce, firstVec );
        }

        for (i = 0; i < 8; i++) {

            int32_t lnid          = elemp->lnid[i];
            fvector_t* nodalForce = mySolver->force + lnid;

            nodalForce->f[0] += localForce[i].f[0];
            nodalForce->f[1] += localForce[i].f[1];
            nodalForce->f[2] += localForce[i].f[2];
        }
    } /* for all the elements */
}


/* -------------------------------------------------------------------------- */
/*                         Efficient Method Utilities                         */
/* -------------------------------------------------------------------------- */


void aTransposeU( fvector_t* un, double* atu )
{
    double temp[24];
    double u[24];
    int    i, j;

    /* arrange displacement values in an array */
    for (i=0; i<8; i++) {
        for(j=0; j<3; j++) {
            temp[i*3 + j] = un[i].f[j];     /* u1 u2 u3 .... v1 v2 v3 ... z1 z2 z3 */
	}
    }

    reformU( temp, u );

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

void firstVector( const double* atu, double* finalVector, double a, double c, double b )
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

void firstVector_kappa( const double* atu, double* finalVector, double kappa)
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

void firstVector_mu( const double* atu, double* finalVector, double b )
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

void au( fvector_t* resVec, const double* u )
{
    int    i, j;
    double finVec[24];
    double temp[24];


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

    reformF( finVec, temp );

    for (j = 0; j<8; j++)
    {
        for (i = 0; i<3; i++)
        {
            resVec[j].f[i] += temp[j*3 + i];
        }
    }
}


void reformF( const double* u, double* newU )
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

void reformU( const double* u, double* newU )
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
