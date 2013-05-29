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
#include <stdio.h>

#include "psolve.h"
#include "octor.h"
#include "util.h"
#include "stiffness.h"
#include "quake_util.h"
#include "commutil.h"
#include "cvm.h"
#include "nonlinear.h"
#include "topography.h"
#include "geometrics.h"

/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

/* Permanent */
static toposolver_t       *myTopoSolver;
static int                theMaxoctlevel, ntp, np_ew, np_ns;
static double             thebase_zcoord = 0, So, theDomainLong_ew, theDomainLong_ns;
static double             *theTopoInfo;
static etreetype_t        theEtreeType;
static int32_t            myTopoElementsCount = 0;
static int32_t            *myTopoElementsMapping;
static topometh_t         theTopoMethod;
static topostation_t      *myTopoStations;


/*  Stiffness Matrices of Tetrahedral Elements */
static fmatrix_t  mytheK1_Tet1[4][4];
static fmatrix_t  mytheK2_Tet1[4][4];

static fmatrix_t  mytheK1_Tet2[4][4];
static fmatrix_t  mytheK2_Tet2[4][4];

static fmatrix_t  mytheK1_Tet3[4][4];
static fmatrix_t  mytheK2_Tet3[4][4];

static fmatrix_t  mytheK1_Tet4[4][4];
static fmatrix_t  mytheK2_Tet4[4][4];

static fmatrix_t  mytheK1_Tet5[4][4];
static fmatrix_t  mytheK2_Tet5[4][4];

static fmatrix_t  mytheK1_SymmTet1[4][4];
static fmatrix_t  mytheK2_SymmTet1[4][4];

static fmatrix_t  mytheK1_SymmTet2[4][4];
static fmatrix_t  mytheK2_SymmTet2[4][4];

static fmatrix_t  mytheK1_SymmTet3[4][4];
static fmatrix_t  mytheK2_SymmTet3[4][4];

static fmatrix_t  mytheK1_SymmTet4[4][4];
static fmatrix_t  mytheK2_SymmTet4[4][4];

static fmatrix_t  mytheK1_SymmTet5[4][4];
static fmatrix_t  mytheK2_SymmTet5[4][4];


/*  ============================================  */
/* TODO: Only for the TopoDRM. Erase later Dorian */
static int32_t            *myDRMFace1ElementsMapping;
static int32_t            *myDRMFace2ElementsMapping;
static int32_t            *myDRMFace3ElementsMapping;
static int32_t            *myDRMFace4ElementsMapping;
static int32_t            *myDRMBottomElementsMapping;

static int32_t            *myDRMBorder1ElementsMapping;
static int32_t            *myDRMBorder2ElementsMapping;
static int32_t            *myDRMBorder3ElementsMapping;
static int32_t            *myDRMBorder4ElementsMapping;

static int32_t            *myDRMBorder5ElementsMapping;
static int32_t            *myDRMBorder6ElementsMapping;
static int32_t            *myDRMBorder7ElementsMapping;
static int32_t            *myDRMBorder8ElementsMapping;

static int32_t            myTopoDRMFaceElementsCount = 0;
static int32_t            myTopoDRMBorderElementsCount = 0;


static double            myTs  = 0;
static double            myFc  = 0;
static int               myDir = 0;
static double            theDRMdepth;

static int32_t            myDRM_Face1Count  = 0;
static int32_t            myDRM_Face2Count  = 0;
static int32_t            myDRM_Face3Count  = 0;
static int32_t            myDRM_Face4Count  = 0;
static int32_t            myDRM_BottomCount = 0;
static int32_t            myDRM_Brd1 = 0;
static int32_t            myDRM_Brd2 = 0;
static int32_t            myDRM_Brd3 = 0;
static int32_t            myDRM_Brd4 = 0;
static int32_t            myDRM_Brd5 = 0;
static int32_t            myDRM_Brd6 = 0;
static int32_t            myDRM_Brd7 = 0;
static int32_t            myDRM_Brd8 = 0;


/* ==============================   */

/* Gauss Points matrix up until 10 integration points. 1st row: gp positions,
                                                       2nd row: gp weights, */
#define  GP  gp[2][54] = { { 0.5773502691896257645091488,  -0.5773502691896257645091488, \
		                     0                          ,   0.7745966692414833770358531,  -0.7745966692414833770358531, \
		                     0.3399810435848562648026658,   0.8611363115940525752239465,  -0.3399810435848562648026658,  -0.8611363115940525752239465, \
		                     0.5384693101056830910363144,   0.9061798459386639927976269,   0                          ,  -0.5384693101056830910363144,   -0.9061798459386639927976269, \
		                     0.2386191860831969086305017,   0.6612093864662645136613996,   0.9324695142031520278123016,  -0.2386191860831969086305017,   -0.6612093864662645136613996,   -0.9324695142031520278123016, \
		                     0.4058451513773971669066064,   0.7415311855993944398638648,   0.9491079123427585245261897,   0                          ,   -0.4058451513773971669066064,   -0.7415311855993944398638648,   -0.9491079123427585245261897, \
		                     0.1834346424956498049394761,   0.5255324099163289858177390,   0.7966664774136267395915539,   0.9602898564975362316835609,   -0.1834346424956498049394761,   -0.5255324099163289858177390,   -0.7966664774136267395915539,   -0.9602898564975362316835609, \
		                     0.3242534234038089290385380,   0.6133714327005903973087020,   0.8360311073266357942994298,   0.9681602395076260898355762,    0                          ,   -0.3242534234038089290385380,   -0.6133714327005903973087020,   -0.8360311073266357942994298,   -0.9681602395076260898355762, \
		                     0.1488743389816312108848260,   0.4333953941292471907992659,   0.6794095682990244062343274,   0.8650633666889845107320967,    0.9739065285171717200779640,   -0.1488743389816312108848260,   -0.4333953941292471907992659,   -0.6794095682990244062343274,   -0.8650633666889845107320967,  -0.9739065285171717200779640}, \
                           { 1                          ,   1                          , \
	                         0.8888888888888888888888889,   0.5555555555555555555555556,   0.5555555555555555555555556, \
	                         0.6521451548625461426269361,   0.3478548451374538573730639,   0.6521451548625461426269361,   0.3478548451374538573730639, \
	                         0.4786286704993664680412915,   0.2369268850561890875142640,   0.5688888888888888888888889,   0.4786286704993664680412915,    0.2369268850561890875142640, \
	                         0.4679139345726910473898703,   0.3607615730481386075698335,   0.1713244923791703450402961,   0.4679139345726910473898703,    0.3607615730481386075698335,    0.1713244923791703450402961, \
	                         0.3818300505051189449503698,   0.2797053914892766679014678,   0.1294849661688696932706114,   0.4179591836734693877551020,    0.3818300505051189449503698,    0.2797053914892766679014678,    0.1294849661688696932706114, \
	                         0.3626837833783619829651504,   0.3137066458778872873379622,   0.2223810344533744705443560,   0.1012285362903762591525314,    0.3626837833783619829651504,    0.3137066458778872873379622,    0.2223810344533744705443560,    0.1012285362903762591525314, \
	                         0.3123470770400028400686304,   0.2606106964029354623187429,   0.1806481606948574040584720,   0.0812743883615744119718922,    0.3302393550012597631645251,    0.3123470770400028400686304,    0.2606106964029354623187429,    0.1806481606948574040584720,    0.0812743883615744119718922, \
	                         0.2955242247147528701738930,   0.2692667193099963550912269,   0.2190863625159820439955349,   0.1494513491505805931457763,    0.0666713443086881375935688,    0.2955242247147528701738930,    0.2692667193099963550912269,    0.2190863625159820439955349,    0.1494513491505805931457763,   0.0666713443086881375935688} }

#define MNOD     mnod[5][4] =     { { 0, 2, 1, 4 }, \
		 	 	 	 	 	        { 3, 1, 2, 7 }, \
		 	 	 	 	            { 6, 4, 7, 2 }, \
		 	 	 	 	            { 5, 7, 4, 1 }, \
		 	 	 	 	            { 2, 4, 7, 1 }	}

#define MNODSYMM  mnodsymm[5][4] ={ { 0, 3, 1, 5 }, \
		 	 	 	 	 	        { 0, 2, 3, 6 }, \
		 	 	 	 	            { 4, 5, 6, 0 }, \
		 	 	 	 	            { 6, 5, 7, 3 }, \
		 	 	 	 	            { 0, 6, 3, 5 }	}

#define TETRACOOR tetraCoor[15][4] = { {0, 0, 1, 0}, \
								       {0, 1, 0, 0}, \
								       {0, 0, 0, 1}, \
								       {1, 1, 0, 1}, \
								       {1, 0, 1, 1}, \
								       {0, 0, 0, 1}, \
								       {0, 0, 1, 0}, \
								       {1, 0, 1, 1}, \
								       {1, 1, 1, 0}, \
								       {1, 1, 0, 1}, \
								       {0, 1, 0, 0}, \
								       {1, 1, 1, 0}, \
								       {0, 0, 1, 1}, \
								       {1, 0, 1, 0}, \
								       {0, 1, 1, 0}   }


#define TETRACOORSYMM tetraCoorSymm[15][4] = { {0, 1, 1, 1}, \
								       	   	   {0, 1, 0, 0}, \
								       	   	   {0, 0, 0, 1}, \
								       	   	   {0, 0, 1, 0}, \
								       	   	   {0, 1, 1, 1}, \
								       	   	   {0, 0, 0, 1}, \
								       	   	   {0, 1, 0, 0}, \
								       	   	   {0, 0, 1, 0}, \
								       	   	   {1, 1, 1, 0}, \
								       	   	   {0, 1, 1, 1}, \
								       	   	   {1, 0, 1, 1}, \
								       	   	   {1, 1, 1, 0}, \
								       	   	   {0, 0, 1, 1}, \
								       	   	   {0, 1, 1, 0}, \
								       	   	   {0, 1, 0, 1}   }


/* -------------------------------------------------------------------------- */
/*                            Basic Functions                                 */
/* -------------------------------------------------------------------------- */

double get_thebase_topo() {
    return thebase_zcoord;
}

//returns YES if the element is air element ( Vp = -1 ), or if it composes the topography surface.
int BelongstoTopography (mesh_t *myMesh, int32_t eindex) {

    int32_t topo_eindex;
    elem_t  *elemp;
    edata_t *edata;

    elemp = &myMesh->elemTable[eindex];
    edata = (edata_t *)elemp->data;

    /* check for air element  */
    if ( edata->Vp == -1 )
    	return YES;

    if ( theTopoMethod == FEM )
    	return NO;

	for ( topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++ ) {

		int32_t          eindexT;

		eindexT = myTopoElementsMapping[topo_eindex];

		if ( eindexT == eindex )
			return YES;

	} /* for all topograhy elements */

	return NO;
}


etreetype_t get_theetree_type() {
    return theEtreeType;
}

int topo_correctproperties ( edata_t *edata )
{
    if ( edata->Vp < 0) {
    	edata->Vs = 1e10;
    	return 1;
    }

    return 0;
}


/* -------------------------------------------------------------------------- */
/*                         Private Method Prototypes                          */
/* -------------------------------------------------------------------------- */

/* Computes the diagonal terms of the Bi(3x3) matrix for a tetrahedral element.
   counter i runs form 0 to 3 (4 nodes tetrahedral) */

void get_AiBiCi ( double x1, double y1, double z1,
				  double x2, double y2, double z2,
				  double x3, double y3, double z3,
				  double x4, double y4, double z4,
                  int i,
				  double *Ai, double *Bi, double *Ci )
{

	double x31 = x3 - x1;
	double y31 = y3 - y1;
	double z31 = z3 - z1;

	double x21 = x2 - x1;
	double y21 = y2 - y1;
	double z21 = z2 - z1;

	double x41 = x4 - x1;
	double y41 = y4 - y1;
	double z41 = z4 - z1;

	double Vo =   x31 * y21 * z41 + x41 * y31 * z21 + z31 * x21 * y41
			  - ( x31 * y41 * z21 + y31 * x21 * z41 + z31 * x41 * y21 );

	double deta_dx  = ( y21 * z41 - z21 * y41 ) / Vo;
	double dpsi_dx  = ( z31 * y41 - y31 * z41 ) / Vo;
	double dgamm_dx = ( y31 * z21 - z31 * y21 ) / Vo;

	double deta_dy  = ( z21 * x41 - x21 * z41 ) / Vo;
	double dpsi_dy  = ( x31 * z41 - x41 * z31 ) / Vo;
	double dgamm_dy = ( z31 * x21 - x31 * z21 ) / Vo;

	double deta_dz  = ( x21 * y41 - y21 * x41 ) / Vo;
	double dpsi_dz  = ( x41 * y31 - x31 * y41 ) / Vo;
	double dgamm_dz = ( x31 * y21 - x21 * y31 ) / Vo;


    switch ( i ) {

        case (0):

            *Ai = - ( deta_dx + dpsi_dx + dgamm_dx );
        	*Bi = - ( deta_dy + dpsi_dy + dgamm_dy );
        	*Ci = - ( deta_dz + dpsi_dz + dgamm_dz );
            break;

        case (1):
			*Ai =   ( dpsi_dx );
			*Bi =   ( dpsi_dy );
			*Ci =   ( dpsi_dz );
            break;

        case (2):
			*Ai =   ( deta_dx );
			*Bi =   ( deta_dy );
			*Ci =   ( deta_dz );
            break;

        case (3):
			*Ai =   ( dgamm_dx );
			*Bi =   ( dgamm_dy );
			*Ci =   ( dgamm_dz );
            break;

    }


}


void kTetr( double x1, double y1, double z1,
		    double x2, double y2, double z2,
		    double x3, double y3, double z3,
		    double x4, double y4, double z4,
		    fmatrix_t k_lambda[4][4], fmatrix_t k_mu[4][4]  )
{

    int i, j; /* indices of the block matrices, i for rows, j for columns */
    int k, l; /* indices of 3 x 3 matrices, k for rows, l for columns     */

    double     Ai, Bi, Ci, Aj, Bj, Cj;
    double     Diag;
	vector3D_t Vi, Vj;

    fmatrix_t *matPtr;
    fmatrix_t *matPtr2;

    for (i = 0 ;  i < 4; i++) {

    	get_AiBiCi ( x1, y1, z1,
    			     x2, y2, z2,
    			     x3, y3, z3,
    				 x4, y4, z4,
    	             i,
    				 &Ai, &Bi, &Ci );
    	Vi.x[0] = Ai;
    	Vi.x[1] = Bi;
    	Vi.x[2] = Ci;

    	for (j = 0 ; j < 4; j++){

        	get_AiBiCi ( x1, y1, z1,
        			     x2, y2, z2,
        				 x3, y3, z3,
        				 x4, y4, z4,
        	             j,
        				 &Aj, &Bj, &Cj );
        	Vj.x[0] = Aj;
        	Vj.x[1] = Bj;
        	Vj.x[2] = Cj;

    		matPtr   = &k_lambda[i][j];
    		matPtr2  = &k_mu[i][j];

    		for (k = 0; k < 3; k++) {

    			for (l = 0; l < 3; l++) {

    				matPtr->f[k][l]  = Vi.x[k] * Vj.x[l];
    				matPtr2->f[l][k] = matPtr->f[k][l];

    			} /* for all l */
    		} /* for all k */

    		Diag = ( Vi.x[0] * Vj.x[0] + Vi.x[1] * Vj.x[1] + Vi.x[2] * Vj.x[2] );
    		matPtr2->f[0][0] +=  Diag;
    		matPtr2->f[1][1] +=  Diag;
    		matPtr2->f[2][2] +=  Diag;

    	} /* for all j */
    } /* for all i */

}




void compute_KTetrah()
{

	double TETRACOOR, TETRACOORSYMM;
    double x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4;

    memset(mytheK1_Tet1, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_Tet1, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_Tet2, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_Tet2, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_Tet3, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_Tet3, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_Tet4, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_Tet4, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_Tet5, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_Tet5, 0, 4 * 4 * sizeof(fmatrix_t));

    memset(mytheK1_SymmTet1, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_SymmTet1, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_SymmTet2, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_SymmTet2, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_SymmTet3, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_SymmTet3, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_SymmTet4, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_SymmTet4, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK1_SymmTet5, 0, 4 * 4 * sizeof(fmatrix_t));
    memset(mytheK2_SymmTet5, 0, 4 * 4 * sizeof(fmatrix_t));


    /* compute: h^2 / ( lambda*Vol ) * K1 for first Tetrahedra ( Lambda dependent matrix)
     *          h^2 / ( mu*Vol     ) * K2 for first Tetrahedra ( Mu dependent matrix) */


    x1 = tetraCoor[0][0];
    y1 = tetraCoor[1][0];
    z1 = tetraCoor[2][0];

    x2 = tetraCoor[0][1];
    y2 = tetraCoor[1][1];
    z2 = tetraCoor[2][1];

    x3 = tetraCoor[0][2];
    y3 = tetraCoor[1][2];
    z3 = tetraCoor[2][2];

    x4 = tetraCoor[0][3];
    y4 = tetraCoor[1][3];
    z4 = tetraCoor[2][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_Tet1, mytheK2_Tet1 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for SECOND Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for SECOND Tetrahedra ( Mu dependent matrix) */


    x1 = tetraCoor[3][0];
    y1 = tetraCoor[4][0];
    z1 = tetraCoor[5][0];

    x2 = tetraCoor[3][1];
    y2 = tetraCoor[4][1];
    z2 = tetraCoor[5][1];

    x3 = tetraCoor[3][2];
    y3 = tetraCoor[4][2];
    z3 = tetraCoor[5][2];

    x4 = tetraCoor[3][3];
    y4 = tetraCoor[4][3];
    z4 = tetraCoor[5][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_Tet2, mytheK2_Tet2 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for THIRD Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for THIRD Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoor[6][0];
    y1 = tetraCoor[7][0];
    z1 = tetraCoor[8][0];

    x2 = tetraCoor[6][1];
    y2 = tetraCoor[7][1];
    z2 = tetraCoor[8][1];

    x3 = tetraCoor[6][2];
    y3 = tetraCoor[7][2];
    z3 = tetraCoor[8][2];

    x4 = tetraCoor[6][3];
    y4 = tetraCoor[7][3];
    z4 = tetraCoor[8][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_Tet3, mytheK2_Tet3 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for FOURTH Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for FOURTH Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoor[9][0];
    y1 = tetraCoor[10][0];
    z1 = tetraCoor[11][0];

    x2 = tetraCoor[9][1];
    y2 = tetraCoor[10][1];
    z2 = tetraCoor[11][1];

    x3 = tetraCoor[9][2];
    y3 = tetraCoor[10][2];
    z3 = tetraCoor[11][2];

    x4 = tetraCoor[9][3];
    y4 = tetraCoor[10][3];
    z4 = tetraCoor[11][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_Tet4, mytheK2_Tet4 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for FIFTH Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for FIFTH Tetrahedra ( Mu dependent matrix) */


    x1 = tetraCoor[12][0];
    y1 = tetraCoor[13][0];
    z1 = tetraCoor[14][0];

    x2 = tetraCoor[12][1];
    y2 = tetraCoor[13][1];
    z2 = tetraCoor[14][1];

    x3 = tetraCoor[12][2];
    y3 = tetraCoor[13][2];
    z3 = tetraCoor[14][2];

    x4 = tetraCoor[12][3];
    y4 = tetraCoor[13][3];
    z4 = tetraCoor[14][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_Tet5, mytheK2_Tet5 );

    /* =================================== */
    /*  Symmetric tetrahedra distribution  */
    /* =================================== */

    /* computes: h^2 / ( lambda*Vol ) * K1 for the FIRST Symmetric Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for the FIRST Symmetric Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoorSymm[0][0];
    y1 = tetraCoorSymm[1][0];
    z1 = tetraCoorSymm[2][0];

    x2 = tetraCoorSymm[0][1];
    y2 = tetraCoorSymm[1][1];
    z2 = tetraCoorSymm[2][1];

    x3 = tetraCoorSymm[0][2];
    y3 = tetraCoorSymm[1][2];
    z3 = tetraCoorSymm[2][2];

    x4 = tetraCoorSymm[0][3];
    y4 = tetraCoorSymm[1][3];
    z4 = tetraCoorSymm[2][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_SymmTet1, mytheK2_SymmTet1 );


    /* computes: h^2 / ( lambda*Vol ) * K1 for the SECOND Symmetric Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for the SECOND Symmetric Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoorSymm[3][0];
    y1 = tetraCoorSymm[4][0];
    z1 = tetraCoorSymm[5][0];

    x2 = tetraCoorSymm[3][1];
    y2 = tetraCoorSymm[4][1];
    z2 = tetraCoorSymm[5][1];

    x3 = tetraCoorSymm[3][2];
    y3 = tetraCoorSymm[4][2];
    z3 = tetraCoorSymm[5][2];

    x4 = tetraCoorSymm[3][3];
    y4 = tetraCoorSymm[4][3];
    z4 = tetraCoorSymm[5][3];


    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_SymmTet2, mytheK2_SymmTet2 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for the THIRD Symmetric Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for the THIRD Symmetric Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoorSymm[6][0];
    y1 = tetraCoorSymm[7][0];
    z1 = tetraCoorSymm[8][0];

    x2 = tetraCoorSymm[6][1];
    y2 = tetraCoorSymm[7][1];
    z2 = tetraCoorSymm[8][1];

    x3 = tetraCoorSymm[6][2];
    y3 = tetraCoorSymm[7][2];
    z3 = tetraCoorSymm[8][2];

    x4 = tetraCoorSymm[6][3];
    y4 = tetraCoorSymm[7][3];
    z4 = tetraCoorSymm[8][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_SymmTet3, mytheK2_SymmTet3 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for the FOURTH Symmetric Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for the FOURTH Symmetric Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoorSymm[9][0];
    y1 = tetraCoorSymm[10][0];
    z1 = tetraCoorSymm[11][0];

    x2 = tetraCoorSymm[9][1];
    y2 = tetraCoorSymm[10][1];
    z2 = tetraCoorSymm[11][1];

    x3 = tetraCoorSymm[9][2];
    y3 = tetraCoorSymm[10][2];
    z3 = tetraCoorSymm[11][2];

    x4 = tetraCoorSymm[9][3];
    y4 = tetraCoorSymm[10][3];
    z4 = tetraCoorSymm[11][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_SymmTet4, mytheK2_SymmTet4 );

    /* computes: h^2 / ( lambda*Vol ) * K1 for the FIFTH Symmetric Tetrahedra ( Lambda dependent matrix)
     *           h^2 / ( mu*Vol     ) * K2 for the FIFTH Symmetric Tetrahedra ( Mu dependent matrix) */

    x1 = tetraCoorSymm[12][0];
    y1 = tetraCoorSymm[13][0];
    z1 = tetraCoorSymm[14][0];

    x2 = tetraCoorSymm[12][1];
    y2 = tetraCoorSymm[13][1];
    z2 = tetraCoorSymm[14][1];

    x3 = tetraCoorSymm[12][2];
    y3 = tetraCoorSymm[13][2];
    z3 = tetraCoorSymm[14][2];

    x4 = tetraCoorSymm[12][3];
    y4 = tetraCoorSymm[13][3];
    z4 = tetraCoorSymm[14][3];

    kTetr( x1, y1, z1,
    	   x2, y2, z2,
    	   x3, y3, z3,
    	   x4, y4, z4,
    	   mytheK1_SymmTet5, mytheK2_SymmTet5 );

}



double magnitude ( vector3D_t v1 )
{
	return sqrt ( v1.x[0] * v1.x[0] + v1.x[1] * v1.x[1] + v1.x[2] * v1.x[2] );
}

void cross_product (vector3D_t v1, vector3D_t v2, vector3D_t *v3)
{
	v3->x[0] = -v1.x[1] * v2.x[2] + v2.x[1] * v1.x[2];
	v3->x[1] =  v1.x[0] * v2.x[2] - v2.x[0] * v1.x[2];
	v3->x[2] = -v1.x[0] * v2.x[1] + v2.x[0] * v1.x[1];

}


/* returns the zp coordinate of a point inside a plane using linear interp   */
double interp_z( double xp, double yp, double xo, double yo, double h, double zcoords[4] )
{
	double x1, y1, mag;
	vector3D_t V1, V2, V3, V4, N;

	V1.x[0] = h;
	V1.x[1] = 0;
	V1.x[2] = zcoords[3] - zcoords[0];

	V2.x[0] = 0;
	V2.x[1] = h;
	V2.x[2] = zcoords[1] - zcoords[0];

	V3.x[0] = h;
	V3.x[1] = h;
	V3.x[2] = zcoords[2] - zcoords[0];

	x1 = xp - xo;
	y1 = yp - yo;

	if ( x1 / y1 > 1) {
		cross_product ( V3, V1,  &V4);
	}
	else
	{
		cross_product ( V2, V3,  &V4);
	}

	mag     = magnitude(V4);
	N.x[0]  = V4.x[0] / mag;
	N.x[1]  = V4.x[1] / mag;
	N.x[2]  = V4.x[2] / mag;

	return zcoords[0] - ( N.x[0] * x1 + N.x[1] * y1) / N.x[2];

}

/* gives the z value of a topography point respect to the surface (Z global)*/
double point_elevation ( double xo, double yo, double esize, double So ) {

	int i, j;
	double xp, yp, x_o, y_o, remi, remj, mesh_cz[4], zp;

	xp = xo;
	yp = yo;

	remi = modf (xp  / So, &x_o );
	remj = modf (yp  / So, &y_o );

	i = x_o;
	j = y_o;

	if ( ( remi == 0 ) && ( remj == 0) )
		zp = ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] );
	else {
		mesh_cz[0] =  theTopoInfo [ np_ew * i + j ];
		mesh_cz[1] =  theTopoInfo [ np_ew * i + j + 1 ];
		mesh_cz[2] =  theTopoInfo [ np_ew * ( i + 1 ) + j + 1 ];
		mesh_cz[3] =  theTopoInfo [ np_ew * ( i + 1 ) + j ];

		zp = thebase_zcoord - interp_z( xp, yp, x_o*So, y_o*So, So, mesh_cz );
	}

	return zp;

}


void get_airprops_topo( octant_t *leaf, double ticksize,
                   edata_t *edata, etree_t *cvm )
{

    edata->Vs  =  1e10;
    /* Assign negative Vp to identify air octants */
    edata->Vp  = -1.0;
    /* Assign zero density */
    edata->rho  = 0.0;

    /* TODO: double check the next commented code. It is not clear its purpose. Dorian*/
    //    int    res;
    //    double x, y, z;
    //    double edgesize;
    //    double halfedge;

    //    cvmpayload_t props;

    //    edgesize = edata->edgesize;
//    halfedge = edgesize * 0.5;
//
//    x = ( leaf->lx * ticksize ) + halfedge;
//    y = ( leaf->ly * ticksize ) + halfedge;
//    z = ( leaf->lz * ticksize ) + halfedge;

    /* Get the Vs at that location on the surface */
//    if ( theEtreeType == FULL )
//		res = cvm_query( cvm, y, x, thebase_zcoord, &props );
//    else
//    	res = cvm_query( cvm, y, x, 0, &props );
//
//    if ( res != 0 ) {
//        return;
//        solver_abort ( __FUNCTION_NAME, "Error from cvm_query: ",
//                       "No properties at east = %f, north = %f", y, x);
//    }


//    if ( ( leaf->lz * ticksize + edgesize ) == thebase_zcoord )
//    	edata->Vs  = props.Vs;
//    else
//    	edata->Vs  = 1e10;


    /* Assign negative Vp to identify air octants */
//    edata->Vp  = -1;

    /* Assign zero density */
//    edata->rho = 0;

//    return;
}


void elem_corners_elev ( double xo, double yo, double esize, double So, double corner_elev[4] ) {

	int m, n, i, j;
	double xp, yp, x_o, y_o, remi, remj, mesh_cz[4];

	for ( m = 0; m < 2; ++m ) {

		xp = xo + m * esize;

		for ( n = 0; n < 2; ++n) {
			yp = yo + n * esize;

			remi = modf (xp  / So, &x_o );
			remj = modf (yp  / So, &y_o );

			i = x_o;
			j = y_o;


			if ( ( remi == 0 ) && ( remj == 0) )
				corner_elev[ 2 * m + n ] = thebase_zcoord - theTopoInfo [ np_ew * i + j ] ;
			else {
				mesh_cz[0] =  theTopoInfo [ np_ew * i + j ];
				mesh_cz[1] =  theTopoInfo [ np_ew * i + j + 1 ];
				mesh_cz[2] =  theTopoInfo [ np_ew * ( i + 1 ) + j + 1 ];
				mesh_cz[3] =  theTopoInfo [ np_ew * ( i + 1 ) + j ];

				corner_elev[ 2 * m + n ] = thebase_zcoord - interp_z( xp, yp, x_o*So, y_o*So, So, mesh_cz );
			}
		}
	}
}

/**
 * Return  1 if the element crosses the surface of the topography.
 * Return  0 if the element is a near topography air element.
 * Return -1 if the element is outside the 2nd topography surface
 * Return  2 if the element is completely inside the topography
 * Note:  X and Y topofile follows the conventional coordinate system this is X=EW, Y=NS     */
int topo_search ( octant_t *leaf, double ticksize, edata_t *edata ) {

	int      m=0, n=0, i=0, j=0, io=0, jo=0, np=0, npx=0, npy=0;
    double   x_m, y_m, z_m, xo, yo, remi, remj, npd, remnp;
    double   esize, oct_topoz[4], sec_surfact;

    sec_surfact = 1.0;           /* Defines the distance from the real surface to the secondary surface  */
    x_m = leaf->lx * ticksize;
    y_m = leaf->ly * ticksize;
    z_m = leaf->lz * ticksize;

    esize  = (double)edata->edgesize;
    double emin  =  50;       /* emin defines the 2nd surface topography used to identify near air elements   */
    double romin =  50;       /* maximum radius for checking the neighborhood of a node   */


//   get topography values at the corners of the element
	elem_corners_elev ( x_m, y_m, esize, So, oct_topoz );

/* Checking for air elements */
    if ( ( oct_topoz[0] >= ( z_m + esize ) ) &&
    	 ( oct_topoz[1] >= ( z_m + esize ) ) &&
    	 ( oct_topoz[2] >= ( z_m + esize ) ) &&
    	 ( oct_topoz[3] >= ( z_m + esize ) ) )   {

		/* Check for surface nodes inside the octant. If so, return 1 if z node < z_m     */
		remi  = modf ( x_m / So, &xo );
		remj  = modf ( y_m / So, &yo );
    	io  = xo;
    	jo  = yo;
		if ( remi != 0 )
			io = io + 1;
		if ( remj != 0 )
			jo = jo + 1;

		remnp = modf ( ( x_m + esize ) / So, &npd );
		npx = npd + 1;
		if ( remi == 0 )
			npx = npx - xo;
		else
			npx = npx - xo - 1;

		remnp = modf ( ( y_m + esize ) / So, &npd );
		npy = npd + 1;
		if ( remj == 0)
			npy = npy - yo;
		else
			npy = npy - yo - 1;

		for ( m = 0; m < npx; ++m ) {
			i = io + m;

			for ( n = 0; n < npy; ++n ) {
				j = jo + n;

				if ( ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] > z_m ) && ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] < z_m + esize ) )
					return 1;  /* Octant with internal topography nodes */
			}
		}


    //Checking for octants inside the imaginary topographic strip
		if ( ( oct_topoz[0] -  emin >= ( z_m + esize ) )   &&
			 ( oct_topoz[1] -  emin >= ( z_m + esize ) )   &&
			 ( oct_topoz[2] -  emin >= ( z_m + esize ) )   &&
			 ( oct_topoz[3] -  emin >= ( z_m + esize ) ) )
			return -1;       /* Element outside topography -- far away topography air element */
		else {

	//		Checking for flat surface. If all the nodes in the neighboring area of the octant have
	//		the same z value then the octant will be considered as part of a flat surface
			remi  = modf ( ( x_m + esize / 2 ) / So, &xo );
			remj  = modf ( ( y_m + esize / 2 ) / So, &yo );
	    	io  = xo;
	    	jo  = yo;
			if ( remi != 0 )
				io = io + 1;
			if ( remj != 0 )
				jo = jo + 1;

			np = romin / So;
			io = io - np;
			jo = jo - np;
			if ( io < 0 )
			io = 0;
			if ( jo < 0 )
			jo = 0;


			if (  ( oct_topoz[0]   == oct_topoz[1] )  &&
				  ( oct_topoz[1]   == oct_topoz[2] )  &&
				  ( oct_topoz[2]   == oct_topoz[3] )  ) {

				for ( m = 0; m < 2 * np; ++m ) {
					i = io + m;

					for ( n = 0; n < 2 * np; ++n ) {
						j = jo + n;

						if ( ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] != oct_topoz[0] ) )
							return 0;  /* Element outside topography -- near topography air element */
					}
				}
				return -1;           /* Air element in flat surface*/
			}
		}

		return 0;            /* Element outside topography -- near topography air element */
    }


    /* Checking for internal octants, i.e. Elements inside the mountain */
    if ( ( oct_topoz[0] <= z_m ) &&
    	 ( oct_topoz[1] <= z_m ) &&
    	 ( oct_topoz[2] <= z_m ) &&
    	 ( oct_topoz[3] <= z_m ) ) {

		/* Check for surface nodes inside the octant. If so, return 1 if z node < z_m     */
		remi  = modf ( x_m / So, &xo );
		remj  = modf ( y_m / So, &yo );
    	io  = xo;
    	jo  = yo;
		if ( remi != 0 )
			io = io + 1;
		if ( remj != 0 )
			jo = jo + 1;

		remnp = modf ( ( x_m + esize ) / So, &npd );
		npx = npd + 1;
		if ( remi == 0)
			npx = npx - xo;
		else
			npx = npx - xo - 1;

		remnp = modf ( ( y_m + esize ) / So, &npd );
		npy = npd + 1;
		if ( remj == 0)
			npy = npy - yo;
		else
			npy = npy - yo - 1;


		for ( m = 0; m < npx; ++m ) {
			i = io + m;

			for ( n = 0; n < npy; ++n ) {
				j = jo + n;

				if ( ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] > z_m ) && ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] < z_m + esize ) )
					return 1;
			}
		}
    		return 2; /* Element inside topography -- normal element */

    }

    return 1;

}

/* Returns the real volume held by the Tetrahedral elements */
void TetraHVol ( double xo, double yo, double zo, double esize, double So,
		         double VolTetr[5]) {

	int i, j, k, m;
	double GP;
	double TETRACOOR, TETRACOORSYMM;
	double lx, ly, lz, qpx, qpy, qpz;
	double xm, ym, zm, zt;


	/* point coordinates for the five internal tetrahedral */
	double MCx[5][4], MCy[5][4], MCz[5][4];


	if ( ( ( xo <  theDomainLong_ns / 2.0  ) && ( yo <  theDomainLong_ew / 2.0 ) ) ||
	     ( ( xo >= theDomainLong_ns / 2.0 ) && ( yo >= theDomainLong_ew / 2.0 ) ) )
	{

		for ( i = 0; i < 5; ++i ) {

			for ( j = 0; j < 4; ++j ) {
				MCx[i][j] = xo + esize * tetraCoor[ 3 * i ][j];
				MCy[i][j] = yo + esize * tetraCoor[ 3 * i + 1 ][j];
				MCz[i][j] = zo + esize * tetraCoor[ 3 * i + 2 ][j];
			}

		}

	} else {

		for ( i = 0; i < 5; ++i ) {

			for ( j = 0; j < 4; ++j ) {
				MCx[i][j] = xo + esize * tetraCoorSymm[ 3 * i ][j];
				MCy[i][j] = yo + esize * tetraCoorSymm[ 3 * i + 1 ][j];
				MCz[i][j] = zo + esize * tetraCoorSymm[ 3 * i + 2 ][j];
			}

		}

	}


	double x21, x31, x41;
	double y21, y31, y41;
	double z21, z31, z41;
	double Vpr, Vo;

	double r, s, t;     /* natural coordinates of the tetrahedron with sides = 2   */
	double eta,psi,gam; /* natural coordinates of the tetrahedron with sides = 1  */

	for ( m = 0; m < 5; ++m ) {

		x21 = MCx[m][1] - MCx[m][0];
		x31 = MCx[m][2] - MCx[m][0];
		x41 = MCx[m][3] - MCx[m][0];

		y21 = MCy[m][1] - MCy[m][0];
		y31 = MCy[m][2] - MCy[m][0];
		y41 = MCy[m][3] - MCy[m][0];

		z21 = MCz[m][1] - MCz[m][0];
		z31 = MCz[m][2] - MCz[m][0];
		z41 = MCz[m][3] - MCz[m][0];

		Vo =   x31 * y21 * z41 + x41 * y31 * z21 + z31 * x21 * y41
		   - ( x31 * y41 * z21 + y31 * x21 * z41 + z31 * x41 * y21 );

		Vpr = 0.0;

		for ( i = 0; i < 10; ++i ) {
			lx  = gp[0][44 + i];
			qpx = gp[1][44 + i];

			for ( j = 0; j < 10; ++j ) {
				ly  = gp[0][44 + j];
				qpy = gp[1][44 + j];

				for ( k = 0; k < 10; ++k ) {
					lz  = gp[0][44 + k];
					qpz = gp[1][44 + k];

					/* Map cube into tetrahedron of side 2 */
	                r = ( 1 + lx ) * ( 1 - lz ) * ( 1 - ly ) / 4;
	                s = ( 1 + ly ) * ( 1 - lz ) / 2;
	                t = lz + 1;

	                /* Map tetrahedron of side 2 into tetrahedron of size 1 */
	                eta = s / 2;
	                psi = r / 2;
	                gam = t / 2;

	                /* get real coord of Gauss point */
	                xm = ( 1 - eta - psi - gam ) * MCx[m][0] + psi * MCx[m][1] + eta * MCx[m][2] + gam * MCx[m][3];
	                ym = ( 1 - eta - psi - gam ) * MCy[m][0] + psi * MCy[m][1] + eta * MCy[m][2] + gam * MCy[m][3];
	                zm = ( 1 - eta - psi - gam ) * MCz[m][0] + psi * MCz[m][1] + eta * MCz[m][2] + gam * MCz[m][3];

					zt = point_elevation ( xm, ym, esize, So );

					if ( zm >= zt ) {
						Vpr +=  ( 1 - lz ) * ( 1 - lz ) * ( 1 - ly ) * qpx * qpy * qpz / 8;
					}
				}
			}
		}

		VolTetr[m] = 3.0 / 4.0 * Vpr; /* percentage of the tetrahedron's volume filled by topography  */

	} /* for each tetrahedra */

}


double interp_lin ( double xi, double yi, double xf, double yf, double xo  ){
	return ( yi + ( yf - yi ) / ( xf - xi ) * xo );
}


/*  Checks the internal topography nodes in a horizontal plane. */
/*  Returns +1 if a node is above the plane, 0 otherwise */
int internal_nodes ( double xo, double yo, double zo, double So, double esize ) {

	double x1, y1, remi, remj, npd, remnp;
	int    i_o, j_o, npx, npy;
	int    m, n, i, j;

	remi   = modf ( xo / So, &x1 );
	remj   = modf ( yo / So, &y1 );
	i_o  = x1 + 1;
	j_o  = y1 + 1;

	remnp = modf ( ( xo + esize ) / So, &npd );
	if ( remnp == 0 )
		npx = npd - i_o;
	else
		npx = npd - i_o + 1;

	remnp = modf ( ( yo + esize ) / So, &npd );
	if ( remnp == 0 )
		npy = npd - j_o;
	else
		npy = npd - j_o + 1;

	for ( m = 0; m < npx; ++m ) {
		i = i_o + m;

		for ( n = 0; n < npy; ++n ) {
			j = j_o + n;

			if ( ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] > zo ) )
				return 0;

//			if ( ( thebase_zcoord - theTopoInfo [ np_ew * i + j ] > zo ) )
//				return -1;
		}
	}
	return 1;
}

/* returns 1 if a crossing exits, 0 otherwise. Face1 has a normal vector towards -X, Face2 towards +X   */
int crossings_faces_12 ( double xp, double yp, double zp, double So, double esize, double zini, double zfin) {

	double x1, y1, y2, remi, remj, remj_f, xo, yo, dx, dy_i, dy_f;
	int    i_o, j_o, j_f, n, j, cnt=0, cnt2=0;
	double zint_1, zint_d, yi, yf;
	int    n_vertcuts, ndiag;

	remi   = modf ( xp / So, &x1 );
	remj   = modf ( yp / So, &y1 );
	i_o    = x1;  /* X origin  */
	j_o    = y1;  /* Y origin */
	xo     = i_o * So;
	yo     = j_o * So;

	if ( remj != 0)
		j_o = j_o + 1;  /* shifts Y origin to the right if rem exists */

	remj_f = modf ( ( yp + esize ) / So, &y2 );
	j_f    = y2; /* Y end */

	n_vertcuts   = j_f - j_o + 1;
	ndiag        = n_vertcuts - 1;
	dx           = xp - xo;
	dy_i         = yp - j_o * So;
	dy_f         = yp + esize - j_f * So;
    cnt2         = n_vertcuts + ndiag + 2;

    if ( ( dx / dy_i >= 1.0 ) && ( remj != 0 ) && ( dx != 0 ) )
    	++cnt2;

    if ( ( dx / dy_f <= 1.0 ) && ( remj_f != 0 ) && ( dx != 0 ) )
    	++cnt2;

	/* checking for element inside topo grid, i.e. Small element */
	if ( ( n_vertcuts == 0 ) || ( ( n_vertcuts == 1 ) && ( ( remj == 0 ) || (remj_f == 0) ) ) ) {
		if  ( ( ( dx / dy_i < 1 ) && ( dx / dy_f < 1 ) )  ||
				( ( dx / dy_i > 1 ) && ( dx / dy_f > 1 ) ) ) {
			return 0; /* element inside topo grid without crossing its diagonal  */
		} else {
			j = y1;

			yi = theTopoInfo [ np_ew * i_o + j ];
			yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j + 1 ];
			zint_d = interp_lin ( 0., yi, So * sqrt (2.0), yf, dx * sqrt (2.0) );

			if ( ( ( thebase_zcoord - zint_d > zp ) && ( thebase_zcoord - zint_d < zp + esize ) ) )
				return 1;
		}
	}

	double crossings[cnt2];

	for ( n = 0; n < n_vertcuts; ++n ) {
		j = j_o + n;

		yi = theTopoInfo [ np_ew * i_o + j ];
		yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j ];
        zint_1 = interp_lin ( 0., yi, So, yf, dx );

        crossings[cnt] =  thebase_zcoord - zint_1;
        ++cnt;
	}

	for ( n = 0; n < ndiag; ++n ) {
		j = j_o + n;

		yi = theTopoInfo [ np_ew * i_o + j ];
		yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j + 1 ];
		zint_d = interp_lin ( 0., yi, So * sqrt (2.0), yf, dx * sqrt (2.0) );

        crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;
	}

	j_o    = y1;  /* recover the original Y origin position */

	/* checking  crossing through last diagonal */
	if ( ( dx / dy_f <= 1.0 ) && ( remj_f != 0 ) && ( dx != 0 ) ) {
		j = j_o + n_vertcuts; /* from the last loop */
		if ( remj == 0 )
			--j;

		yi = theTopoInfo [ np_ew * i_o + j ];
		yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j + 1 ];
		zint_d = interp_lin ( 0, yi, So * sqrt (2.0), yf, dx * sqrt (2.0) );

        crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;
	}

	/* checking  crossing through first diagonal */
	if ( ( dx / dy_i >= 1.0 ) && ( remj != 0) && ( dx != 0 )  ) {
		j  = j_o;
		yi = theTopoInfo [ np_ew * i_o + j ];
		yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j + 1 ];
		zint_d = interp_lin ( 0, yi, So * sqrt (2.0), yf, dx * sqrt (2.0) );

        crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;
	}

	crossings[cnt]     = zini;
	++cnt;
	crossings[cnt]     = zfin;
	++cnt;

	// check for internal crossings
	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] > zp ) && ( crossings[n] < zp + esize ) )
			return 1;
	}

	int sup_nodes = 0, inf_nodes = 0;

	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] <= zp ) )
			++sup_nodes;
	}

	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] >= zp + esize ) )
			++inf_nodes;
	}

	if ( ( sup_nodes == cnt ) || ( inf_nodes == cnt ) )
		return 0;

	return 1;

}


/* returns 1 if a crossing exits, 0 otherwise. Face3 has a normal vector towards -Y, Face4 towards +Y   */
int crossings_faces_34 ( double xp, double yp, double zp, double So, double esize, double zini, double zfin) {

	double x1, y1, x2, remi, remj, remi_f, xo, yo, dy, dx_i, dx_f;
	int    i_o, j_o, i_f, n, i, j, cnt=0, cnt2=0;
	double zint_1, zint_d, yi, yf;
	int    n_vertcuts, ndiag;

	remi   = modf ( xp / So, &x1 );
	remj   = modf ( yp / So, &y1 );
	i_o    = x1;  /* X origin  */
	j_o    = y1;  /* Y origin */
	xo     = i_o * So;
	yo     = j_o * So;

	if ( remi != 0)
		i_o = i_o + 1;  /* shifts X origin to the right if rem exists */

	remi_f = modf ( ( xp + esize ) / So, &x2 );
	i_f    = x2; /* X end */

	n_vertcuts   = i_f - i_o + 1;
	ndiag        = n_vertcuts - 1;
	dy           = yp - yo;
	dx_i         = xp - i_o * So;
	dx_f         = xp + esize - i_f * So;
	cnt2         = n_vertcuts + ndiag + 2;

	if ( ( dx_f / dy >= 1.0 ) && ( remi_f != 0 ) && ( dy != 0 ) )
		++cnt2;

	if ( ( dx_i / dy <= 1.0 ) && ( remi != 0) && ( dy != 0 ) )
		++cnt2;

	/* checking for element inside topo grid */
	if ( ( n_vertcuts == 0 ) || ( ( n_vertcuts == 1 ) && ( ( remi == 0 ) || (remi_f == 0) ) ) ) {
		if  ( ( ( dx_f / dy < 1 ) && ( dx_i / dy < 1 ) )  ||
			  ( ( dx_f / dy > 1 ) && ( dx_i / dy > 1 ) ) ) {
			return 0; /* element inside topo grid without crossing its diagonal  */
		} else {
			j   = j_o;
			i_o = x1;

			yi = theTopoInfo [ np_ew * i_o + j ];
			yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j + 1 ];
			zint_d = interp_lin ( 0., yi, So * sqrt (2.0), yf, dy * sqrt (2.0) );

			if ( ( ( thebase_zcoord - zint_d > zp ) && ( thebase_zcoord - zint_d < zp + esize ) ) )
				return 1;

		}
	}

	double crossings[cnt2];

	for ( n = 0; n < n_vertcuts; ++n ) {
		i = i_o + n;

		yi = theTopoInfo [ np_ew * i + j_o ];
		yf = theTopoInfo [ np_ew * i + j_o + 1 ];
        zint_1 = interp_lin ( 0., yi, So, yf, dy );

        crossings[cnt] =  thebase_zcoord - zint_1;
        ++cnt;

	}

	for ( n = 0; n < ndiag; ++n ) {
		i = i_o + n;

		yi = theTopoInfo [ np_ew * i + j_o ];
		yf = theTopoInfo [ np_ew * ( i + 1 ) + j_o + 1 ];
		zint_d = interp_lin ( 0., yi, So * sqrt (2.0), yf, dy * sqrt (2.0) );

        crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;

	}

	i_o    = x1;  /* recover the original X origin position */


	/* checking  crossing through last diagonal */
	if ( ( dx_f / dy >= 1.0 ) && ( remi_f != 0 ) && ( dy != 0 ) ) {
		i = i_o + n_vertcuts;
		if ( remi == 0 )
			--i;

		yi = theTopoInfo [ np_ew * i + j_o ];
		yf = theTopoInfo [ np_ew * ( i + 1 ) + j_o + 1 ];
		zint_d = interp_lin ( 0, yi, So * sqrt (2.0), yf, dy * sqrt (2.0) );

		crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;

	}

	/* checking  crossing through first diagonal */
	if ( ( dx_i / dy <= 1.0 ) && ( remi != 0) && ( dy != 0 ) ) {

		yi = theTopoInfo [ np_ew * i_o + j_o ];
		yf = theTopoInfo [ np_ew * ( i_o + 1 ) + j_o + 1 ];
		zint_d = interp_lin ( 0, yi, So * sqrt (2.0), yf, dy * sqrt (2.0) );

        crossings[cnt] =  thebase_zcoord - zint_d;
        ++cnt;
	}

	crossings[cnt]     = zini;
	++cnt;
	crossings[cnt]     = zfin;
	++cnt;

	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] > zp ) && ( crossings[n] < zp + esize ) )
			return 1;
	}

	int sup_nodes = 0, inf_nodes = 0;

	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] <= zp ) )
			++sup_nodes;
	}

	for ( n = 0; n < cnt; ++n ) {
		if  ( ( crossings[n] >= zp + esize ) )
			++inf_nodes;
	}

	if ( ( sup_nodes == cnt ) || ( inf_nodes == cnt ) )
		return 0;

	return 1;

}

int check_crossings ( double xo, double yo, double zo, double esize, double So ) {

	double elem_corners[4];
	int sup_nodes = 0, inf_nodes = 0, n;
	elem_corners_elev ( xo, yo, esize, So, elem_corners );

	for ( n = 0; n < 4; ++n ) {
		if  ( ( elem_corners[n] <= zo ) )
			++sup_nodes;
	}

	for ( n = 0; n < 4; ++n ) {
		if  ( ( elem_corners[n] >= zo + esize ) )
			++inf_nodes;
	}

/* 0) checking for mixed elevations in element */
	if ( ( sup_nodes < 4 ) && ( inf_nodes < 4 ) )
		return 1;

/* 1) checking elevation of corners  */
	    if ( ( ( elem_corners[0] > zo ) && ( elem_corners[0] < ( zo + esize ) ) ) ||
	    	 ( ( elem_corners[1] > zo ) && ( elem_corners[1] < ( zo + esize ) ) ) ||
	    	 ( ( elem_corners[2] > zo ) && ( elem_corners[2] < ( zo + esize ) ) ) ||
	    	 ( ( elem_corners[3] > zo ) && ( elem_corners[3] < ( zo + esize ) ) ) )   {
	    	return 1;
	    }

// 2) checking for air elements with topo nodes inside.
	    if (    ( elem_corners[0] > zo + esize ) &&
	    		( elem_corners[1] > zo + esize ) &&
	    		( elem_corners[2] > zo + esize ) &&
	    		( elem_corners[3] > zo + esize ) )   {

	    	if ( internal_nodes ( xo, yo, zo + esize, So, esize ) == 1 )
	    		return 1;

	    	/* 2.1) checking for crossings at the lateral faces:*/
	    	/*face 1, normal vector -X*/
	    	/*face 2, normal vector +X*/
	    	/*face 3, normal vector -Y*/
	    	/*face 4, normal vector +Y*/
	    	if (    crossings_faces_12 ( xo, yo, zo, So, esize, elem_corners[0], elem_corners[1])         ||
	    			crossings_faces_12 ( xo + esize, yo, zo, So, esize, elem_corners[3], elem_corners[2]) ||
	    			crossings_faces_34 ( xo, yo, zo, So, esize, elem_corners[0], elem_corners[3])         ||
	    			crossings_faces_34 ( xo, yo + esize, zo, So, esize, elem_corners[1], elem_corners[2])  )
	    		return 1;

	    	return 0;
	    }

// 3) checking for internal elements with topo nodes inside.
	    if (    ( elem_corners[0] <= zo ) &&
	    		( elem_corners[1] <= zo ) &&
	    		( elem_corners[2] <= zo ) &&
	    		( elem_corners[3] <= zo ) )   {

	    	if ( internal_nodes ( xo, yo, zo, So, esize ) == 0 )
	    		return 1;


	    	/* 3.1) checking for crossings at the lateral faces:*/
	    	/*face 1, normal vector -X*/
	    	/*face 2, normal vector +X*/
	    	/*face 3, normal vector -Y*/
	    	/*face 4, normal vector +Y*/
	    	if (    crossings_faces_12 ( xo, yo, zo, So, esize, elem_corners[0], elem_corners[1])         ||
	    			crossings_faces_12 ( xo + esize, yo, zo, So, esize, elem_corners[3], elem_corners[2]) ||
	    			crossings_faces_34 ( xo, yo, zo, So, esize, elem_corners[0], elem_corners[3])         ||
	    			crossings_faces_34 ( xo, yo + esize, zo, So, esize, elem_corners[1], elem_corners[2])  )
	    		return 1;

	    	return 0;

	    }

/* 4) none of the above, non a superficial element  */
	return 0;
}

int topo_setrec ( octant_t *leaf, double ticksize,
                   edata_t *edata, etree_t *cvm )
{
    int          res;

    res = topo_search( leaf, ticksize, edata );
    if ( ( res == 0 ) || ( res == -1 ) ) {
        get_airprops_topo( leaf, ticksize, edata, cvm);
        return 1;
    }

    return 0;
}

/**
 * Return  1 if an element is in the topography and needs to be refined,
 * Return  0 if an element is air element and does not need to be refined,
 * Return -1 if an element is not a topography element.
 */
int topo_toexpand (  octant_t *leaf,
                     double    ticksize,
                     edata_t  *edata )
{
    int          res;

    res = topo_search( leaf, ticksize, edata );
	if ( ( ( res == 1) || ( res == 0 ) ) && ( leaf->level < theMaxoctlevel ) )
	return 1;


    return -1;

}

int32_t
topography_initparameters ( const char *parametersin )
{
    FILE                *fp;
    FILE                *fp_topo;
    int                 iTopo, Maxoctlevel;
    char                topo_dir[256];
    char                topo_file[256];
    double              L_ew, L_ns, int_np_ew, int_np_ns, fract_np_ew, fract_np_ns;
    char                etree_model[64], fem_meth[64];
    etreetype_t         etreetype;
    topometh_t          topo_method;

    /* Opens parametersin file */

    if ( ( fp = fopen(parametersin, "r" ) ) == NULL ) {
        fprintf( stderr,
                 "Error opening %s\n at topography_initparameters",
                 parametersin );
        return -1;
    }

    /* Parses parametersin to capture topography single-value parameters */

    if ( ( parsetext(fp, "maximum_octant_level",    'i', &Maxoctlevel           ) != 0) ||
         ( parsetext(fp, "computation_method",      's', &fem_meth              ) != 0) ||
         ( parsetext(fp, "topographybase_zcoord",   'd', &thebase_zcoord        ) != 0) ||
         ( parsetext(fp, "topoprahy_directory",     's', &topo_dir              ) != 0) ||
         ( parsetext(fp, "region_length_east_m",    'd', &L_ew                  ) != 0) ||
         ( parsetext(fp, "type_of_etree",           's', &etree_model           ) != 0) ||
         ( parsetext(fp, "region_length_north_m",   'd', &L_ns                  ) != 0) )
    {
        fprintf( stderr,
                 "Error parsing topography parameters from %s\n",
                 parametersin );
        return -1;
    }


    if ( strcasecmp(etree_model, "full") == 0 ) {
        etreetype = FULL;
    } else if ( strcasecmp(etree_model, "flat") == 0 ) {
        etreetype = FLAT;
    } else {
        fprintf(stderr,
                "Illegal etree_type model for topography analysis"
                "(Flat, Full): %s\n", etree_model);
        return -1;
    }

    if ( strcasecmp(fem_meth, "vt") == 0 ) {
        topo_method = VT;
    } else if ( strcasecmp(fem_meth, "fem") == 0 ) {
        topo_method = FEM;
    } else {
        fprintf(stderr,
                "Illegal computation_method for topography analysis"
                "(vt, fem): %s\n", fem_meth);
        return -1;
    }


    /* Performs sanity checks */
    if ( ( Maxoctlevel < 0 ) || ( Maxoctlevel > 30 ) ) {
        fprintf( stderr,
                 "Illegal maximum octant level for topography %d\n",
                 Maxoctlevel );
        return -1;
    }

    if ( ( thebase_zcoord <= 0 ) ) {
        fprintf( stderr,
                 "Illegal z coordinate for the base of the topography %f\n",
                 thebase_zcoord );
        return -1;
    }

    if ( ( L_ew <= 0 ) || ( L_ns <=0 ) ) {
        fprintf( stderr,
                 "Illegal domain's dimensions ew_long=%f ns_long=%f\n",
                 L_ew, L_ns );
        return -1;
    }


    /* Initialize the static global variables */
	theEtreeType        = etreetype;
	theMaxoctlevel      = Maxoctlevel;
	theTopoMethod       = topo_method;
	theDomainLong_ew    = L_ew;
	theDomainLong_ns    = L_ns;

    /* read topography info */
	sprintf( topo_file,"%s/topography.in", topo_dir );

	if ( ( fp_topo   = fopen ( topo_file ,   "r") ) == NULL ) {
	    fprintf(stderr, "Error opening topography file\n" );
	    return -1;
	}

	fscanf( fp_topo,   " %lf ", &So );

	fract_np_ew = modf ( L_ew / So, &int_np_ew );
	fract_np_ns = modf ( L_ns / So, &int_np_ns );

	if ( ( fract_np_ew != 0) || ( fract_np_ns != 0 ) ) {
	    fprintf(stderr, "Error opening topography file - NOT A REGULAR MESH \n" );
	    return -1;
	}

	np_ew              = ( L_ew / So + 1 );
	np_ns              = ( L_ns / So + 1 );
	ntp                = ( L_ew / So + 1 ) * ( L_ns / So + 1 );
	theTopoInfo        = (double*)malloc( sizeof(double) * ntp );


	if ( theTopoInfo           == NULL ) {
		fprintf( stderr, "Error allocating transient array for the topography data"
				"in topography_initparameters " );
		return -1;
	}

	for ( iTopo = 0; iTopo < ntp; ++iTopo) {

	    fscanf(fp_topo,   " %lf ", &(theTopoInfo[iTopo]));

	}

    fclose(fp);
    fclose(fp_topo);

    return 0;
}

void topo_init ( int32_t myID, const char *parametersin ) {

    int     int_message[6];
    double  double_message[4];

    /* Capturing data from file --- only done by PE0 */
    if (myID == 0) {
        if ( topography_initparameters( parametersin ) != 0 ) {
            fprintf(stderr,"Thread 0: topography_local_init: "
                    "topography_initparameters error\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }

    }

    /* Broadcasting data */

    double_message[0]    = thebase_zcoord;
    double_message[1]    = So;
    double_message[2]    = theDomainLong_ew;
    double_message[3]    = theDomainLong_ns;

    int_message   [0]    = theMaxoctlevel;
    int_message   [1]    = ntp;
    int_message   [2]    = np_ew;
    int_message   [3]    = np_ns;
    int_message   [4]    = (int)theEtreeType;
    int_message   [5]    = (int)theTopoMethod;


    MPI_Bcast(double_message, 4, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(int_message,    6, MPI_INT,    0, comm_solver);

    thebase_zcoord       = double_message[0];
    So				     = double_message[1];
    theDomainLong_ew     = double_message[2];
    theDomainLong_ns     = double_message[3];

    theMaxoctlevel       = int_message[0];
    ntp					 = int_message[1];
    np_ew			     = int_message[2];
    np_ns			     = int_message[3];
    theEtreeType         = int_message[4];
    theTopoMethod        = int_message[5];

//    /* allocate table of properties for all other PEs */

    if (myID != 0) {
        theTopoInfo        = (double*)malloc( sizeof(double) * ntp );
    }

    /* Broadcast table of properties */
    MPI_Bcast(theTopoInfo,   ntp, MPI_DOUBLE, 0, comm_solver);

    return;

}

/* returns nodal mass of topography elements */
void toponodes_mass(int32_t eindex, double nodes_mass[8], double M,
		            double xo, double yo, double zo) {

	int32_t myeindex, topo_eindex;
	int i, j, k;
	double MNOD, MNODSYMM;

	/* do not correct mass if the mass calculation method is none*/
	/* or if the element is an air element */
	if ( ( theTopoMethod == FEM ) || ( M == 0.0 ) ) {

		for (j = 0; j < 8; j++) {
			nodes_mass[j] = M;
		}
		return;
	}

	/* Tetrahedra local nodes IDs */
	int32_t M_Nodes[5][4];

	if ( ( ( xo <  theDomainLong_ns / 2.0  ) && ( yo <  theDomainLong_ew / 2.0 ) ) ||
	     ( ( xo >= theDomainLong_ns / 2.0 ) && ( yo >= theDomainLong_ew / 2.0 ) ) )
	{

		for ( i = 0; i < 5; ++i ) {
			for ( j = 0; j < 4; ++j ) {
				M_Nodes[i][j] = mnod[i][j];
			}
		}

	} else {

		for ( i = 0; i < 5; ++i ) {
			for ( j = 0; j < 4; ++j ) {
				M_Nodes[i][j] = mnodsymm[i][j];
			}
		}

	}

	/* end tetrahedra nodes */

	/* look for element in myTopolist elements */
	for (topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++) {

		topoconstants_t  *ecp;

		myeindex = myTopoElementsMapping[topo_eindex];

		if ( myeindex == eindex ) { /* element found */

			ecp    = myTopoSolver->topoconstants + topo_eindex;

			/* adjust mass from each tetrahedra  */
			double VTetr = ecp->h * ecp->h * ecp->h / 6.0; /* full tetrahedron volume */

			for ( k = 0; k < 5; k++ ) {

				if ( k == 4 )
					VTetr = 2.0 * VTetr;

				/* For each tetrahedron node */
				for (j = 0; j < 4; j++) {

					nodes_mass[ M_Nodes[k][j] ] += ecp->tetraVol[k] * VTetr * ecp->rho / 4.0;

				} /* end loop for each node */

			} /* end loop for each tetrahedra */

			return;
		}
	}

	/* Element not found in the topo list, must be a conventional element */
	for (j = 0; j < 8; j++) {
		nodes_mass[j] = M;
	}
	return;

}

void topography_elements_count(int32_t myID, mesh_t *myMesh ) {

    int32_t eindex;
    int32_t count         = 0;

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        elem_t     *elemp;
        edata_t    *edata;
        node_t     *node0dat;
        double      Vp, xo, yo, zo, esize, Vol;
        int32_t	    node0;

        elemp    = &myMesh->elemTable[eindex]; //Takes the information of the "eindex" element
        edata    = (edata_t *)elemp->data;
        node0    = elemp->lnid[0];             //Takes the ID for the zero node in element eindex
        node0dat = &myMesh->nodeTable[node0];

        /* get element Vp */
        Vp       = (double)edata->Vp;

        /* get coordinates of element zero node */
		xo = (node0dat->x)*(myMesh->ticksize);
		yo = (node0dat->y)*(myMesh->ticksize);
		zo = (node0dat->z)*(myMesh->ticksize);


        /* get element size */
		esize = edata->edgesize;
		Vol   = esize * esize *esize;

		if ( ( Vp != -1 ) ) {
			if ( ( check_crossings ( xo, yo, zo, esize, So ) == 1 ) ) {
				count++;
			}
		}
    }

    if ( count > myMesh-> lenum ) {
        fprintf(stderr,"Thread %d: topography_elements_count: "
                "more elements than expected\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    myTopoElementsCount          = count;

    return;
}

void topography_elements_mapping(int32_t myID, mesh_t *myMesh) {

	int32_t eindex;
	int32_t count = 0;

	XMALLOC_VAR_N(myTopoElementsMapping, int32_t, myTopoElementsCount);

	for (eindex = 0; eindex < myMesh->lenum; eindex++) {

		elem_t     *elemp;
		edata_t    *edata;
		node_t     *node0dat;
		double      Vp, xo, yo, zo, esize, Vol;
		int32_t	    node0;

		elemp    = &myMesh->elemTable[eindex]; //Takes the information of the "eindex" element
		edata    = (edata_t *)elemp->data;
		node0    = elemp->lnid[0];             //Takes the ID for the zero node in element eindex
		node0dat = &myMesh->nodeTable[node0];

		/* get element Vp */
		Vp       = (double)edata->Vp;

		/* get coordinates of element zero node */
		xo = (node0dat->x)*(myMesh->ticksize);
		yo = (node0dat->y)*(myMesh->ticksize);
		zo = (node0dat->z)*(myMesh->ticksize);


		/* get element size */
		esize = edata->edgesize;
		Vol   = esize * esize *esize;

		if ( ( Vp != -1 ) ) {

			if  ( check_crossings ( xo, yo, zo, esize, So ) == 1 ) {
				myTopoElementsMapping[count] = eindex;
				count++;
			}
		}
	}

	if ( count != myTopoElementsCount ) {
		fprintf(stderr,"Thread %d: topography_elements_mapping: "
				"more elements than expected\n", myID);
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	return;
}

/*
 * topo_solver_init: Initialize all the structures needed for topography
 *                        analysis and the material/element constants.
 */
void topo_solver_init(int32_t myID, mesh_t *myMesh) {

    int32_t eindex, topo_eindex;

    topography_elements_count   ( myID, myMesh );
    topography_elements_mapping ( myID, myMesh );

//    compute_KTetrah();

    fprintf(stdout, "TopoElem=%d, MyID=%d\n", myTopoElementsCount, myID);

    /* Memory allocation for mother structure */
    myTopoSolver = (toposolver_t *)malloc(sizeof(toposolver_t));

    if (myTopoSolver == NULL) {
        fprintf(stderr, "Thread %d: topography_init: out of memory\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    /* Memory allocation for internal structures */
    myTopoSolver->topoconstants =
        (topoconstants_t *)calloc(myTopoElementsCount, sizeof(topoconstants_t));

    if ( myTopoSolver->topoconstants      == NULL ) {
    	fprintf(stderr, "Thread %d: topography_init: out of memory\n", myID);
    	MPI_Abort(MPI_COMM_WORLD, ERROR);
    	exit(1);
    }

    /* Initialization of element constants */
    for (topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++) {

        elem_t           *elemp;
        edata_t          *edata;
        topoconstants_t  *ecp;
        node_t           *node0dat;
        int32_t	          node0;
        double            xo, yo, zo, esize;
        double            mu, lambda;

        eindex = myTopoElementsMapping[topo_eindex];
        elemp  = &myMesh->elemTable[eindex];
        edata  = (edata_t *)elemp->data;
        ecp    = myTopoSolver->topoconstants + topo_eindex;

        node0    = elemp->lnid[0];             //Takes the ID for the zero node in element eindex
        node0dat = &myMesh->nodeTable[node0];

        /* get coordinates of element zero node */
		xo = (node0dat->x)*(myMesh->ticksize);
		yo = (node0dat->y)*(myMesh->ticksize);
		zo = (node0dat->z)*(myMesh->ticksize);

        /* get full element volume */
		esize      = edata->edgesize;

        /* get element properties */
        mu_and_lambda(&mu, &lambda, edata, eindex);
        ecp->lambda = lambda;
        ecp->mu     = mu;
        ecp->rho    = edata->rho;
        ecp->h      = esize;

        /* get Tetrahedra volumes using 10 points Gauss integation */
        if ( theTopoMethod == VT )
        	TetraHVol ( xo, yo, zo, esize, So, ecp->tetraVol );

    } /* for all elements */

}

void compute_addforce_topoEffective ( mesh_t     *myMesh,
                                      mysolver_t *mySolver,
                                      double      theDeltaTSquared )
{

	int       i;
	int32_t   eindex;
	int32_t   topo_eindex;
	fvector_t localForce[8];
    fvector_t curDisp[8];

	if ( theTopoMethod == FEM )
		return;


	/* Loop on the number of elements */
	for (topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++) {


			elem_t           *elemp;
			edata_t          *edata;
			node_t           *node0dat;

			eindex = myTopoElementsMapping[topo_eindex];

			/* Capture the table of elements from the mesh and the size
			 * This is what gives me the connectivity to nodes */
			elemp                   = &myMesh->elemTable[eindex];
			edata                   = (edata_t *)elemp->data;
			topoconstants_t topo_ec = myTopoSolver->topoconstants[topo_eindex];
			node0dat                = &myMesh->nodeTable[elemp->lnid[0]];

	        /* get coordinates of element zero node */
			double xo = (node0dat->x)*(myMesh->ticksize);
			double yo = (node0dat->y)*(myMesh->ticksize);
			double zo = (node0dat->z)*(myMesh->ticksize);

			memset( localForce, 0, 8 * sizeof(fvector_t) );

			/* get cube's displacements */
	        for (i = 0; i < 8; i++) {
	            int32_t    lnid = elemp->lnid[i];
	            fvector_t* tm1Disp = mySolver->tm1 + lnid;

	            curDisp[i].f[0] = tm1Disp->f[0];
	            curDisp[i].f[1] = tm1Disp->f[1];
	            curDisp[i].f[2] = tm1Disp->f[2];
	        }


	        if (vector_is_zero( curDisp ) != 0)
	        	TetraForces( curDisp, localForce, topo_ec.tetraVol ,
	        			     edata, topo_ec.mu, topo_ec.lambda,
	        			     xo, yo, zo );

			/* Loop over the 8 element nodes:
			 * Add the contribution calculated above to the node
			 * forces carried from the source and stiffness.
			 */

			for (i = 0; i < 8; i++) {

				int32_t    lnid;
				fvector_t *nodalForce;

				lnid = elemp->lnid[i];

				nodalForce = mySolver->force + lnid;

				nodalForce->f[0] -= localForce[i].f[0] * theDeltaTSquared;
				nodalForce->f[1] -= localForce[i].f[1] * theDeltaTSquared;
				nodalForce->f[2] -= localForce[i].f[2] * theDeltaTSquared;

			} /* element nodes */

	}

	return;
}

/* -------------------------------------------------------------------------- */
/*                         Efficient Method Utilities                         */
/* -------------------------------------------------------------------------- */

void TetraForces( fvector_t* un, fvector_t* resVec, double tetraVol[5], edata_t *edata,
		          double mu, double lambda, double xo, double yo, double zo )
{

	int k;
    double prs;
    int32_t N0, N1, N2, N3;

	double VTetr = edata->edgesize * edata->edgesize * edata->edgesize / 6.0; /* full tetrahedron volume */

	/*  distribution for the first and third quadrants */
	if ( ( ( xo <  theDomainLong_ns / 2.0  ) && ( yo <  theDomainLong_ew / 2.0 ) ) ||
			( ( xo >= theDomainLong_ns / 2.0 ) && ( yo >= theDomainLong_ew / 2.0 ) ) )
	{


		for ( k = 0; k < 5; k++ ) { /* for each tetrahedron */

			if ( k == 4 )
				VTetr = 2.0 * VTetr;

			if ( tetraVol[k] != 0 ) {

				double topoC1 = tetraVol[k] * VTetr * lambda / ( edata->edgesize * edata->edgesize );
				double topoC2 = tetraVol[k] * VTetr * mu / ( edata->edgesize * edata->edgesize );

				switch ( k ) {

				case ( 0 ):
				N0 = 0;
				N1 = 2;
				N2 = 1;
				N3 = 4;

				prs = topoC1 * ( un[N0].f[0] + un[N0].f[1] + un[N0].f[2] - un[N1].f[1] - un[N2].f[0] - un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4. * un[N0].f[0] + un[N0].f[1] + un[N0].f[2] - un[N1].f[0] - 2. * un[N2].f[0] - un[N2].f[1] - un[N2].f[2] - un[N3].f[0] );
				resVec[N0].f[1] +=  prs + topoC2 * ( un[N0].f[0] + 4.  * un[N0].f[1] + un[N0].f[2] - un[N1].f[0] - 2.  * un[N1].f[1] - un[N1].f[2] - un[N2].f[1] - un[N3].f[1] );
				resVec[N0].f[2] +=  prs + topoC2 * ( un[N0].f[0] + un[N0].f[1] + 4. * un[N0].f[2] - un[N1].f[2] - un[N2].f[2] - un[N3].f[0] - un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[0] + un[N2].f[1] );
				resVec[N1].f[1] += -prs + topoC2 * ( -2. * ( un[N0].f[1] - un[N1].f[1] ) );
				resVec[N1].f[2] +=        topoC2 * ( -un[N0].f[1] - un[N0].f[2] + un[N1].f[2] + un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( -2. * ( un[N0].f[0] - un[N2].f[0] ) );
				resVec[N2].f[1] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[0] + un[N2].f[1] );
				resVec[N2].f[2] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[2] + un[N2].f[2] + un[N3].f[0]  );
				/* ================ */
				resVec[N3].f[0] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[2] + un[N2].f[2] + un[N3].f[0]  );
				resVec[N3].f[1] +=        topoC2 * ( -un[N0].f[1] - un[N0].f[2] + un[N1].f[2] + un[N3].f[1] );
				resVec[N3].f[2] += -prs + topoC2 * ( -2. * ( un[N0].f[2] - un[N3].f[2] ) );
				break;

				case ( 1 ):
				N0 = 3;
				N1 = 1;
				N2 = 2;
				N3 = 7;

				prs              = topoC1 * ( un[N0].f[0] + un[N0].f[1] - un[N0].f[2] - un[N1].f[1] - un[N2].f[0] + un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4. * un[N0].f[0] + un[N0].f[1] - un[N0].f[2] - un[N1].f[0] - 2. * un[N2].f[0] - un[N2].f[1] + un[N2].f[2] - un[N3].f[0] );
				resVec[N0].f[1] +=  prs + topoC2 * ( un[N0].f[0] + 4. * un[N0].f[1] - un[N0].f[2] - un[N1].f[0] - 2. * un[N1].f[1] + un[N1].f[2] - un[N2].f[1] - un[N3].f[1] );
				resVec[N0].f[2] += -prs + topoC2 * ( -un[N0].f[0] - un[N0].f[1] + 4. * un[N0].f[2] - un[N1].f[2] - un[N2].f[2] + un[N3].f[0] + un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[0] + un[N2].f[1] );
				resVec[N1].f[1] += -prs + topoC2 * ( -2. * ( un[N0].f[1] - un[N1].f[1] ) );
				resVec[N1].f[2] +=        topoC2 * ( un[N0].f[1] - un[N0].f[2] + un[N1].f[2] - un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( -2. * ( un[N0].f[0] - un[N2].f[0] ) );
				resVec[N2].f[1] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[0] + un[N2].f[1] );
				resVec[N2].f[2] +=        topoC2 * ( un[N0].f[0] - un[N0].f[2] + un[N2].f[2] - un[N3].f[0]  );
				/* ================ */
				resVec[N3].f[0] +=       -topoC2 * ( un[N0].f[0] - un[N0].f[2] + un[N2].f[2] - un[N3].f[0]  );
				resVec[N3].f[1] +=       -topoC2 * ( un[N0].f[1] - un[N0].f[2] + un[N1].f[2] - un[N3].f[1] );
				resVec[N3].f[2] +=  prs + topoC2 * ( -2. * ( un[N0].f[2] - un[N3].f[2] ) );
				break;

				case ( 2 ):
				N0 = 6;
				N1 = 4;
				N2 = 7;
				N3 = 2;

				prs              = topoC1 * ( un[N0].f[0] - un[N0].f[1] - un[N0].f[2] + un[N1].f[1] - un[N2].f[0] + un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4. * un[N0].f[0] - un[N0].f[1] - un[N0].f[2] - un[N1].f[0] - 2. * un[N2].f[0] + un[N2].f[1] + un[N2].f[2] - un[N3].f[0] );

				resVec[N0].f[1] += -prs + topoC2 * ( -un[N0].f[0] + 4. * un[N0].f[1] + un[N0].f[2] + un[N1].f[0] - 2. * un[N1].f[1] - un[N1].f[2] - un[N2].f[1] - un[N3].f[1] );
				resVec[N0].f[2] += -prs + topoC2 * ( -un[N0].f[0] + un[N0].f[1] + 4. * un[N0].f[2] - un[N1].f[2] - un[N2].f[2] + un[N3].f[0] - un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * ( -un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N2].f[1] );
				resVec[N1].f[1] +=  prs + topoC2 * ( -2. * ( un[N0].f[1] - un[N1].f[1] ) );
				resVec[N1].f[2] +=        topoC2 * ( -un[N0].f[1] - un[N0].f[2] + un[N1].f[2] + un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( -2. * ( un[N0].f[0] - un[N2].f[0] ) );
				resVec[N2].f[1] +=       -topoC2 * ( -un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N2].f[1] );
				resVec[N2].f[2] +=        topoC2 * (  un[N0].f[0] - un[N0].f[2] + un[N2].f[2] - un[N3].f[0]  );
				/* ================ */
				resVec[N3].f[0] +=       -topoC2 * ( un[N0].f[0] - un[N0].f[2] + un[N2].f[2] - un[N3].f[0]  );
				resVec[N3].f[1] +=        topoC2 * ( -un[N0].f[1] - un[N0].f[2] + un[N1].f[2] + un[N3].f[1] );
				resVec[N3].f[2] +=  prs + topoC2 * ( -2. * ( un[N0].f[2] - un[N3].f[2] ) );
				break;

				case ( 3 ):
				N0 = 5;
				N1 = 7;
				N2 = 4;
				N3 = 1;

				prs              = topoC1 * ( un[N0].f[0] - un[N0].f[1] + un[N0].f[2] + un[N1].f[1] - un[N2].f[0] - un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4. * un[N0].f[0] - un[N0].f[1] + un[N0].f[2] - un[N1].f[0] - 2. * un[N2].f[0] + un[N2].f[1] - un[N2].f[2] - un[N3].f[0] );
				resVec[N0].f[1] += -prs + topoC2 * ( -un[N0].f[0] + 4. * un[N0].f[1] - un[N0].f[2] + un[N1].f[0] - 2. * un[N1].f[1] + un[N1].f[2] - un[N2].f[1] - un[N3].f[1] );
				resVec[N0].f[2] +=  prs + topoC2 * (  un[N0].f[0] - un[N0].f[1] + 4. * un[N0].f[2] - un[N1].f[2] - un[N2].f[2] - un[N3].f[0] + un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * ( -un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N2].f[1] );
				resVec[N1].f[1] +=  prs + topoC2 * ( -2. * ( un[N0].f[1] - un[N1].f[1] ) );
				resVec[N1].f[2] +=        topoC2 * ( un[N0].f[1] - un[N0].f[2] + un[N1].f[2] - un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs - topoC2 * ( un[N0].f[0] - un[N2].f[0] ) * 2.0;
				resVec[N2].f[1] +=       -topoC2 * ( -un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N2].f[1] );
				resVec[N2].f[2] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[2] + un[N2].f[2] + un[N3].f[0] );
				/* ================ */
				resVec[N3].f[0] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[2] + un[N2].f[2] + un[N3].f[0] );
				resVec[N3].f[1] +=       -topoC2 * ( un[N0].f[1] - un[N0].f[2] + un[N1].f[2] - un[N3].f[1] );
				resVec[N3].f[2] += -prs + topoC2 * ( -2. * ( un[N0].f[2] - un[N3].f[2] ) );
				break;

				case ( 4 ):
				N0 = 2;
				N1 = 4;
				N2 = 7;
				N3 = 1;

				topoC1 = topoC1 / 4.;
				topoC2 = topoC2 / 4.;

				prs              = topoC1 * ( un[N0].f[0] - un[N0].f[1] + un[N0].f[2] + un[N1].f[0] + un[N1].f[1] - un[N1].f[2] - un[N2].f[0] - un[N2].f[1] - un[N2].f[2] - un[N3].f[0] + un[N3].f[1] + un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4. * un[N0].f[0] - un[N0].f[1] + un[N0].f[2] - un[N1].f[1] + un[N1].f[2] - 2. * un[N2].f[0] + un[N2].f[1] - un[N2].f[2] - 2. * un[N3].f[0] + un[N3].f[1] - un[N3].f[2] );
				resVec[N0].f[1] += -prs + topoC2 * ( -un[N0].f[0] + 4. * un[N0].f[1] - un[N0].f[2] + un[N1].f[0] - 2. * un[N1].f[1] + un[N1].f[2] - un[N2].f[0] - un[N2].f[2] + un[N3].f[0] - 2. * un[N3].f[1] + un[N3].f[2] );
				resVec[N0].f[2] +=  prs + topoC2 * ( un[N0].f[0] - un[N0].f[1] + 4. * un[N0].f[2] - un[N1].f[0] + un[N1].f[1] - 2. * un[N1].f[2] - un[N2].f[0] + un[N2].f[1] - 2. * un[N2].f[2] + un[N3].f[0] - un[N3].f[1] );
				/* ================ */
				resVec[N1].f[0] +=  prs + topoC2 * (  un[N0].f[1] - un[N0].f[2] + 4. * un[N1].f[0] + un[N1].f[1] - un[N1].f[2] - 2. * un[N2].f[0] - un[N2].f[1] + un[N2].f[2] - 2. * un[N3].f[0] - un[N3].f[1] + un[N3].f[2]  );
				resVec[N1].f[1] +=  prs + topoC2 * ( -un[N0].f[0] - 2. * un[N0].f[1] + un[N0].f[2] + un[N1].f[0] + 4. * un[N1].f[1] - un[N1].f[2] - un[N2].f[0] - 2. * un[N2].f[1] + un[N2].f[2] + un[N3].f[0] - un[N3].f[2]  );
				resVec[N1].f[2] += -prs + topoC2 * (  un[N0].f[0] + un[N0].f[1] - 2. * un[N0].f[2] - un[N1].f[0] - un[N1].f[1] + 4. * un[N1].f[2] - un[N2].f[0]  - un[N2].f[1] + un[N3].f[0] + un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 *  ( - 2. * un[N0].f[0] - un[N0].f[1] - un[N0].f[2] - 2. * un[N1].f[0] - un[N1].f[1] - un[N1].f[2] + 4. * un[N2].f[0] + un[N2].f[1] + un[N2].f[2] + un[N3].f[1] + un[N3].f[2] );
				resVec[N2].f[1] += -prs + topoC2 *  (  un[N0].f[0] + un[N0].f[2] - un[N1].f[0] - 2. * un[N1].f[1] - un[N1].f[2] + un[N2].f[0] + 4. * un[N2].f[1] + un[N2].f[2] - un[N3].f[0] - 2. * un[N3].f[1] - un[N3].f[2] );
				resVec[N2].f[2] += -prs + topoC2 *  ( -un[N0].f[0] - un[N0].f[1] - 2. * un[N0].f[2] + un[N1].f[0] + un[N1].f[1] + un[N2].f[0] + un[N2].f[1] + 4. * un[N2].f[2] - un[N3].f[0] - un[N3].f[1] - 2. * un[N3].f[2] );
				/* ================ */
				resVec[N3].f[0] += -prs + topoC2 *  ( - 2. * un[N0].f[0] + un[N0].f[1] + un[N0].f[2] - 2. * un[N1].f[0] + un[N1].f[1] + un[N1].f[2] - un[N2].f[1] - un[N2].f[2] + 4. * un[N3].f[0] - un[N3].f[1] - un[N3].f[2] );
				resVec[N3].f[1] +=  prs + topoC2 *  (  un[N0].f[0] - 2. * un[N0].f[1] - un[N0].f[2] - un[N1].f[0] + un[N1].f[2] + un[N2].f[0] - 2. * un[N2].f[1] - un[N2].f[2] - un[N3].f[0] + 4. * un[N3].f[1] + un[N3].f[2] );
				resVec[N3].f[2] +=  prs + topoC2 *  ( -un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N1].f[1] - 2.0 * un[N1].f[2] + un[N2].f[0] - un[N2].f[1] - 2.0 * un[N2].f[2] - un[N3].f[0] + un[N3].f[1] + 4.0 * un[N3].f[2] );
				break;
				}
			}
		}
	}  else  {

		/*  distribution for the second and fourth quadrants */
		for ( k = 0; k < 5; k++ ) { /* for each tetrahedron */

			if ( k == 4 )
				VTetr = 2.0 * VTetr;

			if ( tetraVol[k] != 0 ) {

				double topoC1 = tetraVol[k] * VTetr * lambda / ( edata->edgesize * edata->edgesize );
				double topoC2 = tetraVol[k] * VTetr * mu / ( edata->edgesize * edata->edgesize );

				switch ( k ) {

				case ( 0 ):
				N0 = 0;
				N1 = 3;
				N2 = 1;
				N3 = 5;

				prs              = topoC1 * ( un[N0].f[0] - un[N1].f[1] - un[N3].f[2] - un[N2].f[0] + un[N2].f[1] + un[N2].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * 2.0 * ( un[N0].f[0] - un[N2].f[0] );
				resVec[N0].f[1] +=        topoC2 * ( un[N0].f[1] - un[N1].f[0] + un[N2].f[0] - un[N2].f[1] );
				resVec[N0].f[2] +=        topoC2 * ( un[N0].f[2]  + un[N2].f[0] - un[N2].f[2] - un[N3].f[0] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * ( -un[N0].f[1] + un[N1].f[0] - un[N2].f[0] + un[N2].f[1] );
				resVec[N1].f[1] += -prs + topoC2 * 2.0 * ( un[N1].f[1] - un[N2].f[1] );
				resVec[N1].f[2] +=        topoC2 * ( un[N1].f[2] - un[N2].f[1] - un[N2].f[2] + un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( -2.0 * un[N0].f[0] + un[N0].f[1] + un[N0].f[2] - un[N1].f[0] + 4.0 * un[N2].f[0] - un[N2].f[1] - un[N2].f[2] - un[N3].f[0] );
				resVec[N2].f[1] +=  prs + topoC2 * ( -un[N0].f[1] + un[N1].f[0] - 2.0 * un[N1].f[1] - un[N1].f[2] - un[N2].f[0] + 4.0 * un[N2].f[1] + un[N2].f[2] - un[N3].f[1] );
				resVec[N2].f[2] +=  prs + topoC2 * ( -un[N0].f[2] - un[N1].f[2] - un[N2].f[0] + un[N2].f[1] + 4.0 * un[N2].f[2] + un[N3].f[0] - un[N3].f[1] - 2.0 * un[N3].f[2] );
				/* ================ */
				resVec[N3].f[0] +=        topoC2 * ( -un[N0].f[2] - un[N2].f[0] + un[N2].f[2] + un[N3].f[0] );
				resVec[N3].f[1] +=        topoC2 * (  un[N1].f[2] - un[N2].f[1] - un[N2].f[2] + un[N3].f[1] );
				resVec[N3].f[2] += -prs + topoC2 * 2.0 * ( un[N3].f[2] - un[N2].f[2] );

				/* ================ */
				break;

				case ( 1 ):
				N0 = 0;
				N1 = 2;
				N2 = 3;
				N3 = 6;

				prs              = topoC1 * ( un[N0].f[1] - un[N2].f[0] - un[N3].f[2] + un[N1].f[0] - un[N1].f[1] + un[N1].f[2] );

				resVec[N0].f[0] +=        topoC2 * ( un[N0].f[0] - un[N1].f[0] + un[N1].f[1] - un[N2].f[1] );
				resVec[N0].f[1] +=  prs + topoC2 * 2.0 * ( un[N0].f[1] - un[N1].f[1] );
				resVec[N0].f[2] +=        topoC2 * ( un[N0].f[2] + un[N1].f[1] - un[N1].f[2] - un[N3].f[1] );
				/* ================ */
				resVec[N1].f[0] +=  prs + topoC2 * ( -un[N0].f[0] + 4.0 * un[N1].f[0] - un[N1].f[1] + un[N1].f[2] - 2.0 * un[N2].f[0] + un[N2].f[1] - un[N2].f[2] - un[N3].f[0] );
				resVec[N1].f[1] += -prs + topoC2 * ( un[N0].f[0] - 2.0 * un[N0].f[1] + un[N0].f[2] - un[N1].f[0] + 4.0 * un[N1].f[1] - un[N1].f[2] - un[N2].f[1] - un[N3].f[1] );
				resVec[N1].f[2] +=  prs + topoC2 * ( -un[N0].f[2] + un[N1].f[0] - un[N1].f[1] + 4.0 * un[N1].f[2] - un[N2].f[2] - un[N3].f[0] + un[N3].f[1] - 2.0 * un[N3].f[2] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * 2.0 * ( un[N2].f[0] - un[N1].f[0] );
				resVec[N2].f[1] +=        topoC2 * ( -un[N0].f[0] + un[N1].f[0] - un[N1].f[1] + un[N2].f[1] );
				resVec[N2].f[2] +=        topoC2 * ( -un[N1].f[0] - un[N1].f[2] + un[N2].f[2] + un[N3].f[0] );
				/* ================ */
				resVec[N3].f[0] +=        topoC2 * ( -un[N1].f[0] - un[N1].f[2] + un[N2].f[2] + un[N3].f[0] );
				resVec[N3].f[1] +=        topoC2 * ( -un[N0].f[2] - un[N1].f[1] + un[N1].f[2] + un[N3].f[1] );
				resVec[N3].f[2] += -prs + topoC2 * 2.0 * ( un[N3].f[2] - un[N1].f[2] );
				break;

				case ( 2 ):
				N0 = 4;
				N1 = 5;
				N2 = 6;
				N3 = 0;

				prs              = topoC1 * ( un[N3].f[2] - un[N2].f[1] - un[N1].f[0] + un[N0].f[0] + un[N0].f[1] - un[N0].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4.0 * un[N0].f[0] + un[N0].f[1] - un[N0].f[2] - 2.0 * un[N1].f[0] - un[N1].f[1] + un[N1].f[2] - un[N2].f[0] - un[N3].f[0] );
				resVec[N0].f[1] +=  prs + topoC2 * ( 4.0 * un[N0].f[1] + un[N0].f[0] - un[N0].f[2] - un[N1].f[1] - 2.0 * un[N2].f[1] - un[N2].f[0] + un[N2].f[2] - un[N3].f[1] );
				resVec[N0].f[2] += -prs + topoC2 * ( 4.0 * un[N0].f[2] - un[N0].f[0] - un[N0].f[1] - un[N1].f[2] - un[N2].f[2] - 2.0 * un[N3].f[2] + un[N3].f[0] + un[N3].f[1] );
				/* ================ */
				resVec[N1].f[0] += -prs + topoC2 * 2.0 * ( un[N1].f[0] - un[N0].f[0] );
				resVec[N1].f[1] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[1] + un[N2].f[0] );
				resVec[N1].f[2] +=        topoC2 * (  un[N0].f[0] - un[N0].f[2] + un[N1].f[2] - un[N3].f[0] );
				/* ================ */
				resVec[N2].f[0] +=        topoC2 * ( -un[N0].f[0] - un[N0].f[1] + un[N1].f[1] + un[N2].f[0] );
				resVec[N2].f[1] += -prs + topoC2 * 2.0 * ( un[N2].f[1] - un[N0].f[1] );
				resVec[N2].f[2] +=        topoC2 * (  un[N0].f[1] - un[N0].f[2] + un[N2].f[2] - un[N3].f[1] );
				/* ================ */
				resVec[N3].f[0] +=        topoC2 * ( -un[N0].f[0] + un[N0].f[2] - un[N1].f[2] + un[N3].f[0] );
				resVec[N3].f[1] +=        topoC2 * ( -un[N0].f[1] + un[N0].f[2] - un[N2].f[2] + un[N3].f[1] );
				resVec[N3].f[2] +=  prs + topoC2 * 2.0 * ( un[N3].f[2] - un[N0].f[2] );
				break;

				case ( 3 ):
				N0 = 6;
				N1 = 5;
				N2 = 7;
				N3 = 3;

				prs              = topoC1 * ( un[N0].f[0] + un[N1].f[1] + un[N3].f[2] - un[N2].f[0] - un[N2].f[1] - un[N2].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * 2.0 * ( un[N0].f[0] - un[N2].f[0] );
				resVec[N0].f[1] +=        topoC2 * (  un[N0].f[1] + un[N1].f[0] - un[N2].f[0] - un[N2].f[1] );
				resVec[N0].f[2] +=        topoC2 * (  un[N0].f[2] - un[N2].f[0] - un[N2].f[2] + un[N3].f[0] );
				/* ================ */
				resVec[N1].f[0] +=        topoC2 * (  un[N0].f[1] + un[N1].f[0] - un[N2].f[0] - un[N2].f[1] );
				resVec[N1].f[1] +=  prs + topoC2 * 2.0 * ( un[N1].f[1] - un[N2].f[1] );
				resVec[N1].f[2] +=        topoC2 * (  un[N1].f[2] - un[N2].f[1] - un[N2].f[2] + un[N3].f[1] );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( -2.0 * un[N0].f[0] - un[N0].f[1] - un[N0].f[2] - un[N1].f[0] + 4.0 * un[N2].f[0] + un[N2].f[1] + un[N2].f[2] - un[N3].f[0] );
				resVec[N2].f[1] += -prs + topoC2 * ( -un[N0].f[1] - un[N1].f[0] - 2.0 * un[N1].f[1] - un[N1].f[2] + un[N2].f[0] + 4.0 * un[N2].f[1] + un[N2].f[2] - un[N3].f[1] );
				resVec[N2].f[2] += -prs + topoC2 * ( -un[N0].f[2] - un[N1].f[2] + un[N2].f[0] + un[N2].f[1] + 4.0 * un[N2].f[2] - un[N3].f[0] - un[N3].f[1] - 2.0 * un[N3].f[2] );

				/* ================ */
				resVec[N3].f[0] +=        topoC2 * (  un[N0].f[2] - un[N2].f[0] - un[N2].f[2] + un[N3].f[0] );
				resVec[N3].f[1] +=        topoC2 * (  un[N1].f[2] - un[N2].f[1] - un[N2].f[2] + un[N3].f[1] );
				resVec[N3].f[2] +=  prs + topoC2 * 2.0 * ( un[N3].f[2] - un[N2].f[2] );
				break;

				case ( 4 ):
				N0 = 0;
				N1 = 6;
				N2 = 3;
				N3 = 5;

				topoC1 = topoC1 / 4.0;
				topoC2 = topoC2 / 4.0;

				prs              = topoC1 * ( un[N0].f[0] + un[N0].f[1] + un[N0].f[2] + un[N1].f[0] - un[N1].f[1] - un[N1].f[2] - un[N2].f[0] - un[N2].f[1] + un[N2].f[2] - un[N3].f[0] + un[N3].f[1] - un[N3].f[2] );

				resVec[N0].f[0] +=  prs + topoC2 * ( 4.0 * un[N0].f[0] + un[N0].f[1] + un[N0].f[2] + un[N1].f[1] + un[N1].f[2] - un[N2].f[1] - un[N2].f[2] - un[N3].f[1] - un[N3].f[2] - 2.0 * ( un[N2].f[0] + un[N3].f[0] ) );
				resVec[N0].f[1] +=  prs + topoC2 * (  un[N0].f[0] + 4.0 * un[N0].f[1] + un[N0].f[2] - un[N1].f[0] - un[N1].f[2] - un[N2].f[0] - un[N2].f[2] + un[N3].f[0] + un[N3].f[2] - 2.0 * ( un[N1].f[1] + un[N2].f[1] ) );
				resVec[N0].f[2] +=  prs + topoC2 * (  un[N0].f[0] + un[N0].f[1] + 4.0 * un[N0].f[2] - un[N1].f[0] - un[N1].f[1] + un[N2].f[0] + un[N2].f[1] - un[N3].f[0] - un[N3].f[1] - 2.0 * ( un[N1].f[2] + un[N3].f[2] ) );
				/* ================ */
				resVec[N1].f[0] +=  prs + topoC2 * ( -un[N0].f[1] - un[N0].f[2] + 4.0 * un[N1].f[0] -       un[N1].f[1] -       un[N1].f[2] + un[N2].f[1] + un[N2].f[2] + un[N3].f[1] + un[N3].f[2] - 2.0 * ( un[N3].f[0] + un[N2].f[0] ) );
				resVec[N1].f[1] += -prs + topoC2 * (  un[N0].f[0] - un[N0].f[2] -       un[N1].f[0] + 4.0 * un[N1].f[1] +       un[N1].f[2] - un[N2].f[0] + un[N2].f[2] + un[N3].f[0] - un[N3].f[2] - 2.0 * ( un[N3].f[1] + un[N0].f[1] ) );
				resVec[N1].f[2] += -prs + topoC2 * (  un[N0].f[0] - un[N0].f[1] -       un[N1].f[0] +       un[N1].f[1] + 4.0 * un[N1].f[2] +  un[N2].f[0] - un[N2].f[1] - un[N3].f[0] + un[N3].f[1] - 2.0 * ( un[N0].f[2] + un[N2].f[2] ) );
				/* ================ */
				resVec[N2].f[0] += -prs + topoC2 * ( 4.0 * un[N2].f[0] - un[N0].f[1] + un[N0].f[2] - un[N1].f[1] + un[N1].f[2] + un[N2].f[1] - un[N2].f[2] + un[N3].f[1] - un[N3].f[2] - 2.0 * ( un[N0].f[0] + un[N1].f[0] ) );
				resVec[N2].f[1] += -prs + topoC2 * ( 4.0 * un[N2].f[1] - un[N0].f[0] + un[N0].f[2] + un[N1].f[0] - un[N1].f[2] + un[N2].f[0] - un[N2].f[2] - un[N3].f[0] + un[N3].f[2] - 2.0 * ( un[N0].f[1] + un[N3].f[1] ) );
				resVec[N2].f[2] +=  prs + topoC2 * ( 4.0 * un[N2].f[2] - un[N0].f[0] - un[N0].f[1] + un[N1].f[0] + un[N1].f[1] - un[N2].f[0] - un[N2].f[1] + un[N3].f[0] + un[N3].f[1] - 2.0 * ( un[N3].f[2] + un[N1].f[2] ) );
				/* ================ */
				resVec[N3].f[0] += -prs + topoC2 * ( 4.0 * un[N3].f[0] + un[N0].f[1] - un[N0].f[2] + un[N1].f[1] - un[N1].f[2] - un[N2].f[1] + un[N2].f[2] - un[N3].f[1] + un[N3].f[2] - 2.0 * ( un[N0].f[0] + un[N1].f[0] ) );
				resVec[N3].f[1] +=  prs + topoC2 * ( 4.0 * un[N3].f[1] - un[N0].f[0] - un[N0].f[2] + un[N1].f[0] + un[N1].f[2] + un[N2].f[0] + un[N2].f[2] - un[N3].f[0] - un[N3].f[2] - 2.0 * ( un[N1].f[1] + un[N2].f[1] ) );
				resVec[N3].f[2] += -prs + topoC2 * ( 4.0 * un[N3].f[2] - un[N0].f[0] + un[N0].f[1] + un[N1].f[0] - un[N1].f[1] - un[N2].f[0] + un[N2].f[1] + un[N3].f[0] - un[N3].f[1] - 2.0 * ( un[N0].f[2] + un[N2].f[2] ) );
				break;

				}
			}
		}
	}
}

/* -------------------------------------------------------------------------- */
/* TODO: DISCARD DRM FUNCTIONS. THEY ARE JUST MEANT FOR A QUICK CHECKING      */
/* -------------------------------------------------------------------------- */

double Ricker_displ ( double zp, double Ts, double t, double fc, double Vs  ) {

	double alfa1 = ( PI * fc ) * ( PI * fc ) * ( t - zp / Vs - Ts) * ( t - zp / Vs - Ts);
	double alfa2 = ( PI * fc ) * ( PI * fc ) * ( t + zp / Vs - Ts) * ( t + zp / Vs - Ts);

	double uo1 = ( 2.0 * alfa1 - 1.0 ) * exp(-alfa1);
	double uo2 = ( 2.0 * alfa2 - 1.0 ) * exp(-alfa2);

	return (uo1+uo2);
}

double Ricker_Accel ( double zp, double Ts, double t, double fc, double Vs  ) {

	double alfa1 = ( PI * fc ) * ( PI * fc ) * ( t - zp / Vs - Ts) * ( t - zp / Vs - Ts);
	double alfa2 = ( PI * fc ) * ( PI * fc ) * ( t + zp / Vs - Ts) * ( t + zp / Vs - Ts);

	double ao1 = ( 3 - 2 * alfa1 ) * exp(-alfa1);
	       ao1 = ao1 + 2 * alfa1 * ( 2 * alfa1 - 5 ) * exp(-alfa1);

	double ao2 = ( 3 - 2 * alfa2 ) * exp(-alfa2);
	       ao2 = ao2 + 2 * alfa2 * ( 2 * alfa2 - 5 ) * exp(-alfa2);

	return 2 * ( PI * fc ) * ( PI * fc ) * ( ao1 + ao2 );
}

void getRicker ( fvector_t *myDisp, double zp, double t, double Vs ) {

	double Rz = Ricker_displ ( zp, myTs, t, myFc, Vs  );

	if ( myDir == 0 ) {
		myDisp->f[0] = Rz;
		myDisp->f[1] = 0.0;
		myDisp->f[2] = 0.0;
	} else if ( myDir == 1 ) {
		myDisp->f[0] = 0.0;
		myDisp->f[1] = Rz;
		myDisp->f[2] = 0.0;
	} else {
		myDisp->f[0] = 0.0;
		myDisp->f[1] = 0.0;
		myDisp->f[2] = Rz;
	}

}

void DRM_ForcesinElement ( mesh_t     *myMesh,
		mysolver_t *mySolver,
		fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8],
		int *f_nodes, int *e_nodes, int32_t   eindex, double tt, int Nnodes_e, int Nnodes_f )
{

    int       i, j;
    int  CoordArr[8]      = { 0, 0, 0, 0, 1, 1, 1, 1 };


	fvector_t localForce[8];

	elem_t        *elemp;
	edata_t       *edata;
	node_t        *node0dat;
	double        xo, yo, zo;
	int32_t	      node0;
	e_t*          ep;

	/* Capture the table of elements from the mesh and the size
	 * This is what gives me the connectivity to nodes */
	elemp        = &myMesh->elemTable[eindex];
	edata        = (edata_t *)elemp->data;
	node0        = elemp->lnid[0];
	node0dat     = &myMesh->nodeTable[node0];
	ep           = &mySolver->eTable[eindex];

	/* get coordinates of element zero node */
	xo = (node0dat->x)*(myMesh->ticksize);
	yo = (node0dat->y)*(myMesh->ticksize);
	zo = (node0dat->z)*(myMesh->ticksize) - thebase_zcoord;


	/* get material properties  */
	double  h, Vs;
	h    = (double)edata->edgesize;

	if ( ( myDir == 0 ) || ( myDir == 1 ) )
		Vs = edata->Vs;
	else
		Vs = edata->Vp;


	/* Force contribution from external nodes */
	/* -------------------------------
	 * Ku DONE IN THE CONVENTIONAL WAY
	 * ------------------------------- */
	memset( localForce, 0, 8 * sizeof(fvector_t) );

	fvector_t myDisp;
	/* forces over f nodes */
	for (i = 0; i < Nnodes_f; i++) {

		int  nodef = *(f_nodes + i);
		fvector_t* toForce = &localForce[ nodef ];

		/* incoming displacements over e nodes */
		for (j = 0; j < Nnodes_e; j++) {

			int  nodee = *(e_nodes + j);

			double z_ne = zo + h * CoordArr[ nodee ];   /* get zcoord */
			getRicker ( &myDisp, z_ne, tt, Vs ); /* get Displ */

			MultAddMatVec( &theK1[ nodef ][ nodee ], &myDisp, -ep->c1, toForce );
			MultAddMatVec( &theK2[ nodef ][ nodee ], &myDisp, -ep->c2, toForce );
		}
	}

	/* forces over e nodes */
	for (i = 0; i < Nnodes_e; i++) {

		int  nodee = *(e_nodes + i);
		fvector_t* toForce = &localForce[ nodee ];

		/* incoming displacements over f nodes */
		for (j = 0; j < Nnodes_f; j++) {

			int  nodef = *(f_nodes + j);

			double z_nf = zo + h * CoordArr[ nodef ];   /* get zcoord */
			getRicker ( &myDisp, z_nf, tt, Vs ); /* get Displ */

			MultAddMatVec( &theK1[ nodee ][ nodef ], &myDisp, ep->c1, toForce );
			MultAddMatVec( &theK2[ nodee ][ nodef ], &myDisp, ep->c2, toForce );
		}
	}
	/* end Ku */

	/* Loop over the 8 element nodes:
	 * Add the contribution calculated above to the node
	 * forces carried from the source and stiffness.
	 */
	for (i = 0; i < 8; i++) {

		int32_t    lnid;
		fvector_t *nodalForce;

		lnid = elemp->lnid[i];

		nodalForce = mySolver->force + lnid;

		nodalForce->f[0] += localForce[i].f[0] ;
		nodalForce->f[1] += localForce[i].f[1] ;
		nodalForce->f[2] += localForce[i].f[2] ;

	} /* element nodes */

}

void compute_addforce_topoDRM ( mesh_t     *myMesh,
                                mysolver_t *mySolver,
                                double      theDeltaT,
                                int         step,
                                fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8])
{

    int32_t   eindex;
    int32_t   face_eindex;

    int  f_nodes_face1[4] = { 0, 1, 4, 5 };
    int  e_nodes_face1[4] = { 2, 3, 6, 7 };

    int  f_nodes_face2[4] = { 2, 3, 6, 7 };
    int  e_nodes_face2[4] = { 0, 1, 4, 5 };

    int  f_nodes_face3[4] = { 1, 3, 5, 7 };
    int  e_nodes_face3[4] = { 0, 2, 4, 6 };

    int  f_nodes_face4[4] = { 0, 2, 4, 6 };
    int  e_nodes_face4[4] = { 1, 3, 5, 7 };

    int  f_nodes_bottom[4] = { 0, 1, 2, 3 };
    int  e_nodes_bottom[4] = { 4, 5, 6, 7 };

    int  f_nodes_border1[2] = { 1, 5 };
    int  e_nodes_border1[6] = { 0, 2, 3, 4, 6, 7 };

    int  f_nodes_border2[2] = { 0, 4 };
    int  e_nodes_border2[6] = { 1, 2, 3, 5, 6, 7 };

    int  f_nodes_border3[2] = { 3, 7 };
    int  e_nodes_border3[6] = { 0, 1, 2, 4, 5, 6 };

    int  f_nodes_border4[2] = { 2, 6 };
    int  e_nodes_border4[6] = { 0, 1, 3, 4, 5, 7 };

    int  f_nodes_border5[2] = { 0, 1 };
    int  e_nodes_border5[6] = { 2, 3, 4, 5, 6, 7 };

    int  f_nodes_border6[2] = { 2, 3 };
    int  e_nodes_border6[6] = { 0, 1, 4, 5, 6, 7 };

    int  f_nodes_border7[2] = { 1, 3 };
    int  e_nodes_border7[6] = { 0, 2, 4, 5, 6, 7 };

    int  f_nodes_border8[2] = { 0, 2 };
    int  e_nodes_border8[6] = { 1, 3, 4, 5, 6, 7 };

    double tt = theDeltaT * step;
    int  *f_nodes, *e_nodes;

    /* Loop over face1 elements */
    f_nodes = &f_nodes_face1[0];
    e_nodes = &e_nodes_face1[0];
    for ( face_eindex = 0; face_eindex < myDRM_Face1Count ; face_eindex++) {
    	eindex = myDRMFace1ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 4, 4 );
    } /* all elements in face 1*/

    /* Loop over face2 elements */
    f_nodes = &f_nodes_face2[0];
    e_nodes = &e_nodes_face2[0];
    for ( face_eindex = 0; face_eindex < myDRM_Face2Count ; face_eindex++) {
    	eindex = myDRMFace2ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 4, 4 );
    } /* all elements in face 2*/

    /* Loop over face3 elements */
    f_nodes = &f_nodes_face3[0];
    e_nodes = &e_nodes_face3[0];
    for ( face_eindex = 0; face_eindex < myDRM_Face3Count ; face_eindex++) {
    	eindex = myDRMFace3ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 4, 4 );
    } /* all elements in face 3*/

    /* Loop over face4 elements */
    f_nodes = &f_nodes_face4[0];
    e_nodes = &e_nodes_face4[0];
    for ( face_eindex = 0; face_eindex < myDRM_Face4Count ; face_eindex++) {
    	eindex = myDRMFace4ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 4, 4 );
    } /* all elements in face 4*/

    /* Loop over bottom elements */
    f_nodes = &f_nodes_bottom[0];
    e_nodes = &e_nodes_bottom[0];
    for ( face_eindex = 0; face_eindex < myDRM_BottomCount ; face_eindex++) {
    	eindex = myDRMBottomElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 4, 4 );
    } /* all elements in bottom*/

    /* Loop over border1 elements */
    f_nodes = &f_nodes_border1[0];
    e_nodes = &e_nodes_border1[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd1 ; face_eindex++) {
    	eindex = myDRMBorder1ElementsMapping[face_eindex];

    	/* check for bottom element */
    	elem_t        *elemp;
    	int32_t	      node0;
    	node_t        *node0dat;
    	double        zo;

    	elemp        = &myMesh->elemTable[eindex];
    	node0        = elemp->lnid[0];
    	node0dat     = &myMesh->nodeTable[node0];
    	zo           = (node0dat->z)*(myMesh->ticksize);

    	if ( zo != theDRMdepth + thebase_zcoord ) {
    		DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    	} else {
    	    int  f_corner[1] = { 1 };
    	    int  e_corner[7] = { 0, 2, 3, 4, 5, 6, 7 };
    	    int  *fcorner_nodes, *ecorner_nodes;
    	    fcorner_nodes = &f_corner[0];
    	    ecorner_nodes = &e_corner[0];
    	    DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, fcorner_nodes, ecorner_nodes, eindex, tt, 7, 1 );

    	}
    } /* all elements in border1*/

    /* Loop over border2 elements */
    f_nodes = &f_nodes_border2[0];
    e_nodes = &e_nodes_border2[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd2 ; face_eindex++) {
    	eindex = myDRMBorder2ElementsMapping[face_eindex];

    	/* check for bottom element */
    	elem_t        *elemp;
    	int32_t	      node0;
    	node_t        *node0dat;
    	double        zo;

    	elemp        = &myMesh->elemTable[eindex];
    	node0        = elemp->lnid[0];
    	node0dat     = &myMesh->nodeTable[node0];
    	zo           = (node0dat->z)*(myMesh->ticksize);

    	if ( zo != theDRMdepth + thebase_zcoord ) {
    		DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    	} else {
    	    int  f_corner[1] = { 0 };
    	    int  e_corner[7] = { 1, 2, 3, 4, 5, 6, 7 };
    	    int  *fcorner_nodes, *ecorner_nodes;
    	    fcorner_nodes = &f_corner[0];
    	    ecorner_nodes = &e_corner[0];
    	    DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, fcorner_nodes, ecorner_nodes, eindex, tt, 7, 1 );

    	}

    } /* all elements in border2*/

    /* Loop over border3 elements */
    f_nodes = &f_nodes_border3[0];
    e_nodes = &e_nodes_border3[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd3 ; face_eindex++) {
    	eindex = myDRMBorder3ElementsMapping[face_eindex];

    	/* check for bottom element */
    	elem_t        *elemp;
    	int32_t	      node0;
    	node_t        *node0dat;
    	double        zo;

    	elemp        = &myMesh->elemTable[eindex];
    	node0        = elemp->lnid[0];
    	node0dat     = &myMesh->nodeTable[node0];
    	zo           = (node0dat->z)*(myMesh->ticksize);

    	if ( zo != theDRMdepth + thebase_zcoord ) {
    		DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    	} else {
    	    int  f_corner[1] = { 3 };
    	    int  e_corner[7] = { 0, 1, 2, 4, 5, 6, 7 };
    	    int  *fcorner_nodes, *ecorner_nodes;
    	    fcorner_nodes = &f_corner[0];
    	    ecorner_nodes = &e_corner[0];
    	    DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, fcorner_nodes, ecorner_nodes, eindex, tt, 7, 1 );

    	}
    } /* all elements in border3*/

    /* Loop over border4 elements */
    f_nodes = &f_nodes_border4[0];
    e_nodes = &e_nodes_border4[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd4 ; face_eindex++) {
    	eindex = myDRMBorder4ElementsMapping[face_eindex];

    	/* check for bottom element */
    	elem_t        *elemp;
    	int32_t	      node0;
    	node_t        *node0dat;
    	double        zo;

    	elemp        = &myMesh->elemTable[eindex];
    	node0        = elemp->lnid[0];
    	node0dat     = &myMesh->nodeTable[node0];
    	zo           = (node0dat->z)*(myMesh->ticksize);

    	if ( zo != theDRMdepth + thebase_zcoord ) {
    		DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    	} else {
    	    int  f_corner[1] = { 3 };
    	    int  e_corner[7] = { 0, 1, 2, 4, 5, 6, 7 };
    	    int  *fcorner_nodes, *ecorner_nodes;
    	    fcorner_nodes = &f_corner[0];
    	    ecorner_nodes = &e_corner[0];
    	    DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, fcorner_nodes, ecorner_nodes, eindex, tt, 7, 1 );

    	}
    } /* all elements in border4*/

    /* Loop over border5 elements */
    f_nodes = &f_nodes_border5[0];
    e_nodes = &e_nodes_border5[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd5 ; face_eindex++) {
    	eindex = myDRMBorder5ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    } /* all elements in border5*/

    /* Loop over border6 elements */
    f_nodes = &f_nodes_border6[0];
    e_nodes = &e_nodes_border6[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd6 ; face_eindex++) {
    	eindex = myDRMBorder6ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    } /* all elements in border6*/

    /* Loop over border7 elements */
    f_nodes = &f_nodes_border7[0];
    e_nodes = &e_nodes_border7[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd7 ; face_eindex++) {
    	eindex = myDRMBorder7ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    } /* all elements in border7*/

    /* Loop over border8 elements */
    f_nodes = &f_nodes_border8[0];
    e_nodes = &e_nodes_border8[0];
    for ( face_eindex = 0; face_eindex < myDRM_Brd8 ; face_eindex++) {
    	eindex = myDRMBorder8ElementsMapping[face_eindex];
    	DRM_ForcesinElement ( myMesh, mySolver, theK1, theK2, f_nodes, e_nodes, eindex, tt, 6, 2 );
    } /* all elements in border7*/


    return;
}

void topo_DRM_init( mesh_t *myMesh, mysolver_t *mySolver) {


	int32_t halfFace_elem, theBorderElem;  /* half the number of elements in a box's face. Artificially defined by me */
	int32_t theFaceElem, theBaseElem;

	/* Data required by me before running the simmulation */
	double hmin     = 15.625 ; /* 15.625 for the SanchezSeama Ridge
	                              15.625 for the Gaussian Ridge */
	halfFace_elem   = 60;     /* this defines B. See figure in manuscript.
	                             halfFace_elem   = 15 for the SphericalRidge
	                             halfFace_elem   = 60 for the GaussianRidge
	                             halfFace_elem   = 60 for the SanSesma Ridge */

	theBorderElem   = 5;     /* this defines the depth, See figure in manuscript  */
	                         /* 5 for the gaussian ridge, 3 for the SanSesma ridge */
	double theXc           = 1000;   /* domain center coordinates */
	double theYc           = 1000;
	double Ts              = 0.20;  /* 0.18 for the Gaussian Ridge. 0.2 for the SanSesma Ridge  */
	double fc              = 10;  /* 10.26 for the Gaussian Ridge. 10.4 for the SanSesma Ridge  */
	int    wave_dir        = 2;    /*  0:X, 1:Y, 2:Z */

	/* --------  */
	double DRM_D = theBorderElem * hmin;
	double DRM_B = halfFace_elem * hmin;


	theFaceElem = 2 * ( halfFace_elem + 0 ) * theBorderElem;
	theBaseElem = 4 * halfFace_elem * halfFace_elem;
	theDRMdepth = DRM_D;

	myTopoDRMBorderElementsCount = theBorderElem;
	myTopoDRMFaceElementsCount   = theFaceElem;
	myTs                         = Ts;
	myFc                         = fc;
	myDir                        = wave_dir;


	/*  mapping of face1 elements */
	int32_t eindex;
	int32_t countf1 = 0, countf2 = 0, countf3 = 0, countf4 = 0, countbott=0;
	int32_t countb1 = 0, countb2 = 0, countb3 = 0, countb4 = 0;
	int32_t countb5 = 0, countb6 = 0, countb7 = 0, countb8 = 0;

	XMALLOC_VAR_N(myDRMFace1ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMFace2ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMFace3ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMFace4ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMBottomElementsMapping , int32_t, theBaseElem);

	/* border elements*/
	XMALLOC_VAR_N(myDRMBorder1ElementsMapping, int32_t, theBorderElem + 1);
	XMALLOC_VAR_N(myDRMBorder2ElementsMapping, int32_t, theBorderElem + 1);
	XMALLOC_VAR_N(myDRMBorder3ElementsMapping, int32_t, theBorderElem + 1);
	XMALLOC_VAR_N(myDRMBorder4ElementsMapping, int32_t, theBorderElem + 1);

	XMALLOC_VAR_N(myDRMBorder5ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMBorder6ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMBorder7ElementsMapping, int32_t, theFaceElem);
	XMALLOC_VAR_N(myDRMBorder8ElementsMapping, int32_t, theFaceElem);

	for (eindex = 0; eindex < myMesh->lenum; eindex++) {

		elem_t     *elemp;
		node_t     *node0dat;
		edata_t    *edata;
		double      xo, yo, zo;
		int32_t	    node0;

		elemp    = &myMesh->elemTable[eindex]; //Takes the information of the "eindex" element
		node0    = elemp->lnid[0];             //Takes the ID for the zero node in element eindex
		node0dat = &myMesh->nodeTable[node0];
		edata    = (edata_t *)elemp->data;

		/* get coordinates of element zero node */
		xo = (node0dat->x)*(myMesh->ticksize);
		yo = (node0dat->y)*(myMesh->ticksize);
		zo = (node0dat->z)*(myMesh->ticksize);


		if ( ( ( yo - theYc ) == DRM_B ) &&                            /*face 1*/
				( xo >= ( theXc - DRM_B ) ) &&
				( xo <  ( theXc + DRM_B ) ) &&
				( zo <  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMFace1ElementsMapping[countf1] = eindex;
			countf1++;
		} else 	if ( ( ( theYc - ( yo + hmin ) ) == DRM_B ) &&        /* face 2*/
				( xo >= ( theXc - DRM_B ) ) &&
				( xo <  ( theXc + DRM_B ) ) &&
				( zo <  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMFace2ElementsMapping[countf2] = eindex;
			countf2++;

		} else 	if ( ( ( theXc - ( xo + hmin ) ) == DRM_B ) &&      /*face 3*/
				( yo >= ( theYc - DRM_B ) ) &&
				( yo <  ( theYc + DRM_B ) ) &&
				( zo <  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMFace3ElementsMapping[countf3] = eindex;
			countf3++;

		} else 	if ( ( ( xo - theXc ) == DRM_B ) &&             /* face 4*/
				( yo >= ( theYc - DRM_B ) ) &&
				( yo <  ( theYc + DRM_B ) ) &&
				( zo <  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMFace4ElementsMapping[countf4] = eindex;
			countf4++;

		} else 	if ( ( yo >= ( theYc - DRM_B ) ) &&         /*bottom*/
				( yo <  ( theYc + DRM_B ) ) &&
				( xo >= ( theXc - DRM_B ) ) &&
				( xo <  ( theXc + DRM_B ) ) &&
				( zo ==  DRM_D + thebase_zcoord )          ) {

			myDRMBottomElementsMapping[countbott] = eindex;
			countbott++;

		} else 	if ( ( ( yo - theYc ) == DRM_B ) &&      /*border 1*/
				( ( xo + hmin ) == ( theXc - DRM_B ) ) &&
				( zo <=  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMBorder1ElementsMapping[countb1] = eindex;
			countb1++;

		} else if ( ( ( yo - theYc ) == DRM_B  ) &&      /*border 2*/
				( xo  == ( theXc + DRM_B ) ) &&
				( zo <=  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMBorder2ElementsMapping[countb2] = eindex;
			countb2++;

		} else if ( ( ( theYc - ( yo + hmin ) ) == DRM_B ) &&  /* border 3*/
				( ( xo + hmin)  == ( theXc - DRM_B ) ) &&
				( zo <=  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMBorder3ElementsMapping[countb3] = eindex;
			countb3++;

		} else if ( ( ( theYc - ( yo + hmin ) ) == DRM_B ) &&  /* border 4*/
				(  xo  == ( theXc + DRM_B ) ) &&
				( zo <=  DRM_D + thebase_zcoord ) &&
				( zo >=  thebase_zcoord ) ) {

			myDRMBorder4ElementsMapping[countb4] = eindex;
			countb4++;

		} else 	if ( ( ( yo - theYc ) == DRM_B ) &&            /* border 5*/
				( xo >= ( theXc - DRM_B ) ) &&
				( xo <  ( theXc + DRM_B ) ) &&
				( zo ==  DRM_D + thebase_zcoord )          ) {

			myDRMBorder5ElementsMapping[countb5] = eindex;
			countb5++;

		} else if ( ( ( theYc - ( yo + hmin ) ) == DRM_B ) &&          /* border 6*/
				( xo >= ( theXc - DRM_B ) ) &&
				( xo <  ( theXc + DRM_B ) ) &&
				( zo ==  DRM_D + thebase_zcoord )          ) {

			myDRMBorder6ElementsMapping[countb6] = eindex;
			countb6++;

		} else if ( ( ( theXc - ( xo + hmin ) ) == DRM_B ) &&      /* border 7*/
				( yo >= ( theYc - DRM_B ) ) &&
				( yo <  ( theYc + DRM_B ) ) &&
				( zo ==  DRM_D + thebase_zcoord )          ) {

			myDRMBorder7ElementsMapping[countb7] = eindex;
			countb7++;

		} else if ( ( ( xo - theXc ) == DRM_B ) &&             /* border 8*/
				( yo >= ( theYc - DRM_B ) ) &&
				( yo <  ( theYc + DRM_B ) ) &&
				( zo ==  DRM_D + thebase_zcoord )          ) {

			myDRMBorder8ElementsMapping[countb8] = eindex;
			countb8++;
		}
	}


	myDRM_Face1Count  = countf1;
	myDRM_Face2Count  = countf2;
	myDRM_Face3Count  = countf3;
	myDRM_Face4Count  = countf4;
	myDRM_BottomCount = countbott;
	myDRM_Brd1        = countb1;
	myDRM_Brd2        = countb2;
	myDRM_Brd3        = countb3;
	myDRM_Brd4        = countb4;
	myDRM_Brd5        = countb5;
	myDRM_Brd6        = countb6;
	myDRM_Brd7        = countb7;
	myDRM_Brd8        = countb8;

}

/* -------------------------------------------------------------------------- */
/*                        Topography Output to Stations                       */
/* -------------------------------------------------------------------------- */

void topography_stations_init( mesh_t    *myMesh,
                               station_t *myStations,
                               int32_t    myNumberOfStations)
{

    int32_t     eindex, topo_eindex;
    int32_t     iStation=0;
    vector3D_t  point;
    octant_t   *octant;
    int32_t     lnid0;
    elem_t     *elemp;

    if ( ( myNumberOfStations == 0   ) ||
         ( theTopoMethod      == FEM ) )
		return;

    /* Here I allocate memory for all stations */
    XMALLOC_VAR_N( myTopoStations, topostation_t, myNumberOfStations);

    /* initialize topographic stations  */
    for (iStation = 0; iStation < myNumberOfStations; iStation++) {

    	myTopoStations[iStation].TopoStation             = 0;
    	myTopoStations[iStation].local_coord[0]          = 0.0;
    	myTopoStations[iStation].local_coord[1]          = 0.0;
    	myTopoStations[iStation].local_coord[2]          = 0.0;
    	myTopoStations[iStation].nodes_to_interpolate[0] = 0;
    	myTopoStations[iStation].nodes_to_interpolate[1] = 0;
    	myTopoStations[iStation].nodes_to_interpolate[2] = 0;
    	myTopoStations[iStation].nodes_to_interpolate[3] = 0;
    }

    for (iStation = 0; iStation < myNumberOfStations; iStation++) {

        /* capture the stations coordinates */
        point = myStations[iStation].coords;

        /* search the octant */
        if ( search_point(point, &octant) != 1 ) {
            fprintf(stderr,
                    "topography_stations_init: "
                    "No octant with station coords\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }

        for ( topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++ ) {

            eindex = myTopoElementsMapping[topo_eindex];

            lnid0 = myMesh->elemTable[eindex].lnid[0];

            if ( (myMesh->nodeTable[lnid0].x == octant->lx) &&
                 (myMesh->nodeTable[lnid0].y == octant->ly) &&
                 (myMesh->nodeTable[lnid0].z == octant->lz) ) {

                /* I have a match for the element's origin */

                /* Now, perform level sanity check */
                if (myMesh->elemTable[eindex].level != octant->level) {
                    fprintf(stderr,
                            "topo_stations_init: First pass: "
                            "Wrong level of octant\n");
                    MPI_Abort(MPI_COMM_WORLD, ERROR);
                    exit(1);
                }

                myTopoStations[iStation].TopoStation  = 1; /*  Station belongs to topography */

                /* zero node coordinates */
        		double xo = myMesh->nodeTable[lnid0].x * (myMesh->ticksize);
        		double yo = myMesh->nodeTable[lnid0].y * (myMesh->ticksize);
        		double zo = myMesh->nodeTable[lnid0].z * (myMesh->ticksize);

                elemp  = &myMesh->elemTable[eindex];

                topoconstants_t  *ecp;
                ecp    = myTopoSolver->topoconstants + topo_eindex;

                compute_tetra_localcoord ( point, elemp,
                		                   myTopoStations[iStation].nodes_to_interpolate,
                		                   myTopoStations[iStation].local_coord,
                		                   xo, yo, zo, ecp->h );

                break;
            }

        } /* for all my elements */

    } /* for all my stations */

}

void compute_tetra_localcoord ( vector3D_t point, elem_t *elemp,
		                        int32_t *localNode, double *localCoord,
		                        double xo, double yo, double zo, double h )
{

	int i;
	double eta, psi, gamma;
	double xp1, yp1, zp1;

	if ( ( ( xo <  theDomainLong_ns / 2.0  ) && ( yo <  theDomainLong_ew / 2.0 ) ) ||
	     ( ( xo >= theDomainLong_ns / 2.0 ) && ( yo >= theDomainLong_ew / 2.0 ) ) )
	{

		for (i = 0 ;  i < 4; i++) {

			switch ( i ) {

			case ( 0 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = xp1 / h;
				psi   = yp1 / h;
				gamma = zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 0 ];
					*(localNode + 1)  = elemp->lnid[ 2 ];
					*(localNode + 2)  = elemp->lnid[ 1 ];
					*(localNode + 3)  = elemp->lnid[ 4 ];

					return;
				}
				break;

			case ( 1 ):

				xp1 = point.x[0] - ( xo + h );
				yp1 = point.x[1] - ( yo + h );
				zp1 = point.x[2] - zo;

				eta   = -xp1 / h;
				psi   = -yp1 / h;
				gamma =  zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 3 ];
					*(localNode + 1)  = elemp->lnid[ 1 ];
					*(localNode + 2)  = elemp->lnid[ 2 ];
					*(localNode + 3)  = elemp->lnid[ 7 ];

					return;
				}
				break;

			case ( 2 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - ( yo + h );
				zp1 = point.x[2] - ( zo + h );

				eta   =  xp1 / h;
				psi   = -yp1 / h;
				gamma = -zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 6 ];
					*(localNode + 1)  = elemp->lnid[ 4 ];
					*(localNode + 2)  = elemp->lnid[ 7 ];
					*(localNode + 3)  = elemp->lnid[ 2 ];

					return;
				}
				break;

			case ( 3 ):

				xp1 = point.x[0] - ( xo + h );
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - ( zo + h );

				eta   = -xp1 / h;
				psi   =  yp1 / h;
				gamma = -zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 5 ];
					*(localNode + 1)  = elemp->lnid[ 7 ];
					*(localNode + 2)  = elemp->lnid[ 4 ];
					*(localNode + 3)  = elemp->lnid[ 1 ];

					return;
				}
				break;

			case ( 4 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - ( yo + h );
				zp1 = point.x[2] - zo;

				eta   = (  xp1 + yp1 + zp1 ) / ( 2.0 * h );
				psi   = ( -xp1 - yp1 + zp1 ) / ( 2.0 * h );
				gamma = (  xp1 - yp1 - zp1 ) / ( 2.0 * h );

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 2 ];
					*(localNode + 1)  = elemp->lnid[ 4 ];
					*(localNode + 2)  = elemp->lnid[ 7 ];
					*(localNode + 3)  = elemp->lnid[ 1 ];

					return;
				}
				break;

			}

		}

	} else {

		for (i = 0 ;  i < 4; i++) {

			switch ( i ) {

			case ( 0 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = ( xp1 - yp1 - zp1 ) / h;
				psi   = yp1 / h;
				gamma = zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 0 ];
					*(localNode + 1)  = elemp->lnid[ 3 ];
					*(localNode + 2)  = elemp->lnid[ 1 ];
					*(localNode + 3)  = elemp->lnid[ 5 ];

					return;
				}
				break;

			case ( 1 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = xp1 / h;
				psi   = ( -xp1 + yp1 - zp1 ) / h;
				gamma =  zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 0 ];
					*(localNode + 1)  = elemp->lnid[ 2 ];
					*(localNode + 2)  = elemp->lnid[ 3 ];
					*(localNode + 3)  = elemp->lnid[ 6 ];

					return;
				}
				break;

			case ( 2 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - ( zo + h );

				eta   =  yp1 / h;
				psi   =  xp1 / h;
				gamma = -zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 4 ];
					*(localNode + 1)  = elemp->lnid[ 5 ];
					*(localNode + 2)  = elemp->lnid[ 6 ];
					*(localNode + 3)  = elemp->lnid[ 0 ];

					return;
				}
				break;

			case ( 3 ):

				xp1 = point.x[0] - ( xo );
				yp1 = point.x[1] - ( yo + h );
				zp1 = point.x[2] - ( zo + h );

				eta   = ( xp1 + yp1 + zp1 ) / h;
				psi   =  -yp1 / h;
				gamma =  -zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 6 ];
					*(localNode + 1)  = elemp->lnid[ 5 ];
					*(localNode + 2)  = elemp->lnid[ 7 ];
					*(localNode + 3)  = elemp->lnid[ 3 ];

					return;
				}
				break;

			case ( 4 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = (  xp1 + yp1 - zp1 ) / ( 2.0 * h );
				psi   = ( -xp1 + yp1 + zp1 ) / ( 2.0 * h );
				gamma = (  xp1 - yp1 + zp1 ) / ( 2.0 * h );

				if ( ( ( 1.0 - eta - psi - gamma ) >=  0 ) &&
					   ( 1.0 - eta - psi - gamma ) <=  1.0 ) {

					*localCoord       = eta;
					*(localCoord + 1) = psi;
					*(localCoord + 2) = gamma;

					*localNode        = elemp->lnid[ 0 ];
					*(localNode + 1)  = elemp->lnid[ 6 ];
					*(localNode + 2)  = elemp->lnid[ 3 ];
					*(localNode + 3)  = elemp->lnid[ 5 ];

					return;
				}
				break;

			}

		}

	}

	/* Should not get here */
	fprintf(stderr,
			"Topography station error: "
			"Unable to locate tetrahedron for station: "
			"x=%f, y=%f, z=%f\n", point.x[0], point.x[1], point.x[2]);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);

}

int compute_tetra_displ (double *dis_x, double *dis_y, double *dis_z,
						 double *vel_x, double *vel_y, double *vel_z,
						 double *accel_x, double *accel_y, double *accel_z,
						 double theDeltaT, double theDeltaTSquared,
		                 int32_t statID, mysolver_t *mySolver)
{

	if ( theTopoMethod == FEM )
		return 0;

	if ( myTopoStations[statID].TopoStation  == 0 ) {
		return 0;
	} else {


		double ux_0  = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[0] ].f[0];
		double ux_10 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[1] ].f[0] - ux_0;
		double ux_20 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[2] ].f[0] - ux_0;
		double ux_30 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[3] ].f[0] - ux_0;

		double uy_0  = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[0] ].f[1];
		double uy_10 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[1] ].f[1] - uy_0;
		double uy_20 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[2] ].f[1] - uy_0;
		double uy_30 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[3] ].f[1] - uy_0;

		double uz_0  = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[0] ].f[2];
		double uz_10 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[1] ].f[2] - uz_0;
		double uz_20 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[2] ].f[2] - uz_0;
		double uz_30 = mySolver->tm1[ myTopoStations[statID].nodes_to_interpolate[3] ].f[2] - uz_0;

		/* ========= */
		double vx_0  = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[0] ].f[0];
		double vx_10 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[1] ].f[0] - vx_0;
		double vx_20 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[2] ].f[0] - vx_0;
		double vx_30 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[3] ].f[0] - vx_0;

		double vy_0  = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[0] ].f[1];
		double vy_10 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[1] ].f[1] - vy_0;
		double vy_20 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[2] ].f[1] - vy_0;
		double vy_30 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[3] ].f[1] - vy_0;

		double vz_0  = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[0] ].f[2];
		double vz_10 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[1] ].f[2] - vz_0;
		double vz_20 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[2] ].f[2] - vz_0;
		double vz_30 = mySolver->tm2[ myTopoStations[statID].nodes_to_interpolate[3] ].f[2] - vz_0;

		/* ========= */
		double wx_0  = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[0] ].f[0];
		double wx_10 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[1] ].f[0] - wx_0;
		double wx_20 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[2] ].f[0] - wx_0;
		double wx_30 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[3] ].f[0] - wx_0;

		double wy_0  = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[0] ].f[1];
		double wy_10 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[1] ].f[1] - wy_0;
		double wy_20 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[2] ].f[1] - wy_0;
		double wy_30 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[3] ].f[1] - wy_0;

		double wz_0  = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[0] ].f[2];
		double wz_10 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[1] ].f[2] - wz_0;
		double wz_20 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[2] ].f[2] - wz_0;
		double wz_30 = mySolver->tm3[ myTopoStations[statID].nodes_to_interpolate[3] ].f[2] - wz_0;

		 /*   get displacements */
		*dis_x = ux_0 + myTopoStations[statID].local_coord[0] * ux_20
			          + myTopoStations[statID].local_coord[1] * ux_10
			          + myTopoStations[statID].local_coord[2] * ux_30;

		*dis_y = uy_0 + myTopoStations[statID].local_coord[0] * uy_20
			          + myTopoStations[statID].local_coord[1] * uy_10
			          + myTopoStations[statID].local_coord[2] * uy_30;

		*dis_z = uz_0 + myTopoStations[statID].local_coord[0] * uz_20
			          + myTopoStations[statID].local_coord[1] * uz_10
			          + myTopoStations[statID].local_coord[2] * uz_30;

		/*   get velocities */

		*vel_x =    ( *dis_x
				  - ( vx_0   + myTopoStations[statID].local_coord[0] * vx_20
			                 + myTopoStations[statID].local_coord[1] * vx_10
			                 + myTopoStations[statID].local_coord[2] * vx_30 ) ) / theDeltaT;

		*vel_y =    ( *dis_y
				  - ( vy_0   + myTopoStations[statID].local_coord[0] * vy_20
			                 + myTopoStations[statID].local_coord[1] * vy_10
			                 + myTopoStations[statID].local_coord[2] * vy_30 ) ) / theDeltaT;

		*vel_z =    ( *dis_z
				  - ( vz_0   + myTopoStations[statID].local_coord[0] * vz_20
			                 + myTopoStations[statID].local_coord[1] * vz_10
			                 + myTopoStations[statID].local_coord[2] * vz_30 ) ) / theDeltaT;

		/* get accelerations */

		*accel_x  = ( *dis_x
				  - 2.0 * ( vx_0   + myTopoStations[statID].local_coord[0] * vx_20
			                       + myTopoStations[statID].local_coord[1] * vx_10
			                       + myTopoStations[statID].local_coord[2] * vx_30 )
				  +       ( wx_0   + myTopoStations[statID].local_coord[0] * wx_20
								   + myTopoStations[statID].local_coord[1] * wx_10
								   + myTopoStations[statID].local_coord[2] * wx_30 ) ) / theDeltaTSquared;

		*accel_y  = ( *dis_y
				  - 2.0 * ( vy_0   + myTopoStations[statID].local_coord[0] * vy_20
			                       + myTopoStations[statID].local_coord[1] * vy_10
			                       + myTopoStations[statID].local_coord[2] * vy_30 )
				  +       ( wy_0   + myTopoStations[statID].local_coord[0] * wy_20
								   + myTopoStations[statID].local_coord[1] * wy_10
								   + myTopoStations[statID].local_coord[2] * wy_30 ) ) / theDeltaTSquared;

		*accel_z  = ( *dis_z
				  - 2.0 * ( vz_0   + myTopoStations[statID].local_coord[0] * vz_20
			                       + myTopoStations[statID].local_coord[1] * vz_10
			                       + myTopoStations[statID].local_coord[2] * vz_30 )
				  +       ( wz_0   + myTopoStations[statID].local_coord[0] * wz_20
								   + myTopoStations[statID].local_coord[1] * wz_10
								   + myTopoStations[statID].local_coord[2] * wz_30 ) ) / theDeltaTSquared;

		return 1;

	}

return 0;

}


