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
//#include "commutil.h"
#include "cvm.h"
//#include "nonlinear.h"
#include "topography.h"
#include "geometrics.h"


/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

/* Permanent */
static toposolver_t       *myTopoSolver;
static int                ntp, np_ew, np_ns;
static int8_t             theMaxoctlevel;
static double             thebase_zcoord = 0.0, So, theDomainLong_ew, theDomainLong_ns;
static double             *theTopoInfo;
static etreetype_t        theEtreeType;
static int32_t            myTopoElementsCount = 0;
static int32_t            *myTopoElementsMapping;
static topometh_t         theTopoMethod;
static topostation_t      *myTopoStations;

/* Quadrature rule for an equilateral tetrahedron based upon Shunn, L. and Ham, F work.
 * Journal of Computational and Applied Mathematics. 236(2012) 4348-4364  */
/* the first 3 rows are the positions as defined by Shunn and Ham, 4th row contains the weights */

#define  GP56  gp56[5][56] = {  { 0.9551438045408220, 0.0149520651530592, 0.0149520651530592, 0.0149520651530592, 0.7799760084415400, 0.1518319491659370, 0.7799760084415400, \
		                          0.1518319491659370, 0.7799760084415400, 0.1518319491659370, 0.0340960211962615, 0.0340960211962615, 0.0340960211962615, 0.0340960211962615, \
		                          0.0340960211962615, 0.0340960211962615, 0.3549340560639790, 0.5526556431060170, 0.3549340560639790, 0.5526556431060170, 0.3549340560639790, \
		                          0.5526556431060170, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, \
		                          0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, \
		                          0.2281904610687610, 0.2281904610687610, 0.0055147549744775, 0.0055147549744775, 0.0055147549744775, 0.1961837595745600, 0.3523052600879940, \
		                          0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940, \
		                          0.0992057202494530, 0.0992057202494530, 0.0992057202494530, 0.5965649956210170, 0.1344783347929940, 0.1344783347929940, 0.1344783347929940 }, \
		                        { 0.0149520651530592, 0.9551438045408220, 0.0149520651530592, 0.0149520651530592, 0.1518319491659370, 0.7799760084415400, 0.0340960211962615, \
	   	   	   	   	   	   	   	  0.0340960211962615, 0.0340960211962615, 0.0340960211962615, 0.7799760084415400, 0.1518319491659370, 0.7799760084415400, 0.1518319491659370, \
	   	   	   	   	   	   	   	  0.0340960211962615, 0.0340960211962615, 0.5526556431060170, 0.3549340560639790, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, \
	   	   	   	   	   	   	   	  0.0462051504150017, 0.3549340560639790, 0.5526556431060170, 0.3549340560639790, 0.5526556431060170, 0.0462051504150017, 0.0462051504150017, \
	   	   	   	   	   	   	   	  0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.0055147549744775, \
	   	   	   	   	   	   	   	  0.0055147549744775, 0.0055147549744775, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.3523052600879940, 0.1961837595745600, \
	   	   	   	   	   	   	   	  0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.0992057202494530, 0.0992057202494530, 0.0992057202494530, \
	   	   	   	   	   	   	   	  0.1961837595745600, 0.3523052600879940, 0.3523052600879940, 0.1344783347929940, 0.5965649956210170, 0.1344783347929940, 0.1344783347929940 }, \
	   	   	   	   	   	   	   	{ 0.0149520651530592, 0.0149520651530592, 0.9551438045408220, 0.0149520651530592, 0.0340960211962615, 0.0340960211962615, 0.1518319491659370, \
	   	   	   	   	   	   	   	  0.7799760084415400, 0.0340960211962615, 0.0340960211962615, 0.1518319491659370, 0.7799760084415400, 0.0340960211962615, 0.0340960211962615, \
	   	   	   	   	   	   	   	  0.7799760084415400, 0.1518319491659370, 0.0462051504150017, 0.0462051504150017, 0.5526556431060170, 0.3549340560639790, 0.0462051504150017, \
	   	   	   	   	   	   	   	  0.0462051504150017, 0.5526556431060170, 0.3549340560639790, 0.0462051504150017, 0.0462051504150017, 0.3549340560639790, 0.5526556431060170, \
	   	   	   	   	   	   	   	  0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.0055147549744775, 0.0055147549744775, 0.0055147549744775, 0.2281904610687610, \
	   	   	   	   	   	   	   	  0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.3523052600879940, 0.3523052600879940, \
	   	   	   	   	   	   	   	  0.1961837595745600, 0.0992057202494530, 0.0992057202494530, 0.0992057202494530, 0.3523052600879940, 0.1961837595745600, 0.3523052600879940, \
	   	   	   	   	   	   	   	  0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.1344783347929940, 0.1344783347929940, 0.5965649956210170, 0.1344783347929940 }, \
	   	   	   	   	   	   	   	{ 0.0149520651530592, 0.0149520651530592, 0.0149520651530592, 0.9551438045408220, 0.0340960211962615, 0.0340960211962615, 0.0340960211962615, \
	   	   	   	   	   	   	   	  0.0340960211962615, 0.1518319491659370, 0.7799760084415400, 0.0340960211962615, 0.0340960211962615, 0.1518319491659370, 0.7799760084415400, \
	   	   	   	   	   	   	   	  0.1518319491659370, 0.7799760084415400, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.0462051504150017, 0.5526556431060170, \
	   	   	   	   	   	   	   	  0.3549340560639790, 0.0462051504150017, 0.0462051504150017, 0.5526556431060170, 0.3549340560639790, 0.5526556431060170, 0.3549340560639790, \
	   	   	   	   	   	   	   	  0.0055147549744775, 0.0055147549744775, 0.0055147549744775, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.2281904610687610, \
	   	   	   	   	   	   	   	  0.2281904610687610, 0.5381043228880020, 0.2281904610687610, 0.2281904610687610, 0.5381043228880020, 0.0992057202494530, 0.0992057202494530, \
	   	   	   	   	   	   	   	  0.0992057202494530, 0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.3523052600879940, 0.3523052600879940, 0.1961837595745600, \
	   	   	   	   	   	   	   	  0.3523052600879940, 0.3523052600879940, 0.1961837595745600, 0.1344783347929940, 0.1344783347929940, 0.1344783347929940, 0.5965649956210170  }, \
	   	   	   	   	   	   	   	{ 0.0010373112336140, 0.0010373112336140, 0.0010373112336140, 0.0010373112336140, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480, \
	   	   	   	   	   	   	      0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480, 0.0096016645399480, \
	   	   	   	   	   	   	      0.0096016645399480, 0.0096016645399480, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, \
	   	   	   	   	   	   	      0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, 0.0164493976798232, \
	   	   	   	   	   	   	      0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, \
	   	   	   	   	   	   	      0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0153747766513310, 0.0293520118375230, 0.0293520118375230, \
	   	   	   	   	   	   	      0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0293520118375230, \
	   	   	   	   	   	   	      0.0293520118375230, 0.0293520118375230, 0.0293520118375230, 0.0366291366405108, 0.0366291366405108, 0.0366291366405108, 0.0366291366405108  } }


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
/*                                  Functions                                 */
/* -------------------------------------------------------------------------- */

double get_thebase_topo() {
    return thebase_zcoord;
}


//returns YES if the element is air element ( Vp = -1 ), or if it composes the topography surface.
int BelongstoTopography (mesh_t *myMesh, int32_t eindex) {

    int32_t topo_eindex;
    elem_t  *elemp;
    edata_t *edata;

    /* return NO if topography is not considered */
    if ( thebase_zcoord == 0 )
    		return NO;

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

topometh_t get_topo_meth() {
    return theTopoMethod;
}

int topo_correctproperties ( edata_t *edata ) {
    if ( edata->Vp < 0) {
    	edata->Vs = 1e10;
    	return 1;
    }

    return 0;
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


/* distance from a point to a plane   */
/* Note: zp and zo are measured with respect to the local z axis of the topography. i.e: m.a.s.l values  */
double point_to_plane( double xp, double yp, double zp, double xo, double yo, double h, double zcoords[4] )
{
	double x1, y1, z1, mag, dist;
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

	x1 = xp - xo;  /* relative distances with respect to the plane origin */
	y1 = yp - yo;
	z1 = zp - zcoords[0];

	if ( x1 > y1 ) {
		cross_product ( V3, V1,  &V4);
	}
	else
	{
		cross_product ( V2, V3,  &V4);
	}

	mag     = magnitude(V4);

	/*  Sanity check   */
	if ( mag == 0 ) {

        fprintf(stderr,"Error: Found zero magnitude at \n "
                "point_to_plane fnc\n");
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
	}

	N.x[0]  = V4.x[0] / mag;
	N.x[1]  = V4.x[1] / mag;
	N.x[2]  = V4.x[2] / mag;


	dist = ( N.x[0] * x1 + N.x[1] * y1 + N.x[2] * z1 ); /* positive: outside surface;
	                                                       zero    : on surface
	                                                       negative: within surface */

	return dist;

}

/* returns the zp coordinate of a point inside a plane using bi-linear interp   */
double interp_z( double xp, double yp, double xo, double yo, double h, double zcoords[4] )
{

	double eta, psi, zp;

	eta = 2.0 * ( xp - ( xo + h / 2.0 ) ) / h;
	psi = 2.0 * ( yp - ( yo + h / 2.0 ) ) / h;

	zp  = 0.25 * ( ( 1.0 - eta ) * (1.0 - psi ) * zcoords[0] +
			       ( 1.0 - eta ) * (1.0 + psi ) * zcoords[1] +
			       ( 1.0 + eta ) * (1.0 + psi ) * zcoords[2] +
			       ( 1.0 + eta ) * (1.0 - psi ) * zcoords[3] );

	return zp;
}

/* returns the zp coordinate of a point inside a plane using linear interp   */
/*double interp_z( double xp, double yp, double xo, double yo, double h, double zcoords[4] )
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

	if ( y1 == 0 )
		return zcoords[0] + ( zcoords[3] - zcoords[0] ) * x1 / h;

	if ( y1 == h )
		return zcoords[1] + ( zcoords[2] - zcoords[1] ) * x1 / h;

	if ( x1 == 0 )
		return zcoords[0] + ( zcoords[1] - zcoords[0] ) * y1 / h;

	if ( x1 == h )
		return zcoords[3] + ( zcoords[2] - zcoords[3] ) * y1 / h;


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

	  // Sanity check
	if ( N.x[2] == 0 ) {

        fprintf(stderr,"Thread 1: Topography module interp_z fnc: "
                "Zero found when interpolating elevation\n");
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
	}

	return zcoords[0] - ( N.x[0] * x1 + N.x[1] * y1) / N.x[2];

}*/

/* returns the elevation value of a point with plane coordinates (xo,yo).
 * Elevation is measured with respect to Hercules' Z global axis  */
double point_elevation ( double xo, double yo ) {

	int i, j;
	double xp, yp, x_o, y_o, remi, remj, mesh_cz[4] = { 0 }, zp;

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

		/*  Sanity check   */
		if ( (mesh_cz[0] == NAN) || (mesh_cz[1] == NAN) || (mesh_cz[2] == NAN) || (mesh_cz[3] == NAN) ) {

	        fprintf(stderr,"Thread 1: Topography module point_elevation fnc: "
	                "NAN found when computing point elevation\n");
	        MPI_Abort(MPI_COMM_WORLD, ERROR);
	        exit(1);
		}

		zp = thebase_zcoord - interp_z( xp, yp, x_o*So, y_o*So, So, mesh_cz );

	}

	return zp;

}


/* returns the distance of a point to the surface topography */
/* cases: positive distance = point outside topography.
 *        zero distance     = point on topography.
 *        negative distance = point within topography */

double point_PlaneDist ( double xp, double yp, double zp ) {

	int i, j;
	double x_o, y_o, remi, remj, mesh_cz[4]= { 0 }, dist;

	zp = thebase_zcoord - zp; /* sea level elevation  */

	remi = modf (xp  / So, &x_o );
	remj = modf (yp  / So, &y_o );

	i = x_o;
	j = y_o;

	mesh_cz[0] =  theTopoInfo [ np_ew * i + j ];
	mesh_cz[1] =  theTopoInfo [ np_ew * i + j + 1 ];
	mesh_cz[2] =  theTopoInfo [ np_ew * ( i + 1 ) + j + 1 ];
	mesh_cz[3] =  theTopoInfo [ np_ew * ( i + 1 ) + j ];

	if ( (i == 0) || (j == 0) || (i == np_ns - 1 ) || (j == np_ew - 1) ) {

		dist = zp - ( thebase_zcoord - point_elevation(xp,yp) );
	    return dist;
	}

	dist = point_to_plane( xp, yp, zp, x_o*So, y_o*So, So, mesh_cz );

	return dist;

}

void get_airprops_topo( edata_t *edata )
{

    edata->Vs  = 1.0e10;

    /* Assign negative Vp to identify air octants */
    edata->Vp  = -1.0;

    /* Assign zero density */
    edata->rho = 0.0;

    return;
}


/* finds if point (xp,yp,zp) is in the air.
 * -1 if topography is not considered
 *  1 if it is in the air,
 *  0 if it does not   */

int find_topoAirOct( tick_t xTick, tick_t yTick, tick_t zTick,  double  ticksize )
{

    tick_t  x0 = 0, y0 = 0, z0 = 0, edgesize_half;
    int8_t  xbit, ybit, zbit, level = 0;

	 if ( thebase_zcoord == 0 )
		 return -1;

    int node_pos = topo_nodesearch ( xTick, yTick, zTick, ticksize );

    if (node_pos == 1) {

    	/* find octant coordinates */
    	while ( level != theMaxoctlevel ) {
    		edgesize_half = (tick_t)1 << (30 - level - 1);
    		xbit = (( xTick-  x0) >= edgesize_half) ? 1 : 0;
    		ybit = (( yTick - y0) >= edgesize_half) ? 1 : 0;
    		zbit = (( zTick - z0) >= edgesize_half) ? 1 : 0;

    		x0 += xbit * edgesize_half;
    		y0 += ybit * edgesize_half;
    		z0 += zbit * edgesize_half;

    		++level;
    	}

    	int topo_air = topo_crossings ( x0 * ticksize, y0 * ticksize, z0 * ticksize, ticksize * ( (tick_t)1 << (30 - level) ) );

    	if (topo_air == 0)
    		return 1;

    }

    return 0;

}



/**
 * Returns to_topoexpand, and to_toposetrec variables
 *  to_topoexpand: 1(yes), -1(no).
 *  to_toposetrec: 1 (material properties from cvm), -1 air properties.
 * */

void topo_searchII ( octant_t *leaf, double ticksize, edata_t *edata, int *to_topoExpand, int *to_topoSetrec ) {

	double   xo, yo, zo, xp, yp, zp, esize, emin, dist, fact_air=2.0, fact_mat=1.0;
	int      far_air_flag=0, near_air_flag=0, near_mat_flag=0, far_mat_flag=0;
	int      i, j, k, np=3;

	xo = leaf->lx * ticksize;
	yo = leaf->ly * ticksize;
	zo = leaf->lz * ticksize;
	esize  = (double)edata->edgesize;

	/* check for octant outside the domain boundaries */
	if ( ( (xo+esize) > theDomainLong_ns ) || ( (yo + esize) > theDomainLong_ew ) ) {
		*to_topoSetrec = -1;
		*to_topoExpand = -1;
		return;
	}

	emin = ( (tick_t)1 << (PIXELLEVEL - theMaxoctlevel) ) * ticksize;
	double Del = esize / (np - 1);

	/* check for air element with bottom face on flat topo surface */
	double cnt = 0;
	for ( i = 0; i < np; ++i ) {
		xp = xo + Del * i;
		for ( j = 0; j < np; ++j ) {
			yp = yo + Del * j;
			dist = point_PlaneDist( xp, yp, zo + esize );
			if (dist == 0)
				++cnt;
		}
	}

	if (cnt == np * np) { /*air element with bottom side on flat surface */
		*to_topoSetrec = -1;
		*to_topoExpand =  1;
		return;
	}


	for ( i = 0; i < np; ++i ) {
		xp = xo + Del * i;

		for ( j = 0; j < np; ++j ) {
			yp = yo + Del * j;

			for ( k = 0; k < np; ++k ) {
				zp = zo + Del * k;

				dist = point_PlaneDist( xp, yp, zp );

				if ( ( dist > 0 ) && ( dist > fact_air * emin ) && far_air_flag == 0 ) {
					far_air_flag = 1;
				} else if ( ( dist > 0 ) && ( dist <= fact_air * emin ) && near_air_flag == 0 ) {
					near_air_flag = 1;
				} else if ( ( dist <= 0 ) && ( abs(dist) <= fact_mat * emin ) && near_mat_flag == 0 ) {
					near_mat_flag = 1;
				} else if ( ( dist < 0 ) && ( abs(dist) > fact_mat * emin ) && far_mat_flag == 0 ) {
					far_mat_flag = 1;
				}
			} /* for every k */
		} /* for every j */
	} /* for every i */

	/* check 16 combinations */
	if (  far_air_flag == 1  ) {
		if ( ( near_air_flag == 1 ) &&
			 ( near_mat_flag == 0 ) &&
			 ( far_mat_flag == 0 ) ) {
			*to_topoSetrec = -1;
			*to_topoExpand =  1;
			return;
		} else if ( ( near_air_flag == 0 ) &&
				    ( near_mat_flag == 0 ) &&
				    ( far_mat_flag == 0 ) ) {
			*to_topoSetrec = -1;
			*to_topoExpand = -1;
			return;
		} else {
			*to_topoSetrec =  1;
			*to_topoExpand =  1;
			return;
		}
	}

	 if ( ( ( far_air_flag == 0 ) && ( near_air_flag == 0 ) && ( near_mat_flag == 1 ) ) ||
		  ( ( far_air_flag == 0 ) && ( near_air_flag == 1 ) && ( near_mat_flag == 1 ) ) ||
	      ( ( far_air_flag == 0 ) && ( near_air_flag == 1 ) && ( near_mat_flag == 0 ) && ( far_mat_flag == 1 ) ) ) {
		*to_topoSetrec = 1;
		*to_topoExpand = 1;
		return;
	} else if ( ( far_air_flag  == 0 ) &&
			    ( near_air_flag == 1 ) &&
			    ( near_mat_flag == 0 ) &&
			    ( far_mat_flag  == 0 ) ) {
		*to_topoSetrec = -1;
		*to_topoExpand =  1;
		return;
	} else 	if ( ( far_air_flag  == 0 ) &&
			     ( near_air_flag == 0 ) &&
			     ( near_mat_flag == 0 ) &&
			     ( far_mat_flag  == 1 ) ) { /* buried element */
		*to_topoSetrec =  1;
		*to_topoExpand = -1;
		return;
	} else { /* no combination found: This should not happen */

        fprintf(stderr,"Thread 1: topo_search: "
                "Could not find a matching case for octant\n");
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
	}

}


/* Returns the real volume held by the tetrahedral elements */
void TetrVolume ( double xo, double yo, double zo, double esize,
		         double VolTetr[5]) {

	int i, j, m;
	double GP56, NGp=56;
	double TETRACOOR, TETRACOORSYMM;
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

	double eta,psi,gam; /* natural coordinates of the tetrahedron with sides = 1  */
	double x1, x2, x3;

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


		for ( i = 0; i < NGp; ++i ) {

			/* Gauss points wrt the center of the equilateral tetrahedron. See Shun and Ham paper */
			x1 = - 0.50             * gp56[0][i] +            0.50  * gp56[1][i];
			x2 = - sqrt(3.0) / 6.0  * gp56[0][i] - sqrt(3.0) / 6.0  * gp56[1][i] + sqrt(3.0) / 3.0  * gp56[2][i];
			x3 = - sqrt(6.0) / 12.0 * gp56[0][i] - sqrt(6.0) / 12.0 * gp56[1][i] - sqrt(6.0) / 12.0 * gp56[2][i] + sqrt(6.0) / 4.0 * gp56[3][i];

			/* translate coordinates wrt the first node*/
            x1 +=  0.50;
            x2 += sqrt(3.0) / 6.0;
            x3 += sqrt(6.0) / 12.0;

            /*  mapping to the natural rectangular tetrahedron */
            psi = 2.0 / sqrt(3.0) * x2 - 1.0 / sqrt(6.0) * x3;
            eta =                   x1 - 1.0 / sqrt(3.0) * x2 - 1.0 / sqrt(6.0) * x3;
            gam = 3.0 / sqrt(6.0) * x3;

            /* get real coord of Gauss point */
            xm = ( 1 - eta - psi - gam ) * MCx[m][0] + psi * MCx[m][1] + eta * MCx[m][2] + gam * MCx[m][3];
            ym = ( 1 - eta - psi - gam ) * MCy[m][0] + psi * MCy[m][1] + eta * MCy[m][2] + gam * MCy[m][3];
            zm = ( 1 - eta - psi - gam ) * MCz[m][0] + psi * MCz[m][1] + eta * MCz[m][2] + gam * MCz[m][3];

            zt = point_elevation ( xm, ym );

            if ( zm >= zt ) {
            	Vpr +=  gp56[4][i];
            }
		}

		VolTetr[m] = Vpr; /* percentage of the tetrahedron's volume filled by topography  */

	} /* for each tetrahedra */

	/* Check for empty cube   */
	/* Todo: Think of a better way to handle empty tetrahedra */
//	if ( (VolTetr[0] == 0) && (VolTetr[1] == 0) && (VolTetr[2] == 0) && (VolTetr[3] == 0) && (VolTetr[4] == 0) ) {
//
//        fprintf(stdout,"Thread 1: TetrVolume: "
//                "Could not find enclosed volume for topographic element: xo=%f, yo=%f, zo=%f, esize=%f \n",xo, yo, zo, esize );
//
//	}

//		for ( i = 0; i < 5; ++i ) {
//			VolTetr[i] = 0.001;  /* Doriam. I am assuming a 0.5% percent of mass within the tetrahedron in cases when the quadrature scheme fails to detect the volume.
//		                This is done to avoid numerical artifacts */
//		}

}

/**
  * Depending on the position of a node wrt to the surface it returns:
  *  1: node on, or outside topography,
  *  0: node inside topography,
  * -1: topography not considered
  */
 int topo_nodesearch ( tick_t x, tick_t y, tick_t z, double ticksize ) {

	 double xo, yo, zo, dist;

	 if ( thebase_zcoord == 0 )
		 return -1;

	 xo = x * ticksize;
	 yo = y * ticksize;
	 zo = z * ticksize;

	 dist = point_PlaneDist( xo, yo, zo );

	 if ( ( dist >= 0 ) && ( thebase_zcoord > 0 ) )
		 return 1;

	 return 0;

 }


/* Checks the type of element wrt the topography surface:
 * -1: if topography is NOT considered.
 *  0: if air element.
 *  1: if it crosses the topographic surface.
 *  2: if it's buried with top face on flat topography
 *  3: if it's a fully buried element
 * */
int topo_crossings ( double xo, double yo, double zo, double esize ) {

	int i,j,k, np=5, air_flag=0, mat_flag=0, cnt_top=0, cnt_bott=0;
	double xp, yp, zp, dist;


	if ( esize == 0 ) {
        fprintf(stderr, "Error computing topography crossings, esize must be greater that zero: %f\n"
                ,esize );
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	double Del = esize / (np-1);

   if ( thebase_zcoord == 0 ) /* If no topography   */
		   return -1;

	/* 1) Elevation check. np^2 points check */
	for ( i = 0; i < np; ++i ) {
		xp = xo + Del * i;

		for ( j = 0; j < np; ++j ) {
			yp = yo + Del * j;
			zp = point_elevation ( xp, yp );

			if ( ( zp > zo ) && ( zp < (zo + esize) ) ) {
				return 1; /* crossing found it */
			} else if ( zp == zo ) {
				++cnt_top;
			} else if ( zp == zo + esize ) {
				++cnt_bott;
			}
		}
	}

	/* buried element with top face on flat topography   */
	if ( cnt_top == np * np )
		return 2;

	/* air element with bottom face on flat topography   */
	if ( cnt_bott == np * np )
		return 0;

	/* 2) Distance check. Checks the perpendicular distance of np^3 points inside
	 * the element to the external surface */
	for ( i = 0; i < np; ++i ) {
		zp = zo + Del * i;

		for ( j = 0; j < np; ++j ) {
			xp = xo + Del * j;

			for ( k = 0; k < np; ++k ) {
				yp = yo + Del * k;

				dist = point_PlaneDist( xp, yp, zp );

				if ( ( dist > 0 ) && air_flag == 0 ) {
					air_flag = 1;
				} else if ( ( dist < 0 ) && mat_flag == 0 ) {
					mat_flag = 1;
				}

				if ( (air_flag == 1) && (mat_flag==1) ) /* Found crossing  */
					return 1;

			} /* for every k */
		} /* for every j */
	} /* for every i */


	if ( (air_flag == 0) && (mat_flag==1) ) /* Fully buried element  */
		return 3;


	/* Has to be an air element */
	return 0;

}

int topo_setrec ( octant_t *leaf, double ticksize,
                   edata_t *edata, etree_t *cvm ) {

    int       res_exp, res_setr;

    topo_searchII( leaf, ticksize, edata, &res_exp,  &res_setr );

    if (  res_setr == -1  ) {
        get_airprops_topo( edata );
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
                     edata_t  *edata, double theFactor ) {
    int          res_exp, res_setr;

    double emin = ( (tick_t)1 << (PIXELLEVEL - theMaxoctlevel) ) * ticksize;

    topo_searchII( leaf, ticksize, edata, &res_exp,  &res_setr );

    /* check minimum size provided against Vs rule  */
    if ( ( res_exp != -1 ) && ( res_setr != 1 ) ) { /* If not buried element  */
    	if (edata->Vs / theFactor < emin  ) {

            fprintf(stderr,"Thread 2: topo_toexpand: "
                    "Cannot ensure elements of equal size in topography mesh."
                    " Increase the minimum octant level value\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
    	}
    }

	if ( ( res_exp == 1 )  && ( leaf->level < theMaxoctlevel ) )
	return 1;

    return -1;

}

int32_t
topography_initparameters ( const char *parametersin ) {
    FILE                *fp;
    FILE                *fp_topo;
    int                 iTopo;
    int8_t              Maxoctlevel;
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
    } else if ( strcasecmp(etree_model, "sqd") == 0 ) {
        etreetype = SQD;
    } else {
        fprintf(stderr,
                "Illegal etree_type model for topography analysis"
                "(Sqd, Full): %s\n", etree_model);
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
    	double      aux_vol[5] = { 0 };
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

		if ( ( Vp != -1 ) && ( topo_crossings ( xo, yo, zo, esize ) == 1 )  && (
				( xo != 0.0 ) &&
				( xo + esize != theDomainLong_ns ) &&
				( yo != 0.0 ) &&
				( yo + esize != theDomainLong_ew ) ) ) {
			/* Check for enclosed volume   */
			if (theTopoMethod == VT) {
				TetrVolume ( xo, yo, zo, esize, aux_vol );
				if ( ( aux_vol[0]==0 ) && ( aux_vol[1]==0 ) && ( aux_vol[2]==0 ) && ( aux_vol[3]==0 ) && ( aux_vol[4]==0 ) )  /* small enclosed volume */
					get_airprops_topo( edata );  /* consider the element as an  air element */
				else
					count++;
			} else
				count++;
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

    	if ( ( Vp != -1 ) && ( topo_crossings ( xo, yo, zo, esize ) == 1 ) && (
				 ( xo != 0.0 ) &&
			     ( xo + esize != theDomainLong_ns ) &&
			     ( yo != 0.0 ) &&
			     ( yo + esize != theDomainLong_ew ) ) ) {

    		myTopoElementsMapping[count] = eindex;
    		count++;
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

        /* get Tetrahedra volumes using Shunn and Ham quadrature rule */
        if ( theTopoMethod == VT )
        	TetrVolume ( xo, yo, zo, esize, ecp->tetraVol );

    } /* for all elements */

}

void compute_addforce_topoEffective ( mesh_t     *myMesh,
                                      mysolver_t *mySolver,
                                      double      theDeltaTSquared ) {

	int       i;
	int32_t   eindex;
	int32_t   topo_eindex;
	fvector_t localForce[8];
    fvector_t curDisp[8];

	if ( theTopoMethod == FEM )
		return;


	/* Loop on the number of elements */
	for (topo_eindex = 0; topo_eindex < myTopoElementsCount; topo_eindex++) {


			elem_t      *elemp;
			edata_t     *edata;
			node_t      *node0dat;
			e_t         *ep;

			eindex = myTopoElementsMapping[topo_eindex];

			/* Capture the table of elements from the mesh and the size
			 * This is what gives me the connectivity to nodes */
			elemp                   = &myMesh->elemTable[eindex];
			edata                   = (edata_t *)elemp->data;
			topoconstants_t topo_ec = myTopoSolver->topoconstants[topo_eindex];
			node0dat                = &myMesh->nodeTable[elemp->lnid[0]];
			ep                      = &mySolver->eTable[eindex];

	        /* get coordinates of element zero node */
			double xo = (node0dat->x)*(myMesh->ticksize);
			double yo = (node0dat->y)*(myMesh->ticksize);
			double zo = (node0dat->z)*(myMesh->ticksize);

			memset( localForce, 0, 8 * sizeof(fvector_t) );

			double b_over_dt = ep->c3 / ep->c1;
			/* get cube's displacements */
	        for (i = 0; i < 8; i++) {
	            int32_t    lnid = elemp->lnid[i];
	            fvector_t* tm1Disp = mySolver->tm1 + lnid;
	            fvector_t* tm2Disp = mySolver->tm2 + lnid;

//	            curDisp[i].f[0] = tm1Disp->f[0];
//	            curDisp[i].f[1] = tm1Disp->f[1];
//	            curDisp[i].f[2] = tm1Disp->f[2];

	            /* Rayleigh damping is considered simultaneously   */
	            curDisp[i].f[0] = tm1Disp->f[0] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[0];
	            curDisp[i].f[1] = tm1Disp->f[1] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[1];
	            curDisp[i].f[2] = tm1Disp->f[2] * ( 1.0 + b_over_dt ) - b_over_dt * tm2Disp->f[2];
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
		          double mu, double lambda, double xo, double yo, double zo ) {

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
/*                        Topography Output to Stations                       */
/* -------------------------------------------------------------------------- */

void topography_stations_init( mesh_t    *myMesh,
                               station_t *myStations,
                               int32_t    myNumberOfStations) {

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
		                        double xo, double yo, double zo, double h ) {

	int i;
	double eta, psi, gamma;
	double xp1, yp1, zp1, tol=-1.0e-5;

	if ( ( ( xo <  theDomainLong_ns / 2.0  ) && ( yo <  theDomainLong_ew / 2.0 ) ) ||
	     ( ( xo >= theDomainLong_ns / 2.0 ) && ( yo >= theDomainLong_ew / 2.0 ) ) )
	{

		for (i = 0 ;  i < 5; i++) {

			switch ( i ) {

			case ( 0 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = xp1 / h;
				psi   = yp1 / h;
				gamma = zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

		for (i = 0 ;  i < 5; i++) {

			switch ( i ) {

			case ( 0 ):

				xp1 = point.x[0] - xo;
				yp1 = point.x[1] - yo;
				zp1 = point.x[2] - zo;

				eta   = ( xp1 - yp1 - zp1 ) / h;
				psi   = yp1 / h;
				gamma = zp1 / h;

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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

				if ( ( ( 1.0 - eta - psi - gamma ) >=  tol ) &&
					 ( ( 1.0 - eta - psi - gamma ) <=  1.0 - tol ) && ( eta >= 0 ) && ( psi >= 0 ) && ( gamma >= 0) ) {

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
			"x=%f, y=%f, z=%f\n"
			"in octant xo=%f, yo=%f, zo=%f, esize=%f\n",
			point.x[0], point.x[1], point.x[2], xo, yo, zo, h);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);

}

int compute_tetra_displ (double *dis_x, double *dis_y, double *dis_z,
						 double *vel_x, double *vel_y, double *vel_z,
						 double *accel_x, double *accel_y, double *accel_z,
						 double theDeltaT, double theDeltaTSquared,
		                 int32_t statID, mysolver_t *mySolver) {

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


