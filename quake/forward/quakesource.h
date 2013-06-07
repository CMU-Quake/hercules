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

/**
 * quakesource.h: source definition of earthquakes used in a generic finite
 *                element code.
 *

 * Copyright (c) 2005 Leonardo Ramirez-Guzman
 *
 * All rights reserved.  May not be used without permission, modified, or
 *                       copied
 *
 *
 * Contact:
 * Leonardo Ramirez Guzman
 * Civil and Environmental Engineering
 * Carnegie Mellon University
 * 5000 Forbes Avenue
 * Pittsburgh, PA 15213
 * lramirez@andrew.cmu.edu
 *
 */

#include "geometrics.h"



/*------------------------Information data structures --------------------*/


typedef struct numerics_info_t{

  int32_t numberoftimesteps;

  double deltat,validfrequency;

  double minimumh;

  double xlength,ylength,zlength;

}numerics_info_t;


typedef struct mpi_info_t{

  int32_t myid,groupsize;

}mpi_info_t;



/*------------Earthquake source data structure and functions------------------*/


/* typedef struct{ */

/*   double strike, dip, rake, slip, area; */

/*   vector3D_t centroid, p [ 3 ]; */

/* }triangle_t; */




typedef enum {

  RAMP = 0, SINE, QUADRATIC, RICKER, EXPONENTIAL, DISCRETE

} source_function_t;



typedef enum {

  POINT = 0, PLANE, SRFH, PLANEWITHKINKS

} source_type_t;



typedef struct ptsrc_t {

  source_function_t sourceFunctionType;

  int32_t lnid[27];

  vector3D_t localCoords;  /*Local Coords in the plane to be mapped used only
                             for terashake type earthquakes */

  vector3D_t globalCoords;

  vector3D_t domainCoords;

  double x, y, z;                   /* Local coordinate inside a linear element */

  double quad_x, quad_y, quad_z;                   /* Local coordinate inside a quadratic element */

  double strike, dip, rake;

  double delayTime, T0,Ts,Tp;

  double mu, area, maxSlip, muArea;

  double edgesize;                  /* Edge size of the containing element */

  double nodalForce[27][3];          /* nodal forces on the containing elem */

  double *displacement;              /* Displacement vector time dependant  */

  double dt,numberOfTimeSteps;

  double M0;

  double tinit, dtfunction ;

  double *slipfundiscrete;

  int nt1;

  int32_t leid;     				/* Local element id (0...7) within the quadratic element */


} ptsrc_t;




extern int theForcesBufferSize;


/*--------------------------------Functions-----------------------------------*/


/*
 *
 * compute_source_function:
 *
 */
void compute_source_function ( ptsrc_t *pointSource );




/*
 *  compute_initial_time: computes the time when the rupture is initiated in
 *                        the given station. It assumes a homogeneous radial
 *                        dependent rupture time.
 *
 *
 */
double compute_initial_time(vector3D_t station, vector3D_t hypocenter,
			    double rupturevelocity );


/* compute_local_coords: computes the coordinates of a station in the fault
 *                       plane related to the origin of the fault
 *
 *  input: point
 *         totalNodes
 *         gridx, gridy
 *
 * output: vector3D_t pointInCartesian
 *
 */
vector3D_t compute_local_coords(int32_t point, int32_t totalNodes,
				double *gridx, double *gridy );


/*
 *  print_filter_signal:
 *
 */
int PrintFilterSignal( int signalsize, double samplingfrequency,
		       double thresholdfrequency,  int m);




/*
 *  Filters a signal we add zeros such that newsize is 2^n
 *
 */

int FilterSignal ( double *signal, int signalsize,
		   double samplingfrequency, double thresholdfrequency,
		   int m );




/*
 * compute_print_source: computes the forces needed in the calculation
 *
 *    input:
 *          informationdirectory - where source related files are located
 *                      myoctree - octree to be used
 *                        mymesh - information realtive to the mesh
 *                      myforces - the load vector
 *           numericsinformation - see quakesource.h for description
 *                mpiinformation - see quakesource.h for description
 *
 *
 *    It does not check if the source is in the domain. It will IGNORE A SOURCE
 *    NOT CONTAINED.
 *
 */
int compute_print_source (const char *physicsin, octree_t *myoctree,
			  mesh_t *mymesh, numerics_info_t numericsinformation,
			  mpi_info_t mpiinformation, double globalDelayT,
			  double surfaceShift, element_type_t element_type );


void update_forceinprocessor(int32_t iForce, char *inoutprocessor, int onoff);

FILE* source_open_forces_file( const char* flags );

int source_get_local_loaded_nodes_count();
