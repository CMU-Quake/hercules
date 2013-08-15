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

#include <stdlib.h>
#include <string.h>

#include "cvm.h"
#include "psolve.h"
#include "commutil.h"
#include "util.h"
#include "quake_util.h"
#include "damping.h"
#include "buildings.h"
#include "meshformatlab.h"


/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static char      theMeshDirMatlab[256];


static double 	 theMatlabXMax,   theMatlabXMin,
theMatlabYMax,   theMatlabYMin,
theMatlabZMax,   theMatlabZMin;

/* -------------------------------------------------------------------------- */
/*                                 Utilities                                  */
/* -------------------------------------------------------------------------- */

/**
 * Saves coordinates and data of specified elements as an output binary file
 * for MATLAB. also print damping ratios for rayleigh
 */
void saveMeshCoordinatesForMatlab( mesh_t      *myMesh,       int32_t myID,
		const char  *parametersin, double  ticksize,
		damping_type_t theTypeOfDamping, double xoriginm, double yoriginm,
		double zoriginm, noyesflag_t includeBuildings)
{

	int      i, j, k, counter = 0;

	double   double_message[6];
	double   *auxiliar;
	double   x_coord, y_coord, z_coord;

	edata_t  *edata_temp;

	FILE     *meshcoordinates;
	FILE     *meshdata;
	FILE     *fp;

	static char    filename_coord [256];
	static char    filename_data  [256];


	/*Allocate memory to temporarily store coordinates of region to be plotted*/

	auxiliar = (double*)malloc( sizeof(double) * 6 );

	if ( auxiliar == NULL ) {
		fprintf( stderr, "Err allocing mem in saveMeshCoordinatesForMatlab" );
	}

	/**
	 * Parsing the name of the directory of mesh coordinates and corners of
	 * region to be plotted for Matlab .
	 */
	if ( myID == 0 ) {

		if ( (fp = fopen ( parametersin, "r")) == NULL ) {
			solver_abort ("saveMeshCoordinatesForMatlab", parametersin,
					"Error opening parameters.in configuration file");
		}

		if ( parsetext ( fp, "mesh_coordinates_directory_for_matlab",'s',
				theMeshDirMatlab )!= 0 ) {
			solver_abort ( "saveMeshCoordinatesForMatlab", NULL,
					"Error parsing fields from %s\n", parametersin); }

		if ( parsedarray( fp, "mesh_corners_matlab", 6 , auxiliar ) != 0)
		{
			fprintf(stderr, "Error parsing mesh_corners_matlab list from %s\n",
					parametersin);
		}

		/**
		 *We convert physical coordinates into etree coordinates
		 *(divide by ticksize)
		 *
		 */
		theMatlabXMin  =  auxiliar[0] / ticksize;
		theMatlabYMin  =  auxiliar[1] / ticksize;
		theMatlabXMax  =  auxiliar[2] / ticksize;
		theMatlabYMax  =  auxiliar[3] / ticksize;
		theMatlabZMin  =  auxiliar[4] / ticksize;
		theMatlabZMax  =  auxiliar[5] / ticksize;

		free( auxiliar );

		hu_fclose( fp );

	}

	/* Broadcasting data */

	double_message[0]  = theMatlabXMin;
	double_message[1]  = theMatlabYMin;
	double_message[2]  = theMatlabXMax;
	double_message[3]  = theMatlabYMax;
	double_message[4]  = theMatlabZMin;
	double_message[5]  = theMatlabZMax;

	MPI_Bcast( double_message , 6 , MPI_DOUBLE , 0 , comm_solver);

	theMatlabXMin  = double_message[0] ;
	theMatlabYMin  = double_message[1] ;
	theMatlabXMax  = double_message[2] ;
	theMatlabYMax  = double_message[3] ;
	theMatlabZMin  = double_message[4] ;
	theMatlabZMax  = double_message[5] ;

	broadcast_char_array( theMeshDirMatlab, sizeof(theMeshDirMatlab), 0,
			comm_solver );

	/**
	 *Coordinates and data in the specified region are written to a binary file
	 *within each processor
	 */

	for ( i = 0; i < ( myMesh->lenum ); ++i ) {

		/* j is like a node id here */
		j = myMesh->elemTable[i].lnid[0]; /* looking at first node of the element(at top) */

		edata_temp = (edata_t *)myMesh->elemTable[i].data;

		/* these are  lower left  coordinates */
		x_coord = myMesh->nodeTable[j].x;
		y_coord = myMesh->nodeTable[j].y;
		z_coord = myMesh->nodeTable[j].z;


		if ( (z_coord >= theMatlabZMin) && (z_coord < theMatlabZMax) )
		{
			if ( (y_coord >= theMatlabYMin) && (y_coord < theMatlabYMax) )
			{
				if ( (x_coord >= theMatlabXMin) && (x_coord <  theMatlabXMax))
				{

					if ( counter == 0 ) {
						/* In order to open file just once and not creating any
						 * empty files */
						sprintf(filename_coord, "%s/mesh_coordinates.%d",
								theMeshDirMatlab, myID);
						sprintf(filename_data,  "%s/mesh_data.%d",
								theMeshDirMatlab, myID);

						meshcoordinates = hu_fopen ( filename_coord, "wb" );
						meshdata        = hu_fopen ( filename_data,  "wb" );

					}

					counter++;

					for ( k = 0 ; k < 8 ; ++k ) {
						/* j is like a node id here */
						j = myMesh->elemTable[i].lnid[k];

						fwrite( &(myMesh->nodeTable[j].x), sizeof(myMesh->nodeTable[j].x),
								1,meshcoordinates);
						fwrite( &(myMesh->nodeTable[j].y), sizeof(myMesh->nodeTable[j].y),
								1,meshcoordinates);
						fwrite( &(myMesh->nodeTable[j].z), sizeof(myMesh->nodeTable[j].z),
								1,meshcoordinates);
					}

					fwrite( &(edata_temp->Vs),  sizeof(edata_temp->Vs),  1, meshdata );
					fwrite( &(edata_temp->Vp),  sizeof(edata_temp->Vp),  1, meshdata );
					fwrite( &(edata_temp->rho), sizeof(edata_temp->rho), 1, meshdata );

					/**
					 * If you want to have damping ratios, change the 0.
					 */
					if (theTypeOfDamping == RAYLEIGH && 0) {

						int32_t   n_0;
						double    x_m,y_m,z_m;
						float     zeta;

						n_0 = myMesh->elemTable[i].lnid[0];
						z_m = zoriginm + (ticksize)*myMesh->nodeTable[n_0].z;
						x_m = xoriginm + (ticksize)*myMesh->nodeTable[n_0].x;
						y_m = yoriginm + (ticksize)*myMesh->nodeTable[n_0].y;

						/* Shift the domain if buildings are considered */
						if ( includeBuildings == YES ) {
							z_m -= get_surface_shift();
						}

						//if ( softerSoil == YES ) {
						/* Get it from the damping vs strain curves */
						//		zeta = get_damping_ratio(x_m,y_m,z_m,xoriginm,yoriginm,zoriginm);
						//	}
						//	else {
						/* New formula for damping according to Graves */
						//		zeta = 10 / edata_temp->Vs;
						//	}

						//If element is not in the soft-soil box, use the Graves damping formula
						if (zeta == -1) {
							zeta = 10 / edata_temp->Vs;
						}

						//If element is in the building, use 5% damping.
						if (z_m < 0) {
							zeta = 0.05;
						}

						fwrite( &(zeta), sizeof(zeta), 1, meshdata );

					}

				}
			}
		}
	}

	if ( counter != 0 ) {
		hu_fclose( meshcoordinates );
		hu_fclose( meshdata );
	}
}
