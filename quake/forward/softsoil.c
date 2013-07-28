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
 *
 *  Created on: Mar 23, 2012
 *      Author: yigit
 */

#include <mpi.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "cvm.h"
#include "psolve.h"
#include "buildings.h"
#include "softsoil.h"
#include "octor.h"
#include "commutil.h"
#include "util.h"
#include "timers.h"
#include "quake_util.h"

//Trifunac 1996
#define  theScalingFactor  0.22

/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static int         theNumberofStrains;
static double      theBoundX;
static double      theBoundY;
static double      theStrainSpacing;
static int         theCountX;
static int         theCountY;
static int         whichPI;


static double      ***theStrainTables;

static double      theStrainAtPointSevenG[5];
static double      *theStrainDepths;
static char  	   theStrainDir[256];

/* -------------------------------------------------------------------------- */
/*       Initialization of parameters, structures and memory allocations      */
/* -------------------------------------------------------------------------- */

void softsoil_init ( int32_t myID, const char *parametersin )
{
	int     int_message[3],i,j;
	double  double_message[3];

	/* Capturing data from file --- only done by PE0 */
	if (myID == 0) {
		if ( softsoil_initparameters( parametersin ) != 0 ) {
			fprintf(stderr,"Thread 0: softsoil_initparameters: "
					"softsoil_initparameters error\n");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}
	}

	/* Broadcasting data */

	double_message[0] = theBoundX;
	double_message[1] = theBoundY;
	double_message[2] = theStrainSpacing;

	int_message[0]    = theNumberofStrains;
	int_message[1]    = theCountX;
	int_message[2]    = theCountY;

	MPI_Bcast(double_message, 3, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(int_message,    3, MPI_INT,    0, comm_solver);

	theBoundX         = double_message[0];
	theBoundY         = double_message[1];
	theStrainSpacing  = double_message[2];

	theNumberofStrains = int_message[0];
	theCountX          = int_message[1];
	theCountY          = int_message[2];

	/* allocate table of properties for all other PEs */

	if (myID != 0) {
		theStrainDepths   = (double*)malloc( sizeof(double) * (theNumberofStrains+1) );
	}

	/* Broadcast table of properties */
	MPI_Bcast(theStrainDepths, (theNumberofStrains+1), MPI_DOUBLE, 0, comm_solver);

	broadcast_char_array( theStrainDir, sizeof(theStrainDir), 0,
			comm_solver );

	if (myID != 0) {
		/*init table for pes other than 0 */
		theStrainTables = (double***)malloc( sizeof(double**) * theNumberofStrains  );

		for (i = 0; i < theNumberofStrains; i++ ) {
			theStrainTables[i] = (double**)malloc( sizeof(double*) * theCountX  );
			for (j = 0; j < theCountX; j++ ) {
				theStrainTables[i][j]= (double*)malloc( sizeof(double) * theCountY  );
			}
		}
	}

	for (i = 0; i < theNumberofStrains; i++ ) {
		for (j = 0; j < theCountX; j++ ) {
			MPI_Bcast( theStrainTables[i][j], theCountY, MPI_DOUBLE, 0, comm_solver );
		}
	}

	// From EPRI (1993) modulus reduction curves
	theStrainAtPointSevenG[0] =  1.3183e-04;
	theStrainAtPointSevenG[1] =  1.9953e-04;
	theStrainAtPointSevenG[2] =  2.9512e-04;
	theStrainAtPointSevenG[3] =  4.1687e-04;
	theStrainAtPointSevenG[4] =  5.3703e-04;

	//printf("%f %f %f \n",(theStrainTables[4][118])[118],theStrainDepths[0],theStrainDepths[5]);

	//printf("%s \n",theModulusOutputDir);
	return;
}


int32_t
softsoil_initparameters ( const char *parametersin )
{
	FILE   *fp;
	int     numStrains,x_count,y_count,i;
	double  x_lowerbnd,y_lowerbnd,spacing;
	double *straindepths;

	char  strain_output_directory[64];

	/* Opens parametersin file */

	if ( ( fp = fopen(parametersin, "r" ) ) == NULL ) {
		fprintf( stderr,
				"Error opening %s\n at softsoil_initparameters",
				parametersin );
		return -1;
	}

	/* Parses parametersin to capture soft soil single-value parameters */

	if ( ( parsetext(fp, "number_of_strain_files",      'i', &numStrains ) != 0) ||
			( parsetext(fp, "strain_directory",         's', &strain_output_directory ) != 0) ||
			( parsetext(fp, "soft_x_start",             'd', &x_lowerbnd ) != 0) ||
			( parsetext(fp, "soft_y_start",             'd', &y_lowerbnd ) != 0) ||
			( parsetext(fp, "strain_spacing",           'd', &spacing    ) != 0) ||
			( parsetext(fp, "strain_x_count",           'i', &x_count    ) != 0) ||
			( parsetext(fp, "strain_y_count",           'i', &y_count    ) != 0))
	{
		fprintf( stderr,
				"Error parsing softsoil_initparameters parameters from %s\n",
				parametersin );
		return -1;
	}

	/* Performs sanity checks */

	if ( numStrains < 0 ) {
		fprintf( stderr,
				"Illegal number of number_of_strain_files %d\n",
				numStrains );
		return -1;
	}

	if ( x_lowerbnd < 0 ) {
		fprintf( stderr,
				"Illegal soft_x_start  %f\n",
				x_lowerbnd );
		return -1;
	}

	if ( y_lowerbnd < 0 ) {
		fprintf( stderr,
				"Illegal soft_y_start %f\n",
				y_lowerbnd );
		return -1;
	}

	if ( spacing < 0 ) {
		fprintf( stderr,
				"Illegal strain_spacing %f\n",
				spacing );
		return -1;
	}

	if ( x_count < 0 ) {
		fprintf( stderr,
				"Illegal strain_x_count %d\n",
				x_count );
		return -1;
	}

	if ( y_count < 0 ) {
		fprintf( stderr,
				"Illegal strain_y_count %d\n",
				y_count );
		return -1;
	}

	/* Initialize the static global variables */

	theNumberofStrains   = numStrains;
	theBoundX            = x_lowerbnd;
	theBoundY            = y_lowerbnd;
	theStrainSpacing     = spacing;
	theCountX            = x_count;
	theCountY            = y_count;

	straindepths         = (double*)malloc( sizeof(double) * numStrains  );
	theStrainDepths      = (double*)malloc( sizeof(double) * (numStrains+1)  );


	if (straindepths == NULL )

	{
		fprintf( stderr, "Error allocating transient soft_Soil arrays"
				"in softsoil_initparameters " );
		return -1;
	}

	if ( parsedarray( fp, "strain_depths", numStrains, straindepths ) != 0)
	{
		fprintf( stderr,
				"Error parsing softsoil_initparameters list from %s\n",
				parametersin );
		return -1;
	}

	theStrainDepths[0] = 0;
	for( i = 0; i< numStrains; ++i ) {
		theStrainDepths[i+1]=straindepths[i];
	}

	free(straindepths);

	/* We DO NOT convert physical coordinates into etree coordinates here */

	strcpy( theStrainDir, strain_output_directory );

	/* Construct strains table */

	construct_strain_table( parametersin );

	fclose(fp);

	return 0;
}

void construct_strain_table ( const char *parametersin )
{

	int i,j,k;
	char line[20*theCountY];
	/* x is n_s y is e_w in hercules */

	/*             ===> Y (e-w)
	 * 	       _ _ _ _ _ _ _
	 *         |
	 *   |     |
	 *   |     |
	 *   v     |
	 *         |
	 *   X     |
	 *  (s-n)  |
	 *  	   |
	 *         |
	 *
	 *
	 *
	 * input strain file should look like this.
	 *
	 *   */

	FILE* fp;
	char filename[256];

	/*init table */

	theStrainTables = (double***)malloc( sizeof(double**) * theNumberofStrains  );

	for (i = 0; i < theNumberofStrains; i++ ) {
		theStrainTables[i] = (double**)malloc( sizeof(double*) * theCountX  );
		for (j = 0; j < theCountX; j++ ) {
			theStrainTables[i][j]= (double*)malloc( sizeof(double) * theCountY  );
		}
	}

	for (i = 0; i < theNumberofStrains; i++ ) {

		sprintf( filename,"%s/%s.%d", theStrainDir, "strains",i );

		fp = fopen(filename, "r");
		if ( fp == NULL ) {
			solver_abort ( __FUNCTION_NAME, "NULL from fopen",
					"Error opening signals" );
		}


		for ( j = 0; j < theCountX; j++ ) {
			char *strain;

			if ( fgets(line, 20*theCountY, fp) == NULL ) {
				solver_abort ( __FUNCTION_NAME, "NULL from fgets",
						"Error reading signals" );
			}

			strain = strtok(line," ");
			(theStrainTables[i][j])[0] = atof(strain);

			for ( k = 1; k < theCountY; k++ ) {
				strain = strtok(NULL," ");
				(theStrainTables[i][j])[k] = atof(strain);
			}
		}

		fclose(fp);

	}

	//printf("%f \n",(theStrainTables[4][118])[118]);
}

double get_modulus_factor(double x_m,double y_m,double z_m,
		double XMeshOrigin,double YMeshOrigin,double ZMeshOrigin) {

	int i;
	int mape_w,mapn_s;
	double modulusfactor = 1,shearstrain;

	// include lower depth bound, exclude upper depth bound
	for (i = 0; i < theNumberofStrains; i++ ) {
		if ( z_m - ZMeshOrigin >=  theStrainDepths[i] &&  z_m - ZMeshOrigin <  theStrainDepths[i+1] ){

			mape_w = floor((y_m - YMeshOrigin - theBoundY)/ theStrainSpacing);
			mapn_s = floor((x_m - XMeshOrigin - theBoundX)/ theStrainSpacing);

			if(mape_w > theCountY-1 || mapn_s > theCountX-1 || mape_w < 0 || mapn_s < 0) {
				//return 1
				return 1;
			}

			whichPI = i;
			shearstrain = (theStrainTables[i][mapn_s])[mape_w];

			//multiply with a factor
			shearstrain = theScalingFactor *shearstrain;
			modulusfactor = (get_lower_modulus_ratio(i,shearstrain) + get_upper_modulus_ratio(i,shearstrain))/2;

			if ( modulusfactor > 1 ||  modulusfactor < 0) {
				solver_abort ( __FUNCTION_NAME, "something wrong with the modulus factor",
						"modulus factor is either negative or bigger than unity" );
			}

			//printf("%d %d %f \n",mape_w,mapn_s,sqrt(modulusfactor));

			return modulusfactor;

		}
	}

	//return 1
	return 1;

}


/* these are after J.A. SANTOS And A. GOMES CORREIA(2000) */

double get_lower_modulus_ratio(int i, double shearstrain) {

	double gammastar;
	double gamma;

	gammastar = shearstrain/theStrainAtPointSevenG[i];

	//printf("%f \n",gammastar);

	if(gammastar <= 0.01 ) {
		return 1.0;
	}

	gamma = (1-tanh(0.48*log(gammastar/1.9)))/2;
	return gamma;
}

double get_upper_modulus_ratio(int i, double shearstrain) {

	double gammastar;
	double gamma;

	gammastar = shearstrain/theStrainAtPointSevenG[i];

	if(gammastar <= 0.1 ) {
		return 1;
	}

	gamma = (1 - tanh(0.46*log( (gammastar - 0.1)/3.4)))/2;
	return gamma;
}


double get_damping_ratio(double x_m,double y_m,double z_m,
		double XMeshOrigin,double YMeshOrigin,double ZMeshOrigin) {

	double modulusfactor, dampingratio;
	double plasticityindex[5];

	// Ponti et al USGS report 1998
	plasticityindex[0] = 15;
	plasticityindex[1] = 15;
	plasticityindex[2] = 25;
	plasticityindex[3] = 25;
	plasticityindex[4] = 25;

	modulusfactor = get_modulus_factor(x_m,y_m,z_m,XMeshOrigin,YMeshOrigin,ZMeshOrigin);

	//If element is not in the soft-soil box, use the old damping formula
	if (modulusfactor == 1) {
		return -1;
	}

	// Ishibashi Zhang(1993) Kramers Book
	dampingratio = 0.333*( 1 + exp(-0.0145*pow(plasticityindex[whichPI],1.3)))/2*(0.586*pow(modulusfactor,2)-1.547*modulusfactor + 1);
	return dampingratio;

}


