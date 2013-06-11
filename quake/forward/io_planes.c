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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <float.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>

#include "io_planes.h"
#include "psolve.h"
#include "geometrics.h"
#include "quake_util.h"
#include "util.h"
#include "cvm.h"
#include "topography.h"


extern int compute_csi_eta_dzeta( octant_t *octant, vector3D_t pointcoords,
				  vector3D_t *localcoords, int32_t *localNodeID );

#define LINESIZE       512

#define STOP_SERVER    10 /* DATA MESSAGE TO STOP IO SERVER */

#define MAX_STRIPS_PER_PLANE 5000
/*Could be determined dynmamically but would cost search_point calls*/


int  New_planes_print(int32_t myID, mysolver_t* mySolver, int theNumberOfPlanes);
void New_planes_setup(int32_t myID, int32_t *thePlanePrintRate, int theNumberOfPlanes,
		      const char *numericalin, double surfaceShift,
		      double *theSurfaceCornersLong, double *theSurfaceCornersLat,
		      double theDomainX, double theDomainY, double theDomainZ,
		      char* planes_input_file);
void New_planes_close(int32_t myID, int theNumberOfPlanes);
void planes_IO_PES_main(int32_t myID);
static void New_output_planes_construct_strips(int32_t myID, int theNumberOfPlanes);

static void print_planeinfo(int32_t myID, int theNumberOfPlanes);

int  Old_planes_print(int32_t myID, mysolver_t* mySolver, int theNumberOfPlanes);
void Old_planes_setup(int32_t myID, int32_t *thePlanePrintRate, int theNumberOfPlanes, 
		      const char *numericalin, double surfaceShift,
		      double *theSurfaceCornersLong, double *theSurfaceCornersLat,
		      double theDomainX, double theDomainY, double theDomainZ,
		      char* planes_input_file);
void Old_planes_close(int32_t myID, int theNumberOfPlanes);
static int  Old_print_plane_displacements(int32_t myID, int ThisPlane);
static void Old_output_planes_construct_strips(int32_t myID, int theNumberOfPlanes);

typedef struct plane_strip_element_t {

    int32_t nodestointerpolate[8];
    vector3D_t localcoords;  /* csi, eta, dzeta in (-1,1) x (-1,1) x (-1,1)*/

} plane_strip_element_t;


typedef struct plane_t {

    vector3D_t origincoords;              /**< cartesian */

    FILE *fpoutputfile,*fpplanecoordsfile;

    int numberofstepsalongstrike, numberofstepsdowndip;
    double stepalongstrike, stepdowndip, strike, dip, rake;

    int IO_PE_num;
    int numberofstripsthisplane;
    int globalnumberofstripsthisplane, globalnumberofstripsthisplane_recieved; /*valid only on output PE's*/

    int stripstart[MAX_STRIPS_PER_PLANE];
    int stripend[MAX_STRIPS_PER_PLANE];
    int topo_plane;
    plane_strip_element_t * strip[MAX_STRIPS_PER_PLANE];

} plane_t;

static plane_t* thePlanes;
static int planes_GlobalLargestStripCount;
static int planes_LocalLargestStripCount;
static double* planes_output_buffer;
static double* planes_stripMPISendBuffer;
static double* planes_stripMPIRecvBuffer;

/******* Wrapper on routines to maintain backwards compatiblity for single core routines ******/

int planes_print(int32_t myID, int IO_pool_pe_count, int theNumberOfPlanes, mysolver_t* mySolver){
    if (IO_pool_pe_count)
         New_planes_print(myID, mySolver, theNumberOfPlanes);
    else
         Old_planes_print(myID, mySolver, theNumberOfPlanes);
    return 1;
}

void planes_setup(int32_t myID, int32_t *thePlanePrintRate, int IO_pool_pe_count, 
                  int theNumberOfPlanes, const char *numericalin, double surfaceShift,
		  double *theSurfaceCornersLong, double *theSurfaceCornersLat,
		  double theDomainX, double theDomainY, double theDomainZ,
		  char* planes_input_file){

//	IO_pool_pe_count = 1;
    if (IO_pool_pe_count)
	New_planes_setup(myID, thePlanePrintRate, theNumberOfPlanes, numericalin, surfaceShift,
			 theSurfaceCornersLong, theSurfaceCornersLat,
                         theDomainX, theDomainY, theDomainZ,
                         planes_input_file);
    else
	Old_planes_setup(myID, thePlanePrintRate, theNumberOfPlanes, numericalin, surfaceShift,
			 theSurfaceCornersLong, theSurfaceCornersLat,
                         theDomainX, theDomainY, theDomainZ,
                         planes_input_file);
}

void planes_close(int32_t myID, int IO_pool_pe_count, int theNumberOfPlanes){
    if (IO_pool_pe_count)
	New_planes_close(myID, theNumberOfPlanes);
    else
	Old_planes_close(myID, theNumberOfPlanes);
}


/********************* Old Version To Maintain Backwards Compatbility For One Core ******/

/**
 * Interpolate the displacenents and communicate strips to printing PE.
 * Bigben version has no control flow, just chugs through planes.
 */
int Old_planes_print(int32_t myID, mysolver_t* mySolver, int theNumberOfPlanes)
{
    int iPlane, iPhi;
    int stripLength;
    int iStrip, elemnum, rStrip;
    /* Auxiliar array to handle shapefunctions in a loop */
    double  xi[3][8]={ {-1,  1, -1,  1, -1,  1, -1, 1} ,
		       {-1, -1,  1,  1, -1, -1,  1, 1} ,
		       {-1, -1, -1, -1,  1,  1,  1, 1} };

    double      phi[8];
    double      displacementsX, displacementsY, displacementsZ;
    //    int32_t     howManyDisplacements, iDisplacement;
    //    vector3D_t* localCoords; /* convinient renaming */
    //    int32_t*    nodesToInterpolate[8];
    MPI_Status  status;
    //    MPI_Request sendstat;
    int recvStripCount, StartLocation;

    for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	for (iStrip = 0; iStrip < thePlanes[iPlane].numberofstripsthisplane;
	     iStrip++) {

	    /* interpolate points directly into send buffer for this strip */
	    for (elemnum = 0;
		 elemnum < (thePlanes[iPlane].stripend[iStrip]
			    -thePlanes[iPlane].stripstart[iStrip]+1);
		 elemnum++)
		{
		    displacementsX = 0;displacementsY = 0;displacementsZ = 0;


		    for (iPhi = 0; iPhi < 8; iPhi++){
			phi[iPhi] = ( 1 + (xi[0][iPhi]) *
				      (thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[0]) )
			    * ( 1 + (xi[1][iPhi]) *
				(thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[1]) )
			    * ( 1 + (xi[2][iPhi]) *
				(thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[2]) ) /8;

			displacementsX += phi[iPhi]*
			    (mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[0]);
			displacementsY += phi[iPhi]*
			    (mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[1]);
			displacementsZ += phi[iPhi]*
			    (mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[2]);
		    }

		    planes_stripMPISendBuffer[elemnum*3]     = (double) (displacementsX);
		    planes_stripMPISendBuffer[elemnum*3 + 1] = (double) (displacementsY);
		    planes_stripMPISendBuffer[elemnum*3 + 2] = (double) (displacementsZ);

		}

	    /* add start location as last element, add .1 to insure
	       int-float conversion (yes, bad form)*/
            stripLength = thePlanes[iPlane].stripend[iStrip]
		- thePlanes[iPlane].stripstart[iStrip] + 1;

            planes_stripMPISendBuffer[stripLength*3]
		= (double) (thePlanes[iPlane].stripstart[iStrip] + 0.1);

            if (myID==0) { /* don't try to send to same PE, just memcopy */
		memcpy( &(planes_output_buffer[ thePlanes[iPlane].stripstart[iStrip]*3 ] ), planes_stripMPISendBuffer,
			stripLength*3*sizeof(double) );
            }
            else {
		MPI_Send( planes_stripMPISendBuffer, (stripLength*3)+1,
			  MPI_DOUBLE, 0 , iPlane, comm_solver );
            }
	} /*for iStrips*/

	/* IO PE recieves for this plane */
	if(myID == 0){
	    /* Recv all strips not directly memory copied above */
	    for (rStrip = 0;
		 rStrip < (thePlanes[iPlane].globalnumberofstripsthisplane
			   - thePlanes[iPlane].numberofstripsthisplane);
		 rStrip++) {

		MPI_Recv( planes_stripMPIRecvBuffer,
			  planes_GlobalLargestStripCount * 3 + 1, MPI_DOUBLE,
			  MPI_ANY_SOURCE, iPlane, comm_solver, &status );
		MPI_Get_count(&status, MPI_DOUBLE, &recvStripCount);
		StartLocation = (int)planes_stripMPIRecvBuffer[recvStripCount-1];
		memcpy( &(planes_output_buffer[StartLocation * 3]),
			planes_stripMPIRecvBuffer,
			(recvStripCount-1)*sizeof(double) );
	    }
	}

	/* do print for just this plane */
	Old_print_plane_displacements(myID, iPlane );

	/* may not be required */
	MPI_Barrier( comm_solver );
    } /* for iPlane */

    return 1;
}


static int Old_print_plane_displacements(int32_t myID, int ThisPlane)
{
    int ret;

    if  (myID == 0){
	ret = fwrite( planes_output_buffer, sizeof(double),
                      3 * thePlanes[ThisPlane].numberofstepsalongstrike
		      * thePlanes[ThisPlane].numberofstepsdowndip,
		      thePlanes[ThisPlane].fpoutputfile );
	if (ret != (3*thePlanes[ThisPlane].numberofstepsalongstrike
		    * thePlanes[ThisPlane].numberofstepsdowndip) ) {
	    fprintf( stderr, "Error writing all values in planes file %d.\n",
		     ThisPlane );
	    exit(1);
	}
    }

    return 1;
}



/**
 * This is the stripped BigBen version.  It is simplified:
 * 1. It assumes all output is PE0.
 * 2. It does not have a strip size limit.
 */

void Old_planes_setup ( int32_t     myID, int32_t *thePlanePrintRate,
			int theNumberOfPlanes,
                        const char *numericalin,
                        double      surfaceShift,
			double *theSurfaceCornersLong, double *theSurfaceCornersLat,
			double theDomainX, double theDomainY, double theDomainZ,
			char* planes_input_file) {

    char thePlaneDirOut[256];
    static const char* fname = "output_planes_setup()";
    double     *auxiliar;
    char       planedisplacementsout[1024], planecoordsfile[1024];
    int        iPlane, iCorner;
    vector3D_t originPlaneCoords;
    int        largestplanesize;
    FILE*      fp;
    FILE*      fp_planes;

    if (myID == 0) {

	/* obtain the general planes specifications in parameter file */
	if ( (fp = fopen ( numericalin, "r")) == NULL ) {
	    solver_abort (fname, numericalin,
			  "Error opening parameters configuration file");
	}
	if ( (parsetext(fp, "output_planes_print_rate", 'i',
			thePlanePrintRate) != 0) )
	    {
		solver_abort (fname, NULL, "Error parsing output_planes_print_rate field from %s\n",
			      numericalin);
	    }
	auxiliar = (double *)malloc(sizeof(double)*8);
	if ( parsedarray( fp, "domain_surface_corners", 8 ,auxiliar) !=0 ) {
	    solver_abort (fname, NULL, "Error parsing domain_surface_corners field from %s\n",
			  numericalin);
	}
	for ( iCorner = 0; iCorner < 4; iCorner++){
	    theSurfaceCornersLong[ iCorner ] = auxiliar [ iCorner * 2 ];
	    theSurfaceCornersLat [ iCorner ] = auxiliar [ iCorner * 2 +1 ];
	}
	free(auxiliar);

	if ((parsetext(fp, "output_planes_directory", 's',
		       &thePlaneDirOut) != 0))
	    {
		solver_abort( fname, NULL,
			      "Error parsing output planes directory from %s\n",
			      numericalin );
	    }
	thePlanes = (plane_t *) malloc ( sizeof( plane_t ) * theNumberOfPlanes);
	if ( thePlanes == NULL ) {
	    solver_abort( fname, "Allocating memory for planes array",
			  "Unable to create plane information arrays" );
	}
	fclose(fp);


	/* read in the individual planes coords in planes file */
	if ( (fp_planes = fopen ( planes_input_file, "r")) == NULL ) {
	    solver_abort (fname, planes_input_file,
			  "Error opening planes configuration file");
	}

	int found=0;
	while (!found) {
	    char line[LINESIZE];
	    char *name;
	    char delimiters[] = " =\n";
	    char querystring[] = "output_planes";
	    /* Read in one line */
	    if (fgets (line, LINESIZE, fp) == NULL) {
		break;
	    }
	    name = strtok(line, delimiters);
	    if ( (name != NULL) && (strcmp(name, querystring) == 0) ) {
		largestplanesize = 0;
		found = 1;
		for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
		    int fret0, fret1;
		    FILE* fp0;
		    FILE* fp1;
		    fret0 = fscanf( fp_planes," %lf %lf %lf",
				    &thePlanes[iPlane].origincoords.x[0],
				    &thePlanes[iPlane].origincoords.x[1],
				    &thePlanes[iPlane].origincoords.x[2] );

		    /* RICARDO: Correction for buildings case */
		    thePlanes[iPlane].origincoords.x[2] += surfaceShift;

		    /* origin coordinates must be in Lat, Long and depth */
		    if (fret0 == 0) {
			solver_abort (fname, NULL,
				      "Unable to read planes origin in %s",
				      planes_input_file);
		    }
		    /* convert to cartesian refered to the mesh */
                    originPlaneCoords =
			compute_domain_coords_linearinterp(
							   thePlanes[ iPlane ].origincoords.x[1],
							   thePlanes[ iPlane ].origincoords.x[0],
							   theSurfaceCornersLong ,
							   theSurfaceCornersLat,
							   theDomainY, theDomainX );

                    thePlanes[iPlane].origincoords.x[0]=originPlaneCoords.x[0];
                    thePlanes[iPlane].origincoords.x[1]=originPlaneCoords.x[1];

                    fret1 = fscanf(fp_planes," %lf %d %lf %d %lf %lf %d",
                                   &thePlanes[iPlane].stepalongstrike,
                                   &thePlanes[iPlane].numberofstepsalongstrike,
                                   &thePlanes[iPlane].stepdowndip,
				   &thePlanes[iPlane].numberofstepsdowndip,
                                   &thePlanes[iPlane].strike,
                                   &thePlanes[iPlane].dip,
                                   &thePlanes[iPlane].topo_plane);
                    /*Find largest plane for output buffer allocation */
                    if ( (thePlanes[iPlane].numberofstepsdowndip
                          * thePlanes[iPlane].numberofstepsalongstrike)
                         > largestplanesize)
			{
			    largestplanesize
				= thePlanes[iPlane].numberofstepsdowndip
				* thePlanes[iPlane].numberofstepsalongstrike;
			}
                    if (fret1 == 0) {
			solver_abort(fname, NULL,
				     "Unable to read plane specification in %s",
				     planes_input_file);
                    }

                    /* open displacement output files */
                    sprintf( planedisplacementsout, "%s/planedisplacements.%d",
                             thePlaneDirOut, iPlane );
                    fp0 = hu_fopen( planedisplacementsout, "w" );

                    thePlanes[iPlane].fpoutputfile = fp0;
                    sprintf (planecoordsfile, "%s/planecoords.%d",
                             thePlaneDirOut, iPlane);
                    fp1 = hu_fopen (planecoordsfile, "w");
                    thePlanes[iPlane].fpplanecoordsfile = fp1;
		} /* for */
	    } /* if ( (name != NULL) ... ) */
	} /* while */

	fclose(fp_planes);



    } /* if (myID == 0) */

    /* broadcast plane info */
    MPI_Bcast( &theNumberOfPlanes, 1, MPI_INT, 0, comm_solver );
    MPI_Bcast( thePlanePrintRate, 1, MPI_INT, 0, comm_solver );

    /* initialize the local structures */
    if (myID != 0) {
	thePlanes = (plane_t*)malloc( sizeof( plane_t ) * theNumberOfPlanes );
	if (thePlanes == NULL) {
	    solver_abort( "broadcast_planeinfo", NULL,
			  "Error: Unable to create plane information arrays" );
	}
    }

    for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsalongstrike), 1, MPI_INT, 0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsdowndip), 1, MPI_INT, 0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].topo_plane), 1, MPI_INT, 0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].stepalongstrike), 1, MPI_DOUBLE,0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].stepdowndip), 1, MPI_DOUBLE,0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].strike), 1, MPI_DOUBLE,0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].dip), 1, MPI_DOUBLE,0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].rake), 1, MPI_DOUBLE,0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].origincoords.x[0]), 3, MPI_DOUBLE,0, comm_solver);
	MPI_Barrier(comm_solver);
    }

    Old_output_planes_construct_strips(myID, theNumberOfPlanes);

    /* master allocates largest plane */
    if (myID == 0) {
	planes_output_buffer = (double*)malloc( 3 * sizeof(double)
						* largestplanesize );
	if (planes_output_buffer == NULL) {
	    fprintf(stderr, "Error creating buffer for plane output\n");
	    exit(1);
	}
    }

    print_planeinfo(myID, theNumberOfPlanes);

    return;
}

/**
 * This builds all of the strips used in the planes output. BigBen's
 * version does not have strip size limits
 */

static void Old_output_planes_construct_strips(int32_t myID, int theNumberOfPlanes)
{

    //    int iNode, iPlane, iStrike, iDownDip;
    int iPlane, iStrike, iDownDip;
    double xLocal, yLocal;
    vector3D_t origin, pointLocal, pointGlobal;
    octant_t *octant;
    int onstrip ;
    //    int32_t nodesToInterpolate[8];

    planes_LocalLargestStripCount = 0;

    for ( iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++ ){
	thePlanes[ iPlane ].numberofstripsthisplane = 0;
	/* Compute the coordinates of the nodes of the plane */
	origin.x[0] = thePlanes[ iPlane ].origincoords.x[0];
	origin.x[1] = thePlanes[ iPlane ].origincoords.x[1];
	origin.x[2] = thePlanes[ iPlane ].origincoords.x[2];

	/*Dorian. sanity check for topo-planes "Only horizontal planes are supported" */
    if ( ( thePlanes[ iPlane ].topo_plane == 1 ) &&
    	 ( ( thePlanes[ iPlane ].dip > 0 ) && ( thePlanes[ iPlane ].dip < 180 ) ) ) {
        fprintf(stderr,
                "Old_output_planes_construct_strips: "
                "Topo-plane must be horizontal IPlane=%d, dip %f\n",iPlane, thePlanes[ iPlane ].dip);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }


	/*Find limits of consecutive strips*/
	onstrip = 0;
	for ( iStrike = 0;
	      iStrike < thePlanes[iPlane].numberofstepsalongstrike;
	      iStrike++ )
	    {
		xLocal = iStrike*thePlanes[iPlane].stepalongstrike;
		for ( iDownDip = 0;
		      iDownDip < thePlanes[iPlane].numberofstepsdowndip;
		      iDownDip++ )
		    {
			yLocal = iDownDip*thePlanes[ iPlane ].stepdowndip;
			pointLocal.x[0] = xLocal;
			pointLocal.x[1] = yLocal;
			pointLocal.x[2] = 0;
			pointGlobal
			    = compute_global_coords( origin, pointLocal,
						     thePlanes[ iPlane ].dip, 0,
						     thePlanes[ iPlane ].strike );

			/* Dorian: correct z coordinate if topo-plane */
			if ( thePlanes[ iPlane ].topo_plane == 1 ) {
				pointGlobal.x[2] =  point_elevation ( pointGlobal.x[0], pointGlobal.x[1] );
			}

			if (search_point( pointGlobal, &octant ) == 1) {
			    if (!onstrip) { /* start new strip */
				thePlanes[iPlane].stripstart[thePlanes[iPlane].numberofstripsthisplane]
				    = (iStrike * thePlanes[iPlane].numberofstepsdowndip
				       + iDownDip);
				onstrip = 1;
			    }

			}

			else {
			    if (onstrip) { /* close strip */
				onstrip = 0;
				thePlanes[iPlane].stripend[thePlanes[iPlane]
							   .numberofstripsthisplane]
				    = (iStrike * thePlanes[iPlane].numberofstepsdowndip
				       + iDownDip - 1);
				thePlanes[ iPlane ].numberofstripsthisplane++;
				if (thePlanes[ iPlane ].numberofstripsthisplane
				    > MAX_STRIPS_PER_PLANE)
				    {
					fprintf( stderr,
						 "Number of strips on plane exceeds "
						 "MAX_STRIPS_PER_PLANE\n");
					exit(1);
				    }
			    }
			}
		    }
	    }

	if (onstrip){  /*if on a strip at end of loop, close strip */
            thePlanes[iPlane].stripend[thePlanes[iPlane].numberofstripsthisplane]
		= (thePlanes[iPlane].numberofstepsdowndip
		   * thePlanes[iPlane].numberofstepsalongstrike) - 1;
            thePlanes[ iPlane ].numberofstripsthisplane++;
	}

	/* get strip counts to IO PEs */
	MPI_Reduce( &thePlanes[ iPlane ].numberofstripsthisplane,
		    &thePlanes[iPlane].globalnumberofstripsthisplane,
		    1, MPI_INT, MPI_SUM, 0, comm_solver );

	/* allocate strips */
	int stripnum, stripLength;

	for (stripnum = 0;
	     stripnum < thePlanes[ iPlane ].numberofstripsthisplane;
	     stripnum++)
	    {
		stripLength = thePlanes[iPlane].stripend[stripnum]
		    - thePlanes[iPlane].stripstart[stripnum] + 1;

		thePlanes[iPlane].strip[stripnum]
		    = (plane_strip_element_t*)malloc( sizeof(plane_strip_element_t)
						      * stripLength );
		if (thePlanes[ iPlane ].strip[stripnum] == NULL) {
		    fprintf( stderr,
			     "Error malloc'ing array strips for plane output\n"
			     "PE: %d  Plane: %d  Strip: %d  Size: %zu\n",
			     myID, iPlane, stripnum,
			     sizeof(plane_strip_element_t) * stripLength );
		    exit(1);
		}

		if (stripLength>planes_LocalLargestStripCount) {
		    planes_LocalLargestStripCount = stripLength;
		}
	    }

	/* fill strips */
	origin.x[0] = thePlanes[ iPlane ].origincoords.x[0];
	origin.x[1] = thePlanes[ iPlane ].origincoords.x[1];
	origin.x[2] = thePlanes[ iPlane ].origincoords.x[2];

	int elemnum;
	for (stripnum = 0;
	     stripnum < thePlanes[ iPlane ].numberofstripsthisplane;
	     stripnum++)
	    {
		for (elemnum = 0;
		     elemnum < (thePlanes[iPlane].stripend[stripnum]
				-thePlanes[iPlane].stripstart[stripnum] + 1);
		     elemnum++)
		    {
			iStrike  = (elemnum+thePlanes[iPlane].stripstart[stripnum])
			    / thePlanes[iPlane].numberofstepsdowndip;
			iDownDip = (elemnum+thePlanes[iPlane].stripstart[stripnum])
			    % thePlanes[iPlane].numberofstepsdowndip;
			xLocal = iStrike*thePlanes[iPlane].stepalongstrike;
			yLocal = iDownDip*thePlanes[ iPlane ].stepdowndip;
			pointLocal.x[0] = xLocal;
			pointLocal.x[1] = yLocal;
			pointLocal.x[2] = 0;
			pointGlobal
			    =  compute_global_coords( origin, pointLocal,
						      thePlanes[ iPlane ].dip, 0,
						      thePlanes[ iPlane ].strike );

			/* Dorian: correct z coordinate if topo-plane */
			if ( thePlanes[ iPlane ].topo_plane == 1 ) {
				pointGlobal.x[2] =  point_elevation ( pointGlobal.x[0], pointGlobal.x[1] );
			}

			//TODO: local coordinates must be updated to consider tetrahedra elements
			//      in simulations including topography. However, because planes are used mostly
			//      to get general features of the wavefields, conventional interpolation based upon
			//      cubic elements is a fairly good approximation. Dorian
			if (search_point( pointGlobal, &octant) == 1) {
			    compute_csi_eta_dzeta( octant, pointGlobal,
						   &(thePlanes[ iPlane ].strip[stripnum][elemnum].localcoords),
						   thePlanes[ iPlane ].strip[stripnum][elemnum].nodestointerpolate);
			}
		    }
	    }

    } /* end of plane loop: for (stripnum = 0; ...) */

    MPI_Reduce( &planes_LocalLargestStripCount,
		&planes_GlobalLargestStripCount,
		1, MPI_INT, MPI_MAX, 0, comm_solver );

    /* allocate MPI recv buffers for IO PE */
    if (myID == 0) {
	int buf_len0 = (3 * planes_GlobalLargestStripCount) + 1;
	planes_stripMPIRecvBuffer = (double*)malloc(sizeof(double) * buf_len0);

	/* This is large enough for 3 double componants of biggest strip
	   plus location number */
	if (planes_stripMPIRecvBuffer == NULL) {
	    perror("Error creating MPI master buffer strip for plane output\n");
	    exit(1);
	}
    }

    /* allocate MPI send buffers for all PEs */
    int buf_len = (3 * planes_LocalLargestStripCount) + 1;
    planes_stripMPISendBuffer = (double*)malloc( sizeof(double) * buf_len );

    /* This is large enough for 3 double componants of biggest strip plus
       location number */
    if (planes_stripMPISendBuffer == NULL) {
	fprintf(stderr,
		"Error creating MPI send buffer strip for plane output\n");
	exit(1);
    }
}

/** Close planes files. */
void Old_planes_close(int32_t myID, int theNumberOfPlanes)
{
    int iPlane;

    if (myID==0){
	for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	    fclose( thePlanes[iPlane].fpoutputfile );
	    fclose( thePlanes[iPlane].fpplanecoordsfile );
	}
    }
}

/** Print Plane Stats, and possibly old version plane coords file. */
void print_planeinfo(int32_t myID, int theNumberOfPlanes)
{
    int iPlane;
    int MaxStrip;

    if (myID == 0) {
	fprintf( stderr, "\nPlane Output Stats___________________________\n");

	for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	    fprintf( stderr, "Plane dimensions for plane #%d: %d,%d\n", iPlane,
		     thePlanes[iPlane].numberofstepsalongstrike,
		     thePlanes[iPlane].numberofstepsdowndip );
	}

	for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	    fprintf( stderr, "Total strips for plane #%d: %d\n", iPlane,
		     thePlanes[iPlane].globalnumberofstripsthisplane );
	}
    }

    for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	MPI_Reduce( &thePlanes[ iPlane ].numberofstripsthisplane, &MaxStrip,
		    1, MPI_INT, MPI_MAX, 0, comm_solver );
	if (myID==0) {
	    fprintf(stderr, "Max strips for a PE in plane #%d: %d\n",
		    iPlane, MaxStrip);
	}
    }

    if (myID==0) {
	fprintf(stderr,
		"\nDone Plane Output Stats___________________________\n");
	fflush( stderr );
    }
}


/**************************************************************************/
/*New IO Server Version Of Code Below Here*********************************/
/**************************************************************************/


void New_planes_setup( int32_t     PENum, int32_t *thePlanePrintRate,
		       int theNumberOfPlanes,
                       const char *numericalin,
                       double      surfaceShift,
		       double *theSurfaceCornersLong, double *theSurfaceCornersLat,
		       double theDomainX, double theDomainY, double theDomainZ,
                       char* planes_input_file) {

    static const char* fname = "new_output_planes_setup()";
    double     *auxiliar;
    char       thePlaneDirOut[256];
    int        iPlane, iCorner;
    vector3D_t originPlaneCoords;
    int largestplanesize;
    FILE*      fp;
    FILE*      fp_planes;

    if (PENum == 0) {

        /* obtain the general planes specifications in parameter file */
	if ( (fp = fopen ( numericalin, "r")) == NULL ) {
	    solver_abort (fname, numericalin,
			  "Error opening parameters configuration file");
	}
	if ( (parsetext(fp, "output_planes_print_rate", 'i',
			thePlanePrintRate) != 0) )
	    {
		solver_abort (fname, NULL, "Error parsing output_planes_print_rate field from %s\n",
			      numericalin);
	    }
	auxiliar = (double *)malloc(sizeof(double)*8);
	if ( parsedarray( fp, "domain_surface_corners", 8 ,auxiliar) !=0 ) {
	    solver_abort (fname, NULL, "Error parsing domain_surface_corners field from %s\n",
			  numericalin);
	}
	for ( iCorner = 0; iCorner < 4; iCorner++){
	    theSurfaceCornersLong[ iCorner ] = auxiliar [ iCorner * 2 ];
	    theSurfaceCornersLat [ iCorner ] = auxiliar [ iCorner * 2 +1 ];
	}
	free(auxiliar);

	if ((parsetext(fp, "output_planes_directory", 's',
		       &thePlaneDirOut) != 0))
	    {
		solver_abort( fname, NULL,
			      "Error parsing output planes directory from %s\n",
			      numericalin );
	    }
	thePlanes = (plane_t *) malloc ( sizeof( plane_t ) * theNumberOfPlanes);
	if ( thePlanes == NULL ) {
	    solver_abort( fname, "Allocating memory for planes array",
			  "Unable to create plane information arrays" );
	}
	fclose(fp);

        /* read in the individual planes coords in planes file */
        if ( (fp_planes = fopen ( planes_input_file, "r")) == NULL ) {
            solver_abort (fname, planes_input_file,
                          "Error opening planes configuration file");
        }

	int found=0;
	while (!found) {
	    char line[LINESIZE];
	    char *name;
	    char delimiters[] = " =\n";
	    char querystring[] = "output_planes";
	    /* Read in one line */
	    if (fgets (line, LINESIZE, fp) == NULL) {
		break;
	    }
	    name = strtok(line, delimiters);
	    if ( (name != NULL) && (strcmp(name, querystring) == 0) ) {
		largestplanesize = 0;
		found = 1;
		for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
		    int fret0, fret1;
		    fret0 = fscanf( fp_planes," %lf %lf %lf",
				    &thePlanes[iPlane].origincoords.x[0],
				    &thePlanes[iPlane].origincoords.x[1],
				    &thePlanes[iPlane].origincoords.x[2] );

		    /* RICARDO: For buildings correction */
		    thePlanes[iPlane].origincoords.x[2] += surfaceShift;

		    /* origin coordinates must be in Lat, Long and depth */
		    if (fret0 == 0) {
			solver_abort (fname, NULL,
				      "Unable to read planes origin in %s",
				      planes_input_file);
		    }
		    /* convert to cartesian refered to the mesh */
                    originPlaneCoords =
			compute_domain_coords_linearinterp(
							   thePlanes[ iPlane ].origincoords.x[1],
							   thePlanes[ iPlane ].origincoords.x[0],
							   theSurfaceCornersLong ,
							   theSurfaceCornersLat,
							   theDomainY, theDomainX );

                    thePlanes[iPlane].origincoords.x[0]=originPlaneCoords.x[0];
                    thePlanes[iPlane].origincoords.x[1]=originPlaneCoords.x[1];

                    fret1 = fscanf(fp_planes," %lf %d %lf %d %lf %lf %d",
                                   &thePlanes[iPlane].stepalongstrike,
                                   &thePlanes[iPlane].numberofstepsalongstrike,
                                   &thePlanes[iPlane].stepdowndip,
				   &thePlanes[iPlane].numberofstepsdowndip,
                                   &thePlanes[iPlane].strike,
                                   &thePlanes[iPlane].dip,
                                   &thePlanes[iPlane].topo_plane);
                    /*Find largest plane for output buffer allocation */
                    if ( (thePlanes[iPlane].numberofstepsdowndip
                          * thePlanes[iPlane].numberofstepsalongstrike)
                         > largestplanesize)
			{
			    largestplanesize
				= thePlanes[iPlane].numberofstepsdowndip
				* thePlanes[iPlane].numberofstepsalongstrike;
			}
                    if (fret1 == 0) {
			solver_abort(fname, NULL,
				     "Unable to read plane specification in %s",
				     planes_input_file);
                    }

		} /* for */
	    } /* if ( (name != NULL) ... ) */
	} /* while */

	fclose(fp_planes);

    } /* if (PENum == 0) */

    /* broadcast plane info to whole IO group */
    MPI_Bcast( &theNumberOfPlanes, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( thePlanePrintRate, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( &largestplanesize, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( thePlaneDirOut, 256, MPI_CHAR, 0, comm_IO );


    /* initialize the local structures */
    if (PENum != 0) {
	thePlanes = (plane_t*)malloc( sizeof( plane_t ) * theNumberOfPlanes );
	if (thePlanes == NULL) {
	    solver_abort( "broadcast_planeinfo", NULL,
			  "Error: Unable to create plane information arrays" );
	}
    }

    /* broadcast plane params to whole IO group */
    for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsalongstrike), 1, MPI_INT, 0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsdowndip), 1, MPI_INT, 0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].topo_plane), 1, MPI_INT, 0, comm_solver);
	MPI_Bcast( &(thePlanes[iPlane].stepalongstrike), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].stepdowndip), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].strike), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].dip), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].rake), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].origincoords.x[0]), 3, MPI_DOUBLE,0, comm_IO);
	MPI_Barrier(comm_IO);
    }

    /* Assign IO Server PE to send this plane to */
    for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
	/* Simplest possible scheme, get more sophisticated later */
	int IO_Group_Size;
	MPI_Comm_size(comm_IO, &IO_Group_Size);
	thePlanes[iPlane].IO_PE_num = IO_Group_Size-1;
    }

    New_output_planes_construct_strips(PENum, theNumberOfPlanes);

    print_planeinfo(PENum, theNumberOfPlanes);

    return;
}


/* This builds all of the strips used in the planes output. XT5
   version does not have strip size limits */

static void New_output_planes_construct_strips(int32_t myID, int theNumberOfPlanes)
{

    int iPlane, iStrike, iDownDip;
    double xLocal, yLocal;
    vector3D_t origin, pointLocal, pointGlobal;
    octant_t *octant;
    int onstrip ;

    planes_LocalLargestStripCount = 0;

    for ( iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++ ){
	thePlanes[ iPlane ].numberofstripsthisplane = 0;
	/* Compute the coordinates of the nodes of the plane */
	origin.x[0] = thePlanes[ iPlane ].origincoords.x[0];
	origin.x[1] = thePlanes[ iPlane ].origincoords.x[1];
	origin.x[2] = thePlanes[ iPlane ].origincoords.x[2];

	/*Dorian. sanity check for topo-planes "Only horizontal planes are supported" */
    if ( ( thePlanes[ iPlane ].topo_plane == 1 ) &&
    	 ( ( thePlanes[ iPlane ].dip > 0 ) && ( thePlanes[ iPlane ].dip < 180 ) ) ) {
        fprintf(stderr,
                "Old_output_planes_construct_strips: "
                "Topo-plane must be horizontal IPlane=%d, dip %f\n",iPlane, thePlanes[ iPlane ].dip);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

	/*Find limits of consecutive strips*/
	onstrip = 0;
	for ( iStrike = 0;
	      iStrike < thePlanes[iPlane].numberofstepsalongstrike;
	      iStrike++ )
	    {
		xLocal = iStrike*thePlanes[iPlane].stepalongstrike;
		for ( iDownDip = 0;
		      iDownDip < thePlanes[iPlane].numberofstepsdowndip;
		      iDownDip++ )
		    {
			yLocal = iDownDip*thePlanes[ iPlane ].stepdowndip;
			pointLocal.x[0] = xLocal;
			pointLocal.x[1] = yLocal;
			pointLocal.x[2] = 0;
			pointGlobal
			    = compute_global_coords( origin, pointLocal,
						     thePlanes[ iPlane ].dip, 0,
						     thePlanes[ iPlane ].strike );

			/* Dorian: correct z coordinate if topo-plane */
			if ( thePlanes[ iPlane ].topo_plane == 1 ) {
				pointGlobal.x[2] =  point_elevation ( pointGlobal.x[0], pointGlobal.x[1] );
			}

			if (search_point( pointGlobal, &octant ) == 1) {
			    if (!onstrip) { /* start new strip */
				thePlanes[iPlane].stripstart[thePlanes[iPlane].numberofstripsthisplane]
				    = (iStrike * thePlanes[iPlane].numberofstepsdowndip
				       + iDownDip);
				onstrip = 1;
			    }

			}

			else {
			    if (onstrip) { /* close strip */
				onstrip = 0;
				thePlanes[iPlane].stripend[thePlanes[iPlane]
							   .numberofstripsthisplane]
				    = (iStrike * thePlanes[iPlane].numberofstepsdowndip
				       + iDownDip - 1);
				thePlanes[ iPlane ].numberofstripsthisplane++;
				if (thePlanes[ iPlane ].numberofstripsthisplane
				    > MAX_STRIPS_PER_PLANE)
				    {
					fprintf( stderr,
						 "Number of strips on plane exceeds "
						 "MAX_STRIPS_PER_PLANE\n");
					exit(1);
				    }
			    }
			}
		    }
	    }

	if (onstrip){  /*if on a strip at end of loop, close strip */
            thePlanes[iPlane].stripend[thePlanes[iPlane].numberofstripsthisplane]
		= (thePlanes[iPlane].numberofstepsdowndip
		   * thePlanes[iPlane].numberofstepsalongstrike) - 1;
            thePlanes[ iPlane ].numberofstripsthisplane++;
	}

	/* get strip counts to IO PEs */
	MPI_Reduce( &thePlanes[ iPlane ].numberofstripsthisplane,
		    &thePlanes[iPlane].globalnumberofstripsthisplane,
		    1, MPI_INT, MPI_SUM, 0, comm_solver );
	MPI_Bcast( &thePlanes[iPlane].globalnumberofstripsthisplane, 1, MPI_INT, 0, comm_IO );

	/* allocate strips */
	int stripnum, stripLength;

	for (stripnum = 0;
	     stripnum < thePlanes[ iPlane ].numberofstripsthisplane;
	     stripnum++)
	    {
		stripLength = thePlanes[iPlane].stripend[stripnum]
		    - thePlanes[iPlane].stripstart[stripnum] + 1;

		thePlanes[iPlane].strip[stripnum]
		    = (plane_strip_element_t*)malloc( sizeof(plane_strip_element_t)
						      * stripLength );
		if (thePlanes[ iPlane ].strip[stripnum] == NULL) {
		    fprintf( stderr,
			     "Error malloc'ing array strips for plane output\n"
			     "PE: %d  Plane: %d  Strip: %d  Size: %zu\n",
			     myID, iPlane, stripnum,
			     sizeof(plane_strip_element_t) * stripLength );
		    exit(1);
		}

		if (stripLength>planes_LocalLargestStripCount) {
		    planes_LocalLargestStripCount = stripLength;
		}
	    }

	/* fill strips */
	origin.x[0] = thePlanes[ iPlane ].origincoords.x[0];
	origin.x[1] = thePlanes[ iPlane ].origincoords.x[1];
	origin.x[2] = thePlanes[ iPlane ].origincoords.x[2];

	int elemnum;
	for (stripnum = 0;
	     stripnum < thePlanes[ iPlane ].numberofstripsthisplane;
	     stripnum++)
	    {
		for (elemnum = 0;
		     elemnum < (thePlanes[iPlane].stripend[stripnum]
				-thePlanes[iPlane].stripstart[stripnum] + 1);
		     elemnum++)
		    {
			iStrike  = (elemnum+thePlanes[iPlane].stripstart[stripnum])
			    / thePlanes[iPlane].numberofstepsdowndip;
			iDownDip = (elemnum+thePlanes[iPlane].stripstart[stripnum])
			    % thePlanes[iPlane].numberofstepsdowndip;
			xLocal = iStrike*thePlanes[iPlane].stepalongstrike;
			yLocal = iDownDip*thePlanes[ iPlane ].stepdowndip;
			pointLocal.x[0] = xLocal;
			pointLocal.x[1] = yLocal;
			pointLocal.x[2] = 0;
			pointGlobal
			    =  compute_global_coords( origin, pointLocal,
						      thePlanes[ iPlane ].dip, 0,
						      thePlanes[ iPlane ].strike );

			/* Dorian: correct z coordinate if topo-plane */
			if ( thePlanes[ iPlane ].topo_plane == 1 ) {
				pointGlobal.x[2] =  point_elevation ( pointGlobal.x[0], pointGlobal.x[1] );
			}

			//TODO: local coordinates must be updated to consider tetrahedra elements
			//      in simulations including topography. However, because planes are used mostly
			//      to get general features of the wavefields, conventional interpolation based upon
			//      cubic elements is a fairly good approximation. Dorian.
			if (search_point( pointGlobal, &octant) == 1) {
			    compute_csi_eta_dzeta( octant, pointGlobal,
						   &(thePlanes[ iPlane ].strip[stripnum][elemnum].localcoords),
						   thePlanes[ iPlane ].strip[stripnum][elemnum].nodestointerpolate);
			}
		    }
	    }
    } /* end of plane loop: for (stripnum = 0; ...) */

    MPI_Reduce( &planes_LocalLargestStripCount,
		&planes_GlobalLargestStripCount,
		1, MPI_INT, MPI_MAX, 0, comm_solver );
    MPI_Bcast( &planes_GlobalLargestStripCount, 1, MPI_INT, 0, comm_IO );

    /* allocate MPI send buffers for all PEs */
    int buf_len = (3 * planes_LocalLargestStripCount) + 1;
    planes_stripMPISendBuffer = (double*)malloc( sizeof(double) * buf_len );

    /* This is large enough for 3 double componants of biggest strip plus
       location number */
    if (planes_stripMPISendBuffer == NULL) {
	fprintf(stderr,
		"Error creating MPI send buffer strip for plane output\n");
	exit(1);
    }
}

/**
 * Interpolate the displacenents and communicate strips to printing PE.
 * Bigben version has no control flow, just chugs through planes.
 */
int New_planes_print(int32_t PENum, mysolver_t* mySolver, int theNumberOfPlanes)
{
    int iPlane, iPhi;
    int stripLength;
    int iStrip, elemnum;

    /* Auxiliar array to handle shapefunctions in a loop */
    double  xi[3][8]={ {-1,  1, -1,  1, -1,  1, -1, 1} ,
		       {-1, -1,  1,  1, -1, -1,  1, 1} ,
		       {-1, -1, -1, -1,  1,  1,  1, 1} };

    double      phi[8];
    double      displacementsX, displacementsY, displacementsZ;

    for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++) {
	for (iStrip = 0; iStrip < thePlanes[iPlane].numberofstripsthisplane;
	     iStrip++) {

	    /* interpolate points directly into send buffer for this strip */
	    for (elemnum = 0;
		 elemnum < (thePlanes[iPlane].stripend[iStrip]
			    -thePlanes[iPlane].stripstart[iStrip]+1);
		 elemnum++){
		displacementsX = 0;displacementsY = 0;displacementsZ = 0;

		/* Interpolate the element */
		for (iPhi = 0; iPhi < 8; iPhi++){
		    phi[iPhi] = ( 1 + (xi[0][iPhi]) *
				  (thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[0]) )
			* ( 1 + (xi[1][iPhi]) *
			    (thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[1]) )
			* ( 1 + (xi[2][iPhi]) *
			    (thePlanes[iPlane].strip[iStrip][elemnum].localcoords.x[2]) ) /8;

		    displacementsX += phi[iPhi]*
			(mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[0]);
		    displacementsY += phi[iPhi]*
			(mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[1]);
		    displacementsZ += phi[iPhi]*
			(mySolver->tm1[ thePlanes[iPlane].strip[iStrip][elemnum].nodestointerpolate[iPhi] ].f[2]);
		}

		planes_stripMPISendBuffer[elemnum*3]     = (double) (displacementsX);
		planes_stripMPISendBuffer[elemnum*3 + 1] = (double) (displacementsY);
		planes_stripMPISendBuffer[elemnum*3 + 2] = (double) (displacementsZ);

	    } /*for elemnum*/

	    /* add start location as last element, add .1 to insure int-float conversion (yes, bad form)*/
	    stripLength = thePlanes[iPlane].stripend[iStrip]
		- thePlanes[iPlane].stripstart[iStrip] + 1;

	    planes_stripMPISendBuffer[stripLength*3]
		= (double) (thePlanes[iPlane].stripstart[iStrip] + 0.1);

	    MPI_Send( planes_stripMPISendBuffer, (stripLength*3)+1,
		      MPI_DOUBLE, thePlanes[iPlane].IO_PE_num, iPlane, comm_IO );

	} /*for iStrips*/


	MPI_Barrier( comm_solver ); /* may not be required */


    } /* for iPlane */

    return 1;
}


/** Close planes files. */
void New_planes_close(int32_t PEnum, int theNumberOfPlanes){

    int IO_Server_PE, iPlane;

    if (PEnum==0){
	/* Simplest possible scheme, do loop for multiple server PEs later */
	MPI_Comm_size(comm_IO, &IO_Server_PE);
	IO_Server_PE--;
	planes_stripMPISendBuffer[0]= (STOP_SERVER + 0.1);
	for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++)
	    MPI_Send( planes_stripMPISendBuffer, 1, MPI_DOUBLE, IO_Server_PE, iPlane, comm_IO );
    }
}



/* Server IO PEs Main Loop */
void planes_IO_PES_main(int32_t PEnum){

    int        largestplanesize, comm_size, iPlane;
    MPI_Status status;
    char       thePlaneDirOut[256], planedisplacementsout[1024];
    FILE*      fp0;
    int32_t    thePlanePrintRate;
    int        theNumberOfPlanes;

    /* Get basic planes parameters */
    MPI_Bcast( &theNumberOfPlanes, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( &thePlanePrintRate, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( &largestplanesize, 1, MPI_INT, 0, comm_IO );
    MPI_Bcast( thePlaneDirOut, 256, MPI_CHAR, 0, comm_IO );

    if(theNumberOfPlanes==0) return;

    /* Create Planes info structure */
    thePlanes = (plane_t*)malloc( sizeof( plane_t ) * theNumberOfPlanes );
    if (thePlanes == NULL) {
	solver_abort( "broadcast_planeinfo", NULL,
		      "Error: Unable to create plane information arrays" );
    }

    /* Get plane params */
    for ( iPlane = 0; iPlane < theNumberOfPlanes; iPlane++ ) {
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsalongstrike), 1, MPI_INT, 0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].numberofstepsdowndip), 1, MPI_INT, 0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].stepalongstrike), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].stepdowndip), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].strike), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].dip), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].rake), 1, MPI_DOUBLE,0, comm_IO);
	MPI_Bcast( &(thePlanes[iPlane].origincoords.x[0]), 3, MPI_DOUBLE,0, comm_IO);
	MPI_Barrier(comm_IO);
    }

    /* Get strip count for each plane */
    for (iPlane = 0; iPlane < theNumberOfPlanes; iPlane++)
	MPI_Bcast( &thePlanes[iPlane].globalnumberofstripsthisplane, 1, MPI_INT, 0, comm_IO );

    /* Find size of largest strip */
    MPI_Bcast( &planes_GlobalLargestStripCount, 1, MPI_INT, 0, comm_IO );

    /* Idle IO Pool PEs can leave, don't use collectives past here! */
    MPI_Comm_size(comm_IO, &comm_size);
    if (PEnum < (comm_size-1)) return;

    /* Allocate largest required strip recieve buffer */
    int buf_len0 = (3 * planes_GlobalLargestStripCount) + 1;
    planes_stripMPIRecvBuffer = (double*)malloc(sizeof(double) * buf_len0);
    /* large enough for 3 double of biggest strip plus location number */
    if (planes_stripMPIRecvBuffer == NULL) {
	perror("Error creating MPI master buffer strip for plane output\n");
	exit(1);
    }

    /* Allocate largest plane output buffer */
    planes_output_buffer = (double*)malloc( 3 * sizeof(double)
					    * largestplanesize );
    if (planes_output_buffer == NULL) {
	fprintf(stderr, "Error creating buffer for plane output\n");
	exit(1);
    }

    /* open displacement output files */
    /* for multiple IO server PEs, will need to make sure these are handled on each one */
    for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++){
	sprintf( planedisplacementsout, "%s/planedisplacements.%d",
		 thePlaneDirOut, iPlane );
	fp0 = fopen( planedisplacementsout, "w" );
	thePlanes[iPlane].fpoutputfile = fp0;
	/* Not activating planecoored files yet
	   sprintf (planecoordsfile, "%s/planecoords.%d",
	   thePlaneDirOut, iPlane);
	   fp1 = fopen (planecoordsfile, "w");
	   thePlanes[iPlane].fpplanecoordsfile = fp1;
	   Turn this on when and if it is needed */
    }

    while(1){

	int StartLocation, recvStripCount, rStrip;

	/* Get any message from anywhere */
	MPI_Recv( planes_stripMPIRecvBuffer, planes_GlobalLargestStripCount * 3 + 1, MPI_DOUBLE,
		  MPI_ANY_SOURCE, MPI_ANY_TAG, comm_IO, &status );
	MPI_Get_count(&status, MPI_DOUBLE, &recvStripCount);

	/* Check for a stop message */
	if( recvStripCount == 1){
	    if( (int)planes_stripMPIRecvBuffer[0] == STOP_SERVER)
		break;
	}
	/* else pick this plane and fill in first strip */
	else{
	    iPlane = status.MPI_TAG;
	    StartLocation = (int)planes_stripMPIRecvBuffer[recvStripCount-1];
	    memcpy( &(planes_output_buffer[StartLocation * 3]),
		    planes_stripMPIRecvBuffer,
		    (recvStripCount-1)*sizeof(double) );
	}

	/* Fill in rest of this plane until full */
	for (rStrip = 0; rStrip < (thePlanes[iPlane].globalnumberofstripsthisplane - 1); rStrip++) {
	    MPI_Recv( planes_stripMPIRecvBuffer,
		      planes_GlobalLargestStripCount * 3 + 1, MPI_DOUBLE,
		      MPI_ANY_SOURCE, iPlane, comm_IO, &status );
	    MPI_Get_count(&status, MPI_DOUBLE, &recvStripCount);
	    StartLocation = (int)planes_stripMPIRecvBuffer[recvStripCount-1];
	    memcpy( &(planes_output_buffer[StartLocation * 3]),
		    planes_stripMPIRecvBuffer,
		    (recvStripCount-1)*sizeof(double) );
	}

	/* Print just this one plane */
	int ret, items_to_write;
	items_to_write = 3 * thePlanes[iPlane].numberofstepsalongstrike * thePlanes[iPlane].numberofstepsdowndip;
	ret = fwrite( planes_output_buffer, sizeof(double), items_to_write, thePlanes[iPlane].fpoutputfile );
	if (ret != items_to_write)
	    solver_abort( "IO_Server_Main_Loop", NULL, "Error: Unable to output displacement file to disk" );

    }/*while*/

    /* Close all planes.  Will need to make sure these are seperated on multiple PEs*/
    int ret;
    for (iPlane = 0; iPlane < theNumberOfPlanes ;iPlane++){
	ret = fclose( thePlanes[iPlane].fpoutputfile );
	/* if (!ret) printf("Error closing plane file %d!!!\n", iPlane);  Sort this out later*/
    }

    return;

}
