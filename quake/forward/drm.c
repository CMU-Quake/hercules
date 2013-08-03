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
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "cvm.h"
#include "psolve.h"
#include "drm.h"
#include "octor.h"
#include "commutil.h"
#include "util.h"
#include "timers.h"
#include "quake_util.h"
#include "buildings.h"


#define  NONE -1
#define  MAX(a, b) (((a)>(b)) ? (a) : (b))
#define  MIN(a, b) (((a)<(b)) ? (a) : (b))


extern int compute_csi_eta_dzeta( octant_t *octant, vector3D_t pointcoords,
				                  vector3D_t *localcoords, int32_t *localNodeID );


/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static double  theDrmXMin,  theDrmXMax;
static double  theDrmYMin,  theDrmYMax;
static double  theDrmDepth;
static double  theX_Offset;
static double  theY_Offset;
static double  thePart1DeltaT;
static int     theDrmPrintRate;
static double  theDrmEdgeSize;
static double  *myDispWrite;

static double  theSurfaceShift = 0;

/*
 * These are the total drm elems and nodes numbers in all processors.
 */
static int32_t theNumberOfDrmNodes;
static int32_t theNumberOfDrmElements;

/*
 * These are the local drm elems and nodes numbers in each processor.
 */
static int myNumberOfDrmNodes = 0;
static int myNumberOfDupDrmNodes = 0;
static int myNumberOfDrmElements = 0;

static char  theDrmOutputDir[256];

static drmhash_t  theDrmTable;
/*This is to store drm nodes which are not owned by me */
static drmhash_t  theDrmDupTable;
static drm_part_t theDrmPart;

/*To store local ids of drm nodes */
static int64_t      *drmNodeIds;
static int64_t      *drmDupNodesIds;
static int32_t      *drmNodeOwner;
static int32_t      theTableDim = 89;
static int32_t      theTableDupDim = 89;
static int32_t      *pesHaveDrmNodes;
static int32_t      *dupNodesCount;

static fvector_t    *theDrmNodeCoords;
static fvector_t    *theDrmDisplacements1; // For Part 2 only and current timestep
static fvector_t    *theDrmDisplacements2; // For Part 2 only and next timestep

static drm_elem_t   *theDrmElem;
static drm_node_t   *myDrmNodes;

/*Total number of processors that has drm elements */
static int                 drmFilesCount;
static int                 filesToReadCnt = 0;
static pes_drm_nodes_cnt   *partOneDrmNodesCnt;
static drm_file_to_open    drmFileToOpen;
static drm_file_to_read    *drmFilesToRead;

/* this is for storing drm node coords for each processor . part 2*/
static  int32_t  *rank; /*for comm */
static  int32_t  *pes_per_file ;
static  int32_t  *drmNodesCountP2;
static  int  peshavenodes = 0; /* total number of processors with drm nodes--part2*/
static	pes_with_drm_nodes   *pesDrmNodes; /* this is for drm files created in Part2 */

static noyesflag_t  drmImplement = NO;
static noyesflag_t  includeBuildings;

static 	FILE   *drmDisp;

/* -------------------------------------------------------------------------- */
/*                             General Utilities                              */
/* -------------------------------------------------------------------------- */

drm_part_t drm_init (int32_t myID, const char *parametersin, noyesflag_t includeBldgs)
{
	double double_message[9];
	int    int_message[2];

	MPI_Datatype coords_mpi, coordscnt_mpi;

	drmImplement = YES;
	includeBuildings = includeBldgs;
	if ( includeBuildings == YES ) {
		theSurfaceShift = get_surface_shift();
	}

	/* Capturing data from file --- only done by PE0 */
	if (myID == 0) {
		if (drm_initparameters( parametersin) != 0) {
			fprintf(stderr,"Thread 0: drm_init: "
					"drm_initparameters error\n");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}
	}

	/* Broadcasting data */

	int_message[0]  = (int)theDrmPart;
	int_message[1]  = theDrmPrintRate;

	MPI_Bcast(int_message, 2, MPI_INT, 0, comm_solver);

	theDrmPart = int_message[0] ;
	theDrmPrintRate = int_message[1];

	double_message[0] = theDrmXMin;
	double_message[1] = theDrmYMin;
	double_message[2] = theDrmXMax;
	double_message[3] = theDrmYMax;
	double_message[4] = theDrmDepth;
	double_message[5] = theX_Offset;
	double_message[6] = theY_Offset;
	double_message[7] = theDrmEdgeSize;
	double_message[8] = thePart1DeltaT;

	MPI_Bcast(double_message, 9, MPI_DOUBLE, 0, comm_solver);

	theDrmXMin     = double_message[0];
	theDrmYMin     = double_message[1];
	theDrmXMax     = double_message[2];
	theDrmYMax     = double_message[3];
	theDrmDepth    = double_message[4];
	theX_Offset    = double_message[5];
	theY_Offset    = double_message[6];
	theDrmEdgeSize = double_message[7];
	thePart1DeltaT = double_message[8];

	broadcast_char_array( theDrmOutputDir, sizeof(theDrmOutputDir), 0,
			comm_solver );

	MPI_Barrier( comm_solver );


	/* broadcast drm nodal coordinates if in par1 */
	if ( theDrmPart == PART1 ) {

		MPI_Bcast(&theNumberOfDrmNodes, 1, MPI_INT, 0, comm_solver);

		if ( myID != 0 ) {
			XMALLOC_VAR_N( theDrmNodeCoords, fvector_t, theNumberOfDrmNodes );
		}

		MPI_Type_contiguous(3, MPI_DOUBLE, &coords_mpi);
		MPI_Type_commit(&coords_mpi);

		MPI_Bcast(theDrmNodeCoords, theNumberOfDrmNodes, coords_mpi, 0, comm_solver);

	}

	if ( theDrmPart == PART2 ) {

		MPI_Bcast(&theNumberOfDrmNodes, 1, MPI_INT, 0, comm_solver);
		MPI_Bcast(&drmFilesCount, 1, MPI_INT, 0, comm_solver);

		if ( myID != 0 ) {
			XMALLOC_VAR_N( partOneDrmNodesCnt, pes_drm_nodes_cnt, drmFilesCount );
		}

		MPI_Type_contiguous(2, MPI_INT, &coordscnt_mpi);
		MPI_Type_commit(&coordscnt_mpi);

		MPI_Bcast(partOneDrmNodesCnt, drmFilesCount, coordscnt_mpi, 0, comm_solver);

	}

	return theDrmPart;
}

int32_t drm_initparameters (const char *parametersin) {

	FILE   *fp;

	double *auxiliar,
			drm_edgesize,
			x_offset,
			y_offset,
			part1_delta_t;

	int    drm_print_rate;
	char   which_drm_part[64],
       	   drm_output_directory[64];

	drm_part_t drmPart = -1;

	/* Opens parametersin file */

	if ( ( fp = fopen(parametersin, "r" ) ) == NULL ) {
		fprintf( stderr,
				"Error opening %s\n drm_initparameters",
				parametersin );
		return -1;
	}

	if ((parsetext ( fp, "drm_directory",  's', &drm_output_directory ) != 0) ||
	    (parsetext ( fp, "which_drm_part", 's', &which_drm_part       ) != 0) ||
	    (parsetext ( fp, "drm_edgesize",   'd', &drm_edgesize         ) != 0) ||
	    (parsetext ( fp, "drm_offset_x",   'd', &x_offset             ) != 0) ||
	    (parsetext ( fp, "drm_offset_y",   'd', &y_offset             ) != 0) ||
	    (parsetext ( fp, "drm_print_rate", 'i', &drm_print_rate       ) != 0) ||
	    (parsetext ( fp, "part1_delta_t",  'd', &part1_delta_t        ) != 0) )
	{
		solver_abort ( "drm_initparameters", NULL,
				"Error parsing fields from %s\n", parametersin);
	}

	if ( strcasecmp(which_drm_part, "part0") == 0) {
        drmPart = PART0;
    } else if (strcasecmp(which_drm_part, "part1") == 0) {
        drmPart = PART1;
    } else if (strcasecmp(which_drm_part, "part2") == 0) {
        drmPart = PART2;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
        	"Unknown drm type: %s\n",
                drmPart );
    }

	XMALLOC_VAR_N( auxiliar, double, 5 );

	if ( parsedarray( fp, "drm_boundary", 5 , auxiliar ) != 0)
	{
		fprintf( stderr,
				"Error parsing drm_boundaries list from %s\n",
				parametersin );
		return -1;
	}

	/* Init the static global variables */

	/*We DO NOT convert physical coordinates into etree coordinates
	 */
	theDrmXMin  = auxiliar [ 0 ];
	theDrmYMin  = auxiliar [ 1 ];
	theDrmXMax  = auxiliar [ 2 ];
	theDrmYMax  = auxiliar [ 3 ];
	theDrmDepth = auxiliar [ 4 ];

	free( auxiliar );

	thePart1DeltaT = part1_delta_t;
	theDrmPrintRate = drm_print_rate;

	theDrmPart  = drmPart;

	theDrmEdgeSize = drm_edgesize;

	theX_Offset = x_offset;
	theY_Offset = y_offset;

	strcpy( theDrmOutputDir, drm_output_directory );

	fclose(fp);

	/* Read drm nodal coordinates if in Part1 or Part2 */
	if ( theDrmPart == PART1 || theDrmPart == PART2 ) {
		if ( drm_read_coords( ) != 0 ) {
			fprintf(stderr,"Thread 0: drm_read_coords: "
					"drm_read_coords error\n");
			return -1;
		}
	}

	return 0;
}

int32_t drm_read_coords ( ) {

	FILE* fp;

	int32_t    local_number_drm_nodes, total_num_files, i, prev_count=0;
	int32_t    *file_numbers;

	char       filename [256];

	if ( theDrmPart == PART1 ) {
		sprintf(filename, "%s%s", theDrmOutputDir, "/part0/drm_processor_info");
	}

	if ( theDrmPart == PART2 ) {

		sprintf(filename, "%s", "/lustre/scratch/yigit/DRM_new/part1/drm_processor_info");

		//sprintf(filename, "%s%s", theDrmOutputDir, "/part1/drm_processor_info");
		//sprintf(filename, "%s", "/lustre/scratch/yigit/DRM_TEST/outputfiles/DRM/part1/drm_processor_info");

	}

	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error opening %s\n", filename);
		return -1;
	}

	if ( theDrmPart == PART1 ) {

		/* First open drm_processor_info  to read  number of processors that have
		 * drm nodes and their ids from part 0. */
		hu_fread(&total_num_files, sizeof(int32_t), 1, fp );

		XMALLOC_VAR_N( file_numbers, int32_t, total_num_files );

		hu_fread(file_numbers, sizeof(int32_t), total_num_files, fp);

		/*Then read total number of nodes*/
		hu_fread(&theNumberOfDrmNodes, sizeof(int32_t), 1, fp);

		/*Allocate memory for  global array and read coordinates from drm_coordinates_0*/

		XMALLOC_VAR_N( theDrmNodeCoords, fvector_t, theNumberOfDrmNodes );

		fclose(fp);

		/*Read all the  files  */
		for ( i = 0; i < total_num_files; ++i ) {
			sprintf(filename, "%s%s_%d", theDrmOutputDir, "/part0/drm_coordinates", file_numbers[i]);
			if ( (fp = fopen(filename, "r")) == NULL) {
				fprintf(stderr, "Error opening %s\n", filename);
				return -1;
			}
			hu_fread(&local_number_drm_nodes, sizeof(int32_t), 1, fp);
			hu_fread(theDrmNodeCoords+prev_count, sizeof(fvector_t), local_number_drm_nodes, fp);
			prev_count = prev_count + local_number_drm_nodes;
			fclose(fp);
		}

		/* Surface- shift is taken care of in case of buildings */
		if ( includeBuildings == YES )
		for ( i = 0; i < theNumberOfDrmNodes; ++i ) {
			theDrmNodeCoords[i].f[2] += theSurfaceShift;
		}

		free(file_numbers);
	}

	if ( theDrmPart == PART2 ) {

		hu_fread(&theNumberOfDrmNodes, sizeof(int32_t), 1, fp);
		hu_fread(&drmFilesCount, sizeof(int32_t), 1, fp);

		XMALLOC_VAR_N( partOneDrmNodesCnt, pes_drm_nodes_cnt, drmFilesCount );

		for( i = 0; i < drmFilesCount; ++i ) {
			hu_fread(&(partOneDrmNodesCnt[i].id), sizeof(int32_t), 1, fp);
			hu_fread(&(partOneDrmNodesCnt[i].count), sizeof(int32_t), 1, fp);
		}

		fclose(fp);
	}

	return 0;
}

/* If key equals 1 insert mode is on.Otherwise just counts the number of drm elems */
void  search_drm_elems(mesh_t  *myMesh, int32_t key) {

	int32_t   i, n_0, n_7, elementcounter = 0;
	edata_t   *edata_temp;

	/* Loop each element to find drm nodes.
	 */
	for ( i = 0; i < (myMesh->lenum); ++i ) {

		int    keys_array[5];
		double coords_array[6];

		edata_temp = (myMesh)->elemTable[i].data;
		/* n_0 and n_7 are local nodal id's of 0th and 7th node.
		 * Coordinates of 7th node are always bigger than 0th node.
		 * */
		n_0 = myMesh->elemTable[i].lnid[0]; /* looking at first node of the element(at top) */
		n_7 = myMesh->elemTable[i].lnid[7]; /* looking at last node of the element(at bottom) */

		/* these are  coordinates of 0th  node  */
		coords_array[0] = myMesh->nodeTable[n_0].x;
		coords_array[1] = myMesh->nodeTable[n_0].y;
		coords_array[2] = myMesh->nodeTable[n_0].z;

		/* these are  coordinates of 7th node  */
		coords_array[3] = myMesh->nodeTable[n_7].x;
		coords_array[4] = myMesh->nodeTable[n_7].y;
		coords_array[5] = myMesh->nodeTable[n_7].z;

		if ( is_drm_elem(coords_array, keys_array, 1,myMesh->ticksize) ) {
			elementcounter++;
			/* Insert nodes if I am in part0 or part2 */
			if ( theDrmPart != PART1 && key == 1 ) {
				insertNode(myMesh, i);
			}
		}
	}

	if (  theDrmPart != PART1 && key == 1  ) {
		myNumberOfDrmNodes = theDrmTable.tableCnt;
		if (  theDrmPart == PART0 )
			myNumberOfDupDrmNodes = theDrmDupTable.tableCnt;
	}

	myNumberOfDrmElements = elementcounter;
}

/* Return  1 if an element is a drm boundary element
 * Return  0 if an element is not a drm element.
 * theDrm's are in etree coordinates.
 * If only interested with return val , key should be 1.
 * If ,in additon, want to know the face keys in detail , key should be 2.This
 * time it returns  number of faces it belongs to */
/* This also takes care of the surface shift due to buildings */
int is_drm_elem(double *coords_array, int *keys_array, int key, double  ticksize) {

	int    i, cnt = 0;
	double x_coord_0 = coords_array[0],
	       y_coord_0 = coords_array[1],
		   z_coord_0 = coords_array[2],
	       x_coord_7 = coords_array[3],
	       y_coord_7 = coords_array[4],
	       z_coord_7 = coords_array[5];

	if ( includeBuildings == YES ) {
		z_coord_0 -= theSurfaceShift/ticksize;
		z_coord_7 -= theSurfaceShift/ticksize;
	}

	for ( i = 0 ;i < 5; ++i ) {
		keys_array[i] = 0;
	}
	/* these are  total 5 faces where elements needs to be traversed */

	/* This takes care of 1 x face closer to origin */
		if ( z_coord_0 <=  theDrmDepth )
		if ( x_coord_0 <  theDrmXMin && x_coord_7 >= theDrmXMin )
		if ( y_coord_0 <= theDrmYMax && y_coord_7 >= theDrmYMin ) {
			cnt++;
			keys_array[0] = 1;
			if ( key==1 ) {
				return 1;
			}
		}

		/* This takes care of 1 x face far away from origin */
		if ( z_coord_0 <= theDrmDepth )
		if ( x_coord_0 <= theDrmXMax && x_coord_7 >  theDrmXMax )
		if ( y_coord_0 <= theDrmYMax && y_coord_7 >= theDrmYMin){
			cnt++;
			keys_array[1] = 1;
			if ( key==1 ) {
				return 1;
			}
		}

		/* This takes care of 1 y face closer to origin */
		if ( z_coord_0 <=  theDrmDepth )
		if ( y_coord_0 <  theDrmYMin  &&  y_coord_7 >= theDrmYMin )
		if ( x_coord_0 <= theDrmXMax  &&  x_coord_7 >= theDrmXMin ){
			cnt++;
			keys_array[2] = 1;
			if ( key==1 ) {
				return 1;
			}
		}

		/* This takes care of 1 y face far away from origin */
		if ( z_coord_0 <=  theDrmDepth )
		if ( y_coord_0 <= theDrmYMax  &&  y_coord_7 >  theDrmYMax )
		if ( x_coord_0 <= theDrmXMax  &&  x_coord_7 >= theDrmXMin ){
			cnt++;
			keys_array[3] = 1;
			if ( key==1 ) {
				return 1;
			}
		}

		/* This takes care of 1 z face */
		if ( z_coord_0 <= theDrmDepth  && z_coord_7 >  theDrmDepth )
		if ( x_coord_0 <= theDrmXMax   && x_coord_7 >= theDrmXMin )
		if ( y_coord_0 <= theDrmYMax   && y_coord_7 >= theDrmYMin){
			cnt++;
			keys_array[4] = 1;
			if( key==1 ) {
				return 1;
			}
		}

		return cnt;
}

/**
 * Return  1 if an element is a drm boundary element and needs to be refined,
 * Return  0 if an element is a drm boundary element and does not need to be refined,
 * Return -1 if an element is not a drm element.
 * This is for having same size elements in drm boundary to avoid dangling nodes */
int drm_toexpand ( octant_t *leaf, double  ticksize, edata_t  *edata ) {

	/* Send this keys_array even if we dont use it */
	int    keys_array[5];
	double coords_array[6];

	/*These are in etree coordinates */
	coords_array[0] = (leaf->lx);
	coords_array[1] = (leaf->ly);
	coords_array[2] = (leaf->lz);

	coords_array[3] = (leaf->lx) + edata->edgesize/ticksize;
	coords_array[4] = (leaf->ly) + edata->edgesize/ticksize;
	coords_array[5] = (leaf->lz) + edata->edgesize/ticksize;

	if ( is_drm_elem (coords_array, keys_array, 1, ticksize) ) {
		if ( (double)edata->edgesize > theDrmEdgeSize )
			return 1;
		else
			return 0;
	}

	return -1;
}

/*We  convert physical coordinates into etree coordinates*/
void  drm_fix_coordinates(double ticksize) {

	theDrmXMin  = theDrmXMin/ticksize;
	theDrmYMin  = theDrmYMin/ticksize;
	theDrmXMax  = theDrmXMax/ticksize;
	theDrmYMax  = theDrmYMax/ticksize;
	theDrmDepth = theDrmDepth/ticksize;
}

void solver_drm_close () {

	if ( drmImplement == YES && theDrmPart == PART1 ) {

		free(myDispWrite);
		if ( myNumberOfDrmNodes != 0 ) {
			hu_fclose(drmDisp);
			free (myDrmNodes );
		}
	}

	if ( drmImplement == YES && theDrmPart == PART2 ) {

		int32_t i;

		free (theDrmDisplacements1);
	    free (theDrmDisplacements2);

		for ( i = 0; i < myNumberOfDrmElements; i++ ) {
			free( theDrmElem[i].lnid_b);
			free( theDrmElem[i].lnid_e);
		}
		free (theDrmElem );
	}
}

void drm_output(mesh_t *myMesh, int32_t myID, int32_t  theGroupSize,
		        int *array_have_nodes, int64_t  *drm_array, drm_node_t *drm_node)
{
	FILE  *nodecoords;
	FILE  *drminfo;
	FILE  *procinfo;

	double  x_coord, y_coord, z_coord;
	int32_t i, procs_with_drm_nodes = 0;

	static char    filename [256];

	/* this is to avoid opening empty files */
	if ( myNumberOfDrmNodes != 0 ) {

		if ( theDrmPart == PART0 ) {
			sprintf( filename, "%s%s_%d", theDrmOutputDir, "/part0/drm_coordinates", myID);
		}

		if ( theDrmPart == PART1 ) {
			sprintf( filename, "%s%s_%d", theDrmOutputDir, "/part1/drm_coordinates", myID);
		}

		if ( (nodecoords = fopen(filename, "w")) == NULL ) {
			fprintf(stderr, "Error opening %s\n", filename);
		}

		/* All processors which  have drm nodes  prints the number of its drm nodes first,
		 * then prints the coordinates (after adding/subtracting the offset distances)
		 * Also surface-shift is  taken care of */
		hu_fwrite( &myNumberOfDrmNodes,   sizeof(int32_t), 1, nodecoords );

		for ( i = 0; i < myNumberOfDrmNodes; ++i ) {
			if ( theDrmPart == PART0 ) {
				x_coord = myMesh->nodeTable[drm_array[i]].x*(myMesh->ticksize) + theX_Offset;
				y_coord = myMesh->nodeTable[drm_array[i]].y*(myMesh->ticksize) + theY_Offset;

				if ( includeBuildings == YES )
					z_coord = myMesh->nodeTable[drm_array[i]].z*(myMesh->ticksize) - theSurfaceShift;
				else
					z_coord = myMesh->nodeTable[drm_array[i]].z*(myMesh->ticksize);
			}
			if ( theDrmPart == PART1 ) {
				x_coord = drm_node[i].coords.x[0] - theX_Offset;
				y_coord = drm_node[i].coords.x[1] - theY_Offset;
				if ( includeBuildings == YES )
					z_coord = drm_node[i].coords.x[2] - theSurfaceShift;
				else
					z_coord = drm_node[i].coords.x[2];
			}

			hu_fwrite( &x_coord, sizeof(double), 1, nodecoords );
			hu_fwrite( &y_coord, sizeof(double), 1, nodecoords );
			hu_fwrite( &z_coord, sizeof(double), 1, nodecoords );
		}

		fclose(nodecoords);
	}

	/* PE0 prints the number of drm nodes each processor has in a seperate file */
	if ( myID == 0 ) {
		for( i = 0 ;i < theGroupSize; ++i ) {
			if ( array_have_nodes[i] )
				procs_with_drm_nodes ++;
		}

		/* If in part 0  PE0 , first,  prints the number of processors that have drm nodes and their ids.
		 * Second, prints the total number of nodes */
		if ( theDrmPart == PART0 ) {
			sprintf( filename, "%s%s", theDrmOutputDir, "/part0/drm_processor_info");
			if ( (procinfo = fopen(filename, "w")) == NULL )
				fprintf(stderr, "Error opening %s\n", filename);
			hu_fwrite(&procs_with_drm_nodes, sizeof(int32_t), 1, procinfo);
			for( i = 0; i < theGroupSize; ++i ) {
				if ( array_have_nodes[i] ) {
					hu_fwrite(&i, sizeof(int32_t), 1, procinfo);
				}
			}
			hu_fwrite(&theNumberOfDrmNodes, sizeof(int32_t), 1, procinfo);
			fclose(procinfo);
			/* Moreover, PE0 prints the total number of drm nodes and drm elements respectively.
			 * It is going to be used in PART2 */
			sprintf( filename, "%s%s", theDrmOutputDir, "/part0/drm_information");
			drminfo = hu_fopen ( filename, "w" );
			fprintf(drminfo, "drm_numberofnodes = %d \ndrm_numberofelements = %d",
					theNumberOfDrmNodes,theNumberOfDrmElements);
			fclose(drminfo);
		}

		/* If in part 1  PE0 ,first, prints the total number of nodes and the total number of processors with drm nodes*/
		/* Then prints how many drm nodes each processor has with the processor ids.*/
		if ( theDrmPart == PART1 ) {
			sprintf( filename, "%s%s", theDrmOutputDir, "/part1/drm_processor_info");
			if ( (procinfo = fopen(filename, "w")) == NULL )
				fprintf(stderr, "Error opening %s\n", filename);
			hu_fwrite(&theNumberOfDrmNodes, sizeof(int32_t), 1, procinfo);
			hu_fwrite(&procs_with_drm_nodes, sizeof(int32_t), 1, procinfo);
			for( i = 0; i < theGroupSize; ++i ) {
				if ( array_have_nodes[i] ) {
					hu_fwrite(&i, sizeof(int32_t), 1, procinfo);
					hu_fwrite(&array_have_nodes[i], sizeof(int32_t), 1, procinfo);
				}
			}
			fclose(procinfo);
		}
		if ( myNumberOfDrmNodes != 0 ) {
			free(drm_array);
		}
	}
}

/*
 * Prints statistics about number of drm nodes.
 */
void drm_print_stats(int32_t *drmNodesCount, int32_t  theGroupSize,
		double   theXForMeshOrigin, double   theYForMeshOrigin,
		double   theZForMeshOrigin)
{
	FILE *fp;

	int pid;

	peshavenodes = 0;

	if ( theDrmPart == PART0 ) {
		fp = hu_fopen( "stat-drm-part0.txt", "w" );
	}

	if ( theDrmPart == PART1 ) {
		fp = hu_fopen( "stat-drm-part1.txt", "w" );
	}

	if ( theDrmPart == PART2 ) {
		fp = hu_fopen( "stat-drm-part2.txt", "w" );
	}

	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if (drmNodesCount[pid] != 0) {
			peshavenodes++;
		}
	}

	fprintf( fp, " %d in %d pes have drm nodes \n", peshavenodes, theGroupSize );

	fputs( "\n"
			"# ---------------------------------------- \n"
			"# Drm Nodes Count:                         \n"
			"# ---------------------------------------- \n"
			"# Rank       Nodes                         \n"
			"# ---------------------------------------- \n", fp );

	for ( pid = 0; pid < theGroupSize; pid++ ) {

		if (drmNodesCount[pid] != 0) {
			peshavenodes++;
			fprintf( fp, "%06d %11d\n", pid, drmNodesCount[pid] );
		}
	}

	if ( theDrmPart != PART2 ) {

		fprintf( fp,
				"# ---------------------------------------- \n"
				"# Total%11d   \n"
				"# ---------------------------------------- \n\n",
				theNumberOfDrmNodes );
	}


	if ( theDrmPart == PART2 ) {

		fprintf( fp,
				"# ---------------------------------------- \n"
				"# Total%11d (a common node may appear in multiple pes, so likely to be higher than the actual value)  \n"
				"# ---------------------------------------- \n\n",
				theNumberOfDrmNodes );
	}
	hu_fclosep( &fp );

	/* output aggregate information to the monitor file / stdout */
	fprintf( stdout,
			"\nDrm  information\n"
			"Origin of the sub region   x = %f\n"
			"                           y = %f\n"
			"                           z = %f\n"
			"Drm Part:                  Part %d\n"
			"Total number of drm nodes: %d",
			theXForMeshOrigin,
			theYForMeshOrigin,
			theZForMeshOrigin,
			theDrmPart,
			theNumberOfDrmNodes);

	if ( theDrmPart == PART2 ) {
	fprintf( stdout, " (a common node may appear in multiple pes, so likely to be higher than the actual value)" );
	}

	fprintf( stdout,	"\n\n" );

	fflush( stdout );
}

/*
 * Collects statistics about number of drm nodes
 */
void drm_stats(int32_t  myID, int32_t  theGroupSize,
		double   theXForMeshOrigin, double   theYForMeshOrigin,
		double   theZForMeshOrigin)
{
	int32_t *drmNodesCount = NULL;

	if ( myID == 0 ) {
		XMALLOC_VAR_N( drmNodesCount, int32_t, theGroupSize);
	}

	MPI_Gather( &myNumberOfDrmNodes, 1, MPI_INT,
			drmNodesCount,       1, MPI_INT, 0, comm_solver );

	if ( myID == 0 ) {

		drm_print_stats( drmNodesCount, theGroupSize,theXForMeshOrigin,
				theYForMeshOrigin, theZForMeshOrigin);

		xfree_int32_t( &drmNodesCount );
	}

	return;
}

/* -------------------------------------------------------------------------- */
/*                           Functions for Part0                              */
/* -------------------------------------------------------------------------- */

/*Find the drm nodes for PART0, store them in a hashtable avoiding duplicates.
 *And print the coordinates of these nodes
 */
void find_drm_nodes( mesh_t     *myMesh,       int32_t   myID,
					 const char *parametersin, double    ticksize,
					 int32_t    theGroupSize)
{
	int have_drm_nodes = 0;  /* 1 if yes , 0 otherwise */
	int i, j, k, owner;

	/* These are for communication of PE0 with other processors. dup_nodes are
	 * the nodes that I touch but are not owned by me  (owned by other processors.)
	 * */

	/* For PE0 to receive dup_nodes */
	int64_t  *dup_nodes[theGroupSize];
	 /* For PE0 to send dup_nodes after eliminating duplicates */
	int64_t  *dup_nodes_to_send[theGroupSize];
	int32_t   dup_nodes_count_send[theGroupSize]; /* For PE0 */
	int      *dup_owners[theGroupSize]; /* For PE0 to receive dup_nodes owners */

	int       dup_nodes_count = 0; /* For  all processors*/
	int64_t  *dup_nodes_to_recieve; /* For all processors*/

	/* For PE0 to eliminate duplicates from received dup_nodes  */
	drmhash_t  dups[theGroupSize];

	MPI_Status status;

	/* Allocate memory */
	if (myID == 0) {
		XMALLOC_VAR_N( pesHaveDrmNodes, int32_t, theGroupSize );
		XMALLOC_VAR_N( dupNodesCount, int32_t, theGroupSize );
	}

	/* Need to determine TableDim for HashTable. Need to know number of elements
	 * for this. */
	search_drm_elems(myMesh, 0);

	/* Besides storing drm nodes that I own, also store drm nodes owned by other
	 * processors in a seperate hashtable.
	 */
	if (myNumberOfDrmElements != 0)
	{
		/* Get tableDims for theDrmTable and theDrmDupTable */
		get_table_dim(&theTableDim, myNumberOfDrmElements);
		/* Assume theTableDupDim is approximately 1/16 th of theTableDim*/
		get_table_dim(&theTableDupDim, (int32_t)(myNumberOfDrmElements/16));
		/*  Init theDrmTable to store ids of nodes , whose  displacements/
		 * coordinates  will be  stored afterwards */
		drm_table_init(&theDrmTable, theTableDim);
		/*Also init theDrmDupTable */
		drm_table_init(&theDrmDupTable, theTableDupDim);
		/* Find drm nodes that I own  and store them in hashtable avoiding duplicates.
		 * Also find drm nodes that I do not own, store them in another hashtable
		 * along with their owner processor ids. */
		search_drm_elems(myMesh, 1);
		/* construct an array with the non-sorted global ids of drm nodes
		 * which are not owned by me. Also construct an array with the ids of
		 * owner processors.*/
		construct_drm_array(&theDrmDupTable, theTableDupDim, myNumberOfDupDrmNodes,
				&drmDupNodesIds, 2);
	}

	/* This part is mainly the communication b/w PE0 and other processors.First,
	 * all processors send their dup_nodes(nodes owned by other PEs) to PE0 along
	 * with their owner PEids.After merging all dup_nodes ,coming from each PE,
	 * Pe0 checks for duplicates, and sends those node ids to owner PEs. */

	/* Send theDrmDupTable to PE0. Then it sends it to owner processor.*/
	MPI_Gather(&myNumberOfDupDrmNodes, 1, MPI_INT, dupNodesCount,
			1, MPI_INT, 0, comm_solver);

	if ( myID == 0) {
		for ( i = 0; i < theGroupSize; ++i ) {
			if ( dupNodesCount[i] != 0 && i != 0 ) {
				XMALLOC_VAR_N( dup_nodes[i], int64_t, dupNodesCount[i] );
				XMALLOC_VAR_N( dup_owners[i], int32_t, dupNodesCount[i] );
			}
			/* This is used to collect dup_nodes according to owner PE id */
			drm_table_init(&(dups[i]), 89);
		}/* make it 89 for the time being */
	}

	/* Processors (with dup_nodes) send dup_nodes to PE0 along with their owner PE id */
	if ( myID != 0 && myNumberOfDupDrmNodes != 0 ) {
		MPI_Send( drmDupNodesIds, myNumberOfDupDrmNodes, MPI_LONG_LONG_INT, 0, myID, comm_solver );
		MPI_Send( drmNodeOwner, myNumberOfDupDrmNodes, MPI_INT, 0, myID + 1, comm_solver );
	}

	/* PE0 receives what other PEs send to it.*/
	if ( myID == 0) {

		for( i = 1; i < theGroupSize; ++i ) {
			if( dupNodesCount[i] != 0 ) {
				MPI_Recv( dup_nodes[i], dupNodesCount[i], MPI_LONG_LONG_INT, i, i, comm_solver, &status );
				MPI_Recv( dup_owners[i], dupNodesCount[i], MPI_INT, i, i+1, comm_solver, &status );
			}
		}

		/* Then it eliminates the duplicates and merges dup_nodes wrt owner PE id */
		/* This is for dup_nodes PE0 has */
		for ( j = 0; j < myNumberOfDupDrmNodes; ++j ) {
			k = drmDupNodesIds[j] % 89;
			owner = drmNodeOwner[j];
			insert(drmDupNodesIds[j], k, &(dups[owner]), NONE);
		}

		/* This is for dup_nodes coming from other PEs  */
		for ( i = 1; i < theGroupSize; ++i ) {
			for ( j = 0; j < dupNodesCount[i]; ++j ) {
				k = dup_nodes[i][j] % 89;
				owner = dup_owners[i][j];
				insert(dup_nodes[i][j], k, &(dups[owner]), NONE);
			}
		}

		/* It constructs arrays using hashtables ,filled with dup_nodes, for each
		 * PE who owns dup_nodes.Then it sends these arrays to owner PEs*/
		for ( i = 0; i < theGroupSize; ++i ) {
			dup_nodes_count_send[i] = dups[i].tableCnt;

			if ( dup_nodes_count_send[i] != 0 ) {
				construct_drm_array(&(dups[i]),89,dup_nodes_count_send[i],&dup_nodes_to_send[i],1);
			}

			if ( i != 0 ) {
				MPI_Send( &(dup_nodes_count_send[i]), 1 , MPI_INT, i, i, comm_solver );
				if ( dup_nodes_count_send[i] != 0 ) {
					MPI_Send( dup_nodes_to_send[i], dup_nodes_count_send[i], MPI_LONG_LONG_INT, i, i+1, comm_solver );
				}
			}
		}
	}

	/*PEs other than PE0 receives global id of their drm_dup_nodes*/
	if ( myID != 0 ) {
		MPI_Recv( &dup_nodes_count, 1, MPI_INT, 0, myID, comm_solver, &status );
		if ( dup_nodes_count != 0 ) {
			XMALLOC_VAR_N( dup_nodes_to_recieve, int64_t, dup_nodes_count );
			MPI_Recv( dup_nodes_to_recieve, dup_nodes_count, MPI_LONG_LONG_INT, 0, myID+1, comm_solver, &status );
		}
	}

	if ( myID == 0) {
		dup_nodes_count = dup_nodes_count_send[0];
		dup_nodes_to_recieve = dup_nodes_to_send[0];
	}

	/* If I have drm_elements insert dup_nodes to hashtable to eliminate
	 * duplicates */
	if ( myNumberOfDrmElements != 0 ) {
		for ( i = 0; i < dup_nodes_count; ++i ) {
			for ( j = 0; j < myMesh->nharbored; ++j ) {
				if ( dup_nodes_to_recieve[i] == myMesh->nodeTable[j].gnid ) {

					k = j % theTableDim;
					insert(j, k, &theDrmTable, NONE );

				}
			}
		}
		/* Update the number of drm nodes */
		myNumberOfDrmNodes = theDrmTable.tableCnt;
		/* construct an array with the sorted local nodal ids of drm nodes .*/
		construct_drm_array(&theDrmTable, theTableDim, myNumberOfDrmNodes, &drmNodeIds, 1);
	}

	/* If I do not have any drm_elems, then automatically print the coordinates of
	 * the nodes. But first need to find the local ids.No need of hashtable since
	 * duplicates are already checked by PE0 */
	if ( myNumberOfDrmElements == 0 ) {
		k = 0;
		/* only number of nodes is updated, number of elements does not change */
		myNumberOfDrmNodes = dup_nodes_count;
		if ( myNumberOfDrmNodes != 0 ) {
			XMALLOC_VAR_N( drmNodeIds, int64_t, dup_nodes_count );
			for ( i = 0; i < dup_nodes_count; ++i ) {
				for ( j = 0; j < myMesh->nharbored; ++j ) {
					if ( dup_nodes_to_recieve[i] == myMesh->nodeTable[j].gnid ) {
						drmNodeIds[k] = j;
						k++;
					}
				}
			}
			/* PE0 prints later */
			if ( myID != 0 ) {
				drm_output (myMesh, myID, theGroupSize, pesHaveDrmNodes, drmNodeIds, NULL);
			}
		}
	}
	/* Find total number of drm nodes and elements .*/
	MPI_Reduce(&myNumberOfDrmNodes, &theNumberOfDrmNodes, 1, MPI_INT, MPI_SUM, 0, comm_solver);
	MPI_Reduce(&myNumberOfDrmElements, &theNumberOfDrmElements, 1, MPI_INT, MPI_SUM, 0, comm_solver);

	/*This is useful when printing coordinates to avoid opening empty files*/
	if ( myNumberOfDrmNodes != 0 ) {
		have_drm_nodes = 1;
	}
	/*Pe0 gathers the array of pes_have_elems.*/
	MPI_Gather(&have_drm_nodes, 1, MPI_INT, pesHaveDrmNodes,
			1, MPI_INT, 0, comm_solver);

	/* PE0 prints data even it does not have any drm nodes */
	if ( myNumberOfDrmElements != 0 || myID == 0 ) {
		/* output the drm_node_table*/
		drm_output(myMesh, myID, theGroupSize,pesHaveDrmNodes, drmNodeIds, NULL);
	}

	/* Now free all the allocated memory */
	/* Free in PE0 only */
	if ( myID == 0 ) {
		for( i = 0; i < theGroupSize; ++i ) {
			if ( dupNodesCount[i] != 0 && i != 0 ) {
				free(dup_nodes[i]);
				free(dup_owners[i]);
			}

			if ( dup_nodes_count_send[i] != 0 ) {
				free(dup_nodes_to_send[i]);
			}
			freeHashTable(&(dups[i]), 89);
		}
		free(pesHaveDrmNodes);
		free(dupNodesCount);
	}

	/* Free processors other than PE0 only */
	if ( myID != 0 ) {
		if ( dup_nodes_count != 0 ) {
			free(dup_nodes_to_recieve);
		}
	}

	/*free Hashtables */
	if ( myNumberOfDrmElements != 0 ) {
		freeHashTable(&theDrmTable, theTableDim);
		freeHashTable(&theDrmDupTable, theTableDupDim);
		if ( myNumberOfDupDrmNodes != 0 ) {
			free(drmNodeOwner);
			free(drmDupNodesIds);
		}
	}
}

/* -------------------------------------------------------------------------- */
/*                           Functions for Part1                              */
/* -------------------------------------------------------------------------- */
/**
 * Prepare all the info every drm node needs once it is located in a processor
 */
void setup_drm_data(mesh_t *myMesh, int32_t myID,  int32_t   theGroupSize) {

	int      have_drm_nodes = 0;  /* 1 if yes , 0 otherwise */

	int32_t  i, iLocal = 0;
	int32_t  *drmNodesCount = NULL;

	vector3D_t drm_node_coords;
	octant_t   *octant;

	char       filename [256];

	/* look for the drm nodes in the domain each processor has */
	for ( i = 0; i < theNumberOfDrmNodes; i++ ) {
		drm_node_coords.x[0] = theDrmNodeCoords[i].f[0];
		drm_node_coords.x[1] = theDrmNodeCoords[i].f[1];
		drm_node_coords.x[2] = theDrmNodeCoords[i].f[2];
		if ( search_point( drm_node_coords, &octant ) == 1 ) {
			myNumberOfDrmNodes++;
		}
	}

	/* Allocate memory */
	if (myID == 0) {
		XMALLOC_VAR_N( pesHaveDrmNodes, int32_t, theGroupSize );
	}

	/*This is useful when printing coordinates to avoid opening empty files*/
	if ( myNumberOfDrmNodes != 0 ) {
		have_drm_nodes = 1;
	}
	/*Pe0 gathers the array of pes_have_elems.*/
	MPI_Gather(&have_drm_nodes, 1, MPI_INT, pesHaveDrmNodes,
			1, MPI_INT, 0, comm_solver);

	/* allocate memory if necessary and generate the list of drm nodes per
	 * processor. Do not go any further if I do not have any drm nodes */
	if ( myNumberOfDrmNodes != 0 ) {
		/* Open files to write displacements. First print the total number of drm nodes */
		sprintf(filename, "%s%s_%d", theDrmOutputDir, "/part1/drm_disp", myID);
		drmDisp = hu_fopen( filename,"w" );
		hu_fwrite( &myNumberOfDrmNodes,  sizeof(int32_t), 1, drmDisp );
		XMALLOC_VAR_N(myDrmNodes, drm_node_t, myNumberOfDrmNodes);

		for ( i = 0; i < theNumberOfDrmNodes; i++ ) {
			drm_node_coords.x[0] = theDrmNodeCoords[i].f[0];
			drm_node_coords.x[1] = theDrmNodeCoords[i].f[1];
			drm_node_coords.x[2] = theDrmNodeCoords[i].f[2];

			if (search_point( drm_node_coords, &octant ) == 1) {
				myDrmNodes[iLocal].id = i;
				myDrmNodes[iLocal].coords = drm_node_coords;

				compute_csi_eta_dzeta(octant, myDrmNodes[iLocal].coords,
						&(myDrmNodes[iLocal].localcoords),
						myDrmNodes[iLocal].nodestointerpolate);
				iLocal += 1;
			}
		}
	}

	free(theDrmNodeCoords);

	/* These are for drm_output */
	if ( myID == 0 ) {
		XMALLOC_VAR_N( drmNodesCount, int32_t, theGroupSize);
	}

	MPI_Gather( &myNumberOfDrmNodes, 1, MPI_INT,
				drmNodesCount,       1, MPI_INT, 0, comm_solver );

	/* PE0 prints data even if it does not have any drm nodes */
	if ( myNumberOfDrmNodes != 0 || myID == 0 ) {
		/* output the drm_node_table*/
		drm_output(myMesh, myID, theGroupSize, drmNodesCount, NULL, myDrmNodes);
	}
}


void solver_output_drm_nodes ( mysolver_t* mySolver,int step, int32_t totalsteps) {

	Timer_Start("Solver drm output");

	if ( drmImplement == YES && theDrmPart == PART1 && myNumberOfDrmNodes != 0 )
	{
		if (step % theDrmPrintRate == 0 || step == totalsteps - 1) {

			//printf("%d ",step);
			int iPhi;
			/* Auxiliary array to handle shape functions in a loop */
			double  xi[3][8]={ {-1,  1, -1,  1, -1,  1, -1, 1} ,
					{-1, -1,  1,  1, -1, -1,  1, 1} ,
					{-1, -1, -1, -1,  1,  1,  1, 1} };

			double     phi[8];
			double     dis_x, dis_y, dis_z;
			int32_t    i,nodesToInterpolate[8];;
			vector3D_t localCoords; /* convenient renaming */

			if ( step == 0) {
				XMALLOC_VAR_N( myDispWrite, double, myNumberOfDrmNodes*3 );
			}

			for ( i = 0; i < myNumberOfDrmNodes; i++ ) {

				localCoords = myDrmNodes[i].localcoords;

				for ( iPhi = 0; iPhi < 8; iPhi++ ) {
					nodesToInterpolate[iPhi] = myDrmNodes[i].nodestointerpolate[iPhi];
				}

				/* Compute interpolation function (phi) for each node and
				 * load the displacements
				 */
				dis_x = 0;
				dis_y = 0;
				dis_z = 0;

				for ( iPhi = 0; iPhi < 8; iPhi++ ) {
					phi[ iPhi ] = ( 1 + xi[0][iPhi]*localCoords.x[0] )
		             			* ( 1 + xi[1][iPhi]*localCoords.x[1] )
		             			* ( 1 + xi[2][iPhi]*localCoords.x[2] ) / 8;

					dis_x += phi[iPhi] * mySolver->tm1[ nodesToInterpolate[iPhi] ].f[0];
					dis_y += phi[iPhi] * mySolver->tm1[ nodesToInterpolate[iPhi] ].f[1];
					dis_z += phi[iPhi] * mySolver->tm1[ nodesToInterpolate[iPhi] ].f[2];
				}

				myDispWrite [3*i]   = dis_x;
				myDispWrite [3*i+1] = dis_y;
				myDispWrite [3*i+2] = dis_z;
			}

			fwrite( myDispWrite, sizeof(double), myNumberOfDrmNodes*3, drmDisp );

		}
	}
	Timer_Stop("Solver drm output");

}

/* -------------------------------------------------------------------------- */
/*                           Functions for Part2                              */
/* -------------------------------------------------------------------------- */

/* Find which nodes are exterior and which of them are interior in drm elements
 * according to their positions.*/
void proc_drm_elems( mesh_t   *myMesh,     int32_t   myID,
		int32_t theGroupSize, int32_t theTotalSteps) {

	Timer_Start("Find Drm File To Readjust");

	find_drm_file_readjust ( theGroupSize, myID, theTotalSteps  ) ;

	Timer_Stop("Find Drm File To Readjust");
	Timer_Reduce("Find Drm File To Readjust", MAX | MIN | AVERAGE , comm_solver);

	Timer_Start("Fill Drm Struct");

	/* keys are used to determine interior and exterior nodes.x_key_1 is for
	 * x face that is closer to origin, x_key_2 is for far away x face. Other
	 * keys have similar meanings. They are turned on if elements are in any of
	 * these faces */

	int32_t   i, n_0, n_7, iDrmElem = 0;
	int32_t   x_key_1 , x_key_2 , y_key_1, y_key_2, depth_key;

	/* Allocate memory for theDrmElem.First find the number of drm elements I own
	 */
	search_drm_elems(myMesh, 0);

	/* Check if total number of drm elements is same as the total number of elements
	 * from part0.
	 */
	drm_sanity_check(myID);

	if (myNumberOfDrmElements != 0) {

		/* First find drm nodes each processor has.*/

		/* Get tableDims for theDrmTable */
		get_table_dim(&theTableDim, myNumberOfDrmElements);
		/*  Init theDrmTable to store ids of drm nodes. */
		drm_table_init(&theDrmTable, theTableDim);
		/* Find drm nodes that I own  and store them in hashtable avoiding duplicates.*/
		search_drm_elems(myMesh, 1);
		/* construct an array with the sorted local nodal ids of drm nodes .*/
		construct_drm_array(&theDrmTable, theTableDim, myNumberOfDrmNodes, &drmNodeIds, 1);
		/* Drm node local ids are stored in drmNodeIds now. */

		/* Fill drmElem struct now. */

		XMALLOC_VAR_N( theDrmElem, drm_elem_t, myNumberOfDrmElements );

		/* Loop each element to fill drm element struct.
		 */
		for ( i = 0; i < (myMesh->lenum ); ++i ) {

			int    res;
			int    keys_array[5];
			double coords_array[6];

			/* n_0 and n_7 are local nodal id's of 0th and 7th node.
			 * Coordinates of 7th node are always bigger than 0th node.
			 * */

			n_0 = myMesh->elemTable[i].lnid[0]; /* looking at first node of the element(at top) */
			n_7 = myMesh->elemTable[i].lnid[7]; /* looking at last node of the element(at bottom) */

			/* these are  coordinates of 0th  node  */
			coords_array[0] = myMesh->nodeTable[n_0].x;
			coords_array[1] = myMesh->nodeTable[n_0].y;
			coords_array[2] = myMesh->nodeTable[n_0].z;

			/* these are  coordinates of 7th node  */
			coords_array[3] = myMesh->nodeTable[n_7].x;
			coords_array[4] = myMesh->nodeTable[n_7].y;
			coords_array[5] = myMesh->nodeTable[n_7].z;

			/* we need to find the values for the keys.
			 * these are  total 5 faces where elements needs to be traversed
			 * we always try to keep inside boundary as large  as possible if
			 * given limits intersect with element sides.*/

			res = is_drm_elem(coords_array,keys_array,2, myMesh->ticksize);

			x_key_1   = keys_array[0];
			x_key_2   = keys_array[1];
			y_key_1   = keys_array[2];
			y_key_2   = keys_array[3];
			depth_key = keys_array[4];

			/*NOW CONSIDER DIFFERENT CASES.There are 18 different cases */

			/*Do nothing if it is not a drm element */
			if ( res == 0 ) {
				continue;
			}

			/* Start to fill struct if it is drm element */
			(theDrmElem[iDrmElem]).leid = i;

			/* NEAR X FACE */

			/* Element is on near x face, has 4 boundary and exterior nodes */
			if ( x_key_1 == 1 && y_key_1 == 0  && y_key_2 == 0  && depth_key == 0 ){
				int bound_ids[4]    = {1,3,5,7};
				int exterior_ids[4] = {0,2,4,6};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 1, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near x face and near y face intersection, but
			 * away from depth (z) face. It has 2 boundary nodes and 6 exterior
			 * nodes */
			if ( x_key_1 == 1 && y_key_1 == 1  && depth_key == 0 ) {
				int bound_ids[2]    = {3,7};
				int exterior_ids[6] = {0,1,2,4,5,6};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near x face and far y face intersection, but
			 * away from depth (z) face. It has 2 boundary nodes and 6 exterior
			 * nodes */
			if ( x_key_1 == 1 &&  y_key_2 == 1  && depth_key == 0 ) {
				int bound_ids[2]    = {1,5};
				int exterior_ids[6] = {0,2,3,4,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near x face and near y face intersection, also
			 * intersects with depth (z) face. It has 1 boundary node and 7 exterior
			 * nodes */
			if ( x_key_1 == 1 && y_key_1 == 1  && depth_key == 1 ) {
				int bound_ids[1]    = {3};
				int exterior_ids[7] = {0,1,2,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 3, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near x face and far y face intersection, also
			 * intersects with depth (z) face. It has 1 boundary node and 7 exterior
			 * nodes */
			if ( x_key_1 == 1 && y_key_2 == 1  && depth_key == 1 ) {
				int bound_ids[1]    = {1};
				int exterior_ids[7] = {0,2,3,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 3, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near x face and also intersects with depth (z)
			 * face. It has 2 boundary node and 6 exterior nodes */
			if ( x_key_1 == 1 && y_key_1 == 0  && y_key_2 == 0  && depth_key == 1 ) {
				int bound_ids[2]    = {1,3};
				int exterior_ids[6] = {0,2,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2, bound_ids, exterior_ids );
				continue;
			}

			/* NEAR Y FACE */

			/* Element is on near y face, has 4 boundary and exterior nodes */
			if ( y_key_1 == 1 && x_key_1 == 0  && x_key_2 == 0  && depth_key == 0 ) {
				int bound_ids[4]    = {2,3,6,7};
				int exterior_ids[4] = {0,1,4,5};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 1, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  near y face and also intersects with depth (z)
			 * face. It has 2 boundary node and 6 exterior nodes */
			if ( y_key_1 == 1 && x_key_1 == 0  && x_key_2 == 0  && depth_key == 1 ) {
				int bound_ids[2]    = {2,3};
				int exterior_ids[6] = {0,1,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2, bound_ids, exterior_ids );
				continue;
			}

			/* FAR X FACE */

			/* Element is on far x face, has 4 boundary and exterior nodes */
			if ( x_key_2 == 1 && y_key_1 == 0  && y_key_2 == 0  && depth_key == 0 ) {
				int bound_ids[4]    = {0,2,4,6};
				int exterior_ids[4] = {1,3,5,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 1, bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far x face and near y face intersection, but
			 * away from depth (z) face. It has 2 boundary nodes and 6 exterior
			 * nodes */
			if ( x_key_2 == 1 && y_key_1 == 1  && depth_key == 0 ) {
				int bound_ids[2]    = {2,6};
				int exterior_ids[6] = {0,1,3,4,5,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2 , bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far x face and far y face intersection, but
			 * away from depth (z) face. It has 2 boundary nodes and 6 exterior
			 * nodes */
			if ( x_key_2 == 1  && y_key_2 == 1  && depth_key == 0 ) {
				int bound_ids[2]    = {0,4};
				int exterior_ids[6] = {1,2,3,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2 , bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far x face and near y face intersection, also
			 * intersects with depth (z) face. It has 1 boundary node and 7 exterior
			 * nodes */
			if ( x_key_2 == 1 && y_key_1 == 1   && depth_key == 1 ) {
				int bound_ids[1]    = {2};
				int exterior_ids[7] = {0,1,3,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 3 , bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far x face and far y face intersection, also
			 * intersects with depth (z) face. It has 1 boundary node and 7 exterior
			 * nodes */
			if ( x_key_2 == 1  && y_key_2 == 1  && depth_key == 1 ) {
				int bound_ids[1]    = {0};
				int exterior_ids[7] = {1,2,3,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 3 , bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far x face and also intersects with depth (z)
			 * face. It has 2 boundary node and 6 exterior nodes */
			if ( x_key_2 == 1 && y_key_1 == 0  && y_key_2 == 0  && depth_key == 1 ) {
				int bound_ids[2]    = {0,2};
				int exterior_ids[6] = {1,3,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2 , bound_ids, exterior_ids );
				continue;
			}

			/* FAR Y FACE */

			/* Element is on far y face, has 4 boundary and exterior nodes */
			if ( y_key_2 == 1 && x_key_1 == 0  && x_key_2 == 0  && depth_key == 0 ) {
				int bound_ids[4]    = {0,1,4,5};
				int exterior_ids[4] = {2,3,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 1 , bound_ids, exterior_ids );
				continue;
			}

			/* Element is at the  far y face and also intersects with depth (z)
			 * face. It has 2 boundary node and 6 exterior nodes */
			if ( y_key_2 == 1 && x_key_1 == 0  && x_key_2 == 0  && depth_key == 1 ) {
				int bound_ids[2]    = {0,1};
				int exterior_ids[6] = {2,3,4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 2 , bound_ids, exterior_ids );
				continue;
			}

			/* WE DEALT EVERYTHING EXCEPT DEPTH(z) FACE*/

			/* Element is on depth (z) face, has 4 boundary and exterior nodes */
			if ( x_key_1 == 0 && x_key_2 == 0 && y_key_1 == 0  && y_key_2 == 0  && depth_key == 1 ) {
				int bound_ids[4]    = {0,1,2,3};
				int exterior_ids[4] = {4,5,6,7};
				iDrmElem = fill_drm_struct(myMesh, iDrmElem, 1 , bound_ids, exterior_ids );
				continue;
			}
		}
	}


	Timer_Stop("Fill Drm Struct");
	Timer_Reduce("Fill Drm Struct", MAX | MIN | AVERAGE , comm_solver);

	Timer_Start("Comm of Drm Coordinates");

	comm_drm_coordinates ( myMesh, theGroupSize, myID );

	Timer_Stop("Comm of Drm Coordinates");
	Timer_Reduce("Comm of Drm Coordinates", MAX | MIN | AVERAGE , comm_solver);

	rearrange_drm_files ( myMesh, theGroupSize, myID );

}

void find_drm_file_readjust ( int32_t theGroupSize, int32_t myID, int32_t theTotalSteps ) {

	int32_t   i, pid, interval, total_pes_used = 0, max_pes_used_id = 0;
	double   *pes_desired;
	double 	 *pes_remainder;
	int      timesteps_files; // total steps in drm files due to drm_print_rate

	XMALLOC_VAR_N( pes_per_file, int32_t, drmFilesCount );
	XMALLOC_VAR_N( pes_desired, double, drmFilesCount );
	XMALLOC_VAR_N( pes_remainder, double, drmFilesCount );

	timesteps_files = theTotalSteps/theDrmPrintRate;
	/*
	// For testing
	for ( pid = 0; pid < drmFilesCount; pid++ ) {
		partOneDrmNodesCnt[pid].count=10*pid;
	}
	theNumberOfDrmNodes=450;
	 */

	/* need to find out which file I need to open*/
	for ( pid = 0; pid < drmFilesCount; pid++ ) {
		double aux;
		aux = (double)(theGroupSize)
			* (double)(partOneDrmNodesCnt[pid].count)
			/ (double)theNumberOfDrmNodes; /* at this point this is the desired value*/
		pes_desired[pid] = aux;
		pes_remainder[pid] = aux - 1; //Since we assign at least one processor to each file
		pes_per_file[pid] = 1;
	}

	/*
	// For testing
	if (myID == 0)
		for ( pid = 0; pid < drmFilesCount; pid++ ) {
			printf( " %d - %f %f \n",pid,pes_desired[pid],pes_remainder[pid]);
		}
	 */

	/* Distribute the remaining processors to the files */
	for ( i = 0; i < (theGroupSize-drmFilesCount); ++i ) {
		max_pes_used_id = 0;
		for ( pid = 1; pid < drmFilesCount; pid++ ) {
			if ( (double)(pes_remainder[pid] / pes_desired[pid]) >
			(double)(pes_remainder[max_pes_used_id] / pes_desired[max_pes_used_id]) ) {
				max_pes_used_id = pid;
			}
		}
		pes_remainder[max_pes_used_id] = pes_remainder[max_pes_used_id] - 1.0;
		pes_per_file[max_pes_used_id]++;
	}

	/* sanity check */
	for ( pid = 0; pid < drmFilesCount; pid++ ) {
		if ( pes_per_file[pid] <= 0 ) {
			fprintf( stderr,
					"Something went wrong in find_drm_file_readjust1.Negative pes number" );
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}
		total_pes_used += pes_per_file[pid];
	}

	if ( total_pes_used != theGroupSize ) {
		fprintf( stderr,
				"Something went wrong in find_drm_file_readjust2." );
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}
	/* Now find which file to open and the offset  */

	total_pes_used = 0;

	for ( pid = 0; pid < drmFilesCount; pid++ ) {

		total_pes_used += pes_per_file[pid];

		if ( total_pes_used > myID ) {

			/* Necessary for comm */

			XMALLOC_VAR_N( rank, int32_t, pes_per_file[pid] );

			for ( i = 0; i < pes_per_file[pid]; i++ ) {
				rank[i] = total_pes_used - pes_per_file[pid] + i;
			}

			interval = (int)( 1 + (double)timesteps_files / (double)pes_per_file[pid]);

			drmFileToOpen.cid = pid;
			drmFileToOpen.id = partOneDrmNodesCnt[pid].id;
			drmFileToOpen.root_pe = total_pes_used - pes_per_file[pid];
			drmFileToOpen.pos = myID  - drmFileToOpen.root_pe;
			drmFileToOpen.step_to_start = interval * drmFileToOpen.pos;
			drmFileToOpen.step_to_end = drmFileToOpen.step_to_start + interval;

			if ( drmFileToOpen.step_to_start >= timesteps_files ) {
				fprintf( stderr,
						"Something went wrong in find_drm_file_readjust3" );
				MPI_Abort(MPI_COMM_WORLD, ERROR);
				exit(1);
			}

			if( drmFileToOpen.step_to_end > timesteps_files ) {
				drmFileToOpen.step_to_end = timesteps_files  ;
			}
			break;
		}
	}

	free(pes_remainder);
	free(pes_desired);

	//printf(" %d %d %d %d %d\n",myID,drmFileToOpen.cid, drmFileToOpen.step_to_start,drmFileToOpen.step_to_end,drmFileToOpen.step_to_end-drmFileToOpen.step_to_start );
}

/* Each processor should know the which processors has which drm nodes.*/

void comm_drm_coordinates ( mesh_t  *myMesh , int32_t theGroupSize, int32_t myID) {

	MPI_Datatype coords_mpi;

	int32_t   i = 0, j, pid;

	/* Find the total number of drm nodes, then print it in stats.Note that this is likely to be
	 * higher than the actual value, since a single drm node may be counted in multiple processors*/
	MPI_Reduce(&myNumberOfDrmNodes, &theNumberOfDrmNodes, 1, MPI_INT, MPI_SUM, 0, comm_solver);

	XMALLOC_VAR_N( drmNodesCountP2, int32_t, theGroupSize);

	MPI_Allgather( &myNumberOfDrmNodes, 1, MPI_INT,
					drmNodesCountP2,      1, MPI_INT, comm_solver );

	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if (drmNodesCountP2[pid] != 0) {
			peshavenodes++;
		}
	}

	XMALLOC_VAR_N( pesDrmNodes, pes_with_drm_nodes, peshavenodes );

	for ( pid = 0; pid < theGroupSize; pid++ ) {

		if ( drmNodesCountP2[pid] != 0 ) {

			/* Initialize struct */
			pesDrmNodes[i].id = pid;
			pesDrmNodes[i].minx = 0;
			pesDrmNodes[i].miny = 0;
			pesDrmNodes[i].minz = 0;
			pesDrmNodes[i].maxx = 0;
			pesDrmNodes[i].maxy = 0;
			pesDrmNodes[i].maxz = 0;

			XMALLOC_VAR_N( pesDrmNodes[i].drmNodeCoords, vector3D_t, drmNodesCountP2[pid] );

			if ( pid == myID ) {

				for ( j = 0; j < drmNodesCountP2[pid]; j++ ) {

					pesDrmNodes[i].drmNodeCoords[j].x[0] = (myMesh->ticksize)*myMesh->nodeTable[drmNodeIds[j]].x;
					pesDrmNodes[i].drmNodeCoords[j].x[1] = (myMesh->ticksize)*myMesh->nodeTable[drmNodeIds[j]].y;
					pesDrmNodes[i].drmNodeCoords[j].x[2] = (myMesh->ticksize)*myMesh->nodeTable[drmNodeIds[j]].z;
				}
			}
			++i;
		}
	}

	MPI_Type_contiguous(3, MPI_DOUBLE, &coords_mpi);
	MPI_Type_commit(&coords_mpi);

	i = 0;
	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if (drmNodesCountP2[pid] != 0) {
			MPI_Bcast(pesDrmNodes[i].drmNodeCoords, drmNodesCountP2[pid], coords_mpi, pid, comm_solver);
			++i;
		}
	}
}

void rearrange_drm_files ( mesh_t  *myMesh, int32_t theGroupSize, int32_t myID ) {

	FILE* fp;

	MPI_Offset disp;
	MPI_Offset disp2;

	MPI_Status status;
	MPI_Group group_world;
	MPI_Group newgroup[drmFilesCount];
	MPI_Comm  comm_world;
	MPI_Comm  newcomm[drmFilesCount];

	int32_t   local_number_drm_nodes = 0, disp_cnt = 0, pes_i_owe_cnt = 0, pid_cnt = 0, l = 0, m = 0,
			  n = 0, count = 0, i = 0, j = 0, pid = 0, interval = 0, recv_cnt,pes_grp_owe_cnt = 0;
	int32_t   *node_count_to_send_me; /* Find which processors needs how many nodes*/
	int32_t   **coord_i_owe ;      /* the order of coordinates (that I owe to processors )in the array */
	int32_t   *pes_i_owe_id ;      /* the ID of processor that I owe displacements*/
	int32_t   *array_cnt_me;       /* keep track number of elements in each array */

	int32_t   *pes_i_owe_grp_cnt ;     /* has information for all pes in the group. */
	int32_t   pes_i_owe_cnt_send = 0;  /* non zero if i am reading the beginning of the file */

	int32_t   *pes_id_cumulative;

	int64_t   local_id, node_id, elem_id;

	int32_t   *node_count_to_send_grp; /* Find which processors needs how many nodes in group*/
	int32_t   **coord_grp_owe ;        /* the order of coordinates (that group owes to processors )in the array */
	int32_t   *pes_grp_owe_id ;        /* the ID of processor that group owes displacements*/
	int32_t   *array_cnt_grp;          /* keep track number of elements in each array */


	/* these are the coords from the file I need to readjust */
	fvector_t  *drmCoordsAdjust;
	fvector_t  *to_read,  *to_write;

	//off_t   whereToRead;

	double  x, y, z; /* in physical coordinates */
	double  myminx = 0, mymaxx = 0, myminy = 0, mymaxy = 0, myminz = 0, mymaxz = 0;
	int 	total_pes_cnt = 0;

	static char    filename [256];
	comm_world = comm_solver;

	MPI_Comm_group(comm_world, &group_world);
	MPI_Group_incl(group_world, pes_per_file[drmFileToOpen.cid], rank,&(newgroup[drmFileToOpen.cid]));
	MPI_Comm_create ( comm_world, newgroup[drmFileToOpen.cid], &(newcomm[drmFileToOpen.cid]) );

	Timer_Start("Find Which Drm Files To Print");

	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if (drmNodesCountP2[pid] != 0) {

			for ( j = 0; j < drmNodesCountP2[pid]; ++j ) {

				pesDrmNodes[i].minx = MIN(pesDrmNodes[i].minx,
										  pesDrmNodes[i].drmNodeCoords[j].x[0]);
				pesDrmNodes[i].miny = MIN(pesDrmNodes[i].miny,
										  pesDrmNodes[i].drmNodeCoords[j].x[1]);
				pesDrmNodes[i].minz = MIN(pesDrmNodes[i].minz,
										  pesDrmNodes[i].drmNodeCoords[j].x[2]);

				pesDrmNodes[i].maxx = MAX(pesDrmNodes[i].maxx,
										  pesDrmNodes[i].drmNodeCoords[j].x[0]);
				pesDrmNodes[i].maxy = MAX(pesDrmNodes[i].maxy,
										  pesDrmNodes[i].drmNodeCoords[j].x[1]);
				pesDrmNodes[i].maxz = MAX(pesDrmNodes[i].maxz,
										  pesDrmNodes[i].drmNodeCoords[j].x[2]);
			}
			++i;
		}
	}

	/* Now we know which file to open to adjust displacements */

	sprintf( filename, "%s_%d","/lustre/scratch/yigit/DRM_new/part1/drm_coordinates", drmFileToOpen.id);

	//sprintf( filename, "%s%s_%d", theDrmOutputDir, "/part1/drm_coordinates", drmFileToOpen.id);
	//sprintf( filename, "%s_%d","/lustre/scratch/yigit/DRM_TEST/outputfiles/DRM/part1/drm_coordinates", drmFileToOpen.id);

	if ( (fp = fopen(filename, "r")) == NULL ) {
		fprintf(stderr, "Error opening %s\n", filename);
	}

	hu_fread(&local_number_drm_nodes, sizeof(int32_t), 1, fp);

	XMALLOC_VAR_N( drmCoordsAdjust, fvector_t, local_number_drm_nodes );

	hu_fread(drmCoordsAdjust, sizeof(fvector_t), local_number_drm_nodes, fp);

	/* We need to adjust coordinates for Surface Shift */
	for ( j = 0; j < local_number_drm_nodes; ++j ) {
		drmCoordsAdjust[j].f[2] += theSurfaceShift;
	}

	fclose(fp);

	/* Distribute coords among processors as well */

	interval = (int)( 1 + (double)local_number_drm_nodes / (double)pes_per_file[drmFileToOpen.cid]);

	if(pes_per_file[drmFileToOpen.cid] == 1 ) {
		interval = local_number_drm_nodes;
	}

	drmFileToOpen.coord_to_start = interval * drmFileToOpen.pos;
	drmFileToOpen.coord_to_end = interval * drmFileToOpen.pos + interval;

	if ( drmFileToOpen.coord_to_start >= local_number_drm_nodes ) {
		fprintf( stderr,
				"Something went wrong in rearrange_drm_files" );
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	if( drmFileToOpen.coord_to_end > local_number_drm_nodes ) {
		drmFileToOpen.coord_to_end = local_number_drm_nodes  ;
	}

//	printf("%d %d %d \n",drmFileToOpen.coord_to_start,drmFileToOpen.coord_to_end,myID);

	for ( i = drmFileToOpen.coord_to_start; i < drmFileToOpen.coord_to_end; ++i ) {

		myminx = MIN(myminx,drmCoordsAdjust[i].f[0]);
		myminy = MIN(myminy,drmCoordsAdjust[i].f[1]);
		myminz = MIN(myminz,drmCoordsAdjust[i].f[2]);

		mymaxx = MAX(mymaxx,drmCoordsAdjust[i].f[0]);
		mymaxy = MAX(mymaxy,drmCoordsAdjust[i].f[1]);
		mymaxz = MAX(mymaxz,drmCoordsAdjust[i].f[2]);

		//	printf ("%f %f %f \n",drmCoordsAdjust[i].f[0],drmCoordsAdjust[i].f[1], drmCoordsAdjust[i].f[2]);
	}

	/* Now we have everything, just need to know for which processors I will send data */
	node_count_to_send_me = calloc(theGroupSize,sizeof(int32_t));

	for ( pid = 0; pid < theGroupSize; pid++ ) {

		if ( (pesDrmNodes[pid_cnt].minx >=myminx && pesDrmNodes[pid_cnt].minx <=mymaxx) ||
			 (pesDrmNodes[pid_cnt].maxx >=myminx && pesDrmNodes[pid_cnt].maxx <=mymaxx) ||
			 (pesDrmNodes[pid_cnt].minx <=myminx && pesDrmNodes[pid_cnt].maxx >=mymaxx) )
		if ( (pesDrmNodes[pid_cnt].miny >=myminy && pesDrmNodes[pid_cnt].miny <=mymaxy) ||
		     (pesDrmNodes[pid_cnt].maxy >=myminy && pesDrmNodes[pid_cnt].maxy <=mymaxy) ||
			 (pesDrmNodes[pid_cnt].miny <=myminy && pesDrmNodes[pid_cnt].maxy >=mymaxy) )
		if ( (pesDrmNodes[pid_cnt].minz >=myminz && pesDrmNodes[pid_cnt].minz <=mymaxz) ||
			 (pesDrmNodes[pid_cnt].maxz >=myminz && pesDrmNodes[pid_cnt].maxz <=mymaxz) ||
			 (pesDrmNodes[pid_cnt].minz <=myminz && pesDrmNodes[pid_cnt].maxz >=mymaxz) )

			for ( j = 0; j < drmNodesCountP2[pid]; j++ ) {

				if ( pesDrmNodes[pid_cnt].drmNodeCoords[j].x[0] >= myminx &&
					 pesDrmNodes[pid_cnt].drmNodeCoords[j].x[1] >= myminy &&
					 pesDrmNodes[pid_cnt].drmNodeCoords[j].x[2] >= myminz &&
				 	 pesDrmNodes[pid_cnt].drmNodeCoords[j].x[0] <= mymaxx &&
					 pesDrmNodes[pid_cnt].drmNodeCoords[j].x[1] <= mymaxy &&
					 pesDrmNodes[pid_cnt].drmNodeCoords[j].x[2] <= mymaxz ) {

					for ( i = drmFileToOpen.coord_to_start; i < drmFileToOpen.coord_to_end; ++i ) {

						if ( drmCoordsAdjust[i].f[0] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[0]  &&
							 drmCoordsAdjust[i].f[1] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[1]  &&
							 drmCoordsAdjust[i].f[2] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[2] ) {
							node_count_to_send_me[pesDrmNodes[pid_cnt].id]++;
							break;
						}
					}
				}
			}
		if ( drmNodesCountP2[pid] ) pid_cnt++;
	}

	for ( i = 0; i < theGroupSize; ++i ) {
		if(node_count_to_send_me[i]) {
			pes_i_owe_cnt++;
		}
	}

	coord_i_owe = calloc(theGroupSize,sizeof(int32_t*));
	pes_i_owe_id = calloc(pes_i_owe_cnt,sizeof(int32_t));
	array_cnt_me = calloc(pes_i_owe_cnt,sizeof(int32_t));

	j = 0;
	for ( i = 0; i < theGroupSize; ++i ) {
		if(node_count_to_send_me[i]) {
			pes_i_owe_id[j] = i;
			coord_i_owe[i] = calloc(node_count_to_send_me[i],sizeof(int32_t));
			++j;
		}
	}

	l = 0;
	pid_cnt = 0;

	/* this time fill coord_i_owe */
	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if (node_count_to_send_me[pid]) {
			for ( j = 0; j < drmNodesCountP2[pid]; j++ ) {
				for ( i = drmFileToOpen.coord_to_start; i < drmFileToOpen.coord_to_end; ++i ) {
					if ( drmCoordsAdjust[i].f[0] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[0]  &&
							drmCoordsAdjust[i].f[1] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[1]  &&
							drmCoordsAdjust[i].f[2] == pesDrmNodes[pid_cnt].drmNodeCoords[j].x[2] ) {
						coord_i_owe[pid][array_cnt_me[l]] = i ; // the order
						array_cnt_me[l]++;
						break;
					}
				}
				if ( array_cnt_me[l] == node_count_to_send_me[pesDrmNodes[pid_cnt].id]) break;
			}
			l++;
		}
		if (drmNodesCountP2[pid] ) pid_cnt++;
	}

	/* Now, drm coords_I_owe needs to be broadcasted all together */
	node_count_to_send_grp = calloc(theGroupSize,sizeof(int32_t));

	MPI_Allreduce( node_count_to_send_me, node_count_to_send_grp,
				   theGroupSize, MPI_INT, MPI_SUM, newcomm[drmFileToOpen.cid] );

	/*
	for ( i = 0; i < theGroupSize; ++i ) {
		if (node_count_to_send_grp[i] && drmFileToOpen.cid == 4 )
	printf("%d %d %d\n",node_count_to_send_me[i],node_count_to_send_grp[i], i);
	}
	 */

	for ( i = 0; i < theGroupSize; ++i ) {
		if(node_count_to_send_grp[i]) {
			pes_grp_owe_cnt++;
		}
	}

	coord_grp_owe = calloc(pes_grp_owe_cnt,sizeof(int32_t*));
	pes_grp_owe_id = calloc(pes_grp_owe_cnt,sizeof(int32_t));
	array_cnt_grp = calloc(pes_grp_owe_cnt,sizeof(int32_t));

	j = 0;
	for ( i = 0; i < theGroupSize; ++i ) {
		if(node_count_to_send_grp[i]) {
			pes_grp_owe_id[j] = i;
			coord_grp_owe[j] = calloc(node_count_to_send_grp[i],sizeof(int32_t));
			/* Each Processor in the group broadcast coordinates(order) */
			for( l = 0; l < pes_per_file[drmFileToOpen.cid]; ++l ) {

				recv_cnt = node_count_to_send_me[i];
				MPI_Bcast( &recv_cnt, 1, MPI_INT, l, newcomm[drmFileToOpen.cid] );

				if (recv_cnt != 0 ) {
					if (myID == drmFileToOpen.root_pe + l ) {
						for ( m = 0; m < recv_cnt; ++m ) {
							coord_grp_owe[j][array_cnt_grp[j] + m] = coord_i_owe[i][m];
						}
					}
					MPI_Bcast( &coord_grp_owe[j][array_cnt_grp[j]], recv_cnt, MPI_INT, l, newcomm[drmFileToOpen.cid] );

					array_cnt_grp[j]+=recv_cnt;
				}
			}
			++j;
		}
	}

	Timer_Stop("Find Which Drm Files To Print");
	Timer_Reduce("Find Which Drm Files To Print", MAX | MIN | AVERAGE , comm_solver);

	Timer_Start("Read And Rearrange Drm Files");

	/*Important Part -- Read and Arrange Files*/

	MPI_File fp_write[pes_grp_owe_cnt];
	MPI_File fp_grp;
	/* Print header first( total number of nodes and their x-y-z coordinates) */
	for ( pid = 0; pid < pes_grp_owe_cnt; pid++ ) {
		sprintf(filename, "%s%s_%d_%d", theDrmOutputDir, "/part2/drm_disp2", pes_grp_owe_id[pid],drmFileToOpen.id);
		MPI_File_open(newcomm[drmFileToOpen.cid], filename, MPI_MODE_WRONLY | MPI_MODE_CREATE , MPI_INFO_NULL, &fp_write[pid]);

		if (drmFileToOpen.step_to_start == 0) {
			MPI_File_write_at(fp_write[pid],0,&(array_cnt_grp[pid]),1,MPI_INT,&status);
			for ( l = 0; l < array_cnt_grp[pid]; l++ ) {
				disp = ((MPI_Offset)sizeof(int32_t))
		 			 + (MPI_Offset)l *(MPI_Offset) sizeof(double) *(MPI_Offset) 3;
				MPI_File_write_at(fp_write[pid],disp,&drmCoordsAdjust[coord_grp_owe[pid][l]],3,MPI_DOUBLE,&status);
			}
		}

		disp = (( MPI_Offset)sizeof(int32_t))
		     + (MPI_Offset)array_cnt_grp[pid] * (MPI_Offset)sizeof(double) * (MPI_Offset)3
		     + (MPI_Offset)array_cnt_grp[pid] * (MPI_Offset)drmFileToOpen.step_to_start * (MPI_Offset)sizeof(double) * (MPI_Offset)3;

		MPI_File_seek(fp_write[pid],disp,MPI_SEEK_SET);
	}

	sprintf( filename, "%s_%d","/lustre/scratch/yigit/DRM_new/part1/drm_disp", drmFileToOpen.id);

	//sprintf(filename, "%s%s_%d", theDrmOutputDir, "/part1/drm_disp",drmFileToOpen.id );
	//sprintf( filename, "%s_%d","/lustre/scratch/yigit/DRM_TEST/outputfiles/DRM/part1/drm_disp", drmFileToOpen.id);

	MPI_File_open(newcomm[drmFileToOpen.cid], filename, MPI_MODE_RDONLY , MPI_INFO_NULL, &fp_grp);

	/*
	if ( (fp = fopen(filename, "rb")) == NULL) {
		fprintf(stderr, "Error opening %s\n", filename);
	}
	 */
//	hu_fread(&disp_cnt, sizeof(int32_t), 1, fp);

	MPI_File_read_at(fp_grp,0,&disp_cnt,1,MPI_INT,&status);

	for ( i = drmFileToOpen.step_to_start ; i < drmFileToOpen.step_to_end ; ++i ) {

		XMALLOC_VAR_N( to_read, fvector_t, disp_cnt );

		if (i == drmFileToOpen.step_to_start) {

			disp2 = ((MPI_Offset)sizeof(int32_t))
				 + (MPI_Offset) disp_cnt * (MPI_Offset)i * (MPI_Offset)sizeof(double) * (MPI_Offset)3;

			MPI_File_seek(fp_grp,disp2,MPI_SEEK_SET);
			/*
			whereToRead = ((off_t)sizeof(int32_t))
				        + (off_t) disp_cnt * (off_t)i * (off_t)sizeof(double) * (off_t)3;

			hu_fseeko( fp, whereToRead, SEEK_SET );
			 */
		}

		MPI_File_read(fp_grp ,to_read,disp_cnt*3,MPI_DOUBLE,&status);

		for ( pid = 0; pid < pes_grp_owe_cnt; pid++ ) {

			XMALLOC_VAR_N( to_write, fvector_t, array_cnt_grp[pid] );

			for ( l = 0; l < array_cnt_grp[pid]; l++ ) {
				to_write[l] = to_read[coord_grp_owe[pid][l]];
				/*
				to_write[l].f[0] = to_read[coord_grp_owe[pid][l]].f[0];
				to_write[l].f[1] = to_read[coord_grp_owe[pid][l]].f[1];
				to_write[l].f[2] = to_read[coord_grp_owe[pid][l]].f[2];
				 */
			}

			MPI_File_write(fp_write[pid] ,to_write,array_cnt_grp[pid]*3,MPI_DOUBLE,&status);
			free(to_write);
		}
		free(to_read);
	}

	//fclose(fp);
	MPI_File_close(&fp_grp);

	for ( pid = 0; pid < pes_grp_owe_cnt; pid++ ) {
		MPI_File_close(&fp_write[pid]);
	}

	Timer_Stop("Read And Rearrange Drm Files");
	Timer_Reduce("Read And Rearrange Drm Files", MAX | MIN | AVERAGE , comm_solver);

	Timer_Start("Find Which Drm Files To Read");

	pes_i_owe_grp_cnt = calloc(theGroupSize,sizeof(int32_t));

	if (drmFileToOpen.step_to_start == 0) {

		pes_i_owe_cnt_send = pes_grp_owe_cnt ;
	}

	MPI_Allgather( &pes_i_owe_cnt_send, 1, MPI_INT,
			pes_i_owe_grp_cnt, 1, MPI_INT, comm_solver );

	/* Print the id of processors I owe.(print if only I am reading the beginning of file */
	sprintf(filename, "%s%s", theDrmOutputDir, "/part2/drm_file_index" );

	MPI_File_open(comm_solver, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE , MPI_INFO_NULL, &fp_grp);

	for ( pid = 0; pid < theGroupSize; pid++ ) {
		if ( pes_i_owe_grp_cnt[pid] && myID == pid ) {

			disp = ((MPI_Offset)sizeof(int)*total_pes_cnt);

			MPI_File_write_at(fp_grp,disp,pes_grp_owe_id,pes_grp_owe_cnt,MPI_INT,&status);
		}
		total_pes_cnt += pes_i_owe_grp_cnt[pid]  ;
	}

	MPI_File_close(&fp_grp);
	MPI_Barrier(comm_solver);

	/* Find out which files I need to open */

	pes_id_cumulative = calloc(total_pes_cnt,sizeof(int32_t));

	if ( myID == 0 ) {
		sprintf(filename, "%s%s", theDrmOutputDir, "/part2/drm_file_index" );
		if ( (fp = fopen(filename, "rb")) == NULL) {
			fprintf(stderr, "Error opening %s\n", filename);
		}

		hu_fread( pes_id_cumulative, sizeof(int32_t), total_pes_cnt, fp );

		fclose(fp);
	}

	MPI_Bcast( pes_id_cumulative, total_pes_cnt, MPI_INT, 0, comm_solver );

	/* fill drm_file_to_read */

	Timer_Stop("Find Which Drm Files To Read");
	Timer_Reduce("Find Which Drm Files To Read", MAX | MIN | AVERAGE , comm_solver);

	Timer_Start("Locate where I am in file");

	if (myNumberOfDrmNodes) {
		/* First loop pes_id_cumulative to find out how many files I need to open */
		for( i = 0; i < total_pes_cnt; ++i ) {
			if( pes_id_cumulative[i] == myID )
				filesToReadCnt++;
		}
		XMALLOC_VAR_N( drmFilesToRead, drm_file_to_read, filesToReadCnt );
		l = 0;
		j = 0;
		total_pes_cnt = 0;

		for ( pid = 0; pid < theGroupSize; pid++ ) {
			for ( i = 0; i < pes_i_owe_grp_cnt[pid] ; i++ ) {
				if( pes_id_cumulative[total_pes_cnt+i] == myID ) {
					drmFilesToRead[l].file_id = partOneDrmNodesCnt[j].id;
					l++;
					break;
				}
			}
			if ( pes_i_owe_grp_cnt[pid] ) j++;
			total_pes_cnt += pes_i_owe_grp_cnt[pid];
		}

		/* Continue filling drmFilesToRead */

		for ( i = 0; i < filesToReadCnt; ++i) {
			sprintf(filename, "%s%s_%d_%d", theDrmOutputDir, "/part2/drm_disp2", myID,drmFilesToRead[i].file_id);

			if ( (drmFilesToRead[i].fp = fopen(filename, "rb")) == NULL) {
				fprintf(stderr, "Error opening %s\n", filename);
			}

			hu_fread(&(drmFilesToRead[i].nodes_cnt), sizeof(int32_t), 1, drmFilesToRead[i].fp);
			XMALLOC_VAR_N( drmFilesToRead[i].coords, vector3D_t, drmFilesToRead[i].nodes_cnt );
			hu_fseeko( drmFilesToRead[i].fp, (off_t)sizeof(int32_t), SEEK_SET );
			hu_fread(drmFilesToRead[i].coords, sizeof(double), 3*drmFilesToRead[i].nodes_cnt, drmFilesToRead[i].fp);

			if ( i == 0 ) drmFilesToRead[i].total_nodes_cnt = 0;
			else drmFilesToRead[i].total_nodes_cnt = drmFilesToRead[i - 1].total_nodes_cnt
					+ drmFilesToRead[i - 1].nodes_cnt;
		}
		/* Check if everything goes right */
		if ( (drmFilesToRead[filesToReadCnt-1].total_nodes_cnt + drmFilesToRead[filesToReadCnt - 1].nodes_cnt)
				!= myNumberOfDrmNodes) {

			fprintf( stderr, "Internal error.I read more/less drm displacements than needed " );
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}

		/* Fill whereIam in theDrmElem	 */
		for (i = 0; i < myNumberOfDrmElements; i++) {
			/* For both exterior and interior count */
			elem_id = theDrmElem[i].leid;
			for ( l = 0; l < 2; l++ ) {

				if ( l==0 ) count = theDrmElem[i].exteriornodescount;
				if ( l==1 ) count = theDrmElem[i].boundarynodescount;

				for (j = 0; j < count; j++) {

					if ( l==0 ) local_id = theDrmElem[i].lnid_e[j];
					if ( l==1 ) local_id = theDrmElem[i].lnid_b[j];

					node_id  = myMesh->elemTable[elem_id].lnid[local_id];
					x = (myMesh->nodeTable[node_id].x)*(myMesh->ticksize);
					y = (myMesh->nodeTable[node_id].y)*(myMesh->ticksize);
					z = (myMesh->nodeTable[node_id].z)*(myMesh->ticksize);

					for (m = 0; m < filesToReadCnt; m++) {
						for (n = 0; n < drmFilesToRead[m].nodes_cnt; n++) {

							if (drmFilesToRead[m].coords[n].x[0] == x &&
									drmFilesToRead[m].coords[n].x[1] == y &&
									drmFilesToRead[m].coords[n].x[2] == z) {

								theDrmElem[i].whereIam[local_id] = drmFilesToRead[m].total_nodes_cnt + n ;
								m = filesToReadCnt; // so that it breaks the outer loop as well
								break;
							}
						}
					}
				}
			}
		}
	}


	Timer_Stop("Locate where I am in file");
	Timer_Reduce("Locate where I am in file", MAX | MIN | AVERAGE , comm_solver);


	/* Free allocated memory */
	for( i = 0; i < peshavenodes; ++i ) {
		free(pesDrmNodes[i].drmNodeCoords);
	}

	for( i = 0; i < pes_i_owe_cnt; ++i ) {
		free(coord_i_owe[i]);
	}

	for( i = 0; i < pes_grp_owe_cnt; ++i ) {
		free(coord_grp_owe[i]);
	}

	free(pesDrmNodes);
	free(drmNodesCountP2);
	free(drmCoordsAdjust);
	free(pes_i_owe_grp_cnt);
	free(coord_i_owe);
	free(node_count_to_send_me);
	free(pes_i_owe_id);
	free(pes_grp_owe_id);
	free(pes_id_cumulative);
	free(array_cnt_me);
	free(array_cnt_grp);
	free(pes_per_file);
	free(rank);
}

int32_t	 fill_drm_struct( mesh_t  *myMesh, int32_t i, int32_t key, int *bound, int *exterior )
{
	int32_t   j, bound_cnt, exterior_cnt;
	int64_t   elem_id, node_id;

	if ( key==1 ) {
		bound_cnt     = 4;
		exterior_cnt  = 4;
	}
	if ( key==2 ) {
		bound_cnt     = 2;
		exterior_cnt  = 6;
	}
	if ( key==3 ) {
		bound_cnt     = 1;
		exterior_cnt  = 7;
	}

	XMALLOC_VAR_N( theDrmElem[i].lnid_b, int32_t, bound_cnt );
	XMALLOC_VAR_N( theDrmElem[i].lnid_e, int32_t, exterior_cnt );

	if ( theDrmElem[i].lnid_b == NULL || theDrmElem[i].lnid_e == NULL  ) {
		fprintf( stderr, "Err allocing mem in fill_drm_struct" );
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}
	for ( j = 0; j < bound_cnt ; ++j ) {
		(theDrmElem[i].lnid_b)[j] = bound[j];
	}
	for (j = 0; j < exterior_cnt; ++j ) {
		(theDrmElem[i].lnid_e)[j] = exterior[j];
	}
	theDrmElem[i].boundarynodescount = bound_cnt;
	theDrmElem[i].exteriornodescount = exterior_cnt;

	elem_id = theDrmElem[i].leid;
	for ( j = 0; j < 8 ;++j ) {
		/*local id of the node */
		node_id = myMesh->elemTable[elem_id].lnid[j];
	}
	return i+1;
}

/*
 * For the first step read two displacements.
 */
void solver_read_drm_displacements( int32_t step, double deltat, int32_t totalsteps ) {

	Timer_Start("Solver drm read displacements");

	if ( drmImplement == YES && theDrmPart == PART2 && myNumberOfDrmNodes != 0 )
	{
		int aux, i;
        fvector_t* tmpvector;

		aux = (int)(theDrmPrintRate*thePart1DeltaT/deltat);
		if (step % aux == 0 ) {
			step = step / aux;
			if (step != totalsteps /aux - 1 ) {  step++; }
			//printf("%d ", step);
			//step++;
			if ((step - 1) == 0 ) {
				XMALLOC_VAR_N( theDrmDisplacements1, fvector_t, myNumberOfDrmNodes );
				XMALLOC_VAR_N( theDrmDisplacements2, fvector_t, myNumberOfDrmNodes );
			}
			if ((step - 1) != 0 ) {
				tmpvector     = theDrmDisplacements2;
				theDrmDisplacements2 = theDrmDisplacements1;
				theDrmDisplacements1 = tmpvector;
			}

			if ((step - 1) == 0 ) {
				for( i=0; i<filesToReadCnt;++i) {
					off_t   whereToRead;
					whereToRead = ((off_t)sizeof(int32_t))
		    					+ (off_t)drmFilesToRead[i].nodes_cnt * (off_t)sizeof(double) *(off_t) 3;
					hu_fseeko( drmFilesToRead[i].fp, whereToRead, SEEK_SET );
					hu_fread( theDrmDisplacements1 + drmFilesToRead[i].total_nodes_cnt
							,sizeof(double), 3 * drmFilesToRead[i].nodes_cnt,
							drmFilesToRead[i].fp );
				}
			}

			for( i=0; i<filesToReadCnt;++i) {
				off_t   whereToRead;
				whereToRead = ((off_t)sizeof(int32_t))
		    				+(off_t)drmFilesToRead[i].nodes_cnt * (off_t)sizeof(double) *(off_t) 3
		    				+(off_t)drmFilesToRead[i].nodes_cnt *(off_t) step *(off_t) sizeof(double) * (off_t)3;
				hu_fseeko( drmFilesToRead[i].fp, whereToRead, SEEK_SET );
				hu_fread( theDrmDisplacements2 + drmFilesToRead[i].total_nodes_cnt
						,sizeof(double), 3 * drmFilesToRead[i].nodes_cnt,
						drmFilesToRead[i].fp );
			}
		}
	}
	Timer_Stop("Solver drm read displacements");
}


void solver_compute_effective_drm_force( mysolver_t* mySolver , mesh_t* myMesh,
				    fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8], int32_t step,
				    double deltat)
{
	Timer_Start("Solver drm force compute");

	if ( drmImplement == YES && theDrmPart == PART2 && myNumberOfDrmNodes != 0) {

		int32_t    b, e, i ;
		int32_t    lin_eindex;
		int32_t    ln_drm_id_e, ln_drm_id_b;

		int        remainder,aux;
		double 	   fracture;
		fvector_t  localForce[8];
		fvector_t* toForce ;

		/* For Interpolation */

		aux = (int)(theDrmPrintRate*thePart1DeltaT/deltat);
		remainder = step % aux;
		fracture = (double)remainder/(double)aux;

		//printf("%f ", fracture);
		/* loop on the number of drm elements */
		for (lin_eindex = 0; lin_eindex < myNumberOfDrmElements; lin_eindex++) {

			elem_t* elemp;
			e_t*    ep;

			/*Local id of drm elment*/
			int32_t ldrm_id = theDrmElem[lin_eindex].leid;

			elemp  = &myMesh->elemTable[ldrm_id];
			ep     = &mySolver->eTable[ldrm_id];

			memset( localForce, 0, 8 * sizeof(fvector_t) );

			/* step 1: calculate the force contribution at boundary nodes */
			/* contribution by node e to node b */
			for (b = 0; b < theDrmElem[lin_eindex].boundarynodescount; b++)
			{
				/*Local id of boundary node.b is b/w 0-7 */
				ln_drm_id_b = theDrmElem[lin_eindex].lnid_b[b];
				toForce = &localForce[ln_drm_id_b];

				for (e = 0; e < theDrmElem[lin_eindex].exteriornodescount; e++)
				{
					int position,index;
					double low[3], high[3];
					fvector_t  myDisp[1];
					/*id of exterior node.e is b/w 0-7 */
					ln_drm_id_e = theDrmElem[lin_eindex].lnid_e[e];

					position = theDrmElem[lin_eindex].whereIam[ln_drm_id_e];

					/* Interpolation is done here */
					for( index = 0; index < 3; index++) {
						low[index]  = theDrmDisplacements1[position].f[index];
						high[index] = theDrmDisplacements2[position].f[index];
						myDisp[0].f[index] = low[index] + fracture*(high[index] - low[index]);
					}

					/* effective  force contribution ( fb = - deltaT_square * Kbe * Ue )
					 * But if myDisp is zero avoids multiplications
					 */
					if ( (vector_is_all_zero( myDisp ) != 0)) {
						MultAddMatVec( &theK1[ln_drm_id_b][ln_drm_id_e], myDisp, -ep->c1, toForce );
						MultAddMatVec( &theK2[ln_drm_id_b][ln_drm_id_e], myDisp, -ep->c2, toForce );
					}
				}
			}

			/* step 2: calculate the force at exterior nodes */
			/* contribution by node b to node e */
			for (e = 0; e < theDrmElem[lin_eindex].exteriornodescount; e++)
			{
				/*Local id of exterior node.e is b/w 0-7 */
				ln_drm_id_e = theDrmElem[lin_eindex].lnid_e[e];
				toForce = &localForce[ln_drm_id_e];

				for (b = 0; b < theDrmElem[lin_eindex].boundarynodescount; b++)
				{
					int position, index;
					double low[3], high[3];

					fvector_t  myDisp[1];
					/*id of boundary node.b is b/w 0-7 */
					ln_drm_id_b = theDrmElem[lin_eindex].lnid_b[b];

					position = theDrmElem[lin_eindex].whereIam[ln_drm_id_b];

					/* Interpolation is done here */
					for( index = 0; index < 3; index++) {
						low[index]  = theDrmDisplacements1[position].f[index];
						high[index] = theDrmDisplacements2[position].f[index];
						myDisp[0].f[index] = low[index] + fracture*(high[index] - low[index]);
					}

					/* effective  force contribution ( fe = + deltaT_square * Keb * Ub )
					 * But if myDisp is zero avoids multiplications
					 */
					if ( (vector_is_all_zero( myDisp ) != 0)) {
						MultAddMatVec( &theK1[ln_drm_id_e][ln_drm_id_b], myDisp, ep->c1, toForce );
						MultAddMatVec( &theK2[ln_drm_id_e][ln_drm_id_b], myDisp, ep->c2, toForce );
					}
				}
			}

			/* step 3: sum up my contribution to my vertex nodes */
			for (i = 0; i < 8; i++) {
				int32_t    lnid       = elemp->lnid[i];
				fvector_t* nodalForce = mySolver->force + lnid;
				nodalForce->f[0] += localForce[i].f[0];
				nodalForce->f[1] += localForce[i].f[1];
				nodalForce->f[2] += localForce[i].f[2];
			}
		}
	}
	Timer_Stop("Solver drm force compute");
}

void drm_sanity_check(int32_t  myID) {

	FILE  *drminfo;
	static char    filename [256];
	int numberofdrmelements;

	/* Find total number of drm elements.*/
	MPI_Reduce(&myNumberOfDrmElements,&theNumberOfDrmElements,1,MPI_INT,MPI_SUM,0,comm_solver);

	if ( myID == 0 ) {
		sprintf(filename, "%s%s", theDrmOutputDir, "/part0/drm_information");
		drminfo = hu_fopen ( filename, "r" );

		if ((parsetext ( drminfo, "drm_numberofelements", 'i', &numberofdrmelements  ) != 0) ) {
			fprintf( stderr,
					"Error opening %s\n drm_sanity_check",
					filename );
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}

		if ( theNumberOfDrmElements != numberofdrmelements ) {
			fprintf( stderr, "drm boundary has changed"
					"(number of drm elems does not match)");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}

		fclose(drminfo);
	}
}


/* ------------------------------------------------------------------------- *
 *                  Functions related to HashTables				             *
 * ------------------------------------------------------------------------- */

void drm_table_init(drmhash_t *drmTable, int32_t tableDim) {

	int32_t i;

	drmTable->drmBucketArray = calloc(tableDim, sizeof(drmbucket_t));
	drmTable->tableCnt = 0;

	for ( i = 0; i < tableDim; ++i ) {
		(drmTable->drmBucketArray[i]).bucketCnt     = 0;
		(drmTable->drmBucketArray[i]).drmBucketList = NULL;
	}
}

void construct_drm_array( drmhash_t *drmTable, int32_t tableDim,
						  int32_t   arraysize, int64_t **drm_array, int key)
{
	if ( arraysize !=0 ) {

		int32_t i, j=0;

		XMALLOC_VAR_N( (*drm_array) , int64_t, arraysize );

		if ( key == 2) {
			XMALLOC_VAR_N( drmNodeOwner , int32_t, arraysize );
		}

		for ( i = 0; i < tableDim; ++i ) {
			drmbucketelem_t * aux=drmTable->drmBucketArray[i].drmBucketList;

			while (1) {
				if ( (drmTable->drmBucketArray[i]).bucketCnt == 0 ) {
					break;
				}
				(*drm_array)[j] = aux->nodeid;
				if ( key == 2 ) {
					drmNodeOwner[j] = aux->owner;
				}
				j++;
				if ( aux->next == NULL ) {
					break;
				}
				aux=aux->next;
			}
		}

		if ( key == 1 ) {
			qsort((*drm_array), arraysize , sizeof(int64_t), compare);
		}

	}
}

/*Table dimension = (2n+1)*prevguess which obeys the rule, where initialguess=89
 *i.e> tableDim = 89,179,359...*/
void get_table_dim(int32_t *tableDim, int32_t limit) {

	int i = 0;

	while (1) {
		(*tableDim) = (*tableDim) * (2*i+1);
		/* rule assumes an average of 10 nodes in each bucket
		 * also assumes 1 to 2 ratio b/w drm elems and drm nodes*/
		if ( 2*limit <= 10*(*tableDim) ) {
			return;
		}
		++i;
	}
}

/* Insert drmnodes for part0 avoiding duplicates*/
void insertNode(mesh_t  *myMesh, int32_t index) {

	int32_t   j, k, nodeid, owner;

	for ( j = 0; j < 8; ++j ) {
		nodeid = myMesh->elemTable[index].lnid[j];

		/* For part2, i need to insert every single node my drm elements touch. */

		if ( theDrmPart==PART2 ) {
			k = nodeid % theTableDim;
			insert(nodeid, k, &theDrmTable, NONE);
		}

		if ( theDrmPart==PART0 ) {
			/* This is a bit tricky because, if I dont own it, there is a possibility
			 * that drm node may never be inserted.So if I do not own it, send that
			 * node to PE0 and it sends to the owner processor  after checking for
			 * duplicates.*/
			if ( myMesh->nodeTable[nodeid].ismine ) {
				k = nodeid % theTableDim;
				insert(nodeid, k, &theDrmTable, NONE);
			}
			else {
				/* if I dont own this node,  store in a seperate hashtable , and
				 * then send it to PE0(later)  */
				k = nodeid % theTableDupDim;
				owner = myMesh->nodeTable[nodeid].proc.ownerid;
				/* This time insert global id instead of local id and owner processor
				 * id too*/
				insert(myMesh->nodeTable[nodeid].gnid, k, &theDrmDupTable,owner);
			}
		}
	}
}

void insert(int64_t nodeid, int32_t i, drmhash_t *drmTable, int32_t owner) {

	drmbucket_t *bucket_arr= &(drmTable->drmBucketArray[i]);

	if ((bucket_arr->bucketCnt == 0) ||
			searchBucketList(bucket_arr->drmBucketList, nodeid) == 0 )
	{
		insertAtFrontOfBucketList(&(bucket_arr->drmBucketList), nodeid ,owner);
		(bucket_arr->bucketCnt)++;
		(drmTable->tableCnt)++;
		return;
	}

	return;
}

void insertAtFrontOfBucketList(drmbucketelem_t ** bucketList, int64_t nodeid,
							   int32_t owner) {

	drmbucketelem_t *new = malloc ( sizeof(drmbucketelem_t) );

	if ( new == NULL ) {
		fprintf( stderr, "Err allocing mem in insertAtFrontOfBucketList" );
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	}

	new->nodeid = nodeid;
	if ( owner != NONE) {
		new->owner = owner;
	}
	new->next = (*bucketList);
	(*bucketList) = new;

}

int32_t searchBucketList(drmbucketelem_t *bucketList, int64_t nodeid) {

	drmbucketelem_t * aux = bucketList;
	while (1) {
		if ( aux->nodeid == nodeid ) {
			return 1;
		}
		if ( aux->next == NULL ) {
			return 0;
		}
		aux = aux->next;
	}
	// should never reach here
	return 2;

}

void removeAtFrontOfBucketList(drmbucketelem_t ** bucketList) {

	drmbucketelem_t * deadElem = (*bucketList);
	if (*bucketList == NULL) return; /* nothing to remove */
	*bucketList = deadElem->next;
	free( deadElem );

}

void freeHashTable(drmhash_t *drmTable, int32_t tableDim) {

	int32_t i;
	for (i = 0; i < tableDim; ++i ) {
		while( (drmTable->drmBucketArray[i]).drmBucketList ) {
			removeAtFrontOfBucketList(&((drmTable->drmBucketArray[i]).drmBucketList));
		}
	}

	free( drmTable->drmBucketArray);

}

int32_t compare (const void * a, const void * b) {

	return ( *(int32_t*)a - *(int32_t*)b );

}
