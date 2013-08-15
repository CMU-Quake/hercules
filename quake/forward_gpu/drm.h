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

#ifndef DRM_H_
#define DRM_H_

#include "quake_util.h"


/* -------------------------------------------------------------------------- */
/*                                Structures                                  */
/* -------------------------------------------------------------------------- */

typedef enum {

  PART0 = 0, PART1, PART2

} drm_part_t;

typedef struct drmbucketelem_t {

	int64_t  nodeid;
	int32_t  owner;
	struct   drmbucketelem_t * next;

} drmbucketelem_t;


typedef struct drmbucket_t {

	int32_t           bucketCnt;
	drmbucketelem_t * drmBucketList;

} drmbucket_t;

typedef struct drmhash_t {

	drmbucket_t * drmBucketArray;
	int32_t       tableCnt;

} drmhash_t;

typedef struct drm_elem_t {

	int64_t    leid;               /* Local element  */
	int32_t    *lnid_e;            /* exterior node local ids  b/w 0-7 */
	int32_t    *lnid_b;            /* boundary node local ids  b/w 0-7*/
	int32_t    boundarynodescount; /* Number of interior nodes */
	int32_t    exteriornodescount; /* Number of exterior nodes */

	int     whereIam[8];           /* Position of the node in theDrmDisplacements array */

} drm_elem_t;

typedef struct  pes_with_drm_nodes {

	int32_t    id;            /* processor id */

	double     minx, miny, minz,
	     	   maxx, maxy, maxz;   /* limits*/

	vector3D_t *drmNodeCoords;

} pes_with_drm_nodes;

typedef struct  pes_drm_nodes_cnt {

	int32_t    id;            /* processor id */
    int32_t    count;         /* drm node count */

} pes_drm_nodes_cnt;

typedef struct  drm_node_t {

	int32_t    id, nodestointerpolate[8];
    vector3D_t coords;        /* cartesian */
    vector3D_t localcoords;   /* csi, eta, dzeta in (-1,1) x (-1,1) x (-1,1) */

} drm_node_t;

typedef struct  drm_file_to_open {

	int32_t    cid;             /* comm id */
	int32_t    id;              /* which file to open */
	int32_t	   root_pe;			/* leading processor id in the group */
	int32_t	   pos;				/* position in group */
	int32_t    coord_to_start;  /* which coord to start reading*/
	int32_t    coord_to_end;    /* which coord to end reading */
	int32_t    step_to_start;   /* which timestep to start reading*/
	int32_t    step_to_end;     /* which timestep to end reading */

} drm_file_to_open;


/* For part2 */
typedef struct  drm_file_to_read {

	int32_t    file_id;           /* which file to read */
	int32_t    nodes_cnt;         /* how many nodes in this file */
	int32_t    total_nodes_cnt;   /* how many nodes I read before opening this file */
    vector3D_t *coords;           /* coords of the nodes */
	FILE*      fp;                /* File pointers to read disp */


} drm_file_to_read;

/* -------------------------------------------------------------------------- */
/*                                Functions                                  */
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
/*                             General Utilities                              */
/* -------------------------------------------------------------------------- */
drm_part_t drm_init ( int32_t  myID ,  const char *parametersin, noyesflag_t includeBldgs);

int32_t drm_initparameters ( const char *parametersin);

void  search_drm_elems(mesh_t  *myMesh, int32_t key);

int is_drm_elem(double *coords_array, int *keys_array, int key, double  ticksize);

int drm_toexpand ( octant_t *leaf, double  ticksize, edata_t  *edata );

void solver_drm_close ();

void  drm_fix_coordinates(double ticksize);

void drm_stats(int32_t  myID,
			   int32_t  theGroupSize,
			   double   theXForMeshOrigin,
               double   theYForMeshOrigin,
               double   theZForMeshOrigin);

void drm_print_stats(int32_t *drmNodesCount,
                     int32_t  theGroupSize,
                     double   theXForMeshOrigin,
                     double   theYForMeshOrigin,
                     double   theZForMeshOrigin);
/* -------------------------------------------------------------------------- */
/*                           Functions for Part0                              */
/* -------------------------------------------------------------------------- */

void find_drm_nodes( mesh_t     *myMesh,       int32_t   myID,
					 const char *parametersin, double    ticksize,
					 int32_t    theGroupSize) ;

void drm_output(mesh_t *myMesh, int32_t myID, int32_t  theGroupSize,
		        int *array_have_nodes, int64_t  *drm_array, drm_node_t *drm_node);

/* -------------------------------------------------------------------------- */
/*                           Functions for Part1                              */
/* -------------------------------------------------------------------------- */

int32_t drm_read_coords ( ) ;

void setup_drm_data(mesh_t *myMesh, int32_t myID, int32_t   theGroupSize);

void solver_output_drm_nodes ( mysolver_t* mySolver, int step, int32_t totalsteps );

/* -------------------------------------------------------------------------- */
/*                           Functions for Part2                              */
/* -------------------------------------------------------------------------- */

void proc_drm_elems( mesh_t   *myMesh,     int32_t   myID,
		int32_t theGroupSize, int32_t theTotalSteps);

void find_drm_file_readjust ( int32_t theGroupSize, int32_t myID, int32_t theTotalSteps );

void comm_drm_coordinates ( mesh_t  *myMesh , int32_t theGroupSize, int32_t myID);

void rearrange_drm_files ( mesh_t  *myMesh , int32_t theGroupSize, int32_t myID ) ;

int32_t	 fill_drm_struct( mesh_t  *myMesh, int32_t i, int32_t key, int *bound, int *exterior);

void solver_compute_effective_drm_force( mysolver_t* mySolver , mesh_t* myMesh,
				    fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8], int32_t step,
				    double deltat);

void solver_read_drm_displacements( int32_t step, double deltat,int32_t totalsteps ) ;

void drm_sanity_check(int32_t  myID);

/* ------------------------------------------------------------------------- *
 *                  Functions related to HashTables				             *
 * ------------------------------------------------------------------------- */

void drm_table_init(drmhash_t *drmTable, int32_t tableDim);

void construct_drm_array( drmhash_t *drmTable, int32_t tableDim,
						  int32_t   arraysize, int64_t **drm_array, int key);

void get_table_dim(int32_t *tableDim, int32_t limit);

void insertNode(mesh_t  *myMesh, int32_t index);

void insert(int64_t nodeid, int32_t i, drmhash_t *drmTable, int32_t owner);

void insertAtFrontOfBucketList(drmbucketelem_t ** bucketList, int64_t nodeid,
							   int32_t owner);

int32_t searchBucketList(drmbucketelem_t *bucketList, int64_t nodeid);

void removeAtFrontOfBucketList(drmbucketelem_t ** bucketList);

void freeHashTable(drmhash_t *drmTable, int32_t tableDim);

int32_t compare (const void * a, const void * b);


#endif /* DRM_H_ */

