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
#ifndef TOPOGRAPHY_H_
#define TOPOGRAPHY_H_


/* -------------------------------------------------------------------------- */
/*                        Structures and definitions                          */
/* -------------------------------------------------------------------------- */

typedef enum {
    FEM = 0, VT
} topometh_t;

typedef struct topoconstants_t {

	double       lambda;
    double       mu;
    double       rho;
    double       h;
    double       tetraVol[5];

} topoconstants_t;


typedef struct toposolver_t {

    topoconstants_t  *topoconstants;

} toposolver_t;

typedef enum {
  FLAT = 0, FULL
} etreetype_t;

typedef struct topostation_t {

	int32_t  TopoStation;
    int32_t  nodes_to_interpolate[4];
    double   local_coord[3];

} topostation_t;


int    octant_topolocotaion   (mesh_t *myMesh, int32_t eindex);
double get_thebase_topo();
int    BelongstoTopography    (mesh_t *myMesh, int32_t eindex);
etreetype_t get_theetree_type ();
int    topo_correctproperties ( edata_t *edata );

int    topo_toexpand            ( octant_t *leaf, double    ticksize, edata_t  *edata, double VsFactor );
void   topo_init                ( int32_t myID, const char *parametersin );
void   topo_solver_init         ( int32_t  myID, mesh_t *myMesh );
void   toponodes_mass           ( int32_t eindex, double nodes_mass[8], double M, double xo, double yo, double zo);
void   compute_addforce_topo    ( mesh_t *myMesh, mysolver_t *mySolver, double theDeltaTSquared );
void   TetraForces              ( fvector_t* un, fvector_t* resVec, double tetraVol[5], edata_t *edata,
		                          double mu, double lambda, double xo, double yo, double zo  );
void   compute_addforce_topoEffective    ( mesh_t *myMesh, mysolver_t *mySolver, double theDeltaTSquared );
void   compute_tetra_localcoord ( vector3D_t point, elem_t *elemp, int32_t *localNode, double *localCoord, double xo, double yo, double zo, double h );
void   topography_stations_init ( mesh_t    *myMesh, station_t *myStations, int32_t    myNumberOfStations);
int    compute_tetra_displ      (double *dis_x, double *dis_y, double *dis_z,
						 	 	 double *vel_x, double *vel_y, double *vel_z,
						 	 	 double *accel_x, double *accel_y, double *accel_z,
						 	 	 double theDeltaT, double theDeltaTSquared,
						 	 	 int32_t statID, mysolver_t *mySolver);
int    topo_setrec              ( octant_t *leaf, double    ticksize, edata_t  *edata, etree_t  *cvm );
double point_elevation          ( double xo, double yo );
int    topo_nodesearch          ( tick_t x, tick_t y, tick_t z, double ticksize );
int    topo_crossings           ( double xo, double yo, double zo, double esize );

/* ERASE LATER THESE FUNCTIONS, ARE JUST FOR A QUICK CHECKING*/
void topo_DRM_init ( mesh_t *myMesh, mysolver_t *mySolver);

void compute_addforce_topoDRM ( mesh_t     *myMesh,
                                mysolver_t *mySolver,
                                double      theDeltaT,
                                int         step,
                                fmatrix_t (*theK1)[8], fmatrix_t (*theK2)[8]);

#endif /* TOPOGRAPHY_H_ */
