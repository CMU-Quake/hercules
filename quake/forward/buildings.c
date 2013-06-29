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
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cvm.h"
#include "psolve.h"
#include "octor.h"
#include "buildings.h"
#include "util.h"
#include "quake_util.h"
#include "commutil.h"

#define  FENCELIMIT  0.9999

/* -------------------------------------------------------------------------- */
/*                                Structures                                  */
/* -------------------------------------------------------------------------- */

typedef struct bounds_t {

    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;

} bounds_t;

typedef struct bldg_t {

	double       height;
	double       depth;
	bounds_t     bounds;
	bounds_t     bounds_expanded; /* to avoid dangling nodes at the buildings base*/
	cvmpayload_t bldgprops;
	cvmpayload_t fdtnprops;

	/* These two are active if asymmetric_buildings=yes.If so, for buildings, there is a linear
	 * variation in Vs^2 and Vp^2  along the EW direction (y coordinate in Hercules).
	 * Nodes at the leftmost are assigned the values in bldgprops_left and
	 * nodes at the rightmost have the values in bldgprops_right. The values
	 * for the intermediate nodes are linearly interpolated. In other words, stiffness
	 * center is shifted wrt to geometric center.
	 * However, the distribution is different for rho. Mass center is
	 * remained at the same place as the geometric center. The center nodes of the
	 * buildings have the values in bldgprops_right, whereas the leftmost and the rightmost
	 * nodes have the values in bldgprops_left. The values for the rest of the
	 * nodes are linearly interpolated.
	 *   */

	cvmpayload_t bldgprops_left;
	cvmpayload_t bldgprops_right;

} bldg_t;

typedef struct pdown_t {

	double       height;
	double       depth;
	bounds_t     bounds;

} pdown_t;

typedef struct basenode_t {
    lnid_t nindex;
    int    bldg;
} basenode_t;


/* If I am responsible for calculating the average values */
/* rows have information for each node at a given level in a building.
 * columns have the same type of nodal information for different levels.
 * l = # of plan-levels in each building
 * n = # of nodes in each level
 * matrix L, T are constant for each time step.
 */
typedef struct master_constrained_slab_bldg {

	/* general building information */

	int l; /* # of plan-levels in each building */
	int n; /* # of nodes in each level node_x*node_y */
	int node_x; /* # of nodes in x direction for each building */
	int node_y; /* # of nodes in y direction for each building */
	double x_bldg_length; /* length of building in x (m) - NS */
	double y_bldg_length; /* length of building in y (m) - EW */

	/* information needed to calculate the average reference values */

	double area; /* area of the plan section of the building(x_m * y_m) */
	double Ix; /* area moment of inertia of the building top view wrt x axis at
		 the center of building */
	double Iy; /* area moment of inertia of the building top view wrt y axis at
		 the center of building */
	double Ixy; /* area moment of inertia of the building top view */
	double**  tributary_areas; /* matrix TA - tributary areas matrix. same
			 for each level in a building. dim : node_x x node_y */

	/* information needed for communication with the sharers */

	int32_t which_bldg; /*  global id of the building. */

	int32_t  sharer_count; /*  total number of procs that owns a node in the
	building. */
	int32_t* owner_ids; /*  id of the sharer processors who will send its data.
	size = sharer_count */
	int32_t* node_counts; /* total number of nodes that each sharer proc will
	send. size = sharer_count */

	double* average_values; /* average values for each level in the building.
	size = 6*l */

	fvector_t** incoming_disp; /*  displacements sent by the sharers.
	size = sharer_count x node_counts  */
	intvector_t** incoming_disp_map; /* which node is sent. l,i,j index of each
	node. size = sharer_count x node_counts  */

	fvector_t**  dis_by_level;  /* matrix Dis - displacements of each point at
	each plan-level of buildings. dim : l x (node_x x node_y) */

	/* information for updating the displacements */

	lnid_t**  local_ids_by_level;  /* matrix L - local ids of each point at
	each plan-level of buildings. value is set to be  -1 if the node does not
	belong to me. dim :l x (node_x x node_y) */

	double**  distance_to_centroid_xm; /* matrix Dx - distance (x) matrix wrt to
	the centroid. same or each level in a building in meter. dim : node_x x node_y */

	double**  distance_to_centroid_ym; /* matrix Dy - distance (y) matrix wrt to
	the centroid. same or each level in a building in meter. dim : node_x x node_y */

} master_constrained_slab_bldg_t;


/* If I own nodes in a building. I need to send info to buildings master proc.*/

/* rows have information for each node at a given level in a building.
 * columns have the same type of nodal information for different levels.
 * l = # of plan-levels in each building
 * n = # of nodes in each level
 * matrix L, T and R are constant for each time step.
 */
typedef struct sharer_constrained_slab_bldg {
	/* general building information */

	int l; /* # of plan-levels in each building */
	int n; /* # of nodes in each level node_x*node_y */
	int node_x; /* # of nodes in x direction for each building */
	int node_y; /* # of nodes in y direction for each building */


	/* information needed for communication with the master */

	int32_t which_bldg; /*  global id of the building. */
	int32_t responsible_id; /*  global id of the responsible proc. */

	int32_t  my_nodes_count; /*  total number of nodes in the building that I
	own */

	double* average_values; /* average values for each level in the building.
	size = 6*l */

	intvector_t* outgoing_disp_map; /* which node is sent. l,i,j index of each
		node. size = my_nodes_count  */

	fvector_t* outgoing_disp; /* displacements sent to the master  */

	/* information for updating the displacements */

	lnid_t**  local_ids_by_level;  /* matrix L - local ids of each point at
		each plan-level of buildings. value is set to be  -1 if the node does not
		belong to me. dim :l x (node_x x node_y) */

	double**  distance_to_centroid_xm; /* matrix Dx - distance (x) matrix wrt to
		the centroid. same or each level in a building in meter. dim : node_x x node_y */

	double**  distance_to_centroid_ym; /* matrix Dy - distance (y) matrix wrt to
		the centroid. same or each level in a building in meter. dim : node_x x node_y */

} sharer_constrained_slab_bldg_t;



/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static noyesflag_t  areBaseFixed = 0;
static noyesflag_t  asymmetricBuildings = 0;
static noyesflag_t  constrainedSlabs = 0;


static char         theBaseFixedDir[256];
static char         theBaseFixedSufix[64];
static double       theBaseFixedDT;
static int          theBaseFixedStartIndex;
static int32_t      theBaseNodesCount;
static basenode_t  *theBaseNodes;
static fvector_t  **theBaseSignals;
static master_constrained_slab_bldg_t  *theMasterConstrainedSlab;
static sharer_constrained_slab_bldg_t  *theSharerConstrainedSlab;

/* Permanent */

static int      theNumberOfBuildings;
static int      theNumberOfBuildingsMaster = 0;
static int      theNumberOfBuildingsSharer = 0;

static int      theNumberOfPushdowns;
static double   theMinOctSizeMeters;
static double   theSurfaceShift = 0;
static bldg_t   *theBuilding;
static pdown_t  *thePushdown;

//static double  eccentricity = 0;

/* Transient */

static double  *theBuildingsVs;
static double  *theBuildingsVp;
static double  *theBuildingsRho;
static double  *theFoundationsVs;
static double  *theFoundationsVp;
static double  *theFoundationsRho;
static double  *theBuildingsXMin,  *theBuildingsXMax;
static double  *theBuildingsYMin,  *theBuildingsYMax;
static double  *theBuildingsZMin,  *theBuildingsZMax;
static double  *theBuildingsDepth, *theBuildingsHeight;
static double  *thePushDownsXMin,  *thePushDownsXMax;
static double  *thePushDownsYMin,  *thePushDownsYMax;
static double  *thePushDownsZMin,  *thePushDownsZMax;
static double  *thePushDownsDepth, *thePushDownsHeight;

/* Active if asymmetric_buildings=yes */
static double  *theBuildingsVs_left;
static double  *theBuildingsVp_left;
static double  *theBuildingsRho_left;

static double  *theBuildingsVs_right;
static double  *theBuildingsVp_right;
static double  *theBuildingsRho_right;
/* -------------------------------------------------------------------------- */
/*                         Private Method Prototypes                          */
/* -------------------------------------------------------------------------- */

int crossing_rule ( tick_t   tickcoord,
                    double   ticksize,
                    edata_t *edata,
                    double   bound );

bounds_t get_bldgbounds( int i );
bounds_t get_pushdownbounds( int i );

void get_airprops( octant_t *leaf, double ticksize,
                   edata_t *edata, etree_t *cvm,
                   double xoriginm,double yoriginm,
                   double zoriginm);

int inclusivesearch  ( double x, double y, double z, bounds_t bounds );
int exclusivesearch  ( double x, double y, double z, bounds_t bounds );
int onboundarysearch ( double x, double y, double z, bounds_t bounds );
int ininteriorsearch ( double x, double y, double z, bounds_t bounds );

int basenode_search  ( tick_t x, tick_t y, tick_t z, double ticksize );

int bldg_exclusivesearch ( tick_t   x,
                           tick_t   y,
                           tick_t   z,
                           double   ticksize,
                           edata_t *edata,
                           int      bldg );

int pushdown_exclusivesearch ( tick_t   x,
							   tick_t   y,
		                       tick_t   z,
		                       double   ticksize,
		                       int      bldg );


int bldg_meshingsearch ( octant_t *leaf,
                         double    ticksize,
                         edata_t  *edata,
                         int       bldg,
                 		 bounds_t  bounds);


int bldgs_search ( octant_t *leaf, double ticksize, edata_t *edata );

int bldgs_refine ( octant_t *leaf,
                   double    ticksize,
                   edata_t  *edata,
                   int       bldg,
                   double    theFactor,
           		   bounds_t  bounds);

int32_t buildings_initparameters ( const char *parametersin );

void fixedbase_read ( FILE* fp );

double adjust (double input);

void adjust_dimensions ();

/* -------------------------------------------------------------------------- */
/*                             General Utilities                              */
/* -------------------------------------------------------------------------- */

/**
 * Checks if a an octant with left-upper-front origin coordinate and size
 * equal edgesize is being crossed by a defining boundary.
 *
 * Returns
 * 1: True
 * 0: False
 */
int crossing_rule ( tick_t   tickcoord,
                    double   ticksize,
                    edata_t *edata,
                    double   bound ) {

    double LUF, RDB, edgesize;

    edgesize = edata->edgesize;
    LUF = tickcoord * ticksize;
    RDB = LUF + edgesize;

    if ( ( LUF < bound) && ( RDB > bound ) ) {
        return 1;
    }

    return 0;
}

/* -------------------------------------------------------------------------- */
/*                               Get Utilities                                */
/* -------------------------------------------------------------------------- */

double get_surface_shift() {
    return theSurfaceShift;
}

noyesflag_t get_fixedbase_flag() {
    return areBaseFixed;
}

noyesflag_t get_asymmetric_flag() {
    return asymmetricBuildings;
}

bounds_t get_bldgbounds( int i ) {

    return theBuilding[i].bounds;
}

bounds_t get_pushdownbounds( int i ) {

	 return thePushdown[i].bounds;
}

noyesflag_t get_constrained_slab_flag() {
	 return constrainedSlabs;
}

cvmpayload_t get_props( int i, double z ) {

	/* the i-th building is in the i-1 index */
	i--;

	if ( z >= theSurfaceShift ) {
		/* Element is in the foundation */
		return theBuilding[i].fdtnprops;
	}
	/* Element is in the building */
	if ( asymmetricBuildings == NO ) {
		return theBuilding[i].bldgprops;
	}

	/* Return the lowest Vs */
	if ( asymmetricBuildings == YES ) {
		if( theBuilding[i].bldgprops_left.Vs > theBuilding[i].bldgprops_right.Vs )
			return theBuilding[i].bldgprops_right;
		else
			return theBuilding[i].bldgprops_left;
	}

	/* should not reach here */
	return theBuilding[i].bldgprops;
}

void get_airprops( octant_t *leaf, double ticksize, edata_t *edata,
				   etree_t *cvm ,double xoriginm, double yoriginm,
                   double zoriginm)
{
	int    res;
	double x, y, z;
	double edgesize;
	double halfedge;

	cvmpayload_t props;

	edgesize = edata->edgesize;
	halfedge = edgesize * 0.5;

	x = ( leaf->lx * ticksize ) + halfedge + xoriginm;
	y = ( leaf->ly * ticksize ) + halfedge + yoriginm;
	z = ( leaf->lz * ticksize ) + halfedge + zoriginm;


	/* Get the Vs at that location on the surface (z = 0) */
	res = cvm_query( cvm, y, x, 0, &props );

	if ( res != 0 ) {
		return;
		solver_abort ( __FUNCTION_NAME, "Error from cvm_query: ",
				"No properties at east = %f, north = %f", y, x);
	}

    /* Increase Vs as it gets far from the surface */
    edata->Vs  = 2.0 * props.Vs * ( theSurfaceShift - z ) / ticksize;

    /* Assign a negative Vp to identify the octants to carve */
    edata->Vp  = -1;

    /* Assign a zero to the density */
    edata->rho = 0;

    return;
}

/* -------------------------------------------------------------------------- */
/*                          Single Search Utilities                           */
/* -------------------------------------------------------------------------- */

/**
 * Returns 1 if a point is on or within min,max boundaries.
 * Returns 0 otherwise.
 */
int inclusivesearch ( double x, double y, double z, bounds_t bounds )
{
    if ( ( x >= bounds.xmin ) &&
         ( x <= bounds.xmax ) &&
         ( y >= bounds.ymin ) &&
         ( y <= bounds.ymax ) &&
         ( z >= bounds.zmin ) &&
         ( z <= bounds.zmax ) )
    {
        return 1;
    }

    return 0;
}

/**
 * Returns 1 if a point is within min,max boundaries, (including min boundaries
 *           for compliance with etree library.)
 * Returns 0 otherwise.
 */
int exclusivesearch ( double x, double y, double z, bounds_t bounds )
{
    if ( ( x >= bounds.xmin ) &&
         ( x <  bounds.xmax ) &&
         ( y >= bounds.ymin ) &&
         ( y <  bounds.ymax ) &&
         ( z >= bounds.zmin ) &&
         ( z <  bounds.zmax ) )
    {
        return 1;
    }

    return 0;
}

/**
 * Returns 0 if a point is not on one of the boundary faces defining a building.
 * If it is on one or more faces it returns N > 0 as follows:
 *  1: touches the bottom Z face that for this building is also the surface,
 *  2: touches one of face (but it is not also the surface),
 *  3: touches a lateral (X or Y) and the bottom (Z) face (also surface),
 *  4: touches two faces (XY, XZ, or YZ) but Z is not also the surface,
 *  5: touches Z (also surface) and X and Y,
 *  6: touches three faces (XYZ) but Z is not also the surface
 */
int onboundarysearch ( double x, double y, double z, bounds_t bounds )
{
    int onX = 0, onY = 0, onZ = 0;
    int faces = 0;

    if ( inclusivesearch(x, y, z, bounds) ) {
        if ( ( x == bounds.xmin ) ||
             ( x == bounds.xmax ) ) {
            onX = 2;
        }
        if ( ( y == bounds.ymin ) ||
             ( y == bounds.ymax ) ) {
            onY = 2;
        }
        if ( ( z == bounds.zmin ) ||
             ( z == bounds.zmax ) ) {
            onZ = 2;
            if ( z == theSurfaceShift ) {
                onZ = 1;
            }
        }
        faces = onX + onY + onZ;
    }

    return faces;
}

/**
 * Returns 1 if a point is in the interior of a building.
 * Returns 0 otherwise.
 */
int ininteriorsearch ( double x, double y, double z, bounds_t bounds )
{
    if ( ( x > bounds.xmin ) &&
         ( x < bounds.xmax ) &&
         ( y > bounds.ymin ) &&
         ( y < bounds.ymax ) &&
         ( z > bounds.zmin ) &&
         ( z < bounds.zmax ) )
    {
        return 1;
    }

    return 0;
}

/**
 * Return 1 if a node is in a building, 0 otherwise.
 */
int bldg_exclusivesearch ( tick_t   x,
                           tick_t   y,
                           tick_t   z,
                           double   ticksize,
                           edata_t *edata,
                           int      bldg )
{
    /**
     * Checks if a an octant with left-upper-front origin coordinate and size
     * equal edgesize is being crossed by a defining boundary.
     *
     * Returns
     * 1: True
     * 0: False
     */

    double   x_m, y_m, z_m;
    double   esize;
    bounds_t bounds;

    x_m = x * ticksize;
    y_m = y * ticksize;
    z_m = z * ticksize;

    esize = (double)edata->edgesize; /* TODO: Seems like I don't need this */

    bounds = get_bldgbounds( bldg );

    if ( exclusivesearch(x_m, y_m, z_m, bounds) ) {
        return 1;
    }

    return 0;
}


/**
  * Return 1 if a node is in a pushdown, 0 otherwise. with left-upper-front
  */
 int pushdown_exclusivesearch ( tick_t   x,
		 tick_t   y,
		 tick_t   z,
		 double   ticksize,
		 int      pushdown )
 {

	 double   x_m, y_m, z_m;
	 bounds_t bounds;

	 x_m = x * ticksize;
	 y_m = y * ticksize;
	 z_m = z * ticksize;

	 bounds = get_pushdownbounds( pushdown );

	 if ( exclusivesearch(x_m, y_m, z_m, bounds) ) {
		 return 1;
	 }

	 return 0;
 }



/**
 * Return 1 if an element is in the building+foundation, 0 otherwise.
 */
int bldg_meshingsearch ( octant_t *leaf,
                         double    ticksize,
                         edata_t  *edata,
                         int       bldg,
                         bounds_t bounds)
{
	double   x_m, y_m, z_m;
	double   esize;

	x_m = leaf->lx * ticksize;
	y_m = leaf->ly * ticksize;
	z_m = leaf->lz * ticksize;

	esize  = (double)edata->edgesize;

	bounds.xmin -= FENCELIMIT * esize;
	bounds.ymin -= FENCELIMIT * esize;
	bounds.zmin -= FENCELIMIT * esize;
	if ( exclusivesearch(x_m, y_m, z_m, bounds) ) {
		return 1;
	}

	return 0;
}

/* -------------------------------------------------------------------------- */
/*                        Collective Search Utilities                         */
/* -------------------------------------------------------------------------- */

/**
 * Return N it the node is at the base of a building, N being the n-th building
 * Return 0 otherwise
 */
int basenode_search ( tick_t x, tick_t y, tick_t z, double ticksize ) {

    int i;
    double x_m, y_m, z_m;

    z_m = z * ticksize;

    /* Discard nodes not at the surface */
    if ( z_m != theSurfaceShift ) {
        return 0;
    }

    x_m = x * ticksize;
    y_m = y * ticksize;

    for ( i = 0; i < theNumberOfBuildings; i++ ) {
        bounds_t bounds = get_bldgbounds(i);
        if ( inclusivesearch(x_m, y_m, z_m, bounds) ) {
            return i+1;
        }
    }

    return 0;
}

/**
 * Return N if the element belongs to a building, N being the n-th building
 * Return 0 otherwise.
 */
int bldgs_search ( octant_t *leaf, double ticksize, edata_t *edata ) {

    int i;
	bounds_t bounds;

    for ( i = 0; i < theNumberOfBuildings; i++ ) {
		 bounds = get_bldgbounds(i);

        if ( bldg_meshingsearch( leaf, ticksize, edata, i, bounds ) == 1 ) {
            /* the i-th building has index i-1 */
            return i+1;
        }
    }

    return 0;
}


/**
  * Return N if the element belongs to a pushdown, N being the n-th pushdown
  * Return 0 otherwise.
  */
int pushdowns_search ( tick_t x, tick_t y, tick_t z, double ticksize) {

	int i;

	for ( i = 0; i < theNumberOfPushdowns; i++ ) {
		if ( pushdown_exclusivesearch( x, y, z, ticksize, i ) == 1 ) {
			/* the i-th pushdown has index i-1 */
			return i+1;
		}
	}

	return 0;
}


/**
 * Depending on the position of a node wrt to a building it returns:
 *      0: out of building bounds,
 *     -1: in the interior of a building,
 * 1,..,6: depending on the onboundarysearch method.
 */
int bldgs_nodesearch ( tick_t x, tick_t y, tick_t z, double ticksize ) {

    int    i;
    double x_m, y_m, z_m;

    x_m = x * ticksize;
    y_m = y * ticksize;
    z_m = z * ticksize;

    for ( i = 0; i < theNumberOfBuildings; i++ ) {
        bounds_t bounds = get_bldgbounds(i);
        if ( inclusivesearch(x_m, y_m, z_m, bounds) ) {
            if ( ininteriorsearch(x_m, y_m, z_m, bounds) ) {
                return -1;
            }
            int faces = onboundarysearch(x_m, y_m, z_m, bounds);
            if ( faces == 0 ) {
                solver_abort ( __FUNCTION_NAME, NULL,
                               "Wrong node faces touches at "
                               "east = %f, north = %f, depth =%f\n", y, x, z);
            }
            return faces;
        }
    }
    return 0;
}

/* Yigit says-- this function is introduced instead of bldgs_nodesearch
 * TODO : building_nodesearch will be deleted because node_set_property function
 * is now in a "clean" format.
 * */

/**
  * Depending on the position of a node wrt to a building it returns:
  *  0: out of building bounds,
  * -1: in the interior of a building,
  *  1: on a face of a building.
  */
 int bldgs_nodesearch_com ( tick_t x, tick_t y, tick_t z, double ticksize ) {

	 int    i;
	 double x_m, y_m, z_m;

	 x_m = x * ticksize;
	 y_m = y * ticksize;
	 z_m = z * ticksize;

	 for ( i = 0; i < theNumberOfBuildings; i++ ) {
		 bounds_t bounds = get_bldgbounds(i);
		 if ( inclusivesearch(x_m, y_m, z_m, bounds) ) {
			 if ( ininteriorsearch(x_m, y_m, z_m, bounds) ) {
				 return -1;
			 }
			 /* on a face */
			 return 1;
		 }
	 }
	 return 0;
 }


 /**
  * Depending on the position of a node wrt to a pushdown  it returns:
  *  0: out of pushdown bounds,
  * -1: in the interior of a pushdown,
  *  1: on a face of a pushdown.
  */
 int pushdowns_nodesearch ( tick_t x, tick_t y, tick_t z, double ticksize ) {

	 int    i;
	 double x_m, y_m, z_m;

	 x_m = x * ticksize;
	 y_m = y * ticksize;
	 z_m = z * ticksize;

	 for ( i = 0; i < theNumberOfPushdowns; i++ ) {
		 bounds_t bounds = get_pushdownbounds(i);
		 if ( inclusivesearch(x_m, y_m, z_m, bounds) ) {
			 if ( ininteriorsearch(x_m, y_m, z_m, bounds) ) {
				 return -1;
			 }
			 /* on a face */
			 return 1;
		 }
	 }
	 return 0;
 }

/* -------------------------------------------------------------------------- */
/*                              Setrec Packet                                 */
/* -------------------------------------------------------------------------- */

/**
 * Complement for the setrec function in psolve.
 * Return 1 if data is assigned to the octant, 0 otherwise.
 */

int bldgs_setrec ( octant_t *leaf, double ticksize,
                   edata_t *edata, etree_t *cvm,double xoriginm,
                   double yoriginm, double zoriginm)
{
    int          res;
    double       z_m;
    cvmpayload_t props;

    res = bldgs_search( leaf, ticksize, edata );
    if ( res > 0 ) {
        props    = get_props( res, z_m );
        edata->Vp  = props.Vp;
        edata->Vs  = props.Vs;
        edata->rho = props.rho;
        return 1;
    }

    z_m = leaf->lz * ticksize;

    if ( z_m < theSurfaceShift ) {
        get_airprops( leaf, ticksize, edata, cvm,xoriginm,yoriginm,zoriginm);
        return 1;
    }

    return 0;
}


/* -------------------------------------------------------------------------- */
/*                             To Expand Packet                               */
/* -------------------------------------------------------------------------- */

/**
 * Return 1 if an element in building+foundation need to be refined , 0 otherwise.
 */
int bldgs_refine ( octant_t *leaf,
                     double    ticksize,
                     edata_t  *edata,
                     int       bldg,
                     double    theFactor,
                     bounds_t bounds)
{
	/**
	 * Yigit says: I restructured this function to comply with equal sized elems
	 * in the buildings.
	 */
	double   edgesize;
	double   z_m;
	float   Vs_min;

	edgesize = edata->edgesize;
	z_m      = leaf->lz * ticksize;

	/* Same size elements*/
	if ( edgesize != theMinOctSizeMeters ) {
		return 1;
	}

	if ( asymmetricBuildings == NO ) {
		Vs_min = theBuilding[bldg].bldgprops.Vs;
	}

	if ( asymmetricBuildings == YES ) {
		Vs_min = theBuilding[bldg].bldgprops_left.Vs;
		if( theBuilding[bldg].bldgprops_left.Vs > theBuilding[bldg].bldgprops_right.Vs )
			Vs_min = theBuilding[bldg].bldgprops_right.Vs;

	}

	/* Elements not complying with vs-rule */
	if ( z_m >= theSurfaceShift ) {

		/* Element is in the foundation */
		if ( theMinOctSizeMeters > ( theBuilding[bldg].fdtnprops.Vs / theFactor ) &&
				theBuilding[bldg].depth !=0  ) {
			fprintf(stderr, "Error: %s %d: theMinOctSizeMeters should be decreased "
					"to %f to comply with the Vs rule for the %dth foundation \n",
					__FILE__, __LINE__,theBuilding[bldg].fdtnprops.Vs / theFactor,
					bldg+1);
			MPI_Abort(MPI_COMM_WORLD,ERROR);
			exit(1);

		}
	} else {
		/* Element is in the building */
		if ( theMinOctSizeMeters > ( Vs_min / theFactor ) &&
				theBuilding[bldg].height !=0) {
			fprintf(stderr, "Error:%s %d: theMinOctSizeMeters should be decreased "
					"to %f to comply with the Vs rule for the %dth buildings \n",
					__FILE__, __LINE__,theBuilding[bldg].bldgprops.Vs / theFactor,
					bldg+1 );
			MPI_Abort(MPI_COMM_WORLD,ERROR);
			exit(1);

		}
	}

	return 0;
}

/**
  * Return  1 if an element is/crosses in a building+foundation and needs to be refined,
  * Return  0 if an element is in a building+foundation does not need to be refined,
  * Return -1 if an element is not/does not cross in a building+foundation.
 */
int bldgs_toexpand ( octant_t *leaf,
                     double    ticksize,
                     edata_t  *edata,
                     double    theFactor )
{
	int i;
	bounds_t bounds;

	/**
	 * Yigit says: I restructured this function to comply with equal sized elems
	 * in the buildings.
	 */

	/* Elements crossing the surface */
	if ( crossing_rule( leaf->lz, ticksize, edata, theSurfaceShift ) ) {
		return 1;
	}

	/* Elements crossing the building+foundation boundaries */
	for ( i = 0; i < theNumberOfBuildings; i++ ) {

		/* bounds_expanded is used here to avoid dangling nodes in the bldg+fdn */
		bounds = theBuilding[i].bounds_expanded;

		/* Elements inside the building+foundation boundaries */
		if ( bldg_meshingsearch( leaf, ticksize, edata, i,bounds ) ) {
			return bldgs_refine( leaf, ticksize, edata, i, theFactor,bounds );
		}

		/* Elements crossing the buildings adjusted boundaries. Elements inside
		 *  the buildings should return in bldg_meshingsearch*/
		if ( crossing_rule( leaf->lx, ticksize, edata, bounds.xmin ) ||
				crossing_rule( leaf->lx, ticksize, edata, bounds.xmax ) ||
				(leaf->lx*ticksize >= bounds.xmin &&
						leaf->lx*ticksize < bounds.xmax ) )

			if ( crossing_rule( leaf->ly, ticksize, edata, bounds.ymin ) ||
					crossing_rule( leaf->ly, ticksize, edata, bounds.ymax ) ||
					(leaf->ly*ticksize >= bounds.ymin &&
							leaf->ly*ticksize < bounds.ymax ) )


				if ( crossing_rule( leaf->lz, ticksize, edata, bounds.zmin ) ||
						crossing_rule( leaf->lz, ticksize, edata, bounds.zmax ) ||
						(leaf->lz*ticksize >= bounds.zmin &&
								leaf->lz*ticksize < bounds.zmax ) )
				{
					return 1;
				}
	}

	return -1;

}

/* -------------------------------------------------------------------------- */
/*                             Correct Properties                             */
/* -------------------------------------------------------------------------- */

/**
 * Return 1 if an element is in foundation after its properties being corrected,
 * 0 otherwise.
 */
int bldgs_correctproperties ( mesh_t *myMesh, edata_t *edata, int32_t lnid0 )
{
	int    i;
	double ticksize;
	tick_t x, y, z;

	/* Element's lower left coordinates in ticks */
	x = myMesh->nodeTable[lnid0].x;
	y = myMesh->nodeTable[lnid0].y;
	z = myMesh->nodeTable[lnid0].z;

	/* ticks converter to physical coordinates */
	ticksize = myMesh->ticksize;

	/* If there is a pushdown, foundation properties should be set equal
	 * to the those of the soil at the location of a pushdown.  */

	for ( i = 0; i < theNumberOfBuildings; i++ ) {

		if ( bldg_exclusivesearch( x, y, z, ticksize, edata, i) == 1 ) {
			/* artificially set z = theSurfaceShift - 1 for irregular shaped fdnts. */
			if ( ( pushdowns_search( x, y,
					theSurfaceShift/ticksize - 1, ticksize) ) == 0 ) {

				double z_m = z * ticksize;

				if ( z_m >= theSurfaceShift ) {
					/* Element is in the foundation */
					edata->Vp  = theBuilding[i].fdtnprops.Vp;
					edata->Vs  = theBuilding[i].fdtnprops.Vs;
					edata->rho = theBuilding[i].fdtnprops.rho;
				} else {

					if ( asymmetricBuildings == NO ) {
						edata->Vp  = theBuilding[i].bldgprops.Vp;
						edata->Vs  = theBuilding[i].bldgprops.Vs;
						edata->rho = theBuilding[i].bldgprops.rho;
					}

					if ( asymmetricBuildings == YES ) {

						int n, location;
						double y_physical, increment_rho, increment_Vs, increment_Vp ;

						y_physical = y * ticksize;


						/* THIS IS RHO VARIES WRT CENTER - SYMMETRIC */

						/* NOTE: Buildings should have even number of elements in  the y direction. */
						/* non-uniform Vp and Vs distribution in y direction only. - but symmetric */

						/* non-uniform mass distribution in the y direction only (EW). */
						/* number of elements along y */
						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters/2;
						/* increment in rho*/
						increment_rho = (theBuilding[i].bldgprops_left.rho - theBuilding[i].bldgprops_right.rho)/n;
						/* location(wrt left building edge) starts from 0 and goes to n-1*/
						location = (y_physical - (theBuilding[i].bounds.ymin + theBuilding[i].bounds.ymax)/2 )/theMinOctSizeMeters;
						if(location < 0) location++;
						location = abs(location);
						/* final value of rho*/
						edata->rho = theBuilding[i].bldgprops_right.rho + increment_rho*location + increment_rho/2 ;

						//						/* THIS IS VS VP VARIES WRT CENTER - SYMMETRIC */
						//						edata->Vp = pow(pow(800,2)*300/edata->rho,0.5);
						//						edata->Vs = pow(pow(350,2)*300/edata->rho,0.5);


						//						/* THIS IS RHO VARIES - ASYMMETRIC */
						//
						//						/* non-uniform mass distribution in the y direction only (EW). */
						//						/* number of elements along y */
						//						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						//						/* increment in rho*/
						//						increment_rho = (theBuilding[i].bldgprops_right.rho - theBuilding[i].bldgprops_left.rho)/n;
						//						/* location(wrt left building edge) starts from 0 and goes to n-1*/
						//						location = (y_physical - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						//						/* final value of rho*/
						//						edata->rho = theBuilding[i].bldgprops_left.rho + increment_rho*location + increment_rho/2 ;

						/*-----------------------------------------------------*/

						//						/* THIS IS VP VS VARIES WRT CENTER S.T. SUM of THE SQUARES STAYS UNCHANGED AND VARIES LINEARLY  - SYMMETRIC */
						//
						//						/* NOTE: Buildings should have even number of elements in  the y direction. */
						//						/* non-uniform Vp and Vs distribution in y direction only. - but symmetric */
						//
						//						/* number of elements along y up to the center */
						//						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters/2;
						//						/* increment in Vp and Vs*/
						//
						//						increment_Vp = (pow(theBuilding[i].bldgprops_left.Vp,2)
						//								-pow(theBuilding[i].bldgprops_right.Vp,2)) / n;
						//
						//						increment_Vs = (pow(theBuilding[i].bldgprops_left.Vs,2)
						//								-pow(theBuilding[i].bldgprops_right.Vs,2)) / n;
						//
						//						/* location(wrt center) goes from 0 to n-1*/
						//						location = (y_physical - (theBuilding[i].bounds.ymin + theBuilding[i].bounds.ymax)/2 )/theMinOctSizeMeters;
						//						if(location < 0) location++;
						//						location = abs(location);
						//						/* final value of Vp and Vs*/
						//						edata->Vp = pow(pow(theBuilding[i].bldgprops_right.Vp, 2) + increment_Vp*location + increment_Vp/2, 0.5) ;
						//						edata->Vs = pow(pow(theBuilding[i].bldgprops_right.Vs, 2) + increment_Vs*location + increment_Vs/2, 0.5) ;


						/* THIS IS VP VS VARIES ASYMMETRIC */

						/* NOTE: Buildings should have even number of elements in  the y direction. */
						/* non-uniform Vp and Vs distribution in y direction only. - but symmetric */

						/* number of elements along y */
						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						/* increment in Vp and Vs*/

						increment_Vp = (pow(theBuilding[i].bldgprops_right.Vp,2)
								-pow(theBuilding[i].bldgprops_left.Vp,2)) / n;

						increment_Vs = (pow(theBuilding[i].bldgprops_right.Vs,2)
								-pow(theBuilding[i].bldgprops_left.Vs,2)) / n;

						/* location(wrt left building edge) goes from 0 to n-1*/
						location = (y_physical - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						/* final value of Vp and Vs*/
						edata->Vp = pow(pow(theBuilding[i].bldgprops_left.Vp, 2) + increment_Vp*location + increment_Vp/2, 0.5) ;
						edata->Vs = pow(pow(theBuilding[i].bldgprops_left.Vs, 2) + increment_Vs*location + increment_Vs/2, 0.5) ;


//						eccentricity += (y_physical - theBuilding[i].bounds.ymin + theMinOctSizeMeters * 0.5) * pow(edata->Vs,2)  /
//								(pow(350,2) * 6 * 12 * 7);

					}
				}

				return 1;
			}
		}
	}
	/* NOTE: If you want to see the carving process, activate this */
	/*
    double z_m = z * ticksize;
    if ( z_m < theSurfaceShift ) {
        edata->Vp  = -500;
        edata->Vs  = -200;
        edata->rho = -1500;
        return 1;
    }
	 */

	return 0;
}


/* -------------------------------------------------------------------------- */
/*       Initialization of parameters, structures and memory allocations      */
/* -------------------------------------------------------------------------- */


void bldgs_init ( int32_t myID, const char *parametersin )
{
    int     i;
    int     int_message[5];
    double  double_message[2];

    /* Capturing data from file --- only done by PE0 */
    if (myID == 0) {
        if ( buildings_initparameters( parametersin ) != 0 ) {
            fprintf(stderr,"Thread 0: buildings_local_init: "
                    "buildings_initparameters error\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }
        adjust_dimensions();
    }

    /* Broadcasting data */

    double_message[0] = theSurfaceShift;
    double_message[1] = theMinOctSizeMeters;
    int_message[0]    = theNumberOfBuildings;
    int_message[1]    = areBaseFixed;
	int_message[2]    = theNumberOfPushdowns;
	int_message[3]    = asymmetricBuildings;
	int_message[4]    = constrainedSlabs;

    MPI_Bcast(double_message, 2, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(int_message,    5, MPI_INT,    0, comm_solver);

    theSurfaceShift      = double_message[0];
    theMinOctSizeMeters  = double_message[1];
    theNumberOfBuildings = int_message[0];
    areBaseFixed         = int_message[1];
	theNumberOfPushdowns = int_message[2];
	asymmetricBuildings  = int_message[3];
	constrainedSlabs  	 = int_message[4];

    /* allocate table of properties for all other PEs */
    if ( theNumberOfBuildings > 0 ) {

    	if (myID != 0) {

    		theBuildingsXMin   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsXMax   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsYMin   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsYMax   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsZMin   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsZMax   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsDepth  = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsHeight = (double*)malloc( sizeof(double) * theNumberOfBuildings );

    		theFoundationsVp   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theFoundationsVs   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theFoundationsRho  = (double*)malloc( sizeof(double) * theNumberOfBuildings );

    		if(asymmetricBuildings == NO) {
    			theBuildingsVp     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsVs     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsRho    = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		} else {
    			theBuildingsVp_left     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsVs_left     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsRho_left    = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsVp_right     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsVs_right     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    			theBuildingsRho_right    = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		}


    	}

    	/* Broadcast table of properties */
    	MPI_Bcast(theBuildingsXMin,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsXMax,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsYMin,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsYMax,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsZMin,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsZMax,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsDepth,  theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsHeight, theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsVp,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsVs,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsRho,  theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);

    	if(asymmetricBuildings == NO) {
    		MPI_Bcast(theBuildingsVp,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsVs,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsRho,    theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);

    	} else if(asymmetricBuildings == YES) {

    		MPI_Bcast(theBuildingsVp_left,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsVs_left,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsRho_left,    theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);

    		MPI_Bcast(theBuildingsVp_right,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsVs_right,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(theBuildingsRho_right,    theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	}



    	/* Broadcast fixed base data */
    	if ( areBaseFixed == YES ) {
    		MPI_Bcast(&theBaseFixedDT,         1, MPI_DOUBLE, 0, comm_solver);
    		MPI_Bcast(&theBaseFixedStartIndex, 1, MPI_INT,    0, comm_solver);
    		broadcast_char_array( theBaseFixedDir,   sizeof(theBaseFixedDir),
    				0, comm_solver );
    		broadcast_char_array( theBaseFixedSufix, sizeof(theBaseFixedSufix),
    				0, comm_solver );
    	}

    	theBuilding = (bldg_t *)malloc( sizeof(bldg_t) * theNumberOfBuildings );
    	if ( theBuilding == NULL ) {
    		solver_abort ( __FUNCTION_NAME, "NULL from malloc",
    				"Error allocating theBuildings memory" );
    	}

    	for ( i = 0; i < theNumberOfBuildings; i++ ) {

    		theBuilding[i].bounds.xmin = theBuildingsXMin[i];
    		theBuilding[i].bounds.xmax = theBuildingsXMax[i];
    		theBuilding[i].bounds.ymin = theBuildingsYMin[i];
    		theBuilding[i].bounds.ymax = theBuildingsYMax[i];
    		theBuilding[i].bounds.zmin = theBuildingsZMin[i];
    		theBuilding[i].bounds.zmax = theBuildingsZMax[i];

    		/* bounds_expanded is useful for having an additional (equal size
    		 * finite element) layer around the buildings+foundations */

    		theBuilding[i].bounds_expanded.xmin = theBuildingsXMin[i] - theMinOctSizeMeters;
    		theBuilding[i].bounds_expanded.xmax = theBuildingsXMax[i] + theMinOctSizeMeters;
    		theBuilding[i].bounds_expanded.ymin = theBuildingsYMin[i] - theMinOctSizeMeters;
    		theBuilding[i].bounds_expanded.ymax = theBuildingsYMax[i] + theMinOctSizeMeters;
    		theBuilding[i].bounds_expanded.zmin = theBuildingsZMin[i];
    		theBuilding[i].bounds_expanded.zmax = theBuildingsZMax[i] + theMinOctSizeMeters;

    		theBuilding[i].height = theBuildingsHeight[i];
    		theBuilding[i].depth  = theBuildingsDepth[i];

    		if(asymmetricBuildings == NO) {
    			theBuilding[i].bldgprops.Vp  = theBuildingsVp[i];
    			theBuilding[i].bldgprops.Vs  = theBuildingsVs[i];
    			theBuilding[i].bldgprops.rho = theBuildingsRho[i];
    		}
    		else if (asymmetricBuildings == YES) {
    			theBuilding[i].bldgprops_left.Vp  = theBuildingsVp_left[i];
    			theBuilding[i].bldgprops_left.Vs  = theBuildingsVs_left[i];
    			theBuilding[i].bldgprops_left.rho = theBuildingsRho_left[i];

    			theBuilding[i].bldgprops_right.Vp  = theBuildingsVp_right[i];
    			theBuilding[i].bldgprops_right.Vs  = theBuildingsVs_right[i];
    			theBuilding[i].bldgprops_right.rho = theBuildingsRho_right[i];
    		}
    		theBuilding[i].fdtnprops.Vp  = theFoundationsVp[i];
    		theBuilding[i].fdtnprops.Vs  = theFoundationsVs[i];
    		theBuilding[i].fdtnprops.rho = theFoundationsRho[i];

    	}

    	free(theBuildingsXMin);
    	free(theBuildingsXMax);
    	free(theBuildingsYMin);
    	free(theBuildingsYMax);
    	free(theBuildingsZMin);
    	free(theBuildingsZMax);
    	free(theBuildingsHeight);
    	free(theBuildingsDepth);

    	if(asymmetricBuildings == NO) {
    		free(theBuildingsVp);
    		free(theBuildingsVs);
    		free(theBuildingsRho);
    	}

    	if(asymmetricBuildings == YES) {
    		free(theBuildingsVp_left);
    		free(theBuildingsVs_left);
    		free(theBuildingsRho_left);

    		free(theBuildingsVp_right);
    		free(theBuildingsVs_right);
    		free(theBuildingsRho_right);
    	}

    	free(theFoundationsVp);
    	free(theFoundationsVs);
    	free(theFoundationsRho);
    }


    if ( theNumberOfPushdowns > 0 ) {
    	if (myID != 0) {

    		thePushDownsXMin   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsXMax   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsYMin   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsYMax   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsZMin   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsZMax   = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsHeight = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    		thePushDownsDepth  = (double*)malloc( sizeof(double) * theNumberOfPushdowns );
    	}

    	MPI_Bcast(thePushDownsXMin,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsXMax,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsYMin,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsYMax,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsZMin,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsZMax,   theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsHeight, theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(thePushDownsDepth,  theNumberOfPushdowns, MPI_DOUBLE, 0, comm_solver);

    	thePushdown = (pdown_t *)malloc( sizeof(pdown_t) * theNumberOfPushdowns );
    	if ( thePushdown == NULL ) {
    		solver_abort ( __FUNCTION_NAME, "NULL from malloc",
    				"Error allocating thePushdown memory" );
    	}

    	for ( i = 0; i < theNumberOfPushdowns; i++ ) {

    		thePushdown[i].bounds.xmin = thePushDownsXMin[i];
    		thePushdown[i].bounds.xmax = thePushDownsXMax[i];
    		thePushdown[i].bounds.ymin = thePushDownsYMin[i];
    		thePushdown[i].bounds.ymax = thePushDownsYMax[i];
    		thePushdown[i].bounds.zmin = thePushDownsZMin[i];
    		thePushdown[i].bounds.zmax = thePushDownsZMax[i];

    		thePushdown[i].height = thePushDownsHeight[i];
    		thePushdown[i].depth  = thePushDownsDepth[i];
    	}

    	free(thePushDownsXMin);
    	free(thePushDownsXMax);
    	free(thePushDownsYMin);
    	free(thePushDownsYMax);
    	free(thePushDownsZMin);
    	free(thePushDownsZMax);
    	free(thePushDownsHeight);
    	free(thePushDownsDepth);
    }

    return;
}


int32_t
buildings_initparameters ( const char *parametersin )
{
	FILE   *fp;
	int     iBldg, numBldgs, numPushDown, iPDown;
	double  min_oct_size, surface_shift;
	double *auxiliar;
	char    consider_fixed_base[16];
	char    asymmetric_buildings[16];
	char    constrained_slabs[16];

	noyesflag_t fixedbase = -1;
	noyesflag_t asymmetry = -1;
	noyesflag_t constraint = -1;

	/* Opens parametersin file */

	if ( ( fp = fopen(parametersin, "r" ) ) == NULL ) {
		fprintf( stderr,
				"Error opening %s\n at buildings_initparameters",
				parametersin );
		return -1;
	}

	/* Parses parametersin to capture building single-value parameters */

	if ( ( parsetext(fp, "number_of_buildings", 'i', &numBldgs              ) != 0) ||
			( parsetext(fp, "number_of_pushdowns", 'i', &numPushDown        ) != 0) ||
			( parsetext(fp, "min_octant_size_m",   'd', &min_oct_size       ) != 0) ||
			( parsetext(fp, "surface_shift_m",     'd', &surface_shift      ) != 0) ||
			( parsetext(fp, "consider_fixed_base", 's', &consider_fixed_base) != 0) ||
			( parsetext(fp, "constrained_slabs",   's', &constrained_slabs  ) != 0) ||
			( parsetext(fp, "asymmetric_buildings", 's', &asymmetric_buildings) != 0))
	{
		fprintf( stderr,
				"Error parsing building parameters from %s\n",
				parametersin );
		return -1;
	}

	/* Performs sanity checks */

	if ( numBldgs < 0 ) {
		fprintf( stderr,
				"Illegal number of buildings %d\n",
				numBldgs );
		return -1;
	}


	if ( numPushDown < 0 ) {
		fprintf( stderr,
				"Illegal number of push-downs %d\n",
				numPushDown );
		return -1;
	}

	if ( min_oct_size < 0 ) {
		fprintf( stderr,
				"Illegal minimum octant size for buildings %f\n",
				min_oct_size );
		return -1;
	}

	if ( surface_shift < 0 ) {
		fprintf( stderr,
				"Illegal surface_shift for buildings %f\n",
				surface_shift );
		return -1;
	}

	if ( strcasecmp(consider_fixed_base, "yes") == 0 ) {
		fixedbase = YES;
	} else if ( strcasecmp(consider_fixed_base, "no") == 0 ) {
		fixedbase = NO;
	} else {
		solver_abort( __FUNCTION_NAME, NULL,
				"Unknown response for considering"
				" fixed base option (yes or no): %s\n",
				consider_fixed_base );
	}

	if ( strcasecmp(asymmetric_buildings, "yes") == 0 ) {
		asymmetry = YES;
	} else if ( strcasecmp(asymmetric_buildings, "no") == 0 ) {
		asymmetry = NO;
	} else {
		solver_abort( __FUNCTION_NAME, NULL,
				"Unknown response for considering"
				" asymmetric buildings option (yes or no): %s\n",
				consider_fixed_base );
	}


	if ( strcasecmp(constrained_slabs, "yes") == 0 ) {
		constraint = YES;
	} else if ( strcasecmp(constrained_slabs, "no") == 0 ) {
		constraint = NO;
	} else {
		solver_abort( __FUNCTION_NAME, NULL,
				"Unknown response for constrained slabs"
				"option (yes or no): %s\n",
				constraint );
	}

	/* Initialize the static global variables */

	theNumberOfBuildings = numBldgs;
	theNumberOfPushdowns = numPushDown;
	theMinOctSizeMeters  = min_oct_size;
	theSurfaceShift      = surface_shift;
	areBaseFixed         = fixedbase;
	asymmetricBuildings  = asymmetry;
	constrainedSlabs     = constraint;

	/* Detour for fixed base option */
	if ( areBaseFixed == YES ) {
		fixedbase_read( fp );
	}

	if ( numBldgs > 0 ) {

		if(asymmetricBuildings == NO) {

			auxiliar           = (double*)malloc( sizeof(double) * numBldgs * 12 );
			theBuildingsXMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsXMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsYMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsYMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsZMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsZMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsDepth  = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsHeight = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsDepth  = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsVp     = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsVs     = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsRho    = (double*)malloc( sizeof(double) * numBldgs );
			theFoundationsVp   = (double*)malloc( sizeof(double) * numBldgs );
			theFoundationsVs   = (double*)malloc( sizeof(double) * numBldgs );
			theFoundationsRho  = (double*)malloc( sizeof(double) * numBldgs );

			if ( ( auxiliar           == NULL ) ||
					( theBuildingsXMin   == NULL ) ||
					( theBuildingsXMax   == NULL ) ||
					( theBuildingsYMin   == NULL ) ||
					( theBuildingsYMax   == NULL ) ||
					( theBuildingsDepth  == NULL ) ||
					( theBuildingsHeight == NULL ) ||
					( theBuildingsVp     == NULL ) ||
					( theBuildingsVs     == NULL ) ||
					( theBuildingsRho    == NULL ) ||
					( theFoundationsVp   == NULL ) ||
					( theFoundationsVs   == NULL ) ||
					( theFoundationsRho  == NULL ) )
			{
				fprintf( stderr, "Errror allocating transient building arrays"
						"in buildings_initparameters " );
				return -1;
			}

			if ( parsedarray( fp, "building_properties", numBldgs*12, auxiliar ) != 0)
			{
				fprintf( stderr,
						"Error parsing building_properties list from %s\n",
						parametersin );
				return -1;
			}

			/* We DO NOT convert physical coordinates into etree coordinates here */
			for (iBldg = 0; iBldg < numBldgs; iBldg++) {

				theBuildingsXMin   [ iBldg ] = auxiliar [ iBldg * 12      ];
				theBuildingsXMax   [ iBldg ] = auxiliar [ iBldg * 12 +  1 ];
				theBuildingsYMin   [ iBldg ] = auxiliar [ iBldg * 12 +  2 ];
				theBuildingsYMax   [ iBldg ] = auxiliar [ iBldg * 12 +  3 ];
				theBuildingsDepth  [ iBldg ] = auxiliar [ iBldg * 12 +  4 ];
				theBuildingsHeight [ iBldg ] = auxiliar [ iBldg * 12 +  5 ];
				theBuildingsVp     [ iBldg ] = auxiliar [ iBldg * 12 +  6 ];
				theBuildingsVs     [ iBldg ] = auxiliar [ iBldg * 12 +  7 ];
				theBuildingsRho    [ iBldg ] = auxiliar [ iBldg * 12 +  8 ];
				theFoundationsVp   [ iBldg ] = auxiliar [ iBldg * 12 +  9 ];
				theFoundationsVs   [ iBldg ] = auxiliar [ iBldg * 12 + 10 ];
				theFoundationsRho  [ iBldg ] = auxiliar [ iBldg * 12 + 11 ];
			}

			free( auxiliar );
		}

		/* Three more values are to be read from the parameters. */
		if(asymmetricBuildings == YES) {

			auxiliar           = (double*)malloc( sizeof(double) * numBldgs * 15 );
			theBuildingsXMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsXMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsYMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsYMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsZMin   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsZMax   = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsDepth  = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsHeight = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsDepth  = (double*)malloc( sizeof(double) * numBldgs );

			theBuildingsVp_left     = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsVp_right     = (double*)malloc( sizeof(double) * numBldgs );

			theBuildingsVs_left     = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsVs_right     = (double*)malloc( sizeof(double) * numBldgs );

			theBuildingsRho_left    = (double*)malloc( sizeof(double) * numBldgs );
			theBuildingsRho_right    = (double*)malloc( sizeof(double) * numBldgs );

			theFoundationsVp   = (double*)malloc( sizeof(double) * numBldgs );
			theFoundationsVs   = (double*)malloc( sizeof(double) * numBldgs );
			theFoundationsRho  = (double*)malloc( sizeof(double) * numBldgs );

			if ( ( auxiliar           == NULL ) ||
					( theBuildingsXMin   == NULL ) ||
					( theBuildingsXMax   == NULL ) ||
					( theBuildingsYMin   == NULL ) ||
					( theBuildingsYMax   == NULL ) ||
					( theBuildingsDepth  == NULL ) ||
					( theBuildingsHeight == NULL ) ||
					( theBuildingsVp_left     == NULL ) ||
					( theBuildingsVs_left     == NULL ) ||
					( theBuildingsRho_left    == NULL ) ||
					( theBuildingsVp_right     == NULL ) ||
					( theBuildingsVs_right     == NULL ) ||
					( theBuildingsRho_right    == NULL ) ||
					( theFoundationsVp   == NULL ) ||
					( theFoundationsVs   == NULL ) ||
					( theFoundationsRho  == NULL ) )
			{
				fprintf( stderr, "Errror allocating transient building arrays"
						"in buildings_initparameters " );
				return -1;
			}

			if ( parsedarray( fp, "building_properties", numBldgs*15, auxiliar ) != 0)
			{
				fprintf( stderr,
						"Error parsing building_properties list from %s. Note that "
						"asymmetric_buildings = yes \n",
						parametersin );
				return -1;
			}

			/* We DO NOT convert physical coordinates into etree coordinates here */
			for (iBldg = 0; iBldg < numBldgs; iBldg++) {

				theBuildingsXMin      [ iBldg ] = auxiliar [ iBldg * 15     ];
				theBuildingsXMax      [ iBldg ] = auxiliar [ iBldg * 15 + 1 ];
				theBuildingsYMin      [ iBldg ] = auxiliar [ iBldg * 15 + 2 ];
				theBuildingsYMax      [ iBldg ] = auxiliar [ iBldg * 15 + 3 ];
				theBuildingsDepth     [ iBldg ] = auxiliar [ iBldg * 15 + 4 ];
				theBuildingsHeight    [ iBldg ] = auxiliar [ iBldg * 15 + 5 ];
				theBuildingsVp_left   [ iBldg ] = auxiliar [ iBldg * 15 + 6 ];
				theBuildingsVp_right  [ iBldg ] = auxiliar [ iBldg * 15 + 7 ];

				theBuildingsVs_left   [ iBldg ] = auxiliar [ iBldg * 15 + 8 ];
				theBuildingsVs_right  [ iBldg ] = auxiliar [ iBldg * 15 + 9 ];

				theBuildingsRho_left  [ iBldg ] = auxiliar [ iBldg * 15 + 10 ];
				theBuildingsRho_right [ iBldg ] = auxiliar [ iBldg * 15 + 11 ];
				theFoundationsVp      [ iBldg ] = auxiliar [ iBldg * 15 + 12 ];
				theFoundationsVs      [ iBldg ] = auxiliar [ iBldg * 15 + 13 ];
				theFoundationsRho     [ iBldg ] = auxiliar [ iBldg * 15 + 14 ];
			}

			free( auxiliar );

		}


	}

	if ( numPushDown > 0 ) {

		auxiliar           = (double*)malloc( sizeof(double) * numPushDown * 6 );
		thePushDownsXMin   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsXMax   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsYMin   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsYMax   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsZMin   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsZMax   = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsHeight = (double*)malloc( sizeof(double) * numPushDown );
		thePushDownsDepth  = (double*)malloc( sizeof(double) * numPushDown );

		if   ( ( auxiliar         == NULL ) ||
				( thePushDownsXMin   == NULL ) ||
				( thePushDownsXMax   == NULL ) ||
				( thePushDownsYMin   == NULL ) ||
				( thePushDownsYMax   == NULL ) ||
				( thePushDownsZMin   == NULL ) ||
				( thePushDownsZMax   == NULL ) ||
				( thePushDownsHeight == NULL ) ||
				( thePushDownsDepth  == NULL ))

		{
			fprintf( stderr, "Errror allocating transient push-down arrays"
					"in buildings_initparameters " );
			return -1;
		}


		if ( parsedarray( fp, "pushdown_properties", numPushDown*6, auxiliar ) != 0)
		{
			fprintf( stderr,
					"Error parsing pushdown_properties list from %s\n",
					parametersin );
			return -1;
		}

		/* We DO NOT convert physical coordinates into etree coordinates here */
		for (iPDown = 0; iPDown < numPushDown; iPDown++) {

			thePushDownsXMin   [ iPDown ] = auxiliar [ iPDown * 6      ];
			thePushDownsXMax   [ iPDown ] = auxiliar [ iPDown * 6 +  1 ];
			thePushDownsYMin   [ iPDown ] = auxiliar [ iPDown * 6 +  2 ];
			thePushDownsYMax   [ iPDown ] = auxiliar [ iPDown * 6 +  3 ];
			thePushDownsDepth  [ iPDown ] = auxiliar [ iPDown * 6 +  4 ];
			thePushDownsHeight [ iPDown ] = auxiliar [ iPDown * 6 +  5 ];

			/* thePushDownsDepth is set to 0, whatever the value is given
			 * in the parameters file.*/

			thePushDownsDepth  [ iPDown ] = 0;

		}

		free( auxiliar );
	}

	fclose(fp);
	return 0;
}

/* -------------------------------------------------------------------------- */
/*                             Fixed Base Option                              */
/* -------------------------------------------------------------------------- */

void fixedbase_read ( FILE* fp ) {

    if ( ( parsetext(fp, "fixedbase_input_dt",         'd', &theBaseFixedDT          ) != 0) ||
         ( parsetext(fp, "fixedbase_input_dir",        's', &theBaseFixedDir         ) != 0) ||
         ( parsetext(fp, "fixedbase_input_startindex", 'i', &theBaseFixedStartIndex  ) != 0) ||
         ( parsetext(fp, "fixedbase_input_sufix",      's', &theBaseFixedSufix       ) != 0) )
    {
        solver_abort ( __FUNCTION_NAME, "Parsetext returned non-zero",
                       "Error parsing fixed-based options" );
    }

    return;
}

int32_t count_base_nodes ( mesh_t *myMesh ) {

    lnid_t  nindex;
    double  ticksize;
    int32_t count = 0;

    ticksize = myMesh->ticksize;

    for ( nindex = 0; nindex < myMesh->nharbored; nindex++ ) {

        node_t node = myMesh->nodeTable[nindex];

        int res = basenode_search(node.x, node.y, node.z, ticksize);

        if ( res != 0 ) {
            count++;
        }
    }

    return count;
}

int32_t map_base_nodes ( mesh_t *myMesh ) {

    lnid_t  nindex;
    double  ticksize;
    int32_t count = 0;

    ticksize = myMesh->ticksize;

    for ( nindex = 0; nindex < myMesh->nharbored; nindex++ ) {

        node_t node = myMesh->nodeTable[nindex];

        int bldg = basenode_search(node.x, node.y, node.z, ticksize);

        if ( bldg != 0 ) {
            theBaseNodes[count].nindex = nindex;
            theBaseNodes[count].bldg   = bldg;
            count++;
        }
    }

    return count;
}

void read_base_input(double simTime) {

    int i,j;
    int steps = (int)(simTime / theBaseFixedDT);

    theBaseSignals = (fvector_t**)malloc( sizeof(fvector_t*) *
                                           theNumberOfBuildings );
    if ( theBaseSignals == NULL ) {
        solver_abort ( __FUNCTION_NAME, "NULL from malloc",
                       "Error allocating theBaseSignals memory" );
    }

    for ( i = 0; i < theNumberOfBuildings; i++ ) {

        theBaseSignals[i] = (fvector_t*)malloc( sizeof(fvector_t) * steps );
        if ( theBaseSignals[i] == NULL ) {
            solver_abort ( __FUNCTION_NAME, "NULL from malloc",
                           "Error allocating base signal memory" );
        }

        FILE* fp;
        char filename[256];

        sprintf( filename,
                 "%s/%s.%d",
                 theBaseFixedDir,
                 theBaseFixedSufix,
                 i + theBaseFixedStartIndex );
        fp = fopen(filename, "r");
        if ( fp == NULL ) {
            solver_abort ( __FUNCTION_NAME, "NULL from fopen",
                           "Error opening signals" );
        }

        char line[512];
        for ( j = 0; j < 1; j++ ) {
            if ( fgets(line, 512, fp) == NULL ) {
                solver_abort ( __FUNCTION_NAME, "NULL from fgets",
                               "Error reading signals" );
            }
        }

        for ( j = 0; j < steps; j++ ) {
            float aux, x,y,z;
            if ( fgets(line, 512, fp) == NULL ) {
                solver_abort ( __FUNCTION_NAME, "NULL from fgets",
                               "Error reading signals" );
            }
            sscanf(line,"%g %g %g %g",&aux,&x,&y,&z);
            theBaseSignals[i][j].f[0] = (double)x;
            theBaseSignals[i][j].f[1] = (double)y;
            theBaseSignals[i][j].f[2] = (double)z;
        }
    }
}

void bldgs_fixedbase_init ( mesh_t *myMesh, double simTime ) {

    int32_t recount;

    /* Get the number of nodes at the base of a building */
    theBaseNodesCount = count_base_nodes(myMesh);

    /* Allocate memory for the mapper */
    theBaseNodes = (basenode_t*)malloc(sizeof(basenode_t) * theBaseNodesCount);
    if ( theBaseNodes == NULL ) {
        solver_abort ( __FUNCTION_NAME, "NULL from malloc",
                       "Error allocating theBaseNodes memory" );
    }

    /* Map node indices and bldgs assingments */
    recount = map_base_nodes(myMesh);

    if ( recount != theBaseNodesCount ) {
        solver_abort ( __FUNCTION_NAME, "NULL from malloc",
                       "Error allocating theBaseNodes memory" );
    }

    read_base_input(simTime);

    return;
}

void constrained_slabs_init ( mesh_t *myMesh, double simTime, int32_t group_number, int32_t myID ) {

	int iBldg, i, j, k, l, iMaster = -1, iSharer = -1;
	int nodes_x, nodes_y;
	double ticksize;
	int* masterproc; /* how many nodes do I have in each building only if I am master. 0 otherwise*/
	int* sharerproc; /* how many nodes do I have in each building if I am the sharer*/

	int* masterproc_all; /* how many nodes does the master proc have in each building. 0 if I am the sharer of the building.
	First the information of Proc0 for each building, then Proc1 for each building and goes on like this... */
	int* sharerproc_all; /* how many nodes does the procs( only sharers)  have in each building. 0 if I am the master.
	First the information of Proc0 for each building, then Proc1 for each building and goes on like this... */

	ticksize = myMesh->ticksize;


	masterproc = calloc(theNumberOfBuildings,sizeof(int));
	sharerproc = calloc(theNumberOfBuildings,sizeof(int));

	masterproc_all = (int*)malloc( sizeof(int) * theNumberOfBuildings * group_number);
	if ( masterproc_all == NULL ) {
		solver_abort ( __FUNCTION_NAME, "NULL from malloc",
				"Error allocating masterproc_all memory" );
	}


	sharerproc_all = (int*)malloc( sizeof(int) * theNumberOfBuildings * group_number);
	if ( sharerproc_all == NULL ) {
			solver_abort ( __FUNCTION_NAME, "NULL from malloc",
					"Error allocating sharerproc_all memory" );
		}


	/* Find out if I am a master or a sharer */
	for ( iBldg = 0; iBldg < theNumberOfBuildings; iBldg++ ) {
		int sharer = 0, master = 0;

		for ( i = 0; i < myMesh->nharbored; ++i ) {
			if ( myMesh->nodeTable[i].ismine == 1) {

				double x_m, y_m, z_m;

				x_m = (myMesh->nodeTable[i].x) * ticksize;
				y_m = (myMesh->nodeTable[i].y) * ticksize;
				z_m = (myMesh->nodeTable[i].z) * ticksize;


				/* If the buildings S-W roof corner node belongs to me, I am the master */
				if ( ( z_m == theBuilding[iBldg].bounds.zmin) &&
						( x_m == theBuilding[iBldg].bounds.xmin) &&
						( y_m == theBuilding[iBldg].bounds.ymin) )
				{
					master = 1;
				}

				/* If I am a sharer */
				if (( z_m <= theSurfaceShift) &&
						inclusivesearch( x_m, y_m, z_m, theBuilding[iBldg].bounds) == 1 )
				{
					sharer++;
				}

			}
		}

		/* means I am a sharer of this building */
		if (sharer != 0 && master == 0) {
			sharerproc[iBldg] = sharer;
			theNumberOfBuildingsSharer++;
		}

		/* means I am the master of this building */
		if (master != 0) {
			masterproc[iBldg] = sharer;
			theNumberOfBuildingsMaster++;
		}

	}

	/*  information sharing among processors */
	/*  building_info_proc1  building_info_proc2  building_info_proc3 ...*/
	MPI_Allgather( masterproc, theNumberOfBuildings, MPI_INT,
			masterproc_all, theNumberOfBuildings, MPI_INT, comm_solver );

	MPI_Allgather( sharerproc, theNumberOfBuildings, MPI_INT,
			sharerproc_all, theNumberOfBuildings, MPI_INT, comm_solver );


//	printf("\nmyID = %d theNumberOfBuildingsSharer = %d\n",myID,theNumberOfBuildingsSharer);
//
//		fprintf(stdout, "\n  Master %d \n\n", myID);
//		for ( i = 0 ; i < group_number ; i++){
//			for ( j = 0; j < theNumberOfBuildings; j++) {
//				fprintf(stdout, "%5d   ",
//						masterproc_all[i*theNumberOfBuildings + j]);
//			}
//			fprintf(stdout, "\n");
//		}
//		fprintf(stdout, "\n\n");



	/* Masters calculate the avearge dis for the buildings and then sends it back to
	 * the sharers.
	 */
	/* Allocate memory for the  theMasterConstrainedSlab */
	if (theNumberOfBuildingsMaster != 0 ) {

		theMasterConstrainedSlab =
				(master_constrained_slab_bldg_t*)malloc(sizeof(master_constrained_slab_bldg_t)*theNumberOfBuildingsMaster);
		if ( theMasterConstrainedSlab == NULL ) {
			solver_abort ( __FUNCTION_NAME, "NULL from malloc",
					"Error allocating theMasterConstrainedSlab memory" );
		}

		/* Fill theConstrainedSlab struct*/

		for (iBldg = 0; iBldg < theNumberOfBuildings; iBldg++) {

			if( masterproc[iBldg] != 0  ) {

				iMaster++;

				int sharer_count = 0;
				double x_m;
				double y_m;

				/* These are general building information */
				theMasterConstrainedSlab[iMaster].l = (theBuilding[iBldg].height)/
						theMinOctSizeMeters + 1;
				nodes_x = (theBuilding[iBldg].bounds.xmax -
						theBuilding[iBldg].bounds.xmin)/theMinOctSizeMeters + 1;
				nodes_y = (theBuilding[iBldg].bounds.ymax -
						theBuilding[iBldg].bounds.ymin)/theMinOctSizeMeters + 1;

				theMasterConstrainedSlab[iMaster].node_x = nodes_x;
				theMasterConstrainedSlab[iMaster].node_y = nodes_y;
				theMasterConstrainedSlab[iMaster].n = nodes_y * nodes_x;
				theMasterConstrainedSlab[iMaster].x_bldg_length =
						(theBuilding[iBldg].bounds.xmax - theBuilding[iBldg].bounds.xmin);
				theMasterConstrainedSlab[iMaster].y_bldg_length =
						(theBuilding[iBldg].bounds.ymax - theBuilding[iBldg].bounds.ymin);

				x_m = theMasterConstrainedSlab[iMaster].x_bldg_length;
				y_m = theMasterConstrainedSlab[iMaster].y_bldg_length;

				theMasterConstrainedSlab[iMaster].area = x_m*y_m;

				theMasterConstrainedSlab[iMaster].Ix =  0;
				theMasterConstrainedSlab[iMaster].Iy =  0;
				theMasterConstrainedSlab[iMaster].Ixy = 0;


				/* These are communication information */
				theMasterConstrainedSlab[iMaster].which_bldg = iBldg;

				for ( i = 0 ; i < group_number ; i++) {
					if ( sharerproc_all[i*theNumberOfBuildings + iBldg] != 0 && i != myID) {
						sharer_count++;
					}
				}

				theMasterConstrainedSlab[iMaster].sharer_count = sharer_count;

				theMasterConstrainedSlab[iMaster].owner_ids =
						(int32_t*)malloc(sizeof(int32_t) * sharer_count);
				if ( theMasterConstrainedSlab[iMaster].owner_ids == NULL ) {
					solver_abort ( __FUNCTION_NAME, "NULL from malloc",
							"Error allocating theMasterConstrainedSlab[iMaster].owner_ids memory" );
				}

				theMasterConstrainedSlab[iMaster].node_counts =
						(int32_t*)malloc(sizeof(int32_t) * sharer_count);
				if ( theMasterConstrainedSlab[iMaster].node_counts == NULL ) {
					solver_abort ( __FUNCTION_NAME, "NULL from malloc",
							"Error allocating theMasterConstrainedSlab[iMaster].node_counts memory" );
				}


				theMasterConstrainedSlab[iMaster].average_values =
						(double*)malloc(sizeof(double) * 6 * theMasterConstrainedSlab[iMaster].l);
				if ( theMasterConstrainedSlab[iMaster].average_values == NULL ) {
					solver_abort ( __FUNCTION_NAME, "NULL from malloc",
							"Error allocating theMasterConstrainedSlab[iMaster].average_values memory" );
				}

				int total_nodes = 0;
				j = 0;
				for ( i = 0 ; i < group_number ; i++) {
					if ( sharerproc_all[i*theNumberOfBuildings + iBldg] != 0 && i != myID ) {

						theMasterConstrainedSlab[iMaster].owner_ids[j] = i;
						theMasterConstrainedSlab[iMaster].node_counts[j] =
								sharerproc_all[i*theNumberOfBuildings + iBldg];
						j++;
						total_nodes += sharerproc_all[i*theNumberOfBuildings + iBldg];
					}
				}

				//				printf(" \n myId = %d iBldg = %d sharercount = %d \n", myID, iBldg, theMasterConstrainedSlab[iMaster].sharer_count);
				//				j=0;
				//				for ( i = 0 ; i < group_number ; i++) {
				//					if ( sharerproc_all[i*theNumberOfBuildings + iBldg] != 0 && i != myID ) {
				//
				//						printf( " \n myId = %d iBldg = %d owner_id = %d node_counts = %d \n" , myID, iBldg,
				//								theMasterConstrainedSlab[iMaster].owner_ids[j],
				//								theMasterConstrainedSlab[iMaster].node_counts[j]);
				//						j++;
				//					}
				//				}


				/* sanity check */
				if ( total_nodes + masterproc[iBldg] != theMasterConstrainedSlab[iMaster].n *
						theMasterConstrainedSlab[iMaster].l) {
					solver_abort ( __FUNCTION_NAME, "sanity check fails",
							"total number of nodes is not correct" );
				}


				/* Incoming displacements */
				theMasterConstrainedSlab[iMaster].incoming_disp =
						(fvector_t**)malloc(sharer_count * sizeof(fvector_t*));
				for ( i = 0; i < sharer_count; i++) {
					theMasterConstrainedSlab[iMaster].incoming_disp[i] =
							(fvector_t*)malloc(theMasterConstrainedSlab[iMaster].node_counts[i] * sizeof(fvector_t));
				}

				/* Incoming displacements map*/
				theMasterConstrainedSlab[iMaster].incoming_disp_map =
						(intvector_t**)malloc(sharer_count * sizeof(intvector_t*));
				for ( i = 0; i < sharer_count; i++) {
					theMasterConstrainedSlab[iMaster].incoming_disp_map[i] =
							(intvector_t*)malloc(theMasterConstrainedSlab[iMaster].node_counts[i] * sizeof(intvector_t));
				}

				/* information for updating the displacements */

				/* Matrix L */
				theMasterConstrainedSlab[iMaster].local_ids_by_level =
						(lnid_t**)malloc(theMasterConstrainedSlab[iMaster].l * sizeof(lnid_t*));
				for ( i = 0; i < theMasterConstrainedSlab[iMaster].l; i++) {
					theMasterConstrainedSlab[iMaster].local_ids_by_level[i] =
							(lnid_t*)malloc(theMasterConstrainedSlab[iMaster].n * sizeof(lnid_t));
				}

				/* Matrix TA */
				theMasterConstrainedSlab[iMaster].tributary_areas =
						(double**)malloc(theMasterConstrainedSlab[iMaster].node_x * sizeof(double*));
				for ( i = 0; i < theMasterConstrainedSlab[iMaster].node_x; i++) {
					theMasterConstrainedSlab[iMaster].tributary_areas[i] =
							(double*)malloc( theMasterConstrainedSlab[iMaster].node_y * sizeof(double));
				}

				/* Matrix Dx */
				theMasterConstrainedSlab[iMaster].distance_to_centroid_xm =
						(double**)malloc(theMasterConstrainedSlab[iMaster].node_x * sizeof(double*));
				for ( i = 0; i < theMasterConstrainedSlab[iMaster].node_x; i++) {
					theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i] =
							(double*)malloc( theMasterConstrainedSlab[iMaster].node_y * sizeof(double));
				}

				/* Matrix Dy */
				theMasterConstrainedSlab[iMaster].distance_to_centroid_ym =
						(double**)malloc(theMasterConstrainedSlab[iMaster].node_x * sizeof(double*));
				for ( i = 0; i < theMasterConstrainedSlab[iMaster].node_x; i++) {
					theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i] =
							(double*)malloc( theMasterConstrainedSlab[iMaster].node_y * sizeof(double));
				}



				/* Find local ids of nodes at each level for each building.( level0 -> base ) - matrix L  */
				/* l is from basement to roof. If I do not have the node set -1 in matrix L */
				/* Also find tributary areas and Dx Dy and Ix Iy Ixy */

				double bldg_center_x_m, bldg_center_y_m;

				bldg_center_x_m = (theBuilding[iBldg].bounds.xmax +
						theBuilding[iBldg].bounds.xmin)/2;
				bldg_center_y_m = (theBuilding[iBldg].bounds.ymax +
						theBuilding[iBldg].bounds.ymin)/2;

				for ( l = 0; l < theMasterConstrainedSlab[iMaster].l; l++) {

					tick_t   z_tick;
					z_tick = (theSurfaceShift - theMinOctSizeMeters*l)/ticksize;

					for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x; i++ ) {
						for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y; j++) {

							double   dist_x_center_m, dist_y_center_m;
							double   x_m, y_m;

							lnid_t index;

							index = j+i*theMasterConstrainedSlab[iMaster].node_y;

							/* coordinate of the node */
							x_m = (theBuilding[iBldg].bounds.xmin + theMinOctSizeMeters*i);
							y_m = (theBuilding[iBldg].bounds.ymin + theMinOctSizeMeters*j);


							/* distance of the node to the geometric center of the building  */
							dist_x_center_m = x_m  - bldg_center_x_m;
							dist_y_center_m = y_m  - bldg_center_y_m;


							/* calculate these only once since are same for each level */
							if ( l == 0 ) {
								/* distance to centroid -- matrix Dx*/
								theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] =
										dist_x_center_m;

								/* distance to centroid -- matrix Dy*/
								theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] =
										dist_y_center_m;

								/* Also find tributary areas. TA */

								/* tributary_areas -- matrix TA*/
								theMasterConstrainedSlab[iMaster].tributary_areas[i][j] =
										theMinOctSizeMeters * theMinOctSizeMeters;

								if ( (y_m == theBuilding[iBldg].bounds.ymin) ||
										(y_m == theBuilding[iBldg].bounds.ymax) ||
										(x_m == theBuilding[iBldg].bounds.xmin) ||
										(x_m == theBuilding[iBldg].bounds.xmax) ) {
									/* tributary_areas -- matrix TA*/
									theMasterConstrainedSlab[iMaster].tributary_areas[i][j] =
											theMinOctSizeMeters * theMinOctSizeMeters / 2;
								}

								if ( ((y_m == theBuilding[iBldg].bounds.ymax) && (x_m == theBuilding[iBldg].bounds.xmin) ) ||
										((y_m == theBuilding[iBldg].bounds.ymin) && (x_m == theBuilding[iBldg].bounds.xmin) ) ||
										((y_m == theBuilding[iBldg].bounds.ymax) && (x_m == theBuilding[iBldg].bounds.xmax) ) ||
										((y_m == theBuilding[iBldg].bounds.ymin) && (x_m == theBuilding[iBldg].bounds.xmax) )
								) {

									/* tributary_areas -- matrix TA*/
									theMasterConstrainedSlab[iMaster].tributary_areas[i][j] =
											theMinOctSizeMeters * theMinOctSizeMeters / 4;
								}

								theMasterConstrainedSlab[iMaster].Ix += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
										theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
										theMasterConstrainedSlab[iMaster].tributary_areas[i][j];
								theMasterConstrainedSlab[iMaster].Iy += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
										theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
										theMasterConstrainedSlab[iMaster].tributary_areas[i][j];
								theMasterConstrainedSlab[iMaster].Ixy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
										theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
										theMasterConstrainedSlab[iMaster].tributary_areas[i][j];
							}

							/* -1 if I do not have the node */
							theMasterConstrainedSlab[iMaster].local_ids_by_level[l][index] = -1;

							/* Loop each node.
							 */
							for ( k = 0; k < myMesh->nharbored; ++k ) {
								if( myMesh->nodeTable[k].ismine == 1 ) {
									if ( myMesh->nodeTable[k].z == z_tick) {
										if ( myMesh->nodeTable[k].x == x_m/ticksize) {
											if ( myMesh->nodeTable[k].y == y_m/ticksize) {

												/* Local ids -- matrix L*/
												theMasterConstrainedSlab[iMaster].local_ids_by_level[l][index] = k;

												break;
											}
										}
									}
								}

							}
						}
					}
				}


				//				fprintf(stdout, "\n areas my id %d  building %d \n\n", myID, iBldg);
				//
				//				for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x; i++ ) {
				//					for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y; j++) {
				//
				//						fprintf(stdout, "%7.2f     ",
				//								theMasterConstrainedSlab[iMaster].tributary_areas[i][j]);
				//					}
				//					fprintf(stdout, "\n");
				//				}



				//				fprintf(stdout, "\n  local ids my id %d  building %d Ix  %f  Iy %f  Ixy  %f \n\n", myID, iBldg,
				//						theMasterConstrainedSlab[iMaster].Ix,
				//						theMasterConstrainedSlab[iMaster].Iy,
				//						theMasterConstrainedSlab[iMaster].Ixy);
				//
				//				for ( l = 0; l < theMasterConstrainedSlab[iMaster].l; l++) {
				//					fprintf(stdout, "\n  level %d \n\n", l);
				//
				//					for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x; i++ ) {
				//						for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y; j++) {
				//
				//							fprintf(stdout, "%5d   ",
				//									theMasterConstrainedSlab[iMaster].local_ids_by_level[l][j+i*theMasterConstrainedSlab[iMaster].node_y]);
				//
				//						}
				//						fprintf(stdout, "\n");
				//
				//					}
				//				}

				//				fprintf(stdout, "\n Trib Area my id %d  building %d \n\n", myID, iBldg);
				//
				//				for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x; i++ ) {
				//					for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y; j++) {
				//
				//						fprintf(stdout, "%7.2f     ",
				//								theMasterConstrainedSlab[iMaster].tributary_areas[i][j]);
				//					}
				//					fprintf(stdout, "\n");
				//				}

			}
		}
	}

	/* Sharers send the displacements to the master at each time step. Then
	 * they receive the average values and update the displacements.
	 */
	/* Allocate memory for the  theSharerConstrainedSlab */
	if (theNumberOfBuildingsSharer != 0 ) {

		theSharerConstrainedSlab =
				(sharer_constrained_slab_bldg_t*)malloc(sizeof(sharer_constrained_slab_bldg_t)*theNumberOfBuildingsSharer);
		if ( theSharerConstrainedSlab == NULL ) {
			solver_abort ( __FUNCTION_NAME, "NULL from malloc",
					"Error allocating theNumberOfBuildingsSharer memory" );
		}

		/* Fill theSharerConstrainedSlab struct*/

		for (iBldg = 0; iBldg < theNumberOfBuildings; iBldg++) {

			if( sharerproc[iBldg] != 0 && masterproc[iBldg] == 0 ) {

				iSharer++;

				/* These are general building information */
				theSharerConstrainedSlab[iSharer].l = (theBuilding[iBldg].height)/
						theMinOctSizeMeters + 1;
				nodes_x = (theBuilding[iBldg].bounds.xmax -
						theBuilding[iBldg].bounds.xmin)/theMinOctSizeMeters + 1;
				nodes_y = (theBuilding[iBldg].bounds.ymax -
						theBuilding[iBldg].bounds.ymin)/theMinOctSizeMeters + 1;

				theSharerConstrainedSlab[iSharer].node_x = nodes_x;
				theSharerConstrainedSlab[iSharer].node_y = nodes_y;
				theSharerConstrainedSlab[iSharer].n = nodes_y * nodes_x;


				/* These are communication information */
				theSharerConstrainedSlab[iSharer].which_bldg = iBldg;

				for ( i = 0 ; i < group_number ; i++) {
					if ( masterproc_all[i*theNumberOfBuildings + iBldg] != 0) {
						theSharerConstrainedSlab[iSharer].responsible_id = i;
						break;
					}
				}

				theSharerConstrainedSlab[iSharer].my_nodes_count = sharerproc[iBldg];

				/* Outgoing displacements map*/

				theSharerConstrainedSlab[iSharer].outgoing_disp_map =
						(intvector_t*)malloc(theSharerConstrainedSlab[iSharer].my_nodes_count * sizeof(intvector_t));

				theSharerConstrainedSlab[iSharer].outgoing_disp =
						(fvector_t*)malloc(theSharerConstrainedSlab[iSharer].my_nodes_count * sizeof(fvector_t));


				theSharerConstrainedSlab[iSharer].average_values =
						(double*)malloc(sizeof(double) * 6 * theSharerConstrainedSlab[iSharer].l);
				if ( theSharerConstrainedSlab[iSharer].average_values == NULL ) {
					solver_abort ( __FUNCTION_NAME, "NULL from malloc",
							"Error allocating theSharerConstrainedSlab[iSharer].average_values memory" );
				}

				/* information for updating the displacements */

				/* Matrix L */
				theSharerConstrainedSlab[iSharer].local_ids_by_level =
						(lnid_t**)malloc(theSharerConstrainedSlab[iSharer].l * sizeof(lnid_t*));
				for ( i = 0; i < theSharerConstrainedSlab[iSharer].l; i++) {
					theSharerConstrainedSlab[iSharer].local_ids_by_level[i] =
							(lnid_t*)malloc(theSharerConstrainedSlab[iSharer].n * sizeof(lnid_t));
				}

				/* Matrix Dx */
				theSharerConstrainedSlab[iSharer].distance_to_centroid_xm =
						(double**)malloc(theSharerConstrainedSlab[iSharer].node_x * sizeof(double*));
				for ( i = 0; i < theSharerConstrainedSlab[iSharer].node_x; i++) {
					theSharerConstrainedSlab[iSharer].distance_to_centroid_xm[i] =
							(double*)malloc( theSharerConstrainedSlab[iSharer].node_y * sizeof(double));
				}

				/* Matrix Dy */
				theSharerConstrainedSlab[iSharer].distance_to_centroid_ym =
						(double**)malloc(theSharerConstrainedSlab[iSharer].node_x * sizeof(double*));
				for ( i = 0; i < theSharerConstrainedSlab[iSharer].node_x; i++) {
					theSharerConstrainedSlab[iSharer].distance_to_centroid_ym[i] =
							(double*)malloc( theSharerConstrainedSlab[iSharer].node_y * sizeof(double));
				}



				/* Find local ids of nodes at each level for each building.( level0 -> base ) - matrix L  */
				/* l is from basement to roof. If I do not have the node set -1 in matrix L */
				/* Also find Dx Dy  and outgoing_disp_map*/
				int    counter = 0;
				double bldg_center_x_m, bldg_center_y_m;

				bldg_center_x_m = (theBuilding[iBldg].bounds.xmax +
						theBuilding[iBldg].bounds.xmin)/2;
				bldg_center_y_m = (theBuilding[iBldg].bounds.ymax +
						theBuilding[iBldg].bounds.ymin)/2;

				for ( l = 0; l < theSharerConstrainedSlab[iSharer].l; l++) {

					tick_t   z_tick;
					z_tick = (theSurfaceShift - theMinOctSizeMeters*l)/ticksize;

					for ( i = 0; i <  theSharerConstrainedSlab[iSharer].node_x; i++ ) {
						for ( j = 0; j <  theSharerConstrainedSlab[iSharer].node_y; j++) {

							double   dist_x_center_m, dist_y_center_m;
							double   x_m, y_m;

							lnid_t index;

							index = j+i*theSharerConstrainedSlab[iSharer].node_y;

							/* coordinate of the node */
							x_m = (theBuilding[iBldg].bounds.xmin + theMinOctSizeMeters*i);
							y_m = (theBuilding[iBldg].bounds.ymin + theMinOctSizeMeters*j);


							/* distance of the node to the geometric center of the building  */
							dist_x_center_m = x_m  - bldg_center_x_m;
							dist_y_center_m = y_m  - bldg_center_y_m;


							/* calculate these only once since are same for each level */
							if ( l == 0 ) {
								/* distance to centroid -- matrix Dx*/
								theSharerConstrainedSlab[iSharer].distance_to_centroid_xm[i][j] =
										dist_x_center_m;

								/* distance to centroid -- matrix Dy*/
								theSharerConstrainedSlab[iSharer].distance_to_centroid_ym[i][j] =
										dist_y_center_m;

							}

							/* -1 if I do not have the node */
							theSharerConstrainedSlab[iSharer].local_ids_by_level[l][index] = -1;

							/* Loop each node.
							 */
							for ( k = 0; k < myMesh->nharbored; ++k ) {
								if( myMesh->nodeTable[k].ismine == 1 ) {
									if ( myMesh->nodeTable[k].z == z_tick) {
										if ( myMesh->nodeTable[k].x == x_m/ticksize) {
											if ( myMesh->nodeTable[k].y == y_m/ticksize) {

												/* Local ids -- matrix L*/
												theSharerConstrainedSlab[iSharer].local_ids_by_level[l][index] = k;

												/* outgoing_disp_map*/

												theSharerConstrainedSlab[iSharer].outgoing_disp_map[counter].f[0] = l;
												theSharerConstrainedSlab[iSharer].outgoing_disp_map[counter].f[1] = i;
												theSharerConstrainedSlab[iSharer].outgoing_disp_map[counter].f[2] = j;

												counter++;
												break;
											}
										}
									}
								}

							}
						}
					}
				}


				//				fprintf(stdout, "\n Dy my id %d  building %d \n\n", myID, iBldg);
				//
				//				for ( i = 0; i <  theSharerConstrainedSlab[iSharer].node_x; i++ ) {
				//					for ( j = 0; j <  theSharerConstrainedSlab[iSharer].node_y; j++) {
				//
				//						fprintf(stdout, "%7.2f     ",
				//								theSharerConstrainedSlab[iSharer].distance_to_centroid_ym[i][j]);
				//					}
				//					fprintf(stdout, "\n");
				//				}



				//				fprintf(stdout, "\n Dy my id %d  building %d \n\n", myID, iBldg);
				//				for ( l = 0; l < theSharerConstrainedSlab[iSharer].l; l++) {
				//					fprintf(stdout, "\n  level %d \n\n", l);
				//
				//					for ( i = 0; i <  theSharerConstrainedSlab[iSharer].node_x; i++ ) {
				//						for ( j = 0; j <  theSharerConstrainedSlab[iSharer].node_y; j++) {
				//
				//							fprintf(stdout, "%5d   ",
				//									theSharerConstrainedSlab[iSharer].local_ids_by_level[l][j+i*theSharerConstrainedSlab[iSharer].node_y]);
				//
				//						}
				//						fprintf(stdout, "\n");
				//
				//					}
				//				}


//				if (myID == 4) {
//					fprintf(stdout, "\n my id %d  building %d  master %d count %d \n\n",
//							myID, iBldg,
//							theSharerConstrainedSlab[iSharer].responsible_id,
//							theSharerConstrainedSlab[iSharer].my_nodes_count);
//
//					for ( i = 0; i <  theSharerConstrainedSlab[iSharer].my_nodes_count; i++ ) {
//
//						fprintf(stdout, " %d  %d  %d \n",
//								theSharerConstrainedSlab[iSharer].outgoing_disp_map[i].f[0],
//								theSharerConstrainedSlab[iSharer].outgoing_disp_map[i].f[1],
//								theSharerConstrainedSlab[iSharer].outgoing_disp_map[i].f[2]);
//					}
//				}

			}

		}
	}


	/* communication between sharers and masters */
	MPI_Status   status;
	MPI_Request  isendreqs;
	MPI_Datatype index_sharer;

	MPI_Type_contiguous(3, MPI_INT, &index_sharer);
	MPI_Type_commit(&index_sharer);


	/* First, sharers send their information to masters */

	for (iSharer = 0; iSharer < theNumberOfBuildingsSharer; iSharer++) {

		MPI_Isend(theSharerConstrainedSlab[iSharer].outgoing_disp_map,
				theSharerConstrainedSlab[iSharer].my_nodes_count,
				index_sharer,
				theSharerConstrainedSlab[iSharer].responsible_id,
				theSharerConstrainedSlab[iSharer].which_bldg,
				comm_solver, &isendreqs);
	}


	/* Second, masters receive the information */

	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {

		for (i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++) {

			MPI_Recv(theMasterConstrainedSlab[iMaster].incoming_disp_map[i],
					theMasterConstrainedSlab[iMaster].node_counts[i],
					index_sharer,
					theMasterConstrainedSlab[iMaster].owner_ids[i],
					theMasterConstrainedSlab[iMaster].which_bldg,
					comm_solver, &status);
		}
	}

	MPI_Barrier(comm_solver);


//	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {
//		for (i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++) {
//
//			fprintf(stdout, "\n my id %d  building %d  sharer %d count %d \n\n",
//					myID, theMasterConstrainedSlab[iMaster].which_bldg,
//					theMasterConstrainedSlab[iMaster].owner_ids[i],
//					theMasterConstrainedSlab[iMaster].node_counts[i]);
//
//			for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_counts[i]; j++ ) {
//
//				fprintf(stdout, " %d  %d  %d \n",
//						theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[0],
//						theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[1],
//						theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[2]);
//			}
//		}
//	}


	return;
}


double interpolatedisp ( double low, double high, double frac) {

    return low + frac * (high - low);
}

fvector_t bldgs_get_base_disp ( int bldg, double simDT, int step ) {

    double    truetime, supposedstep, frac;
    int       i, lowstep, highstep;
    fvector_t lowdisp, highdisp, disp;

    truetime     = step * simDT;
    supposedstep = truetime / (double)theBaseFixedDT;

    lowstep  = (int)supposedstep;
    highstep = lowstep + 1;
    frac     = supposedstep - (double)lowstep;

    lowdisp  = theBaseSignals[bldg][lowstep];
    highdisp = theBaseSignals[bldg][highstep];

    for ( i = 0; i < 3; i++ ) {
        disp.f[i] = interpolatedisp( lowdisp.f[i], highdisp.f[i], frac);
    }

    return disp;
}

void bldgs_load_fixedbase_disps ( mysolver_t* solver, double simDT, int step ) {

    lnid_t bnode;
    lnid_t nindex;

    for ( bnode = 0; bnode < theBaseNodesCount; bnode++ ) {

        nindex = theBaseNodes[bnode].nindex;
        int bldg = theBaseNodes[bnode].bldg-1;
        fvector_t* dis = solver->tm2 + nindex;
        *dis = bldgs_get_base_disp ( bldg, simDT, step );
    }

    return;
}

/* Displacements are calculated using the T * average values(A)  */

void bldgs_update_constrainedslabs_disps ( mysolver_t* solver, double simDT, int step, int32_t myID ) {

	int32_t  iSharer, iMaster, i, j, l;

	MPI_Status  status;
	MPI_Request isendreqs;
	MPI_Datatype dis_sharer;

	MPI_Type_contiguous(3, MPI_DOUBLE, &dis_sharer);
	MPI_Type_commit(&dis_sharer);

	/* First sharers send displacements */

	for (iSharer = 0; iSharer < theNumberOfBuildingsSharer; iSharer++) {

		int counter = 0;

		for ( l = 0; l < theSharerConstrainedSlab[iSharer].l; l++) {
			for ( i = 0; i <  theSharerConstrainedSlab[iSharer].node_x ; i++ ) {
				for ( j = 0; j <  theSharerConstrainedSlab[iSharer].node_y ; j++) {
					lnid_t  nindex, index;
					fvector_t* dis;

					index = j + i*theSharerConstrainedSlab[iSharer].node_y;

					nindex = theSharerConstrainedSlab[iSharer].local_ids_by_level[l][index];

					if ( nindex != -1 ) {
						dis = solver->tm2 + nindex;

						theSharerConstrainedSlab[iSharer].outgoing_disp[counter] = *(dis);
						counter++;
					}
				}
			}
		}


		MPI_Isend(theSharerConstrainedSlab[iSharer].outgoing_disp,
				theSharerConstrainedSlab[iSharer].my_nodes_count,
				dis_sharer,
				theSharerConstrainedSlab[iSharer].responsible_id,
				theSharerConstrainedSlab[iSharer].which_bldg,
				comm_solver, &isendreqs);
	}

	/* Second, masters receive the information */

	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {

		for (i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++) {

			MPI_Recv(theMasterConstrainedSlab[iMaster].incoming_disp[i],
					theMasterConstrainedSlab[iMaster].node_counts[i],
					dis_sharer,
					theMasterConstrainedSlab[iMaster].owner_ids[i],
					theMasterConstrainedSlab[iMaster].which_bldg,
					comm_solver, &status);

		}
	}


//	if(step == 60)
//	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {
//		for (i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++) {
//
//			fprintf(stdout, "\n my id %d  building %d  sharer %d count %d timestep %d\n\n",
//					myID, theMasterConstrainedSlab[iMaster].which_bldg,
//					theMasterConstrainedSlab[iMaster].owner_ids[i],
//					theMasterConstrainedSlab[iMaster].node_counts[i],
//					step);
//
//			for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_counts[i]; j++ ) {
//
//				fprintf(stdout, " %8e % 8e % 8e  %f  %f \n",
//						theMasterConstrainedSlab[iMaster].incoming_disp[i][j].f[0],
//						theMasterConstrainedSlab[iMaster].incoming_disp[i][j].f[1],
//						theMasterConstrainedSlab[iMaster].incoming_disp[i][j].f[2]);
//			}
//		}
//	}


	//MPI_Barrier(comm_solver);

	/* Calculate average values if I am the master */

	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {

		/* Calculate for each level */

		for ( l = 0; l < theMasterConstrainedSlab[iMaster].l; l++) {

			double    average_values[6];
			/* order: x, y, z, theta_z, theta_x, theta_y */
			int  m;

			for ( m = 0; m < 6; m++) {
				average_values[m] = 0;
			}

			/* Moment around the centeral node */
			double   Mwy = 0, Mwx = 0;
			double   Muy = 0, Moy = 0;
			double   Mvx = 0, Mox = 0;

			double   Iy = 0, Ix = 0, Ixy = 0;
			double   total_area;

			Ix  = theMasterConstrainedSlab[iMaster].Ix;
			Iy  = theMasterConstrainedSlab[iMaster].Iy;
			Ixy = theMasterConstrainedSlab[iMaster].Ixy;

			total_area = theMasterConstrainedSlab[iMaster].area;

			/* Contribution from my nodes  - average translations */
			for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x ; i++ ) {
				for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y ; j++) {
					lnid_t  nindex, index;
					fvector_t* dis;
					double trib_area;

					index = j + i*theMasterConstrainedSlab[iMaster].node_y;
					nindex = theMasterConstrainedSlab[iMaster].local_ids_by_level[l][index];

					/* First my nodes contribute */
					if (nindex != -1 ) {
						dis = solver->tm2 + nindex;
						trib_area = theMasterConstrainedSlab[iMaster].tributary_areas[i][j];

						average_values[0] += dis->f[0] * trib_area ; /* average x */
						average_values[1] += dis->f[1] * trib_area ; /* average y */
						average_values[2] += dis->f[2] * trib_area ; /* average z */
					}
				}
			}


			/* Contribution from sharers nodes  -  average translations */
			for ( i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++ ) {
				for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_counts[i]; j++ ) {
					fvector_t dis;
					int l_index, i_index, j_index;
					double trib_area;

					l_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[0];
					i_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[1];
					j_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[2];

					if ( l_index == l ) {
						dis = theMasterConstrainedSlab[iMaster].incoming_disp[i][j];
						trib_area = theMasterConstrainedSlab[iMaster].tributary_areas[i_index][j_index];

						average_values[0] += dis.f[0] * trib_area ; /* average x */
						average_values[1] += dis.f[1] * trib_area ; /* average y */
						average_values[2] += dis.f[2] * trib_area ; /* average z */
					}
				}
			}

			average_values[0] = average_values[0] / total_area; /* average x */
			average_values[1] = average_values[1] / total_area; /* average y */
			average_values[2] = average_values[2] / total_area; /* average z */

			theMasterConstrainedSlab[iMaster].average_values[6*l + 0] = average_values[0];
			theMasterConstrainedSlab[iMaster].average_values[6*l + 1] = average_values[1];
			theMasterConstrainedSlab[iMaster].average_values[6*l + 2] = average_values[2];

			/* Contribution from my nodes -  average rotations */
			for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x ; i++ ) {
				for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y ; j++) {
					lnid_t  nindex, index;
					fvector_t* dis;
					double trib_area;

					index = j + i*theMasterConstrainedSlab[iMaster].node_y;
					nindex = theMasterConstrainedSlab[iMaster].local_ids_by_level[l][index];

					/* First my nodes contribute */
					if (nindex != -1 ) {
						dis = solver->tm2 + nindex;
						trib_area = theMasterConstrainedSlab[iMaster].tributary_areas[i][j];

						Muy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
								trib_area * (dis->f[0]);
						Mvx += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
								trib_area * (dis->f[1]);


						Mox += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
								trib_area * average_values[1];

						Moy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
								trib_area * average_values[0];


						Mwx += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] *
								trib_area * (dis->f[2]);

						Mwy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
								trib_area * (dis->f[2]);

					}
				}
			}

			/* Contribution from sharers nodes  -  average rotations */
			for ( i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++ ) {
				for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_counts[i]; j++ ) {
					fvector_t dis;
					int l_index, i_index, j_index;
					double trib_area;

					l_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[0];
					i_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[1];
					j_index = theMasterConstrainedSlab[iMaster].incoming_disp_map[i][j].f[2];

					if ( l_index == l ) {
						dis = theMasterConstrainedSlab[iMaster].incoming_disp[i][j];
						trib_area = theMasterConstrainedSlab[iMaster].tributary_areas[i_index][j_index];

						Muy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i_index][j_index] *
								trib_area * (dis.f[0]);
						Mvx += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i_index][j_index] *
								trib_area * (dis.f[1]);


						Mox += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i_index][j_index] *
								trib_area * theMasterConstrainedSlab[iMaster].average_values[6*l + 1];

						Moy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i_index][j_index] *
								trib_area * theMasterConstrainedSlab[iMaster].average_values[6*l + 0];


						Mwx += theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i_index][j_index] *
								trib_area * (dis.f[2]);

						Mwy += theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i_index][j_index] *
								trib_area * (dis.f[2]);
					}
				}
			}


			average_values[3] = ( (Muy) / Ix + (- Mvx) / Iy ) / 2.0;  /* average theta_z */
			average_values[4] =  Mwy / Ix;  /* average theta_x */
			average_values[5] = -1.0 * Mwx / Iy;  /* average theta_y */

			//average_values[3] = ( (Muy - Moy) / Ix + (Mox - Mvx) / Iy ) / 2.0;  /* average theta_z */
			//average_values[4] = (Iy*Mwy  - Ixy*Mwx)/(Ix*Iy - Ixy*Ixy);
			//average_values[5] = (Ixy*Mwy - Ix*Mwx)/(Ix*Iy - Ixy*Ixy);

			theMasterConstrainedSlab[iMaster].average_values[6*l + 3] = average_values[3];
			theMasterConstrainedSlab[iMaster].average_values[6*l + 4] = average_values[4];
			theMasterConstrainedSlab[iMaster].average_values[6*l + 5] = average_values[5];
		}
	}


	/* Now, masters sends the average dis to sharers */

	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {

		for (i = 0; i < theMasterConstrainedSlab[iMaster].sharer_count; i++) {

			MPI_Isend(theMasterConstrainedSlab[iMaster].average_values,
					theMasterConstrainedSlab[iMaster].l * 6,
					MPI_DOUBLE,
					theMasterConstrainedSlab[iMaster].owner_ids[i],
					theMasterConstrainedSlab[iMaster].which_bldg,
					comm_solver, &isendreqs);
		}
	}


	/* Second sharers receives the average dis */

	for (iSharer = 0; iSharer < theNumberOfBuildingsSharer; iSharer++) {

		MPI_Recv(theSharerConstrainedSlab[iSharer].average_values,
				theSharerConstrainedSlab[iSharer].l * 6,
				MPI_DOUBLE,
				theSharerConstrainedSlab[iSharer].responsible_id,
				theSharerConstrainedSlab[iSharer].which_bldg,
				comm_solver, &status);
	}



	/* Masters and sharers update their displacements */

	/* First masters update displacements */

	for (iMaster = 0; iMaster < theNumberOfBuildingsMaster; iMaster++) {

		for ( l = 0; l < theMasterConstrainedSlab[iMaster].l; l++) {
			for ( i = 0; i <  theMasterConstrainedSlab[iMaster].node_x ; i++ ) {
				for ( j = 0; j <  theMasterConstrainedSlab[iMaster].node_y ; j++) {
					lnid_t  nindex, index;
					fvector_t* dis;

					index = j + i*theMasterConstrainedSlab[iMaster].node_y;

					nindex = theMasterConstrainedSlab[iMaster].local_ids_by_level[l][index];

					if ( nindex != -1 ) {
						dis = solver->tm2 + nindex;

						dis->f[0] = theMasterConstrainedSlab[iMaster].average_values[6*l + 0] +
								theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
								theMasterConstrainedSlab[iMaster].average_values[6*l + 3];

						dis->f[1] = theMasterConstrainedSlab[iMaster].average_values[6*l + 1] +
								theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] * -1 *
								theMasterConstrainedSlab[iMaster].average_values[6*l + 3];

						dis->f[2] = theMasterConstrainedSlab[iMaster].average_values[6*l + 2] +
								theMasterConstrainedSlab[iMaster].distance_to_centroid_ym[i][j] *
								theMasterConstrainedSlab[iMaster].average_values[6*l + 4] +
								theMasterConstrainedSlab[iMaster].distance_to_centroid_xm[i][j] * -1 *
								theMasterConstrainedSlab[iMaster].average_values[6*l + 5] ;
					}
				}
			}
		}
	}


	/* Second sharers update displacements */

	for (iSharer = 0; iSharer < theNumberOfBuildingsSharer; iSharer++) {

		for ( l = 0; l < theSharerConstrainedSlab[iSharer].l; l++) {
			for ( i = 0; i <  theSharerConstrainedSlab[iSharer].node_x ; i++ ) {
				for ( j = 0; j <  theSharerConstrainedSlab[iSharer].node_y ; j++) {
					lnid_t  nindex, index;
					fvector_t* dis;

					index = j + i*theSharerConstrainedSlab[iSharer].node_y;

					nindex = theSharerConstrainedSlab[iSharer].local_ids_by_level[l][index];

					if ( nindex != -1 ) {
						dis = solver->tm2 + nindex;

						dis->f[0] = theSharerConstrainedSlab[iSharer].average_values[6*l + 0] +
								theSharerConstrainedSlab[iSharer].distance_to_centroid_ym[i][j] *
								theSharerConstrainedSlab[iSharer].average_values[6*l + 3];

						dis->f[1] = theSharerConstrainedSlab[iSharer].average_values[6*l + 1] +
								theSharerConstrainedSlab[iSharer].distance_to_centroid_xm[i][j] * -1 *
								theSharerConstrainedSlab[iSharer].average_values[6*l + 3];

						dis->f[2] = theSharerConstrainedSlab[iSharer].average_values[6*l + 2] +
								theSharerConstrainedSlab[iSharer].distance_to_centroid_ym[i][j] *
								theSharerConstrainedSlab[iSharer].average_values[6*l + 4] +
								theSharerConstrainedSlab[iSharer].distance_to_centroid_xm[i][j] * -1 *
								theSharerConstrainedSlab[iSharer].average_values[6*l + 5] ;
					}
				}
			}
		}
	}



	//	for ( iBldg = 0; iBldg < theNumberOfBuildings; iBldg++ ) {
	//		for ( l = 0; l < theConstrainedSlab[iBldg].l; l++) {
	//

	//
	//
	//			for ( i = 0; i <  theConstrainedSlab[iBldg].node_x ; i++ ) {
	//				for ( j = 0; j <  theConstrainedSlab[iBldg].node_y ; j++) {
	//					lnid_t  nindex;
	//					fvector_t* dis_slab_node;
	//
	//					nindex = theConstrainedSlab[iBldg].local_ids_by_level[l][counter];
	//					dis_slab_node = solver->tm2 + nindex;
	//
	//					dis_slab_node->f[0] = average_values[0] +
	//							theConstrainedSlab[iBldg].transformation_matrix[counter*3][3] * average_values[3]  ;
	//					dis_slab_node->f[1] = average_values[1] +
	//							theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][3] * average_values[3] ;
	//					dis_slab_node->f[2] = average_values[2] +
	//							theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][4] * average_values[4] +
	//							theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][5] * average_values[5] ;
	//
	//					counter++;
	//				}
	//			}
	//		}
	//	}

	return;
}

/* -------------------------------------------------------------------------- */
/*                             Dimensions Adjust                              */
/* -------------------------------------------------------------------------- */

double adjust (double input) {

    return theMinOctSizeMeters * round(input / theMinOctSizeMeters);
}

/**
 * Adjust the surface shift and the buildings' boundaries to the etree grid.
 * The user is responsible for providing a valid value for the minimum
 * octant size to be used as reference.
 *
 */
void adjust_dimensions ( ) {

	int iBldg,iPDown;

	theSurfaceShift = adjust( theSurfaceShift );

	for (iBldg = 0; iBldg < theNumberOfBuildings; iBldg++) {

		/* Compute Z limits based on surface shift and building vertical info */
		theBuildingsZMin[iBldg] = theSurfaceShift - theBuildingsHeight[iBldg];
		theBuildingsZMax[iBldg] = theSurfaceShift + theBuildingsDepth[iBldg];

		/* Correct Z-min to avoid negative coords. This occurs if the height
		 * of a building is greater than the surface shift (pushdown) */
		if ( theBuildingsZMin[iBldg] < 0 ) {
			theBuildingsZMin[iBldg] = 0;
		}

		theBuildingsXMin   [ iBldg ] = adjust( theBuildingsXMin   [ iBldg ] );
		theBuildingsXMax   [ iBldg ] = adjust( theBuildingsXMax   [ iBldg ] );
		theBuildingsYMin   [ iBldg ] = adjust( theBuildingsYMin   [ iBldg ] );
		theBuildingsYMax   [ iBldg ] = adjust( theBuildingsYMax   [ iBldg ] );
		theBuildingsZMin   [ iBldg ] = adjust( theBuildingsZMin   [ iBldg ] );
		theBuildingsZMax   [ iBldg ] = adjust( theBuildingsZMax   [ iBldg ] );
		theBuildingsDepth  [ iBldg ] = adjust( theBuildingsDepth  [ iBldg ] );
		theBuildingsHeight [ iBldg ] = adjust( theBuildingsHeight [ iBldg ] );

	}

	for (iPDown = 0; iPDown < theNumberOfPushdowns; iPDown++) {

		/* Compute Z limits based on surface shift and building vertical info */
		thePushDownsZMin[iPDown] = theSurfaceShift - thePushDownsHeight[iPDown];
		thePushDownsZMax[iPDown] = theSurfaceShift + thePushDownsDepth[iPDown];

		/* Correct Z-min to avoid negative coords. This occurs if the height
		 * of a pushdown is greater than the surface shift (pushdown) */
		if ( thePushDownsZMin[iPDown] < 0 ) {
			thePushDownsZMin[iPDown] = 0;
		}

		thePushDownsXMin   [ iPDown ] = adjust( thePushDownsXMin   [ iPDown ] );
		thePushDownsXMax   [ iPDown ] = adjust( thePushDownsXMax   [ iPDown ] );
		thePushDownsYMin   [ iPDown ] = adjust( thePushDownsYMin   [ iPDown ] );
		thePushDownsYMax   [ iPDown ] = adjust( thePushDownsYMax   [ iPDown ] );
		thePushDownsZMin   [ iPDown ] = adjust( thePushDownsZMin   [ iPDown ] );
		thePushDownsZMax   [ iPDown ] = adjust( thePushDownsZMax   [ iPDown ] );
		thePushDownsDepth  [ iPDown ] = adjust( thePushDownsDepth  [ iPDown ] );
		thePushDownsHeight [ iPDown ] = adjust( thePushDownsHeight [ iPDown ] );

	}

	return;
}

/* -------------------------------------------------------------------------- */
/*                                  Finalize                                  */
/* -------------------------------------------------------------------------- */

void bldgs_finalize() {

	if ( theNumberOfBuildings > 0 ) {
		free( theBuilding );
	}

	if ( theNumberOfPushdowns > 0 ) {
		free( thePushdown );
	}
	return;
}

/* -------------------------------------------------------------------------- */
/*                                  Obsolete                                  */
/* -------------------------------------------------------------------------- */

/**
 * Calls suitable functions based upon whether element is in foundation or not.
 * In any case, returns 1 if element is needed to be expanded or 0 otherwise.
 */

//int buildingmagic ( octant_t *leaf,  double ticksize,
//                    edata_t  *edata, int    theNumberOfBuildings ,
//                    double    theFactor )
//{
//    int returnval;
//
//    returnval = bldgs_toexpand ( leaf, ticksize, (edata_t *)edata,
//                                   theNumberOfBuildings, theFactor );
//    if ( returnval != -1 ) {
//        return returnval;
//    } else {
//        return vsrule( edata, theFactor );
//    }
//}

/**
 * Return 1 if the octant's origin is in the air, 0 otherwise.
 */
//int bldgs_airoctant ( octant_t *leaf, double ticksize ) {
//
//    int    i, air;
//    double x, y, z;
//
//    z = leaf->lz * ticksize;
//
//    /* an octant may be in the air only if it is above the surface */
//    if ( z >= theSurfaceShift ) {
//        return 0;
//    }
//
//    x = leaf->lx * ticksize;
//    y = leaf->ly * ticksize;
//
//    air = 0;
//    for ( i = 0; i < theNumberOfBuildings; i++ ) {
//        air += outofbuilding(x, y, z, i);
//    }
//
//    /* an octant is in the air only if it is out of bounds for all buildings */
//    if ( air == theNumberOfBuildings ) {
//        return 1;
//    }
//
//    return 0;
//}

/**
 * Returns 1 if a point is out of the i-th building bounds, 0 otherwise.
 */
//int outofbuilding ( double x, double y, double z, int i )
//{
//    bounds_t bounds = get_bldgbounds(i);
//
//    /* Air above a building */
//    if ( ( x >= bounds.xmin ) &&
//         ( x <= bounds.xmax ) ) {
//
//        if ( ( y >= bounds.ymin ) &&
//             ( y <= bounds.ymax ) ) {
//
//            if ( z < bounds.zmin ) {
//                return 1;
//            }
//        }
//    }
//
//    /* Air around buildings and above surface */
//    if ( ( x < bounds.xmin ) &&
//         ( x > bounds.xmax ) ) {
//
//        if ( ( y < bounds.ymin ) &&
//             ( y > bounds.ymax ) ) {
//
//            if ( z < theSurfaceShift ) {
//                return 1;
//            }
//        }
//    }
//
//    return 0;
//}

