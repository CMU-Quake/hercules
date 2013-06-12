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
	 * variation in rho along the EW direction (y coordinate in Hercules).
	 * Nodes at the leftmost are assigned the values in bldgprops_left and
	 * nodes at the rightmost have the values in bldgprops_right. The values
	 * for the intermediate nodes are linearly interpolated. In other words, mass
	 * center is shifted wrt to geometric center.
	 * However, the distribution is different for Vp and Vs. Stiffness center is
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



/* rows have information for each node at a given level in a building.
 * columns have the same type of nodal information for different levels.
 * l = # of plan-levels in each building
 * n = # of nodes in each level
 * matrix L, T and R are constant for each time step.
 */
typedef struct constrained_slab_bldg {
	int l; /* # of plan-levels in each building */
	int n; /* # of nodes in each level node_x*node_y */
	int node_x; /* # of nodes in x direction for each building */
	int node_y; /* # of nodes in y direction for each building */
	double x_bldg_length; /* length of building in x (m) - NS */
	double y_bldg_length; /* length of building in y (m) - EW */
	double Ix; /* area moment of inertia of the building top view wrt x axis at
	 the center of building */
	double Iy; /* area moment of inertia of the building top view wrt y axis at
	 the center of building */
	double Ixy; /* area moment of inertia of the building top view */

	lnid_t**  local_ids_by_level;  /* matrix L - local ids of each point at
	each plan-level of buildings. dim : l x n */
	double**  transformation_matrix; /* matrix T - transformation matrix. same
	 for each level in a building. dim : 3n x 6 */
	lnid_t**  local_ids_reference_nodes; /* matrix R - local ids of 5 reference
	 nodes at each plan-level. l x 5 - (center node is also included) */

	double**  tributary_areas; /* matrix TA - tributary areas matrix. same
		 for each level in a building. dim : node_x x node_y */

	double**  distance_to_centroid_xm; /* matrix Dx - distance (x) matrix wrt to
	the centroid. same or each level in a building in meter. dim : node_x x node_y */

	double**  distance_to_centroid_ym; /* matrix Dy - distance (y) matrix wrt to
	the centroid. same or each level in a building in meter. dim : node_x x node_y */

} constrained_slab_bldg_t;


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
static constrained_slab_bldg_t  *theConstrainedSlab;

/* Permanent */

static int      theNumberOfBuildings;
static int      theNumberOfPushdowns;
static double   theMinOctSizeMeters;
static double   theSurfaceShift = 0;
static bldg_t   *theBuilding;
static pdown_t  *thePushdown;

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
	} else {
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
	}
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

						/* non-uniform mass distribution in the y direction only (EW). */
						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						increment_rho = (theBuilding[i].bldgprops_right.rho - theBuilding[i].bldgprops_left.rho)/n;
						location = (y_physical - theBuilding[i].bounds.ymin)/theMinOctSizeMeters;
						edata->rho = theBuilding[i].bldgprops_left.rho + increment_rho*location + increment_rho/2 ;

						/* non-uniform Vp and Vs distribution in y direction only. - but symmetric */
						n = (theBuilding[i].bounds.ymax - theBuilding[i].bounds.ymin)/theMinOctSizeMeters/2;
						increment_Vp = (theBuilding[i].bldgprops_right.Vp - theBuilding[i].bldgprops_left.Vp)/n;
						increment_Vs = (theBuilding[i].bldgprops_right.Vs - theBuilding[i].bldgprops_left.Vs)/n;
						location = (y_physical - (theBuilding[i].bounds.ymin + theBuilding[i].bounds.ymax)/2 )/theMinOctSizeMeters;
						if(location < 0) location++;
						location = abs(location);

						edata->Vp = theBuilding[i].bldgprops_right.Vp - increment_Vp*location - increment_Vp/2 ;
					    edata->Vs = theBuilding[i].bldgprops_right.Vs - increment_Vs*location - increment_Vs/2 ;

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

void constrained_slabs_init ( mesh_t *myMesh, double simTime ) {

	int iBldg,i,j;
	int nodes_x, nodes_y;
	double ticksize;
	ticksize = myMesh->ticksize;

	/* Allocate memory for the  theConstrainedSlab */
	theConstrainedSlab =
			(constrained_slab_bldg_t*)malloc(sizeof(constrained_slab_bldg_t)*theNumberOfBuildings);
	if ( theConstrainedSlab == NULL ) {
		solver_abort ( __FUNCTION_NAME, "NULL from malloc",
				"Error allocating theConstrainedSlab memory" );
	}

	/* Fill theConstrainedSlab struct*/

	for (iBldg = 0; iBldg < theNumberOfBuildings; iBldg++) {
		double x_m;
		double y_m;

		theConstrainedSlab[iBldg].l = (theBuilding[iBldg].height)/
				theMinOctSizeMeters + 1;
		nodes_x = (theBuilding[iBldg].bounds.xmax -
				theBuilding[iBldg].bounds.xmin)/theMinOctSizeMeters + 1;
		nodes_y = (theBuilding[iBldg].bounds.ymax -
				theBuilding[iBldg].bounds.ymin)/theMinOctSizeMeters + 1;

		theConstrainedSlab[iBldg].node_x = nodes_x;
		theConstrainedSlab[iBldg].node_y = nodes_y;
		theConstrainedSlab[iBldg].n = nodes_y * nodes_x;
		theConstrainedSlab[iBldg].x_bldg_length =
				(theBuilding[iBldg].bounds.xmax - theBuilding[iBldg].bounds.xmin);
		theConstrainedSlab[iBldg].y_bldg_length =
				(theBuilding[iBldg].bounds.ymax - theBuilding[iBldg].bounds.ymin);

		x_m = theConstrainedSlab[iBldg].x_bldg_length;
		y_m = theConstrainedSlab[iBldg].y_bldg_length;

		theConstrainedSlab[iBldg].Ix  = x_m*y_m*y_m*y_m/12;
		theConstrainedSlab[iBldg].Iy  = y_m * x_m * x_m * x_m/12;
		theConstrainedSlab[iBldg].Ixy = 0;

		printf ("\n l=%d nodes_x=%d nodes_y=%d n=%d x_bldg_length=%f y_bldg_length=%f"
				"Ix=%f Iy=%f Ixy=%f  \n",
				theConstrainedSlab[iBldg].l, theConstrainedSlab[iBldg].node_x,
				theConstrainedSlab[iBldg].node_y,theConstrainedSlab[iBldg].n,
				theConstrainedSlab[iBldg].x_bldg_length ,
				theConstrainedSlab[iBldg].y_bldg_length,
				theConstrainedSlab[iBldg].Ix,
				theConstrainedSlab[iBldg].Iy,
				theConstrainedSlab[iBldg].Ixy );
	}

	/* Allocate memory for the two dimensional arrays; L T R  and TA */
	for (iBldg = 0; iBldg < theNumberOfBuildings; iBldg++) {

		/* Matrix L */
		theConstrainedSlab[iBldg].local_ids_by_level =
				(lnid_t**)malloc(theConstrainedSlab[iBldg].l * sizeof(lnid_t*));
		for ( i = 0; i < theConstrainedSlab[iBldg].l; i++) {
			theConstrainedSlab[iBldg].local_ids_by_level[i] =
					(lnid_t*)malloc(theConstrainedSlab[iBldg].n * sizeof(lnid_t));
		}

		/* Matrix T */
		theConstrainedSlab[iBldg].transformation_matrix =
				(double**)malloc(theConstrainedSlab[iBldg].n * 3 * sizeof(double*));
		for ( i = 0; i < theConstrainedSlab[iBldg].n * 3; i++) {
			theConstrainedSlab[iBldg].transformation_matrix[i] =
					(double*)malloc(6 * sizeof(double));
		}

		/* Matrix R */
		theConstrainedSlab[iBldg].local_ids_reference_nodes =
				(lnid_t**)malloc(theConstrainedSlab[iBldg].l * sizeof(lnid_t*));
		for ( i = 0; i < theConstrainedSlab[iBldg].l; i++) {
			theConstrainedSlab[iBldg].local_ids_reference_nodes[i] =
					(lnid_t*)malloc( 5 * sizeof(lnid_t));
		}

		/* Matrix TA */
		theConstrainedSlab[iBldg].tributary_areas =
				(double**)malloc(theConstrainedSlab[iBldg].node_x * sizeof(double*));
		for ( i = 0; i < theConstrainedSlab[iBldg].node_x; i++) {
			theConstrainedSlab[iBldg].tributary_areas[i] =
					(double*)malloc( theConstrainedSlab[iBldg].node_y * sizeof(double));
		}

		/* Matrix Dx */
		theConstrainedSlab[iBldg].distance_to_centroid_xm =
				(double**)malloc(theConstrainedSlab[iBldg].node_x * sizeof(double*));
		for ( i = 0; i < theConstrainedSlab[iBldg].node_x; i++) {
			theConstrainedSlab[iBldg].distance_to_centroid_xm[i] =
					(double*)malloc( theConstrainedSlab[iBldg].node_y * sizeof(double));
		}

		/* Matrix Dy */
		theConstrainedSlab[iBldg].distance_to_centroid_ym =
				(double**)malloc(theConstrainedSlab[iBldg].node_x * sizeof(double*));
		for ( i = 0; i < theConstrainedSlab[iBldg].node_x; i++) {
			theConstrainedSlab[iBldg].distance_to_centroid_ym[i] =
					(double*)malloc( theConstrainedSlab[iBldg].node_y * sizeof(double));
		}


	}

	/* Find local ids of nodes at each level for each building.( level0 -> base ) - matrix L  */
	/* Find local_ids_reference_nodes matrix for each building and each level. - matrix R  */
	/* l is from basement to roof */

	for ( iBldg = 0; iBldg < theNumberOfBuildings; iBldg++ ) {
		int32_t l;
		tick_t bldg_center_x, bldg_center_y;

		bldg_center_x = (theBuilding[iBldg].bounds.xmax +
				theBuilding[iBldg].bounds.xmin)/2/ticksize;
		bldg_center_y = (theBuilding[iBldg].bounds.ymax +
				theBuilding[iBldg].bounds.ymin)/2/ticksize;

		for ( l = 0; l < theConstrainedSlab[iBldg].l; l++) {
			tick_t    z_tick, counter = 0;
			z_tick = (theSurfaceShift - theMinOctSizeMeters*l)/ticksize;
			for ( i = 0; i <  theConstrainedSlab[iBldg].node_x; i++ ) {
				for ( j = 0; j <  theConstrainedSlab[iBldg].node_y; j++) {

					int32_t   k;
					tick_t    x_tick, y_tick;
					x_tick = (theBuilding[iBldg].bounds.xmin + theMinOctSizeMeters*i)/ticksize;
					y_tick = (theBuilding[iBldg].bounds.ymin + theMinOctSizeMeters*j)/ticksize;

					/* Loop each node.
					 */
					for ( k = 0; k < myMesh->nharbored; ++k ) {
						if( myMesh->nodeTable[k].ismine == 1 ) {
							if ( myMesh->nodeTable[k].z == z_tick) {
								if ( myMesh->nodeTable[k].x == x_tick) {
									if ( myMesh->nodeTable[k].y == y_tick) {

										/* Local ids -- matrix L*/
										theConstrainedSlab[iBldg].local_ids_by_level[l][counter] = k;
										counter++;

										/* tributary_areas -- matrix TA*/
										theConstrainedSlab[iBldg].tributary_areas[i][j] =
												theMinOctSizeMeters * theMinOctSizeMeters;

										if ( (y_tick == theBuilding[iBldg].bounds.ymin/ticksize) ||
												(y_tick == theBuilding[iBldg].bounds.ymax/ticksize) ||
												(x_tick == theBuilding[iBldg].bounds.xmin/ticksize) ||
												(x_tick == theBuilding[iBldg].bounds.xmax/ticksize) ) {
											/* tributary_areas -- matrix TA*/
											theConstrainedSlab[iBldg].tributary_areas[i][j] =
													theMinOctSizeMeters * theMinOctSizeMeters / 2;
										}

										if ( ((y_tick == theBuilding[iBldg].bounds.ymax/ticksize) && (x_tick == theBuilding[iBldg].bounds.xmin/ticksize) ) ||
												((y_tick == theBuilding[iBldg].bounds.ymin/ticksize) && (x_tick == theBuilding[iBldg].bounds.xmin/ticksize) ) ||
												((y_tick == theBuilding[iBldg].bounds.ymax/ticksize) && (x_tick == theBuilding[iBldg].bounds.xmax/ticksize) ) ||
												((y_tick == theBuilding[iBldg].bounds.ymin/ticksize) && (x_tick == theBuilding[iBldg].bounds.xmax/ticksize) )
										) {

											/* tributary_areas -- matrix TA*/
											theConstrainedSlab[iBldg].tributary_areas[i][j] =
													theMinOctSizeMeters * theMinOctSizeMeters / 4;
										}

										/* For reference nodes -- matrix R*/
										/* For reference point 1 (E)*/
										if ( (y_tick == theBuilding[iBldg].bounds.ymax/ticksize) &&
												(x_tick == bldg_center_x) ) {
											theConstrainedSlab[iBldg].local_ids_reference_nodes[l][0] = k;
											break;
										}
										/* For reference point 2 (N)*/
										if ( (x_tick == theBuilding[iBldg].bounds.xmax/ticksize) &&
												(y_tick == bldg_center_y) ) {
											theConstrainedSlab[iBldg].local_ids_reference_nodes[l][1] = k;
											break;
										}
										/* For reference point 3 (W)*/
										if ( (y_tick == theBuilding[iBldg].bounds.ymin/ticksize) &&
												(x_tick == bldg_center_x) ) {
											theConstrainedSlab[iBldg].local_ids_reference_nodes[l][2] = k;
											break;
										}
										/* For reference point 4 (S)*/
										if ( (x_tick == theBuilding[iBldg].bounds.xmin/ticksize) &&
												(y_tick == bldg_center_y) ) {
											theConstrainedSlab[iBldg].local_ids_reference_nodes[l][3] = k;
											break;
										}
										/* For reference point 5 -- center point*/
										if ( (x_tick == bldg_center_x) &&
												(y_tick == bldg_center_y) ) {
											theConstrainedSlab[iBldg].local_ids_reference_nodes[l][4] = k;
											break;
										}

										break;
									}
								}
							}
						}
					}
				}
			}
		}
	}


	printf("roof pt1_x_y_z= %f %f %f pt2_x_y_z= %f %f %f pt3_x_y_z= %f %f %f  pt4_x_y_z= %f %f %f pt5_x_y_z= %f %f %f\n\n"
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][0] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][0] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][0] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][1] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][1] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][1] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][2] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][2] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][2] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][3] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][3] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][3] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][4] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][4] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[theConstrainedSlab[0].l-1][4] ].z* ticksize );

	printf("base pt1_x_y_z= %f %f %f pt2_x_y_z= %f %f %f pt3_x_y_z= %f %f %f  pt4_x_y_z= %f %f %f pt5_x_y_z= %f %f %f\n\n"
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][0] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][0] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][0] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][1] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][1] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][1] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][2] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][2] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][2] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][3] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][3] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][3] ].z* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][4] ].x* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][4] ].y* ticksize
			, myMesh->nodeTable[  theConstrainedSlab[0].local_ids_reference_nodes[0][4] ].z* ticksize );

	/* Find transformation matrix for each building. Only calculate for one
	 * level (level = 1)(constant for each level) - matrix T  */

	for ( iBldg = 0; iBldg < theNumberOfBuildings; iBldg++ ) {

		int counter = 0;
		double bldg_center_x, bldg_center_y;

		bldg_center_x = (theBuilding[iBldg].bounds.xmax + theBuilding[iBldg].bounds.xmin)/2;
		bldg_center_y = (theBuilding[iBldg].bounds.ymax + theBuilding[iBldg].bounds.ymin)/2;

		//printf("\n");

		for ( i = 0; i < theConstrainedSlab[iBldg].node_x; i++ ) {
			for ( j = 0; j < theConstrainedSlab[iBldg].node_y; j++) {
				lnid_t nindex;
				/* distance of the node to the geometric center of the building (resistance center)  */
				double dist_x, dist_y;

				nindex = theConstrainedSlab[iBldg].local_ids_by_level[1][counter];
				dist_x = myMesh->nodeTable[nindex].x * ticksize - bldg_center_x;
				dist_y = myMesh->nodeTable[nindex].y * ticksize - bldg_center_y;


				/* distance to centroid -- matrix Dx*/
				theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] =
						dist_x;

				/* distance to centroid -- matrix Dy*/
				theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] =
						dist_y;

				theConstrainedSlab[iBldg].transformation_matrix[counter*3][0] = 1;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3][1] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3][2] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3][3] = dist_y;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3][4] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3][5] = 0;

				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][0] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][1] = 1;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][2] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][3] = -1*dist_x;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][4] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][5] = 0;

				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][0] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][1] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][2] = 1;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][3] = 0;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][4] = dist_y;
				theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][5] = -1*dist_x;

				//printf("x = %d y = %d  disx = %f  disy = %f \n",i,j,dist_x,dist_y);

				counter++;
			}
		}
	}


	int i_mat, j_mat;

	fprintf(stdout, "\n Transformation Matrix T \n\n");
	for ( i_mat = 0 ; i_mat < theConstrainedSlab[0].n * 3 ; i_mat++){
		for ( j_mat = 0; j_mat < 6; j_mat++) {
			fprintf(stdout, "%3.2f ",
					theConstrainedSlab[0].transformation_matrix[i_mat][j_mat]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "\n\n");

	tick_t nin;
	fprintf(stdout, "\n Reference Matrix R \n\n");
	for ( i_mat = 0 ; i_mat < theConstrainedSlab[0].l ; i_mat++){
		for ( j_mat = 0; j_mat < 4; j_mat++) {
			nin = theConstrainedSlab[0].local_ids_reference_nodes[i_mat][j_mat];
			fprintf(stdout, "%3.2f-%3.2f-%3.2f  ",
					myMesh->nodeTable[nin].x * ticksize,
					myMesh->nodeTable[nin].y * ticksize,
					myMesh->nodeTable[nin].z * ticksize);
		}
		fprintf(stdout, "\n");
	}


	fprintf(stdout, "\n Tributary Areas Matrix T \n\n");
	for ( i_mat = 0 ; i_mat < theConstrainedSlab[0].node_x ; i_mat++){
		for ( j_mat = 0; j_mat < theConstrainedSlab[0].node_y; j_mat++) {
			fprintf(stdout, "%3.2f ",
					theConstrainedSlab[0].tributary_areas[i_mat][j_mat]);
		}
		fprintf(stdout, "\n");
	}


	fprintf(stdout, "\n Distance x \n\n");
	for ( i_mat = 0 ; i_mat < theConstrainedSlab[0].node_x ; i_mat++){
		for ( j_mat = 0; j_mat < theConstrainedSlab[0].node_y; j_mat++) {
			fprintf(stdout, "%3.2f ",
					theConstrainedSlab[0].distance_to_centroid_xm[i_mat][j_mat]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "\n Distance y  \n\n");
	for ( i_mat = 0 ; i_mat < theConstrainedSlab[0].node_x ; i_mat++){
		for ( j_mat = 0; j_mat < theConstrainedSlab[0].node_y; j_mat++) {
			fprintf(stdout, "%3.2f ",
					theConstrainedSlab[0].distance_to_centroid_ym[i_mat][j_mat]);
		}
		fprintf(stdout, "\n");
	}

	fprintf(stdout, "\n\n");

//	for ( iBldg = 0; iBldg < 4; iBldg++ ) {
//		static char stationFile[256];
//		sprintf(stationFile, "%s/station.%d","outputfiles/ref_stations",iBldg);
//		fp_deneme[iBldg] = hu_fopen( stationFile,"w" );
//		fputs( "#  Time(s)         X|(m)         Y-(m)         Z.(m)",
//				fp_deneme[iBldg] );
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

void bldgs_update_constrainedslabs_disps ( mysolver_t* solver, double simDT, int step ) {

	int32_t   iBldg, i, j, k, m, l;
	double    average_values[6]; /*order:  x,y,z,theta_z,theta_x,theta_y*/

	for ( iBldg = 0; iBldg < theNumberOfBuildings; iBldg++ ) {
		for ( l = 0; l < theConstrainedSlab[iBldg].l; l++) {
			int 	    counter = 0;
			lnid_t      ref_index[5];
			fvector_t*  ref_dis[5];

			for ( m = 0; m < 6; m++) {
				average_values[m] = 0;
			}

//			double Factor = 1.0/1.0;
//
//
//			/* Fill average displacements vector (A) */
//			for ( k = 0; k < 4; k++) {
//				ref_index[k] =  theConstrainedSlab[iBldg].local_ids_reference_nodes[l][k];
//				ref_dis[k] = solver->tm2 + ref_index[k];
//
////				average_values[0] += ref_dis[k]->f[0]/4; /* average x */
////				average_values[1] += ref_dis[k]->f[1]/4; /* average y */
////				average_values[2] += ref_dis[k]->f[2]/4; /* average z */
//			}
//
//			/* average translation is from the center */
//			ref_index[4] = theConstrainedSlab[iBldg].local_ids_reference_nodes[l][4];
//			ref_dis[4] = solver->tm2 + ref_index[4];
////			average_values[0] = ref_dis[4]->f[0]; /* average x */
////			average_values[1] = ref_dis[4]->f[1]; /* average y */
//			//average_values[2] = ref_dis[4]->f[2]; /* average z */
//			/* average z displacement is zero */
//			average_values[2] = 0;
//
//
//			/* First approach */
//			/* average theta_z */
//			average_values[3] = 1.0*
//					(ref_dis[0]->f[0] - ref_dis[2]->f[0])/2/theConstrainedSlab[iBldg].y_bldg_length +
//					(ref_dis[3]->f[1] - ref_dis[1]->f[1])/2/theConstrainedSlab[iBldg].x_bldg_length;
//
//			/* average theta_x */
//			average_values[4] = Factor *
//					(ref_dis[0]->f[2] - ref_dis[2]->f[2])/theConstrainedSlab[iBldg].y_bldg_length;
//
//			/* average theta_y */
//			average_values[5] = Factor *
//					(ref_dis[3]->f[2] - ref_dis[1]->f[2])/theConstrainedSlab[iBldg].x_bldg_length;

		//	double Factor = 1.0/1.01;
			double Factor = 1.0/1.0;

			/* Second approach */
			int32_t  counter2 = 0, counter_ave = 0 ;
			double   Mwy = 0, Mwx = 0;
			double   Muy = 0, Moy = 0;
			double   Mvx = 0, Mox = 0;

			double   Iy = 0, Ix = 0, Ixy = 0;


			for ( i = 0; i <  theConstrainedSlab[iBldg].node_x ; i++ ) {
				for ( j = 0; j <  theConstrainedSlab[iBldg].node_y ; j++) {
					lnid_t  nindex;
					fvector_t* dis;

					nindex = theConstrainedSlab[iBldg].local_ids_by_level[l][counter_ave];
					dis = solver->tm2 + nindex;

					average_values[0] += dis->f[0]/theConstrainedSlab[iBldg].n; /* average x */
					average_values[1] += dis->f[1]/theConstrainedSlab[iBldg].n; /* average y */
					average_values[2] += dis->f[2]/theConstrainedSlab[iBldg].n; /* average z */
					counter_ave++;
				}
			}

			average_values[2] = 0;

			for ( i = 0; i <  theConstrainedSlab[iBldg].node_x ; i++ ) {
				for ( j = 0; j <  theConstrainedSlab[iBldg].node_y ; j++) {
					lnid_t  nindex;
					fvector_t* dis;

					nindex = theConstrainedSlab[iBldg].local_ids_by_level[l][counter2];
					dis = solver->tm2 + nindex;

					Muy += theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j] *
							(dis->f[0]);
//					Moy += theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
//							theConstrainedSlab[iBldg].tributary_areas[i][j] *
//							average_values[0];


					Mvx += theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j] *
							(dis->f[1]);
//					Mox += theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
//							theConstrainedSlab[iBldg].tributary_areas[i][j] *
//							average_values[1];


					Mwx += theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j] *
							(dis->f[2]);
					Mwy += theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j] *
							(dis->f[2]);
					Ix += theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
							theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j];
					Iy += theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
							theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
							theConstrainedSlab[iBldg].tributary_areas[i][j];
//					Ixy += theConstrainedSlab[iBldg].distance_to_centroid_ym[i][j] *
//							theConstrainedSlab[iBldg].distance_to_centroid_xm[i][j] *
//							theConstrainedSlab[iBldg].tributary_areas[i][j];
					counter2++;
				}
			}

			average_values[3] = ( (Muy) / Ix + (- Mvx) / Iy ) / 2.0;  /* average theta_z */
			average_values[4] = Factor * Mwy / Ix;  /* average theta_x */
			average_values[5] = Factor * -1.0 * Mwx / Iy;  /* average theta_y */

//			average_values[3] = 1.0/1.035 * ( (Muy - Moy) / Ix + (Mox - Mvx) / Iy ) / 2.0;  /* average theta_z */
//			average_values[4] = (Iy*Mwy  - Ixy*Mwx)/(Ix*Iy - Ixy*Ixy);
//			average_values[5] = (Ixy*Mwy - Ix*Mwx)/(Ix*Iy - Ixy*Ixy);


			//        double time = simDT * step;

			//			// roof ref disp
			//			if ( l == theConstrainedSlab[iBldg].l - 1) {
			//				for ( i = 0; i < 1; i++ ) {
			//					fprintf( fp_deneme[i],
			//							"\n%10.6f % 8e % 8e % 8e",
			//							time, average_values[3], average_values[4], average_values[5] );
			//					fflush(fp_deneme[i]);
			//				}
			//			}


			for ( i = 0; i <  theConstrainedSlab[iBldg].node_x ; i++ ) {
				for ( j = 0; j <  theConstrainedSlab[iBldg].node_y ; j++) {
					lnid_t  nindex;
					fvector_t* dis_slab_node;

					nindex = theConstrainedSlab[iBldg].local_ids_by_level[l][counter];
					dis_slab_node = solver->tm2 + nindex;

					dis_slab_node->f[0] = average_values[0] +
							theConstrainedSlab[iBldg].transformation_matrix[counter*3][3] * average_values[3]  ;
					dis_slab_node->f[1] = average_values[1] +
							theConstrainedSlab[iBldg].transformation_matrix[counter*3+1][3] * average_values[3] ;
					dis_slab_node->f[2] = average_values[2] +
							theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][4] * average_values[4] +
							theConstrainedSlab[iBldg].transformation_matrix[counter*3+2][5] * average_values[5] ;

					counter++;
				}
			}
		}
	}

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

