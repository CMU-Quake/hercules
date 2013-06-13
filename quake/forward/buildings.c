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

/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static noyesflag_t  areBaseFixed = 0;
static char         theBaseFixedDir[256];
static char         theBaseFixedSufix[64];
static double       theBaseFixedDT;
static int          theBaseFixedStartIndex;
static int32_t      theBaseNodesCount;
static basenode_t  *theBaseNodes;
static fvector_t  **theBaseSignals;

/* Permanent */

static int      theNumberOfBuildings;
static int      theNumberOfPushdowns;
static double   theMinOctSizeMeters;
static double   theSurfaceShift = 0;
static bldg_t  *theBuilding;
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

bounds_t get_bldgbounds( int i ) {

    return theBuilding[i].bounds;
}

bounds_t get_pushdownbounds( int i ) {

	 return thePushdown[i].bounds;
}


cvmpayload_t get_props( int i, double z ) {

    /* the i-th building is in the i-1 index */
    i--;

    if ( z >= theSurfaceShift ) {
        /* Element is in the foundation */
        return theBuilding[i].fdtnprops;
    } else {
        /* Element is in the foundation */
        return theBuilding[i].bldgprops;
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

	edgesize = edata->edgesize;
	z_m      = leaf->lz * ticksize;

	/* Same size elements*/
	if ( edgesize != theMinOctSizeMeters ) {
		return 1;
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
		if ( theMinOctSizeMeters > ( theBuilding[bldg].bldgprops.Vs / theFactor ) &&
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
					/* Element is in the building */
					edata->Vp  = theBuilding[i].bldgprops.Vp;
					edata->Vs  = theBuilding[i].bldgprops.Vs;
					edata->rho = theBuilding[i].bldgprops.rho;
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
    int     int_message[3];
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

    MPI_Bcast(double_message, 2, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(int_message,    3, MPI_INT,    0, comm_solver);

    theSurfaceShift      = double_message[0];
    theMinOctSizeMeters  = double_message[1];
    theNumberOfBuildings = int_message[0];
    areBaseFixed         = int_message[1];
	theNumberOfPushdowns = int_message[2];

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
    		theBuildingsVp     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsVs     = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theBuildingsRho    = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theFoundationsVp   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theFoundationsVs   = (double*)malloc( sizeof(double) * theNumberOfBuildings );
    		theFoundationsRho  = (double*)malloc( sizeof(double) * theNumberOfBuildings );

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
    	MPI_Bcast(theBuildingsVp,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsVs,     theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theBuildingsRho,    theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsVp,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsVs,   theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);
    	MPI_Bcast(theFoundationsRho,  theNumberOfBuildings, MPI_DOUBLE, 0, comm_solver);

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

    		theBuilding[i].bldgprops.Vp  = theBuildingsVp[i];
    		theBuilding[i].bldgprops.Vs  = theBuildingsVs[i];
    		theBuilding[i].bldgprops.rho = theBuildingsRho[i];

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
    	free(theBuildingsVp);
    	free(theBuildingsVs);
    	free(theBuildingsRho);
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

	noyesflag_t fixedbase = -1;

	/* Opens parametersin file */

	if ( ( fp = fopen(parametersin, "r" ) ) == NULL ) {
		fprintf( stderr,
				"Error opening %s\n at buildings_initparameters",
				parametersin );
		return -1;
	}

	/* Parses parametersin to capture building single-value parameters */

	if ( ( parsetext(fp, "number_of_buildings", 'i', &numBldgs           ) != 0) ||
			( parsetext(fp, "number_of_pushdowns", 'i', &numPushDown        ) != 0) ||
			( parsetext(fp, "min_octant_size_m",   'd', &min_oct_size       ) != 0) ||
			( parsetext(fp, "surface_shift_m",     'd', &surface_shift      ) != 0) ||
			( parsetext(fp, "consider_fixed_base", 's', &consider_fixed_base) != 0) )
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
				"fixed base option (yes or no): %s\n",
				consider_fixed_base );
	}

	/* Initialize the static global variables */

	theNumberOfBuildings = numBldgs;
	theNumberOfPushdowns = numPushDown;
	theMinOctSizeMeters  = min_oct_size;
	theSurfaceShift      = surface_shift;
	areBaseFixed         = fixedbase;

	/* Detour for fixed base option */
	if ( areBaseFixed == YES ) {
		fixedbase_read( fp );
	}

	if ( numBldgs > 0 ) {

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

