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

/***
 *
 * octor.h: Parallel unstructured octree mesh generation library.
 *
 *
 */


#ifndef OCTOR_H
#define OCTOR_H

#include <inttypes.h> 
#include <mpi.h>

#include "ocutil.h"

#define TOTALLEVEL       31
#define PIXELLEVEL       30


/**
 * tick_t: The unit of the embeded integer domain 
 *
 */
typedef int32_t tick_t;


/**
 * Two different search mode.
 *
 */
#define  EXACT_SEARCH     1
#define  AGGREGATE_SEARCH 2

/**
 * octant_t: octant structure;
 *
 */
#define LEAF       0
#define INTERIOR   1
#define T_UNDEFINED  2

#define LOCAL      0xFF
#define REMOTE     0x00
#define GLOBAL     0xcc

typedef struct octant_t {
    /* structural information */
    unsigned char type;     /* LEAF or INTERIOR */
    unsigned char where;    /* Where is this octants */
    int8_t which;           /* which child of the parent */
    int8_t level;           /* Root is at level 0 */
    tick_t lx, ly, lz;      /* left-lower corner coordinate */
    struct octant_t *parent;/* pointer to the parent */
    void *appdata;          /* pointer to application-specific data */
} octant_t;


/**
 * octree_t: 
 *
 */
typedef struct octree_t {
    octant_t *root;        /* point to the root octant */
    tick_t nearendp[3];    /* near-end coordinate of the domain, set to 0 */
    tick_t farendp[3];     /* far-end coordinate of the domain */

    /* RICARDO */
    tick_t surfacep;

    double ticksize;       /* map a tick to physical space meters */
    int32_t recsize;       /* size of the leaf record */
    MPI_Comm comm_tree;  /* MPI communicator to cover Octree PEs */
} octree_t;
    

/**
 * point_t:
 *
 */
typedef struct point_t {
    tick_t x, y, z;
} point_t;



/**
 * elem_t:
 *
 */
typedef struct elem_t{
    int64_t geid;          /* Global element id */
    int32_t lnid[8];       /* Local node ids */
    int8_t level;          /* Level of the corresponding octants */
    void *data;            /* Application-specific data */
} elem_t; 


/**
 * int32link_t: Link list for int32_t's.
 *
 */
typedef struct int32link_t {
    int32_t id;
    struct int32link_t *next;    
} int32link_t;

   
/**
 * node_t:
 *
 *
 */
typedef struct node_t {
    tick_t x, y, z;        /* Coordinate of the node */
    int8_t ismine;         /* Whether this node is owned by me */
    int8_t isanchored;     /* 1 if anchored, 0 otherwise */

    int64_t gnid;          /* Global node id */

    union {
        int32_t ownerid;    /* If (!ismine) */
        int32link_t *share; /* If (ismine) */
    } proc;
} node_t;


/**
 * dnode_t: An entry of a dangling node exists on a processor only
 *          if the dangling node is owned by the processor. i.e.
 *          ((nodeTable[lnid].ismine) && (!nodeTable[lnid].isanchored)) 
 *
 */
typedef struct dnode_t {
    int32_t ldnid;          /* Local node id of the dangling node */
    uint32_t deps;            /* Number of dependences, 2 or 4 */
    int32link_t *lanid;     /* Local node id list of the anchored nodes
                               which are depended on */
} dnode_t; 
    
    

/**
 * mesh_t:
 *
 */
typedef struct mesh_t {
    int32_t lenum;         /* Number of elements this processor owns */
    int32_t lnnum;         /* Number of nodes this processor owns */
    int32_t ldnnum;        /* Number of dangling nodes this processor owns */
    int32_t nharbored;     /* Number of nodes this processor harbors*/

    double ticksize;       /* The edge size of a pixel in physical
                              coordiate system. Same as
                              octree->ticksize */

    elem_t *elemTable;     
    node_t *nodeTable;
    dnode_t *dnodeTable;
} mesh_t;
    

/**
 * Application provided plugins:
 *
 */
typedef int32_t toexpand_t ( octant_t *leaf, double ticksize,
                             const void *data );
typedef int32_t toshrink_t ( octant_t *leaf[8], double ticksize,
                             const void *data[8] );
typedef void setrec_t ( octant_t *leaf, double ticksize, void *data, boolean useSetrec2);

typedef int bldgs_nodesearch_t ( tick_t x, tick_t y, tick_t z,
                                 double ticksize );


/***************************/
/* Octant-level operations */
/***************************/

/**
 * Return the pointer to the parent octant 
 *
 */
#define OCTOR_GETPARENT(octant) ((octant)->parent)



#define OCTOR_GETOCTTICKS(octant) ((tick_t)1 << (PIXELLEVEL - (octant)->level))


/**
 * Set the center coordinate of an octant. Coord should be double[3]
 *
 */

#define OCTOR_SETOCTCENTER(octree, octant, coord) \
do { \
    (coord)[0] =  \
        octree->ticksize * ((octant)->lx + OCTOR_GETOCTTICKS(octant) / 2); \
\
    (coord)[1] = \
        octree->ticksize * ((octant)->ly + OCTOR_GETOCTTICKS(octant) / 2); \
\
    (coord)[2] = \
        octree->ticksize * ((octant)->lz + OCTOR_GETOCTTICKS(octant) / 2); \
} while (0) 


/**
 * Return the edge size of an octant (either LEAF or INTERIOR)
 *
 */
#define OCTOR_GETOCTSIZE(octree, octant) \
    ((octree)->ticksize * OCTOR_GETOCTTICKS(octant))



/**
 * octor_getchild: Return the specified child octant. If parentoctant is a 
 *                 LEAF or the specified child does not exist. Return NULL.
 *
 */
extern octant_t *
octor_getchild(octant_t *parentoctant, int32_t whichchild);


/**
 * octor_getfirstleaf: Return the first LOCAL leaf.
 *
 */
extern octant_t *
octor_getfirstleaf(octree_t *octree);


/**
 * octor_getnextleaf: Return the next leaf octant (may be a REMOTE one).
 *
 */
extern octant_t *
octor_getnextleaf(octant_t *octant);


/**
 * octor_getleafrec: Retrieve the record stored in a leaf octant. Return
 *                    0 if OK, -1 on error. 
 * 
 *  Note that the appdata associated an octant should be directly
 *  accessed via the pointer defined in octant_t. 
 *
 */
extern int32_t 
octor_getleafrec(octree_t *octree, octant_t *octant, void *data);

/**
 * octor_searchoctant: Look for the octant specified by (x, y, z, level).
 *                     Return a pointer to the octant if found (exact or 
 *                     aggregate). NULL, if not found.
 *
 */
extern octant_t *
octor_searchoctant(octree_t *octree, tick_t x, tick_t y, tick_t z, 
                   int8_t level, int32_t searchtype);
                   

extern int32_t
octor_getmaxleaflevel(const octree_t* octree, int where);

extern int32_t
octor_getminleaflevel(const octree_t* octree, int where);

extern int64_t
octor_getleavescount(const octree_t* octree, int where);
extern int64_t
octor_getminleavescount(const octree_t* octree, int where);
extern int64_t
octor_getmaxleavescount(const octree_t* octree, int where);

/*************************/
/* Tree-level operations */
/*************************/

extern octree_t *
octor_newtree(double x, double y, double z, int32_t recsize,
              int32_t myid, int32_t groupsize, MPI_Comm solver_comm,
              double surface_shift);

extern void 
octor_deletetree(octree_t * octree);

extern int32_t 
octor_refinetree(octree_t *octree, toexpand_t *toexpand, setrec_t *setrec, boolean useSetrec2);

extern int32_t 
octor_coarsentree(octree_t *octree, toshrink_t *toshrink, setrec_t *setrec);

extern int32_t 
octor_balancetree(octree_t *octree, setrec_t *setrec, boolean useSetrec2, int theStepMeshingFactor);

extern void
octor_carvebuildings(octree_t *octree, int flag,
                     bldgs_nodesearch_t *bldgs_nodesearch);

extern int32_t
octor_partitiontree(octree_t *octree, bldgs_nodesearch_t *bldgs_nodesearch);

extern mesh_t *
octor_extractmesh(octree_t *octree, bldgs_nodesearch_t *bldgs_nodesearch);

extern void
octor_deletemesh(mesh_t *mesh);


/*************************/
/* Statistics            */
/*************************/
extern void
octor_showstat(octree_t *octree, mesh_t *mesh);

/*************************/
/* Auxilliary functions  */
/*************************/
extern int32_t octor_zcompare(const void *p1, const void *p2);
extern const point_t * octor_getintervaltable(octree_t *octree);
    

#endif /* OCTOR_H */
