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

/**
 * Parallel unstructured octree mesh generation library.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "octor.h"

/* #define TREE_VERBOSE */
/* #define MESH_VERBOSE */

#define XFACE            1  /* On the center of a face vertical to X axis */
#define YFACE            2
#define ZFACE            3
#define XEDGE            5  /* On the midpoint of an edge parallel to X axis */
#define YEDGE            6
#define ZEDGE            7

#define DETAILS          1
#define BRIEF            2


static double theSurfaceShift = 0;
static double theTotalDepth;


/**
 * Application provided payload:
 *
 */

typedef struct edata_t {
    float edgesize, Vp, Vs, rho;
} edata_t;


/****************************/
/* Memory management        */
/****************************/

static size_t theAllocatedMemSum = 0;


typedef struct hdr_t hdr_t;

/**
 * hdr_t: header of a memory chunk.
 */
struct hdr_t {
    hdr_t* nextchunk;
    int    capacity;
    int    used;
    void*  nextobj;
};


/**
 * tlr_t:  trailer of a memory chunk.
 *
 */
struct tlr_t {
    hdr_t* nextchunk;
};


typedef struct tlr_t tlr_t;


/**
 * cursor_t: An offset in a memory chunk.
 *
 */
typedef struct cursor_t {
    int    inited;
    hdr_t* chunkptr;
    void*  objptr;
    size_t objoffset;
} cursor_t;


/**
 * mem_t: memory object manager
 *
 */
typedef struct mem_t {

    size_t objsize;       /* size of each individual data object */
    size_t chunkcapacity;
    size_t chunkdatasize; /* size of the real data, excl. hdr_t and tlr_t */
    size_t chunkbytesize; /* size of a chunk (data + hdr_t + tlr_t */
    int    chunkcount;    /* Number of chunks allocated */
    unsigned objcount;   /* Number of objects in this pool */

    void*  recycled;      /* pointer to the recycled objects */
    hdr_t* first;         /* pointer to the first allocated memory chunk */
    hdr_t* last;          /* pointer to the last allocated memory chunk */

    /** count of all the object allocation calls for this memory pool */
    unsigned long total_count;
    cursor_t cursor;      /* Iterator over the memory buffer. Undefined
			     until mem_initcursor() is called. */
} mem_t;


/**
 * Global static memory pool that stores application-specific data
 *
 */
static mem_t* theRecordPool;
static int32_t theRecordPoolRefs;


/* static routines: mem_ */

static mem_t*  mem_new(size_t objsize, int chunkcapacity);
static void    mem_clean(mem_t* mem);
static void    mem_delete(mem_t* mem);

static void*   mem_newobj(mem_t* mem);
static void    mem_recycleobj(mem_t* mem, void* obj);

static void    mem_initcursor(mem_t* mem);
static void    mem_advcursor(mem_t* mem);
static void*   mem_getcursor(mem_t* mem);


/***************************************/
/*  Mesh data structure                */
/***************************************/


/**
 * link_t: auxilliary data structure for linked list.
 *
 */

typedef struct link_t {
    void *record;
    struct link_t *next;
} link_t;


/**
 * vertex_t: Internal mesh node representation
 *
 */

typedef struct vertex_t {
    tick_t x, y, z;         /* Coordinate of the node in integer address
                               space */
    int32_t owner;          /* Who owns this vertex for output */
    int32_t lnid;           /* Local node id */
    int8_t touches;         /* How many octants share this vertex */
    int8_t level;           /* Lowest level of the octants who share this
                               vertex */
    unsigned char property; /* Indicate ANCHORED/DANGLING, and in the
                               latter case, the location and oriention */
    int32link_t *share;     /* Link list to who also harbor this vertex, valid
                               only if I own this vertex */
} vertex_t;


/**
 * vertex_info_t: Vertex info sent between processors.
 *
 */
typedef struct vertex_info_t {
    tick_t x, y, z;
    int32_t owner;
    int8_t touches;
    int8_t level;
} vertex_info_t;


/**
 * mess_t: Internal mesh representation
 *
 */
typedef struct mess_t {
    int32_t lenum;         /* Number of elements this processor owns */
    int32_t lnnum;         /* Number of nodes this processor owns */
    int32_t ldnnum;        /* Number of dangling nodes this processor owns */
    int32_t nharbored;     /* Number of nodes this processor harbors*/

    double ticksize;
    elem_t *elemTable;
    node_t *nodeTable;
    dnode_t *dnodeTable;

    mem_t *int32linkpool;  /* Memory pool out of int32link_t is allocated */

} mess_t;


/**************************************/
/* Octant- and tree-level  operations */
/**************************************/
struct com_t;


/**
 * oct_t: internal octant representations
 *
 */
struct oct_t;

typedef struct leaf_t {
    struct oct_t *next;    /* ptr to the next local leaf at the same level */
    struct oct_t *prev;    /* ptr to the previous one at the same level */
    void *data;            /* ptr to application specific data */
} leaf_t;


/**
 * interior_t: interior octant payload
 *
 */
typedef struct interior_t {
    struct oct_t *child[8];
} interior_t;


typedef  struct oct_t {
    /* structural information */
    unsigned char type;     /* LEAF or INTERIOR */
    unsigned char where;    /* Where is a LEAF octants */
    int8_t which;           /* which child of the parent */
    int8_t level;           /* Root is at level 0 */
    tick_t lx, ly, lz;      /* left-lower corner coordinate */
    struct oct_t *parent;   /* pointer to the parent */
    void *appdata;          /* pointer to application-specific data */

    /* internal data type */
    union {
        leaf_t     *leaf;
        interior_t *interior;
    } payload;

} oct_t;


/**
 * tree_t: internal tree representation.
 *
 */
typedef struct tree_t {
    oct_t *root;           /* point to the root octant (malloc'ed) */
    tick_t nearendp[3];    /* near-end coordinate of the domain, set to 0 */
    tick_t farendp[3];     /* far-end coordinate of the domain */

    /* RICARDO */
    tick_t surfacep;

    double ticksize;       /* map a tick to physical space meters */
    int32_t recsize;       /* size of the leaf payload */
    MPI_Comm comm_tree;  /* MPI communicator to cover Octree PEs */

    /*------------------ Internal fields  ------------------------------*/

    /* Persistent static fields */
    tick_t farbound[3];    /* farendp - 1 */
    int32_t procid;        /* my processor id */
    int32_t groupsize;     /* number of cooperating processors */
    mem_t *octmem;         /* octant memory manager (malloc'ed) */
    mem_t *leafmem;        /* leaf payload memory manager (malloc'ed) */
    mem_t *interiormem;    /* interior payload memory manager (malloc'ed) */

    /* Persistent dynamcal fields */
    oct_t *firstleaf;      /* point to the first (preorder) LOCAL leaf oct */
    int32_t leafcount[TOTALLEVEL]; /* number of LOCAL leaves at each level */
    struct com_t * com;    /* communication manager (malloc'ed) */

    /* Transient fields used by octor_balancetree() only */
    oct_t *toleaf[TOTALLEVEL];  /* lists of LOCAL leaves at the same level */

} tree_t;


/**
 * dir_t: Directions in consecutive ascending order. Easy to use
 *        in a for-loop. More importantly, LDB -- RUF maps to 0 -- 7,
 *        respectively.
 */
typedef enum {
    LDB = 0, RDB, LUB, RUB, LDF, RDF, LUF, RUF,
    L, R, D, U, B, F,
    LD, LU, LB, LF, RD, RU, RB, RF, DB, DF, UB, UF,
    OMEGA
} dir_t;


/* Set bits of the directions in a 32-bit unsigned integer */
#define LEFT             0x0001
#define RIGHT            0x0002
#define DOWN             0x0004
#define UP               0x0008
#define BACK             0x0010
#define FRONT            0x0020

#define L_bits           LEFT
#define R_bits           RIGHT
#define D_bits           DOWN
#define U_bits           UP
#define B_bits           BACK
#define F_bits           FRONT

#define LD_bits          (LEFT | DOWN)
#define LU_bits          (LEFT | UP)
#define LB_bits          (LEFT | BACK)
#define LF_bits          (LEFT | FRONT)
#define RD_bits          (RIGHT | DOWN)
#define RU_bits          (RIGHT | UP)
#define RB_bits          (RIGHT | BACK)
#define RF_bits          (RIGHT | FRONT)
#define DB_bits          (DOWN | BACK)
#define DF_bits          (DOWN | FRONT)
#define UB_bits          (UP | BACK)
#define UF_bits          (UP | FRONT)

#define LDB_bits         (LEFT | DOWN | BACK)
#define RDB_bits         (RIGHT | DOWN | BACK)
#define LUB_bits         (LEFT | UP | BACK)
#define RUB_bits         (RIGHT | UP | BACK)
#define LDF_bits         (LEFT | DOWN | FRONT)
#define RDF_bits         (RIGHT | DOWN | FRONT)
#define LUF_bits         (LEFT | UP | FRONT)
#define RUF_bits         (RIGHT | UP | FRONT)


/**
 * theDirBitRep: map a direction (dir_t) to a bit-representation.
 *
 *
 */
static const uint32_t theDirBitRep[] = {
    /* corners */
    LDB_bits, RDB_bits, LUB_bits, RUB_bits,
    LDF_bits, RDF_bits, LUF_bits, RUF_bits,

    /* faces */
    L_bits, R_bits, D_bits, U_bits, B_bits, F_bits,

    /* edges */
    LD_bits, LU_bits, LB_bits, LF_bits,
    RD_bits, RU_bits, RB_bits, RF_bits,
    DB_bits, DF_bits, UB_bits, UF_bits
};


/**
 * oct_stack_t: This stack should never overflow in this program.
 *
 */
typedef struct oct_stack_t {
    int8_t top;
    int8_t dir[PIXELLEVEL];
} oct_stack_t;


/**
 * theDirReflectTable[O][I]:
 *
 * - O: sontype of an octant; I: direction of reflection.
 * - Maps to the sontype of the block of equal size that shares the
 *   Ith face, edge, or vertex of a block of sontype O.
 *
 */
static const dir_t theDirReflectTable[8][26] = {
    /* O = LDB */
    {RUF, RUF, RUF, RUF, RUF, RUF, RUF, RUF,
     RDB, RDB, LUB, LUB, LDF, LDF,
     RUB, RUB, RDF, RDF, RUB, RUB, RDF, RDF, LUF, LUF, LUF, LUF},

    /* O = RDB */
    {LUF, LUF, LUF, LUF, LUF, LUF, LUF, LUF,
     LDB, LDB, RUB, RUB, RDF, RDF,
     LUB, LUB, LDF, LDF, LUB, LUB, LDF, LDF, RUF, RUF, RUF, RUF},

    /* O = LUB */
    {RDF, RDF, RDF, RDF, RDF, RDF, RDF, RDF,
     RUB, RUB, LDB, LDB, LUF, LUF,
     RDB, RDB, RUF, RUF, RDB, RDB, RUF, RUF, LDF, LDF, LDF, LDF},

    /* O = RUB */
    {LDF, LDF, LDF, LDF, LDF, LDF, LDF, LDF,
     LUB, LUB, RDB, RDB, RUF, RUF,
     LDB, LDB, LUF, LUF, LDB, LDB, LUF, LUF, RDF, RDF, RDF, RDF},

    /* O = LDF */
    {RUB, RUB, RUB, RUB, RUB, RUB, RUB, RUB,
     RDF, RDF, LUF, LUF, LDB, LDB,
     RUF, RUF, RDB, RDB, RUF, RUF, RDB, RDB, LUB, LUB, LUB, LUB},

    /* O = RDF */
    {LUB, LUB, LUB, LUB, LUB, LUB, LUB, LUB,
     LDF, LDF, RUF, RUF, RDB, RDB,
     LUF, LUF, LDB, LDB, LUF, LUF, LDB, LDB, RUB, RUB, RUB, RUB},

    /* O = LUF */
    {RDB, RDB, RDB, RDB, RDB, RDB, RDB, RDB,
     RUF, RUF, LDF, LDF, LUB, LUB,
     RDF, RDF, RUB, RUB, RDF, RDF, RUB, RUB, LDB, LDB, LDB, LDB},

    /* O = RUF */
    {LDB, LDB, LDB, LDB, LDB, LDB, LDB, LDB,
     LUF, LUF, RDF, RDF, RUB, RUB,
     LDF, LDF, LUB, LUB, LDF, LDF, LUB, LUB, RDB, RDB, RDB, RDB}
};


#define TRUE             1
#define FALSE            0


/**
 * theIsAdjacentTable[O][I]:
 *
 * - O: sontype of an octant; I: direction.
 * - TRUE if and only if octant O is adjacent to the Ith face or
 *   edge or vertex of O's containing block.
 *
 */
static const int32_t theIsAdjacentTable[8][26] = {
    /* O == LDB */
    {TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
     TRUE, FALSE, TRUE, FALSE, TRUE, FALSE,
     TRUE, FALSE, TRUE, FALSE,
     FALSE, FALSE, FALSE, FALSE,
     TRUE, FALSE, FALSE, FALSE},

    /* O == RDB */
    {FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
     FALSE, TRUE, TRUE, FALSE, TRUE, FALSE,
     FALSE, FALSE, FALSE, FALSE,
     TRUE, FALSE, TRUE, FALSE,
     TRUE, FALSE, FALSE, FALSE},

    /* O == LUB */
    {FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
     TRUE, FALSE, FALSE, TRUE, TRUE, FALSE,
     FALSE, TRUE, TRUE, FALSE,
     FALSE, FALSE, FALSE, FALSE,
     FALSE, FALSE, TRUE, FALSE},

    /* O == RUB */
    {FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE,
     FALSE, TRUE, FALSE, TRUE, TRUE, FALSE,
     FALSE, FALSE, FALSE, FALSE,
     FALSE, TRUE, TRUE, FALSE,
     FALSE, FALSE, TRUE, FALSE},

    /* O == LDF */
    {FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE,
     TRUE, FALSE, TRUE, FALSE, FALSE, TRUE,
     TRUE, FALSE, FALSE, TRUE,
     FALSE, FALSE, FALSE, FALSE,
     FALSE, TRUE, FALSE, FALSE},

    /* O == RDF */
    {FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE,
     FALSE, TRUE, TRUE, FALSE, FALSE, TRUE,
     FALSE, FALSE, FALSE, FALSE,
     TRUE, FALSE, FALSE, TRUE,
     FALSE, TRUE, FALSE, FALSE},

    /* O == LUF */
    {FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
     TRUE, FALSE, FALSE, TRUE, FALSE, TRUE,
     FALSE, TRUE, FALSE, TRUE,
     FALSE, FALSE, FALSE, FALSE,
     FALSE, FALSE, FALSE, TRUE},

    /* O == RUF */
    {FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE,
     FALSE, TRUE, FALSE, TRUE, FALSE, TRUE,
     FALSE, FALSE, FALSE, FALSE,
     FALSE, TRUE, FALSE, TRUE,
     FALSE, FALSE, FALSE, TRUE}
};

/**
 * theCommonFace[O][I]:
 *
 * - O: sontype of an octant; I: direction.
 * - Maps to the type of the face of O's containing block, that is common to
 *   octant O and its neigbhor in the Ith direction

 *
 */
static const dir_t theCommonFace[8][26] = {
    /* O == LDB */
    {OMEGA, OMEGA, OMEGA, B, OMEGA, D, L, OMEGA,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     OMEGA, L, OMEGA, L, D, OMEGA, B, OMEGA, OMEGA, D, B, OMEGA},

    /* O == RDB */
    {OMEGA, OMEGA, B, OMEGA, D, OMEGA, OMEGA, R,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     D, OMEGA, B, OMEGA, OMEGA, R, OMEGA, R, OMEGA, D, B, OMEGA},

    /* O == LUB */
    {OMEGA, B, OMEGA, OMEGA, L, OMEGA, OMEGA, U,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     L, OMEGA, OMEGA, L, OMEGA, U, B, OMEGA, B, OMEGA, OMEGA, U},

    /* O == RUB */
    {B, OMEGA, OMEGA, OMEGA, OMEGA, R, U, OMEGA,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     OMEGA, U, B, OMEGA, R, OMEGA, OMEGA, R, B, OMEGA, OMEGA, U},

    /* O == LDF */
    {OMEGA, D, L, OMEGA, OMEGA, OMEGA, OMEGA, F,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     OMEGA, L, L, OMEGA, D, OMEGA, OMEGA, F, D, OMEGA, OMEGA, F},

    /* O == RDF */
    {D, OMEGA, OMEGA, R, OMEGA, OMEGA, F, OMEGA,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     D, OMEGA, OMEGA, F, OMEGA, R, R, OMEGA, D, OMEGA, OMEGA, F},

    /* O == LUF */
    {L, OMEGA, OMEGA, U, OMEGA, F, OMEGA, OMEGA,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     L, OMEGA, L, OMEGA, OMEGA, U, OMEGA, F, OMEGA, F, U, OMEGA},

    /* O == RUF */
    {OMEGA, R, U, OMEGA, F, OMEGA, OMEGA, OMEGA,
     OMEGA, OMEGA, OMEGA, OMEGA, OMEGA, OMEGA,
     OMEGA, U, OMEGA, F, R, OMEGA, R, OMEGA, OMEGA, F, U, OMEGA}
};


/* static routine: oct_ */

static oct_t * oct_getleftmost(oct_t *oct);
static oct_t * oct_getnextleaf(oct_t *oct);
static void    oct_linkleaf(oct_t *oct, oct_t *toleaf[]);
static void    oct_unlinkleaf(oct_t *oct, oct_t *toleaf[]);
static oct_t * oct_findneighbor(oct_t *oct, dir_t dir, oct_stack_t *stackp);

static int32_t oct_installleaf(tick_t lx, tick_t ly, tick_t lz,
                               int8_t level, void *rec, tree_t *tree);


static int32_t oct_sprout(oct_t *leafoct, tree_t *tree);
static void    oct_prune(oct_t *oct, tree_t *tree);
static int32_t oct_expand(oct_t *leafoct, tree_t *tree,
                          toexpand_t *toexpand, setrec_t *setrec);
static oct_t * oct_shrink(oct_t *oct, tree_t *tree, unsigned char where,
                         toshrink_t *toshrink, setrec_t *setrec);


/* static routine: tree_ */

static int8_t  tree_getminleaflevel(const tree_t *tree);
static int8_t  tree_getmaxleaflevel(const tree_t *tree);
static int64_t tree_countleaves(tree_t *tree);
static void    tree_setdistribution(tree_t *tree, int64_t **pCountTable,
                                    int64_t **pStartTable, int64_t mycount);

// static void  tree_showstat(tree_t *tree, int32_t mode, const char *comment);

static int32_t tree_setcom(tree_t *tree, int32_t msgsize,
                           bldgs_nodesearch_t *bldgs_nodesearch);
static void    tree_deletecom(tree_t *tree);

static oct_t * tree_ascend(oct_t *oct, dir_t I, oct_stack_t *stackp);
static oct_t * tree_descend(oct_t *oct, oct_stack_t *stackp);
static int32_t tree_pushdown(oct_t *nbr, tree_t *tree, oct_stack_t *stackp,
                             setrec_t *setrec);
static oct_t * tree_searchoct(tree_t *tree, tick_t lx, tick_t ly,
                              tick_t lz, int8_t level, int32_t searchtype);



/****************************/
/* Communications           */
/****************************/
#define UNEXPECTED_ERR    -1
#define OUTOFMEM_ERR      -2
#define COMM_ERR          -3
#define INTERNAL_ERR      -4

#define POINT_MSG         100
#define DESCENT_MSG       200
#define COUNT_MSG         300
#define OCT_MSG           400
#define VERTEX_INFO_MSG   500
#define GNID_MSG          600
#define ANCHORED_MSG      700




/**
 * descent_t: Tree descending instruction.
 *
 */
typedef struct descent_t {
    tick_t lx, ly, lz;
    int8_t level;
    struct oct_stack_t stack;
} descent_t;


/**
 * touches_t: Number of touches a vertex has.
 *
 */
typedef struct touch_t {
    tick_t x, y, z;
    int32_t cnt;
} touch_t;


/** Global node identifier message. */
struct gnid_info_t {
    tick_t x, y, z;
    int64_t gnid;
};

typedef struct gnid_info_t gnid_info_t;


/**
 * RICARDO: This is Julio's magic pause-tool
 */

//static void
//wait_for_debugger( const char* progname, int peid, int cond )
//{
//    if (cond != 0) {
//        int i = 0;
//        char hostname[256];
//
//        gethostname( hostname, sizeof(hostname) );
//        fprintf( stderr, "PE %d waiting for debugger to attach\n %s: gdb %s %d\n",
//                 peid, hostname, progname, getpid() );
//        fflush( stderr );
//
//        while (0 == i) {
//            sleep( 60 );
//            i = 1;
//        }
//    }
//}



/**
 * Obtain the element with the lowest id that belongs to a subset
 * (partition) of a larger set that has been partitioned into subsets
 * of approximately equal integral sizes.
 *
 * \param task_id	Id of an element in the subset.
 * \param group_size	Number of subsets.
 * \param n		Number of elements in the whole set.
 *
 * \return The id of the first element in the same subset as the element
 *	given as a parameter.
 */
static inline int64_t
block_low( int64_t task_id, int group_size, int64_t n )
{
    return ((int64_t)task_id) * n / group_size;
}

/**
 * Obtain the element with the highest id that belongs to a subset
 * (partition) of a larger set that has been partitioned into subsets
 * of approximately equal integral sizes.
 *
 * \param task_id	Id of an element in the subset.
 * \param group_size	Number of subsets.
 * \param n		Number of elements in the whole set.
 *
 * \return The id of the last element in the same subset as the element
 *	given as a parameter.
 */
static inline int64_t
block_high( int64_t task_id, int group_size, int64_t n )
{
    return block_low(task_id + 1, group_size, n) - 1;
}


/**
 * Obtain the size of a subset or partition of a larges set
 * that has been partitioned into subsets of approximately equal integral
 * sizes.
 *
 * \param task_id	Id of an element in the subset.
 * \param group_size	Number of subsets.
 * \param n		Number of elements in the whole set.
 *
 * \return Subset (partition) size for the subset the element
 * in question belongs to.
 */
static inline int64_t
block_size( int64_t task_id, int group_size, int64_t n)
{
    return block_low(task_id + 1, group_size, n)
        - block_low(task_id, group_size, n);
}


/**
 * Obtain the owner (subset or partition id) of an element of a large set
 * that has been partitioned into subsets of approximately equal integral
 * sizes.
 *
 * \param task_id	Id of the element to lookup.
 * \param group_size	Number of subsets.
 * \param n		Number of elements in the whole set.
 *
 * \return Subset id (partition id) for the element in question.
 */
static inline unsigned int
block_owner( int64_t task_id, int group_size, int64_t n )
{
    return (((task_id + 1) * group_size) - 1) / n;
}


static int
block_validate( int64_t low, int64_t high, int64_t block_size, int64_t n,
                int group_size )
{
    int ret = 0;
    const char* fn_name = "block_validate";
    int64_t max_block_size = (n / group_size) + 1;

    /* check block size */
    if (block_size < 0 || block_size > max_block_size) {
        fprintf(stderr, "%s: invalid block size = %lld\n", fn_name, block_size);
        ret = -1;
    }

    /* check low id */
    if (low < 0 || low >= n) {
        fprintf( stderr, "%s: invalid low id = %lld\n", fn_name, low );
        ret = -1;
    }

    /* check high id */
    if (high < 0 || high >= n) {
        fprintf( stderr, "%s: invalid high id = %lld\n", fn_name, high );
        ret = -1;
    }

    if (ret != 0) {
        fprintf( stderr, "%s: Found an inconsistency, aborting\n", fn_name );
        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR );
        exit( 1 );
    }

    return ret;
}

/**
 * BLOCK_XXX: Excerpted from Quinn's book pp 120.
 *
 * Code to partition a set of size n into p subsets.  The subsets are
 * roughly of the same size, the sizes are integral.
 */
#define BLOCK_LOW(id, p, n)     block_low(id, p, n)
#define BLOCK_HIGH(id, p, n)    block_high(id, p, n)
#define BLOCK_SIZE(id, p, n)    block_size(id, p, n)
#define BLOCK_OWNER(id, p, n)   block_owner(id,p, n)


/**
 * (Remote) processor controller
 */
typedef struct pctl_t {
    int32_t procid;            /* (remote) processor id */
    int32_t msgsize;           /* messeges size */

    mem_t * sndmem;            /* Send message buffer manager */
    void  * sndactive;         /* Pointer to the current memory chunk
                                  to be sent; set to &sndmem when done. */

    mem_t * rcvmem;            /* Receive message buffer manager */

    void *cursor;              /* Pointer to a received message */

    struct pctl_t *next;
} pctl_t;


/**
 * com_t: Communication manager.
 *
 */

typedef struct com_t {
    int32_t procid;            /* My processor id */
    int32_t groupsize;         /* Number of processors */

    point_t *interval;         /* interval table (malloc'ed) */

    int32_t nbrcnt;            /* Number of neighboring processors */
    pctl_t **pctltab;          /* Keep track of processor controllers */
    pctl_t *firstpctl;         /* A link list of neighboring processors */

} com_t;


/* static routines: pctl_ */
static pctl_t *  pctl_new(int32_t procid, int32_t msgsize);
static void      pctl_delete(pctl_t *pctl);
static pctl_t *  pctl_reset(pctl_t *pctl, int32_t msgsize);


/* static routines: com_ */
static com_t *   com_new(int32_t procid, int32_t groupsize);
static void      com_delete(com_t *com);

static int32_t   com_allocpctl(com_t *com, int32_t msgsize, oct_t *first,
                               tick_t nearendp[3], tick_t farendp[3],
                               tick_t surfacep, double ticksize,
                               bldgs_nodesearch_t *bldgs_nodesearch);
static int32_t   com_resetpctl(com_t *com, int32_t msgsize);


/* calls that might hang */
static void      com_OrchestrateExchange(com_t *com, int32_t msgtag, MPI_Comm comm_for_this_tree);




/***************************/
/* Math routines           */
/***************************/

/**
 * theLogLookupTable: quick result for log(v) (v <= 255)
 *
 * Note: log(0) is defined to be -1.
 *
 */
static const char theLogLookupTable[] =
{
    0xff, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};



#define LOG2_16u(v) \
(((v) >> 24)    ? 24 + theLogLookupTable[(v) >> 24] : \
             16 + theLogLookupTable[((v) >> 16) & 0xff])

#define LOG2_16l(v) \
(((v) & 0xff00) ?  8 + theLogLookupTable[((v) & 0xff00) >> 8] :\
                 theLogLookupTable[(v) & 0xff])

#define LOG2_32b(v) (((v) >> 16) ? LOG2_16u(v) : LOG2_16l(v))

/* static routines: math_ */
static int32_t math_gcd(int32_t a, int32_t b);
static int32_t math_zsearch(const point_t *interval, int32_t size,
                            const point_t *key);
static int32_t math_bsearch(const int64_t *start, int32_t count, int64_t key);
static uint32_t math_hashuint32(const void *start, int32_t count);



/****************************/
/* Memory management        */
/****************************/

/**
 * newmem: Return a handle (pointer) to an object manager. No data memory
 *          is allocated yet. Return NULL on error.
 */
static mem_t *
mem_new( size_t objsize, int chunkcapacity )
{
    mem_t* mem = (mem_t*)malloc(sizeof(mem_t));

    if (mem != NULL) {
        theAllocatedMemSum += sizeof(mem_t);

        /* Adjust the objsize to accomodate at least a pointer */
        objsize = (objsize < (int)sizeof(void *)) ?
            (int)sizeof(void *) : objsize;

        mem->objsize       = objsize;
        mem->chunkcapacity = chunkcapacity;
        mem->chunkdatasize = objsize * chunkcapacity;
        mem->chunkbytesize = mem->chunkdatasize + sizeof(hdr_t)
            + sizeof(tlr_t);
        mem->chunkcount    = 0;
        mem->objcount      = 0;
        mem->recycled      = NULL;
        mem->first         = NULL;
        mem->last          = NULL;
        mem->total_count   = 0;

        mem->cursor.inited = 0;
    }

    return mem;
}


/**
 * cleanmem: Free all the memory chunks managed by this mem.
 *
 */
static void
mem_clean(mem_t *mem)
{
    if (mem != NULL) {
        hdr_t *hdr, *next;
        tlr_t *tlr;

        hdr = (hdr_t *)mem->first;

        while (hdr != NULL) {
            tlr = (tlr_t *)((char *)hdr + sizeof(hdr_t) + mem->chunkdatasize);
            next = tlr->nextchunk;

            free(hdr);  /* a chunk of memory has been cast to hdr_t * type */
            hdr = next;
        }

        mem->chunkcount = 0;
        mem->objcount   = 0;
        mem->total_count = 0;
        mem->recycled   = NULL;
        mem->first      = NULL;
        mem->last       = NULL;
    }

    return;
}


/**
 * mem_delete: Release all the memory associated with this mem
 *
 */
static void
mem_delete(mem_t *mem)
{
    if (mem != NULL) {
        mem_clean(mem);
        free(mem);
    }

    return;
}


/**
 * new: Return a data object from an object manager. NULL on error.
 *
 */
static void *
mem_newobj(mem_t *mem)
{
    void *newobj;

    if (mem->recycled != NULL) {
        /* Try to make use of the recycled objects first */
        void *next_recycled;

        newobj = mem->recycled;

        next_recycled = (void *)(*(void **)mem->recycled);
        mem->recycled = next_recycled;

    } else {
        hdr_t *hdr;
        tlr_t *tlr;

        if (mem->last != NULL) {
            hdr = (hdr_t *)mem->last;
            tlr = (tlr_t *)((char *)hdr +
                            sizeof(hdr_t) + mem->chunkdatasize);
        } else {
            hdr = NULL;
            tlr = NULL;
        }

        if ((hdr == NULL) || (hdr->used == hdr->capacity)) {
            /* Allocate a new chunk */

            void *newchunk;
            hdr_t *newhdr;
            tlr_t *newtlr;

            newchunk = malloc(mem->chunkbytesize);

            if (newchunk == NULL) {
                fprintf(stderr, "mem_newobj: out of memory\n");
                fprintf(stderr, "mem_newobj: attemping allocating %d bytes\n",
                        (int)mem->chunkbytesize);
                fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                        (int)(theAllocatedMemSum));
                return NULL;
            } else {
                theAllocatedMemSum += mem->chunkbytesize;
            }

            /* Initialize the new chunk header */
            newhdr = (hdr_t *)newchunk;
            newhdr->capacity = mem->chunkcapacity;
            newhdr->used = 0;
            newhdr->nextchunk = NULL;

            /* Initialize the new chunk trailer */
            newtlr = (tlr_t *)((char *)newchunk + sizeof(hdr_t) +
                               mem->chunkdatasize);

            newhdr->nextobj = (char *)newchunk + sizeof(hdr_t);
            newtlr->nextchunk = NULL;

            if (hdr == NULL) {
                /* This is the first chunk */
                mem->first = newchunk;
            } else {
                /* Link in the new chunk */
                hdr->nextchunk = newhdr;
                tlr->nextchunk = newhdr;
            }

            mem->chunkcount++;
            mem->last = newchunk;

            hdr = newhdr;
            tlr = newtlr;
        }

        /* We can use this chunk safely */
        newobj = hdr->nextobj;

        hdr->nextobj = (char *)hdr->nextobj + mem->objsize;
        hdr->used++;
    }

    mem->total_count++;
    mem->objcount++;

    return newobj;
}


/**
 * recycle: Put an allcoated obj in a recycle list.
 *
 */
static void
mem_recycleobj(mem_t *mem, void *obj)
{
    void **blob;

    if (obj != NULL) {
        blob = (void **)obj;

        *(blob) = mem->recycled;
        mem->recycled = obj;
        mem->objcount--;
    }

    return;
}


/**
 * mem_initcursor: Set pointer to the beginning of a memory buffer.
 *                 Assume there are no holes in the memory buffer.
 */
static void
mem_initcursor(mem_t *mem)
{
    mem->cursor.inited = 1;

    if (mem->first == NULL) {
        mem->cursor.chunkptr = NULL;
        mem->cursor.objptr = NULL;

    } else {
        mem->cursor.chunkptr = mem->first;
        mem->cursor.objptr = (unsigned char *)mem->first + sizeof(hdr_t);
        mem->cursor.objoffset = 0;
    }

    return;
}


/**
 * mem_getcursor: Get an object from the (rcv) memory buffer.
 *                 Assume there are no holes in the memory buffer.
 */
static void *
mem_getcursor(mem_t *mem)
{
    if (mem->cursor.inited == 0)
        return NULL;
    else
        return mem->cursor.objptr;
}


/**
 * mem_advcursor: Move to the next allocated obj (message) in a (rcv) buffer.
 *                 Assume there are no holes in the memory buffer.
 *
 */
static void
mem_advcursor(mem_t *mem)
{
    if (mem->cursor.inited == 0)
        /* Do nothing */
        return;
    else if (mem->cursor.chunkptr == NULL)
        return;
    else {
        hdr_t *hdr;

        hdr = (hdr_t *)mem->cursor.chunkptr;

        mem->cursor.objoffset++;
        if (mem->cursor.objoffset < hdr->used) {
            mem->cursor.objptr =
                (unsigned char *)mem->cursor.objptr + mem->objsize;

        } else {
            /* We have visisted all the objects on the current chunk,
               move to the next one */
            tlr_t * tlr;

            tlr = (tlr_t *)((char *)hdr + sizeof(hdr_t) + mem->chunkdatasize);

            if (tlr->nextchunk == NULL) {
                mem->cursor.chunkptr = NULL;
                mem->cursor.objptr = NULL;

            } else {
                mem->cursor.chunkptr = tlr->nextchunk;
                mem->cursor.objptr =
                    (unsigned char *)tlr->nextchunk + sizeof(hdr_t);
                mem->cursor.objoffset = 0;
            }
        }
        return;
    }
}




/***************************/
/* Octant-level operations */
/***************************/

/**
 * oct_getleftmost: Return the leftmost LOCAL leaf oct.
 *
 */
static oct_t *
oct_getleftmost(oct_t *oct)
{
    if (oct->type == LEAF) {
        if (oct->where ==  LOCAL)
            return oct;
        else
            return NULL;

    } else {
        /* this oct is an interior octant */
        int which;

        for (which = 0; which < 8; which++) {
            if (oct->payload.interior->child[which] != NULL) {
                oct_t *lmoct;

                lmoct = oct_getleftmost(oct->payload.interior->child[which]);
                if (lmoct != NULL)
                    return lmoct;
                else
                    continue;
            }
        } /* for */
    }

    /* The point should not be reached */
    return NULL;
}


/**
 * oct_getnext: get the next leaf oct in the preorder/postorder.
 *
 * - The input parameter 'oct' may be an interior oct.
 */
static oct_t *
oct_getnextleaf(oct_t *oct)
{
    oct_t *parent;
    int8_t whoami;
    int32_t which;

    parent = oct->parent;

    if (parent == NULL)
        /* We are at the root, no next leaf oct exists */
        return NULL;

    whoami = oct->which;;

    for (which = whoami + 1; which < 8; which++) {
        /* Move to the next one at the same level */
//yigit
        if (parent->payload.interior->child[which] != NULL && parent->payload.interior->child[which]->where !=REMOTE) {
            return oct_getleftmost(parent->payload.interior->child[which]);
        }
    }

    /* No more siblings on the same level. Go one level up */
    return oct_getnextleaf(parent);
}


/**
 * oct_linkleaf: put a leaf oct into the level list it belongs to
 *               and update the stats
 *
 */
static void
oct_linkleaf(oct_t *oct, oct_t *toleaf[])
{
    oct_t *first;

    first = toleaf[(int32_t)oct->level];

    if (first != NULL) {
        first->payload.leaf->prev = oct;
    }

    oct->payload.leaf->next = first;
    oct->payload.leaf->prev = NULL;

    toleaf[(int32_t)oct->level] = oct;

    return;
}


/**
 * oct_unlinkleaf: remove a leaf oct from its level list
 *                 and update the stats
 *
 */
static void
oct_unlinkleaf(oct_t *oct, oct_t *toleaf[])
{
    oct_t *next, *prev;

    next = oct->payload.leaf->next;
    prev = oct->payload.leaf->prev;

    if (next != NULL)
        next->payload.leaf->prev = prev;

    if (prev != NULL)
        prev->payload.leaf->next = next;

    if (toleaf[(int32_t)oct->level] == oct)
        toleaf[(int32_t)oct->level] = next;

    return;
}


/**
 * oct_findneighbor: Return a pointer to the neigbhroing octants. NULL, if
 *               the neigbhor does not exist.
 *
 */
static oct_t *
oct_findneighbor(oct_t *oct, dir_t dir, oct_stack_t *stackp)
{
    oct_t *firstoctant, *ancestor, *mirror;

    /* Find the first octant of direction O on the path s.t.
       theIsAdjacentTable[O][dir] == FALSE  or oct->parent == NULL */

    firstoctant = tree_ascend(oct, dir, stackp);
    ancestor = firstoctant->parent;

    if (ancestor == NULL) {
        /* An indication that this octant does not have a neighboring
           octant in direction "dir" */
        return NULL;

    } else {
        /* Find the neighbor's ancestor */
        if ((dir >= L) && (dir <= F)) {
            /* A face neighbor shares a common ancestor */
            mirror = tree_descend(ancestor, stackp);

        } else {
            /* An edge neighbor may or may not share a common acnestor */
            dir_t O, sharedface;

            O = (dir_t)firstoctant->which;
            sharedface = theCommonFace[O][dir];

            if (sharedface == OMEGA) {
                /* This is the easy case when the neighbor ancestor is
                   the same as the ancestor of the current octant */
                mirror = tree_descend(ancestor, stackp);

            } else {
                /* This happens when the neighbor's ancestor shares a
                   face with the ancestor of the current octant. The
                   strategy is to find the neighbor ancestor, and then
                   go further down to find the expected neighbor. */

                /* Note: stackp now carries the (reflected) hops
                   recorded in the preceding tree_ascend() call. The
                   net effect of the following call is to ascend
                   further up from the ancestor octant until we reach
                   an octant from which we can descend all the way to
                   the neighbor we expect to find. */

                mirror = oct_findneighbor(ancestor, sharedface, stackp);
            }
        }

        return mirror;
    }

}

/**
 * oct_installeaf: Traverse down the tree and expand if necessary
 *                  until reach an octant where the intended leaf could
 *                  be installed.
 *
 */
static int32_t
oct_installleaf(tick_t lx, tick_t ly, tick_t lz, int8_t level, void *data,
                tree_t *tree)
{
    int8_t parentlevel, childlevel;
    oct_t *parentoct, *childoct;

    parentoct = tree->root;

    for (parentlevel = 0; parentlevel < level; parentlevel++) {
        int8_t which;
        int32_t xbit, ybit, zbit;
        int8_t offset;


        /* Parent oct may be a LEAF marked as REMOTE */
        if (parentoct->type == LEAF) {

            /* Sanity check */ //yigit
            if (parentoct->where != REMOTE && parentoct->where != T_UNDEFINED) {
            	fprintf(stderr, "oct_installleaf: fatal error\n");
            	return -1;
            }

            parentoct->where = T_UNDEFINED;  //yigit for buildings

            if (parentoct->payload.leaf != NULL) {

                mem_recycleobj(theRecordPool, parentoct->payload.leaf->data);
                mem_recycleobj(tree->leafmem, parentoct->payload.leaf);
            }

            parentoct->type = INTERIOR;
            parentoct->payload.interior =
                (interior_t *)mem_newobj(tree->interiormem);
            if (!parentoct->payload.interior)
                return -1;

            memset(parentoct->payload.interior, 0, sizeof(interior_t));
        }

        childlevel = parentlevel + 1;
        offset = PIXELLEVEL - childlevel;
        xbit = (lx & (1 << offset)) >> offset;
        ybit = (ly & (1 << offset)) >> offset;
        zbit = (lz & (1 << offset)) >> offset;
        which = (zbit << 2) | (ybit << 1) | (xbit);

        childoct = parentoct->payload.interior->child[(int32_t)which];

        if (childoct == NULL) {
            /* Allocate an oct for a non-existent branch */
            childoct = (oct_t *)mem_newobj(tree->octmem);
            if (!childoct) {
                /* out of memory */
                return -1;
            }
            parentoct->payload.interior->child[(int32_t)which] = childoct;


            /* Initialize the fields common to LEAF and INTERIOR*/
            childoct->which = which;
            childoct->level = childlevel;
            childoct->lx = parentoct->lx | (xbit << offset);
            childoct->ly = parentoct->ly | (ybit << offset);
            childoct->lz = parentoct->lz | (zbit << offset);
            childoct->appdata = NULL;

            childoct->parent = parentoct;

            /* Property-specific assignment */
            if (childlevel == level) {
                childoct->type = LEAF;
                childoct->where = LOCAL;   /* Applicable to LEAF only */

                childoct->payload.leaf = (leaf_t *)mem_newobj(tree->leafmem);
                if (!childoct->payload.leaf)
                    return -1;

                childoct->payload.leaf->next = childoct->payload.leaf->prev
                    = NULL;

                childoct->payload.leaf->data = mem_newobj(theRecordPool);
                if (!childoct->payload.leaf->data)
                    return -1;

                memcpy(childoct->payload.leaf->data, data, tree->recsize);

            } else {
                childoct->type = INTERIOR; /* We have to go further down */
                childoct->where = T_UNDEFINED;

                childoct->payload.interior =
                    (interior_t *)mem_newobj(tree->interiormem);
                if (!childoct->payload.interior)
                    return -1;

                memset(childoct->payload.interior, 0, sizeof(interior_t));
            }
        } /* if childoct == NULL */

        parentoct = childoct;

    } /* for parentlevel < level */

    return 0;
}



/**
 * oct_sprout: sprout a leaf octant to an interior octant. Create
 *             children octants if they are inside the domain boudary.
 *             The record payloads of the newly creately children leaf
 *             octants are not allocated.
 *
 * - Return the number of children; -1 on error.
 */
static int32_t
oct_sprout(oct_t *leafoct, tree_t *tree)
{
    int32_t which;
    int8_t childlevel;
    int32_t childcount;
    oct_t *child = NULL;

    /* Release leaf payload memory if necessary */
    if (leafoct->payload.leaf != NULL) {
        mem_recycleobj(theRecordPool, leafoct->payload.leaf->data);
        mem_recycleobj(tree->leafmem, leafoct->payload.leaf);
    }

    /* change current node to an interior node and allocate memory  */
    leafoct->type = INTERIOR;
    leafoct->payload.interior = (interior_t *)mem_newobj(tree->interiormem);
    if (leafoct->payload.interior == NULL) {
        /* Out of memory */
        return -1;
    }

    tree->leafcount[(int32_t)leafoct->level]--;

    /* move one level down */
    childlevel = leafoct->level + 1;

    childcount = 0;

    for (which = 7; which >= 0; which--) {
        tick_t childlx, childly, childlz;

        childlx = leafoct->lx +
            ((0x1 & which) ? ((tick_t)1 << (PIXELLEVEL - childlevel)) : 0);

        childly = leafoct->ly +
            ((0x2 & which) ? ((tick_t)1 << (PIXELLEVEL - childlevel)) : 0);

        childlz = leafoct->lz +
            ((0x4 & which) ? ((tick_t)1 << (PIXELLEVEL - childlevel)) : 0);

        if ( ( childlx >= tree->farendp[0] ) ||
             ( childly >= tree->farendp[1] ) ||
             ( childlz >= tree->farendp[2] ) )
        {
            /* this child is out of the domain */
            leafoct->payload.interior->child[which] = NULL;

        } else {
            /* create a new oct for the child */

            if ((child = (oct_t *)mem_newobj(tree->octmem)) == NULL)
                return -1;

            tree->leafcount[(int32_t)childlevel]++;
            childcount++;

            child->type = LEAF;
            child->where =  leafoct->where; /* inherit the whereabout */
            child->which = which;
            child->level = childlevel;
            child->lx = childlx;
            child->ly = childly;
            child->lz = childlz;
            child->appdata = NULL;

            /* The payload of a leaf oct is not initialized */
            child->payload.leaf = NULL;

            /* link into the tree structure */
            child->parent = leafoct;
            leafoct->payload.interior->child[which] = child;
        }
    } /* for which */

    /* Adjust the firstleaf pointer to point to the first child */
    tree->firstleaf = (tree->firstleaf == leafoct) ? child : tree->firstleaf;

    return childcount;
}


/**
 * oct_prune: Prune an interior oct to a leaf oct. All the children
 *         must exist.
 */
static void
oct_prune(oct_t *oct, tree_t *tree)
{
    oct_t *child;
    int8_t which;
    unsigned char where;

    /* Release all the child octs */
    for (which = 7; which >=0; which--) {
        child = oct->payload.interior->child[(int32_t)which];
        where = child->where;

        if (child->payload.leaf != NULL) {
            mem_recycleobj(theRecordPool, child->payload.leaf->data);
            mem_recycleobj(tree->leafmem, child->payload.leaf);
        }

        mem_recycleobj(tree->octmem, child);
    }

    /* Mark the oct as leaf */
    oct->type = LEAF;
    oct->where = where;
    oct->payload.leaf = NULL; /* uinitialized */

    if (oct->where ==  LOCAL) {
        tree->leafcount[(int32_t)child->level] -= 8;
        tree->leafcount[(int32_t)oct->level]++;

        /* Ajust the firstleaf pointer to point to the new leaf oct  */
        tree->firstleaf = (tree->firstleaf == child) ? oct : tree->firstleaf;
    }

    return;
}


/**
 * oct_expand: On entering this function, oct must have intersect
 *         the domain of interest.
 *
 * - Return 0 if succeeds. -1 on error.
 */
static int32_t
oct_expand(oct_t *leafoct, tree_t *tree, toexpand_t *toexpand,
           setrec_t *setrec)
{
    int8_t which;
    const void *data;
    int32_t isOverlapped;
    tick_t ur[3];

    /* Point *data to the leaf payload if the user has initialized
       the data */
    if (leafoct->payload.leaf == NULL)
        data = NULL;
    else
        data = leafoct->payload.leaf->data;

    ur[0] = leafoct->lx + ((tick_t)1 << (PIXELLEVEL - leafoct->level));
    ur[1] = leafoct->ly + ((tick_t)1 << (PIXELLEVEL - leafoct->level));
    ur[2] = leafoct->lz + ((tick_t)1 << (PIXELLEVEL - leafoct->level));

    if ((ur[0] > tree->farendp[0]) ||
        (ur[1] > tree->farendp[1]) ||
        (ur[2] > tree->farendp[2])) {
        /* this leaf octant lies cross the boundary of the domain */
        isOverlapped = 1;
    } else {
        isOverlapped = 0;
    }


    if (isOverlapped ||
        (toexpand((octant_t *)leafoct, tree->ticksize, data) == 1)) {

        /* strutural change */
        if (oct_sprout(leafoct, tree) == -1) {
            /* Out of memory */
            return -1;
        }

        for (which = 0; which < 8; which++) {
            oct_t *child;

            child = leafoct->payload.interior->child[(int32_t)which];

            if (child != NULL) {
                void *data;

                /* child->payload.leaf == NULL */
                if ((child->payload.leaf =
                     (leaf_t *)mem_newobj(tree->leafmem)) == NULL){
                    /*Out of memory */
                    return -1;
                }

                child->payload.leaf->next = child->payload.leaf->prev = NULL;

                data = child->payload.leaf->data = mem_newobj(theRecordPool);
                if (data == NULL) {
                    return -1;
                }

                /* instantiate the child */
                setrec((octant_t *)child, tree->ticksize, data);

                /* This child intersects the domain of interest */
                if (oct_expand(child, tree, toexpand, setrec) != 0)
                    return -1;
            }
        }
    }

    return 0;
}



/**
 * oct_shrink: Shrink (leaf) octants as specified by the toshrink
 *            function.
 *
 */
static oct_t *
oct_shrink(oct_t *oct, tree_t *tree, unsigned char where,
           toshrink_t *toshrink, setrec_t *setrec)
{
    int8_t which;
    oct_t *child;
    int8_t childrentype;

    if (oct->type == LEAF) {
        /* We cannot shrink a leaf */
        return oct;
    }

    childrentype = LEAF;
    for (which = 0; which < 8; which++) {
        child = oct->payload.interior->child[(int32_t)which];
        if (child == NULL) {
            /* Mark the parent as non-aggregatable */
            childrentype = INTERIOR;
            continue;
        }

        child = oct_shrink(child, tree, where, toshrink, setrec);

        if (child == NULL)
            return NULL;

        childrentype =
            (child->type == INTERIOR) ? INTERIOR : childrentype;
    }

    if (childrentype == LEAF) { /* All children exist */
        unsigned char childrenwhereabout;
        octant_t *leafoctant[8];
        const void *data[8];
        unsigned char opposite;

        opposite = (where == LOCAL) ? REMOTE : LOCAL;

        childrenwhereabout = where;
        for (which = 7; which >= 0; which--) {
            child = oct->payload.interior->child[(int32_t)which];
            childrenwhereabout =
                (child->where == opposite) ? opposite : childrenwhereabout;

            if (where == LOCAL) {
                /* Only invoke user supplied toshrink() function
                   if shrinking LOCAL octants */
                leafoctant[(int32_t)which] = (octant_t *)child;
                data[(int32_t)which] = child->payload.leaf->data;
            }
        }

        if (((where ==  REMOTE) &&
             (childrenwhereabout ==  REMOTE)) ||
            ((where ==  LOCAL) &&
             (childrenwhereabout == LOCAL) &&
             (toshrink(leafoctant, tree->ticksize, data) == 1))) {

            oct_prune(oct, tree);

            if (where == LOCAL) { /* same as oct->where */
                void *data;

                /* oct->payload.leaf == NULL */
                oct->payload.leaf = (leaf_t *)mem_newobj(tree->leafmem);
                if (oct->payload.leaf == NULL) {
                    /* out of memory */
                    return NULL;
                }

                oct->payload.leaf->next = oct->payload.leaf->prev = NULL;
                data = oct->payload.leaf->data = mem_newobj(theRecordPool);
                if (data == NULL) {
                    return NULL;
                }

                setrec((octant_t *)oct, tree->ticksize, data);
            }
        }
    }

    return oct;
}


/**
 * octor_getchild: Return the specified child octant. If parentoctant is a
 *                 LEAF or the specified child does not exist. Return NULL.
 *
 */
extern octant_t *
octor_getchild(octant_t *parentoctant, int32_t whichchild)
{
    oct_t *poct = (oct_t *)parentoctant;

    if ((poct->type == LEAF) || (whichchild > 7) || (whichchild < 0))
        return NULL;

    return (octant_t *)(poct->payload.interior->child[whichchild]);
}


/**
 * octor_getfirstleaf: Get the first LOCAL leaf
 *
 */
extern octant_t *
octor_getfirstleaf(octree_t *octree)
{
    return (octant_t *)oct_getleftmost((oct_t *)octree->root);
}


/**
 * octor_getnextleaf: Get the next leaf octant.
 *
 */
extern octant_t *
octor_getnextleaf(octant_t *octant)
{
    return (octant_t *)oct_getnextleaf((oct_t *)octant);
}


/**
 * octor_getleafrec: Retrieve the record stored in a leaf octant. Return
 *                    0 if OK, -1 on error.
 */
extern int32_t
octor_getleafrec(octree_t *octree, octant_t *octant, void *data)
{
    oct_t *oct = (oct_t *)octant;
    tree_t *tree = (tree_t *)octree;

    if (oct->type == INTERIOR) {
        return -1;
    } else {
        if (oct->payload.leaf == NULL) {
            /* Undefined leaf payload */
            return -1;
        } else {
            memcpy(data, oct->payload.leaf->data, tree->recsize);
            return 0;
        }
    }
}


/**
 * octor_searchoctant: Look for the octant specified by (x, y, z, level).
 *                     Return a pointer to the octant if found (exact or
 *                     aggregate). NULL, if not found.
 *
 */
extern octant_t *
octor_searchoctant(octree_t *octree, tick_t x, tick_t y, tick_t z,
                   int8_t level, int32_t searchtype)
{
    tree_t *tree = (tree_t *)octree;

    return (octant_t *)tree_searchoct(tree, x, y, z, level, searchtype);
}


extern int32_t
octor_getminleaflevel(const octree_t* octree, int where)
{
    const tree_t* tree = (const tree_t*)octree;
    int32_t lmin, gmin;

    lmin = tree_getminleaflevel(tree);
    if (where == LOCAL) {
        return lmin;
    } else {
        if (tree->groupsize > 1) {
            MPI_Allreduce(&lmin, &gmin, 1, MPI_INT, MPI_MIN, tree->comm_tree);
            return gmin;
        } else {
            return lmin;
        }
    }
}


extern int32_t
octor_getmaxleaflevel(const octree_t* octree, int where)
{
    const tree_t *tree = (const tree_t*)octree;
    int32_t lmax, gmax;

    lmax = tree_getmaxleaflevel(tree);
    if (where == LOCAL) {
        return lmax;
    } else {
        if (tree->groupsize > 1) {
            MPI_Allreduce(&lmax, &gmax, 1, MPI_INT, MPI_MAX, tree->comm_tree);
            return gmax;
        } else {
            return lmax;
        }
    }
}

extern int64_t
octor_getleavescount(const octree_t* octree, int where)
{
    tree_t *tree = (tree_t*)octree;
    int64_t lcount, gcount;

	lcount = tree_countleaves(tree);
	if (where == LOCAL) {
		return lcount;
	} else {
		if (tree->groupsize > 1) {
			MPI_Allreduce(&lcount, &gcount, 1, MPI_INT, MPI_SUM, tree->comm_tree);
			return gcount;
		} else {
			return lcount;
		}
	}
}

extern int64_t
octor_getminleavescount(const octree_t* octree, int where)
{
    tree_t *tree = (tree_t*)octree;
    int64_t lcount, gcount;

	lcount = tree_countleaves(tree);
	if (where == LOCAL) {
		return lcount;
	} else {
		if (tree->groupsize > 1) {
			MPI_Allreduce(&lcount, &gcount, 1, MPI_INT, MPI_MIN, tree->comm_tree);
			return gcount;
		} else {
			return lcount;
		}
	}
}

extern int64_t
octor_getmaxleavescount(const octree_t* octree, int where)
{
    tree_t *tree = (tree_t*)octree;
    int64_t lcount, gcount;

	lcount = tree_countleaves(tree);
	if (where == LOCAL) {
		return lcount;
	} else {
		if (tree->groupsize > 1) {
			MPI_Allreduce(&lcount, &gcount, 1, MPI_INT, MPI_MAX, tree->comm_tree);
			return gcount;
		} else {
			return lcount;
		}
	}
}


/*************************/
/* Tree-level operations */
/*************************/



/**
 * tree_getminleaflevel: Return the minimum LOCAL leaf level.
 *
 */
static int8_t tree_getminleaflevel(const tree_t* tree)
{
    int8_t level;

    for (level = 0; level < TOTALLEVEL; level++) {
        if (tree->leafcount[(int32_t)level] > 0)
            return level;
    }

    return -1; /* indicate non-existence */
}


/**
 * tree_getmaxleaflevel: return the max LOCAL leaf level.
 *
 */
static int8_t tree_getmaxleaflevel(const tree_t* tree)
{
    int8_t level;

    for (level = TOTALLEVEL - 1; level >=0; level--) {
        if (tree->leafcount[(int32_t)level] > 0)
            return level;
    }

    return TOTALLEVEL; /* indicate non-existence */

}


/**
 * tree_countleaves: Return the number of LOCAL leaf octants.
 *
 */
static int64_t tree_countleaves(tree_t *tree)
{
    int8_t level;
    int64_t localtotal;

    localtotal = 0;

    for (level = 0; level < TOTALLEVEL; level++) {
        localtotal += tree->leafcount[(int32_t)level];
    }

    return localtotal;
}


/**
 * tree_setdistribution:
 *
 */
static void  tree_setdistribution(tree_t *tree, int64_t **pCountTable,
                                  int64_t **pStartTable, int64_t localcount)
{
    int64_t *counttable, *starttable;
    int32_t procid;

    if ((counttable = (int64_t *)malloc(sizeof(int64_t) * tree->groupsize))
        == NULL) {
        /* Out of memory */
        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                tree->procid, __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %dMB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));

        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        /* Update the memory allocation stat */
        theAllocatedMemSum += sizeof(int64_t) * tree->groupsize;
    }

    /* Global communication */
    MPI_Allgather(&localcount, sizeof(localcount), MPI_CHAR,
                  counttable, sizeof(localcount), MPI_CHAR,
                  tree->comm_tree);

    /* Determine how the octants are distributed among the processors */
    if ((starttable = (int64_t *)malloc(sizeof(int64_t) * tree->groupsize))
        == NULL) {
        /* Out of memory */
        fprintf(stderr, "Thread %d: %s %d: out of memory\n", tree->procid,
                __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %dMB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));

        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        /* Update the memory allocation stat */
        theAllocatedMemSum += sizeof(int64_t) * tree->groupsize;
    }


    starttable[0] = 0;
    for (procid = 1; procid < tree->groupsize; procid++)
        starttable[procid] = starttable[procid - 1] + counttable[procid - 1];

    *pCountTable = counttable;
    *pStartTable = starttable;

    return;
}

#ifdef TREE_VERBOSE
/**
 * tree_showstat: Print on the stdout the statistics of the octant
 *                distribution among processors.
 *
 */
static void
tree_showstat(tree_t *tree, int32_t mode, const char *comment)
{
    int32_t localcount;
    int32_t *counttable;

    localcount = (int32_t)tree_countleaves(tree);

    if (tree->procid == 0) {
        counttable = (int32_t *)calloc(sizeof(int32_t), tree->groupsize);
        if (counttable == NULL) {
            fprintf(stderr, "Thread 0: tree_showstat: out of memory\n");
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        }
    }

    MPI_Gather(&localcount, 1, MPI_INT, counttable, 1, MPI_INT, 0,
               tree->comm_tree);

    if (tree->procid == 0) {
        int32_t procid, min, max, total, min_id = 0, max_id = 0;
        double squaredsum, mean, meansquared, var, coeff;

        total = 0;
        min = (~0) ^ (0x80000000);
        max = 0;
        squaredsum = 0;

        for (procid = 0; procid < tree->groupsize; procid++) {
            total += counttable[procid];

            if (min > counttable[procid]) {
                min = counttable[procid];
                min_id = procid;
            }

            if (max < counttable[procid]) {
                max = counttable[procid];
                max_id = procid;
            }

            squaredsum += 1.0 * counttable[procid] * counttable[procid];
        }

        mean = total * 1.0 / tree->groupsize;
        meansquared = mean * mean;

        var = (squaredsum / tree->groupsize) - meansquared;
        coeff = sqrt(var) / mean;

        printf("\nTREE_INFO: %s = %d, %s = %.3f, %s(proc %d)= %d, %s(proc %d) = %d\n",
               "#leaves (avg)", (int32_t)mean,
                "coef", coeff, "max", max_id, max, "min", min_id, min);

        if (mode == DETAILS) {
            for (procid = 0; procid < tree->groupsize; procid++) {
                fprintf(stdout, "Proc %d: %d\n", procid, counttable[procid]);
            }
        }

        fprintf(stdout, 0);

        free(counttable);
    }

    return;
}


#endif /* TREE_VERBOSE */




/**
 * tree_setcom: Allocate and initialize data structure to support
 *               inter-subtree communication.
 *
 * - Return 0 if OK, -1 on error.
 *
 */
static int32_t
tree_setcom(tree_t *tree, int32_t msgsize, bldgs_nodesearch_t *bldgs_nodesearch)
{
    /* Allocate a communication manager */
    tree->com = com_new(tree->procid, tree->groupsize);
    if (tree->com == NULL)
        return -1;

    /* Global communication */
    MPI_Allgather(&tree->firstleaf->lx, sizeof(point_t), MPI_CHAR,
                  tree->com->interval, sizeof(point_t), MPI_CHAR,
                  tree->comm_tree);

    /*
       LOCAL op: check neighbors for each LOCAL leaf oct. This routine
       assume that the distributed octree conforms to the 2-to-1
       constraint (balanced in mesh generation term).
    */
    if (com_allocpctl(tree->com, msgsize, tree->firstleaf,
                      tree->nearendp, tree->farendp, tree->surfacep,
                      tree->ticksize, bldgs_nodesearch) != 0)
        return -1;

    return 0;
}


/**
 * tree_deletecom: delete the current communication manager.
 *                  Useful if the geometry boundary of each parition
 *                  changes over the time.
 *
 */
static void
tree_deletecom(tree_t *tree)
{
    if (tree->com == NULL)
        return;
    else {
        com_delete(tree->com);
        tree->com = NULL;
    }

    return;
}



/**
 * tree_ascendtree: Ascend the octree recursively.
 *
 * - In any possible case, go up one level first (conceptually) by
 *   pushing the value of theDirReflectTable[O][I] into the stack.
 * - Return a pointer to the first encountered  octant
 *   (of direction) O on the path to the root octant, s.t.
 *   theIsAdjacentTable[O][I] == FALSE; or NULL if none exists.
 * - NOTE: the reflected hop from the returned octant to its parent
 *   has been pushed onto the stack already!
 *
 */
static oct_t *
tree_ascend(oct_t *oct, dir_t I, oct_stack_t *stackp)
{
    dir_t O, opposite;

    if (oct->parent != NULL) {
        /* Get the direction of 'oct' and find its opposite direction
           wrt. direction I */
        O = (dir_t)oct->which;
        opposite = theDirReflectTable[O][I];

        /* Push the opposite direction to the stack */
        stackp->dir[(int32_t)stackp->top] = opposite;
        stackp->top++;

        /* Conduct isadjacent test */
        if (theIsAdjacentTable[O][I]) {
            return tree_ascend(oct->parent, I, stackp);
        }
    }

    return oct;

}


/**
 * tree_descend: descend the local octree either till we cannot
 *              go further down as required by the path info in the stack
 *
 */
static oct_t *
tree_descend(oct_t *oct, oct_stack_t *stackp)
{
    while (stackp->top > 0) {
        if ((oct == NULL) || (oct->type == LEAF)) {
            break;
        } else {
            dir_t dir;

            stackp->top--;
            dir = (dir_t)stackp->dir[(int32_t)stackp->top];

            oct = oct->payload.interior->child[dir];
        }
    } /* descend until we cannot go further down */

    return oct;
}


/**
 * tree_pushdown: go down the tree as many level necessary. Sprout leaf
 *               octants on the path.
 *
 *  - Return 0 if OK, -1 if out of memory.
 */
static int32_t
tree_pushdown(oct_t *nbr, tree_t *tree, oct_stack_t *stackp, setrec_t *setrec)
{
    dir_t dir;

    while (stackp->top > 1) {
        int8_t which;

        if (nbr->type == LEAF) {
            oct_unlinkleaf(nbr, tree->toleaf);
            if (oct_sprout(nbr, tree) == -1) {
                /* out of memory */
                return -1;
            }

            for (which = 0; which < 8; which++) {
                oct_t *child;
                void *data;

                /* instantiate the child */
                child = nbr->payload.interior->child[(int32_t)which];

                /* child->payload.leaf == NULL */
                child->payload.leaf =
                    (leaf_t *)mem_newobj(tree->leafmem);
                if (child->payload.leaf == NULL) {
                    return -1;
                }

                child->payload.leaf->next = child->payload.leaf->prev = NULL;
                data = child->payload.leaf->data = mem_newobj(theRecordPool);
                if (!data) {
                    return -1;
                }

                setrec((octant_t *)child, tree->ticksize, data);

                /* add child to the corresponding link level list */
                oct_linkleaf(child, tree->toleaf);
            }
        }

        /* nbr->type == INTERIOR. So move one level down comfortably */
        stackp->top--;
        dir = (dir_t)stackp->dir[(int32_t)stackp->top];
        nbr = nbr->payload.interior->child[dir];
    } /* while stackp->top */

    return 0;
}


/**
 * searchoct: Given the location (lx, ly, lz, level) of an oct, return
 *            a pointer to the oct. If not found, return NULL
 *
 */
static oct_t *
tree_searchoct(tree_t *tree, tick_t lx, tick_t ly, tick_t lz, int8_t level,
               int32_t searchtype)
{
    oct_t *oct;

    oct = tree->root;
    while (oct != NULL) {
        if (oct->level == level) {
            if ((oct->lx == lx) && (oct->ly == ly) && (oct->lz == lz))
                return oct;
            else
                /* Something must be wrong */
                return NULL;
        } else {
            if (oct->type == LEAF) {
                if (searchtype == EXACT_SEARCH) {
                    /* Fail to locate the oct exactly */
                    return NULL;
                } else {
                    /* Still this is an aggregate hit */
                    return oct;
                }
            } else {
                /* We still have hope. Find the branch to descend */

                int8_t which, xbit, ybit, zbit;
                tick_t edgesize_half;

                edgesize_half = (tick_t)1 << (PIXELLEVEL - oct->level - 1);
                xbit = ((lx - oct->lx) >= edgesize_half) ? 1 : 0;
                ybit = ((ly - oct->ly) >= edgesize_half) ? 1 : 0;
                zbit = ((lz - oct->lz) >= edgesize_half) ? 1 : 0;

                which = (zbit << 2) | (ybit << 1) | xbit;

                oct = oct->payload.interior->child[(int32_t)which];
            }
        }
    }

    return NULL;
}



/****************************/
/* Communications           */
/****************************/

/**
 * pctl_new:
 *
 */
static pctl_t *
pctl_new(int32_t procid, int32_t msgsize)
{
    pctl_t *pctl;

    if ((pctl = (pctl_t *)malloc(sizeof(pctl_t))) == NULL) {
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        return NULL;
    } else {
        theAllocatedMemSum += sizeof(pctl_t);
    }

    pctl->procid = procid;
    pctl->msgsize = msgsize;

    if (((pctl->sndmem = mem_new(msgsize, 1 << 10)) == NULL) ||
        ((pctl->rcvmem = mem_new(msgsize, 1 << 10)) == NULL))
        return NULL;

    pctl->next = NULL;

    return pctl;
}

/**
 * pctl_delete: Release all the memory associated with the pctl
 *
 */
static void
pctl_delete(pctl_t *pctl)
{
    mem_delete(pctl->sndmem);
    mem_delete(pctl->rcvmem);

    free(pctl);

    return;
}

/**
 * pctl_reset: Release the memory used by memory managers and reset
 *             the memory manager for the new type of messages. Return
 *             a pointer to the pctl reset. NULL on error.
 *
 */
static pctl_t *
pctl_reset(pctl_t *pctl, int32_t msgsize)
{
    if (pctl == NULL)
        return NULL;

    pctl->msgsize = msgsize;

    mem_delete(pctl->sndmem);
    mem_delete(pctl->rcvmem);

    if (((pctl->sndmem = mem_new(msgsize, 1 << 10)) == NULL) ||
        ((pctl->rcvmem = mem_new(msgsize, 1 << 10)) == NULL))
        return NULL;

    return pctl;
}





/**
 * com_new: Allocate memory for a communication manager and initialize
 *          properly. Return a pointer to the com_t if OK, NULL on error.
 *
 */
static com_t *
com_new(int32_t procid, int32_t groupsize)
{
    com_t *com;

    com = (com_t *)malloc(sizeof(com_t));
    if (com == NULL) {
        fprintf(stderr, "Total memory allocated by Octor : %d MB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));
        return NULL;
    } else {
        theAllocatedMemSum += sizeof(com_t);
    }

    com->procid = procid;
    com->groupsize = groupsize;

    com->interval = (point_t *)calloc(groupsize, sizeof(point_t));
    if (!com->interval) {
        fprintf(stderr, "Total memory allocated by Octor : %dMB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));
        return NULL;
    } else {
        theAllocatedMemSum +=  groupsize * sizeof(point_t);
    }

    com->nbrcnt = 0;
    com->pctltab = (pctl_t **)calloc(groupsize, sizeof(pctl_t *));
    if (com->pctltab == NULL) {
        fprintf(stderr, "Total memory allocated by Octor : %dMB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));
        return NULL;
    } else {
        theAllocatedMemSum +=  groupsize * sizeof(pctl_t *);
    }

    com->firstpctl = NULL;

    return com;
}


/**
 * com_delete: Release any memory associated with this com.
 *
 */
static void
com_delete(com_t *com)
{
    pctl_t *pctl, *nextctl;

    if (com == NULL)
        return;

    free(com->interval);

    pctl = com->firstpctl;
    while (pctl != NULL) {
        nextctl = pctl->next;

        pctl_delete(pctl);
        pctl = nextctl;
    }

    free(com->pctltab);
    free(com);

    return;
}


/**
 * com_allocpctl: Create processor controllers for neighboring processor.
 *            Return 0 if OK, -1 on error.
 *
 *  - Assume com->interval has been initialized and the octree is globally
 *    balanced in mesh generation term.
 *  - Return 0 if OK , -1 on error.
 */
static int32_t
com_allocpctl(com_t *com, int32_t msgsize, oct_t *firstleaf,
              tick_t nearendp[3], tick_t farendp[3], tick_t surfacep,
              double ticksize, bldgs_nodesearch_t *bldgs_nodesearch)
{
    oct_t *oct;
    tick_t ox, oy, oz, increment;
    point_t pt;
    int32_t procid;

    /* Search neighbor for each LOCAL leaf oct */

    oct = firstleaf;

    while ((oct != NULL) && (oct->where ==  LOCAL)) {
        int32_t i, j, k;
        increment = (tick_t)1 << (PIXELLEVEL - (oct->level + 1));

        ox = oct->lx - increment;
        oy = oct->ly - increment;
        oz = oct->lz - increment;

        for (k = 0; k < 4; k++) {

            pt.z = oz + increment * k;

            /* Discard out of bounds */
            if ( ( pt.z < nearendp[2] ) || ( pt.z >= farendp[2] ) ) {
                continue;
            }

            for (j = 0; j < 4; j++) {

                pt.y = oy + increment * j;

                /* Discard out of bounds */
                if ( ( pt.y < nearendp[1] ) || ( pt.y >= farendp[1] ) ) {
                    continue;
                }

                for (i = 0; i < 4; i++) {

                    pt.x = ox + increment * i;

                    /* Discard out of bounds */
                    if ( ( pt.x < nearendp[0] ) || ( pt.x >= farendp[0] ) ) {
                        continue;
                    }

                    /* If we have pushed the surface down for buildings... */
                    if ( surfacep > 0 ) {

                        /* ...and the point is above the surface... */
                        if ( pt.z < surfacep ) {

                            /* ...and it does not belong to any building... */
                            int res = bldgs_nodesearch(
                                    pt.x, pt.y, pt.z, ticksize);

                            if ( res == 0 ) {

                                /* ...then, discard 'air' nodes! */
                                continue;
                            }
                        }
                    }

                    /* Search the interval table */
                    procid = math_zsearch(com->interval, com->groupsize, &pt);

                    /* Sanity check introduced after the buildings
                     * options were incorporated. Should never occur */
                    if ( ( procid < 0 ) || ( procid > com->groupsize ) ) {
                        fprintf(stderr,
                                "Thread %d: wrong procid from math search at "
                                "com_allocpctl in vertex with coords "
                                "x,y,z = %f %f %f\n",
                                procid,
                                pt.x*ticksize,
                                pt.y*ticksize,
                                pt.z*ticksize);
                    }

                    if ( ( procid == com->procid        /* LOCAL */        ) ||
                         ( com->pctltab[procid] != NULL /* KNOWN REMOTE */ ) )
                        continue;

                    /* We have found a new neighbor */

                    com->pctltab[procid] = pctl_new(procid, msgsize);
                    if (!com->pctltab[procid]) {
                        return -1;
                    } else {
                        /* link into the pctl list */
                        com->pctltab[procid]->next = com->firstpctl;
                        com->firstpctl = com->pctltab[procid];

                        com->nbrcnt++;
                    }
                } /* for i */
            } /* for j */
        } /* for k */

        oct = oct_getnextleaf(oct);
    }

    return 0;
}


/**
 * com_resetpctl: Reset each existing pctl. However, the communication
 *            STRUCTURE (who are my neighbors) does not change.
 *            Return 0 if OK, -1 on error.
 *
 */
static int32_t
com_resetpctl(com_t *com, int32_t newmsgsize)
{
    pctl_t *pctl;

    /* interval is not reset */

    pctl = com->firstpctl;

    while (pctl != NULL) {
        if (pctl_reset(pctl, newmsgsize) != pctl)
            return -1;
        pctl = pctl->next;
    }

    return 0;
}


/**
 * com_OrchestrateExchange: Return if OK, abort and exit otherwise.
 *
 */
static void
com_OrchestrateExchange(com_t *com, int32_t msgtag, MPI_Comm comm_for_this_tree)
{

    int32_t receive_finished, send_returned;
    pctl_t *pctl;
    int32_t isendcount, isendnum;
    MPI_Request *isendreqs;
    MPI_Status *isendstats;

    receive_finished = 0;
    send_returned = 0;

    /* Initialize snd memory chunk cursor and record how many packets
       (data chunks plus termination messages) need to be isent */
    pctl = com->firstpctl;
    isendcount = 0;

    while (pctl != NULL) {
        isendcount += (pctl->sndmem->chunkcount + 1);
        pctl->sndactive = pctl->sndmem->first;
        pctl = pctl->next;
    }


    /* Create MPI request and stat objects */
    isendreqs = (MPI_Request *)malloc(isendcount * sizeof(MPI_Request));
    isendstats = (MPI_Status *)malloc(isendcount * sizeof(MPI_Status));
    if ((isendreqs == NULL) || (isendstats == NULL)) {
        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                com->procid, __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %dMB\n",
                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));
        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        theAllocatedMemSum += (isendcount) *
            (sizeof(MPI_Request) + sizeof(MPI_Status));
    }


    /* Repeat rounds of snd/rcv until all done */
    isendnum = 0;
    while ((send_returned < com->nbrcnt) ||
           (receive_finished < com->nbrcnt)) {

        if (send_returned < com->nbrcnt) {

            /* Send first, if there is data left to be sent */

            pctl = com->firstpctl;
            while (pctl != NULL) {

                if (pctl->sndactive != &pctl->sndmem) {
                    /* There are more messages to send */

                    if (pctl->sndactive == NULL) {
                        /* Send termination message */
                        int32_t chunkssent;

                        chunkssent = pctl->sndmem->chunkcount;

                        MPI_Isend(&chunkssent, 4, MPI_CHAR, pctl->procid,
                                  msgtag, comm_for_this_tree,
                                  &isendreqs[isendnum]);
                        isendnum++;

                        /* Short-circuit the sndactive pointer */
                        pctl->sndactive = &pctl->sndmem;
                        send_returned++;

                    } else {
                        tlr_t * tlr;
                        hdr_t *hdr;
                        int32_t sendbytes;

                        /* Determine how many bytes of useful data I need
                           to send */
                        hdr = (hdr_t *)pctl->sndactive;
                        sendbytes = sizeof(hdr_t) +
                            hdr->used * pctl->sndmem->objsize;

                        /* Send a msgtag page. We don't send the tlr_t of
                           of outgoing page */
                        MPI_Isend(pctl->sndactive, sendbytes, MPI_CHAR,
                                  pctl->procid, msgtag, comm_for_this_tree,
                                  &isendreqs[isendnum]);
                        isendnum++;

                        /* Move to the next chunk. Note that we are not
                           referencing the application send buffer */
                        tlr = (tlr_t *)((char *)pctl->sndactive + sizeof(hdr_t)
                                        + pctl->sndmem->chunkdatasize);
                        pctl->sndactive = tlr->nextchunk;
                    }
                }

                pctl = pctl->next;
            } /* while more pctl for sending to be processed */
        } /* more to send */

        if (receive_finished < com->nbrcnt) {

            MPI_Status status;
            int32_t fromwhom;
            int32_t recvbytes;
            int32_t flag;

            /* Stall until there is at least one message */
            MPI_Probe(MPI_ANY_SOURCE, msgtag, comm_for_this_tree, &status);
            flag = 1; /* indicate that the message is valid */

            while (flag != 0) {

                /* The message is valie */
                MPI_Get_count(&status, MPI_CHAR, &recvbytes);

                fromwhom = status.MPI_SOURCE;

                if (recvbytes == 4) {
                    /* It's a termination message */
                    int32_t chunkssent;

                    MPI_Recv(&chunkssent, 1, MPI_INT, fromwhom, msgtag,
                             comm_for_this_tree, &status);

                    receive_finished++;

#ifdef DEBUG
                    /* Sanity check: MPI requires non-overtaking. See
                       MPI 1.1 standard Section 3.5: Semantics of
                       point-to-point communication. */

                    pctl = com->pctltab[fromwhom];
                    if (chunkssent != pctl->rcvmem->chunkcount) {
                        fprintf(stderr, "Thread %d: %s %d: ",
                                com->procid, __FILE__, __LINE__);
                        fprintf(stderr, "Termination message from Thread %d ",
                                fromwhom);
                        fprintf(stderr, "arrives before data message\n");
                        MPI_Abort(MPI_COMM_WORLD, UNEXPECTED_ERR);
                        exit(1);
                    }
#endif /* DEBUG */

                } else {
                    /* Receive a message page */
                    void *lastchunk, *newchunk;
                    tlr_t *lasttlr, *newtlr;

                    pctl = com->pctltab[fromwhom];

                    if (pctl == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: ",
                                com->procid, __FILE__, __LINE__);
                        fprintf(stderr, "Unexpected message from Thread %d\n",
                                fromwhom);
                        MPI_Abort(MPI_COMM_WORLD, UNEXPECTED_ERR);
                        exit(1);
                    }

                    newchunk = malloc(pctl->rcvmem->chunkbytesize);

                    if (newchunk == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                com->procid, __FILE__, __LINE__);
                        fprintf(stderr,
                                "Total memory allocated by Octor : %dMB\n",
                                (int)(theAllocatedMemSum * 1.0 / (1 << 20)));
                        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                        exit(1);
                    } else {
                        theAllocatedMemSum += pctl->rcvmem->chunkbytesize;
                    }

                    /* Receive the data page into newchunk. the tlr_t
                       is not sent by the sender.  */
                    MPI_Recv(newchunk, recvbytes, MPI_CHAR, pctl->procid,
                             msgtag, comm_for_this_tree, &status);

                    /* Link into the rcvmem */
                    lastchunk = pctl->rcvmem->last;

                    if (lastchunk == NULL) {
                        /* This is the first recieved chunk */
                        pctl->rcvmem->first = newchunk;
                        pctl->rcvmem->last = newchunk;

                    } else {
                        /* Put the new chunk to the end of the receive
                           buffer pool */

                        lasttlr = (tlr_t *)
                            ((char *)lastchunk + sizeof(hdr_t) +
                             pctl->rcvmem->chunkdatasize);
                        lasttlr->nextchunk = (hdr_t *)newchunk;

                        pctl->rcvmem->last = newchunk;
                    }

                    /* Update the chunkcount */
                    pctl->rcvmem->chunkcount++;

                    /* Set the trailer of the newchunk. I have
                       allocated the memory for tlr_t but it's uninited. */
                    newtlr = (tlr_t *)((char *)newchunk + sizeof(hdr_t) +
                                       pctl->rcvmem->chunkdatasize);

                    //newtlr->nextobj = NULL; /* We never use it ! */
                    newtlr->nextchunk = NULL; /* Currently last chunk */

                }  /* Receive a msgtag page */

                /* Optimistically probe the next message */
                MPI_Iprobe(MPI_ANY_SOURCE, msgtag, comm_for_this_tree,
                           &flag, &status);
            }
        } /* more to receive */
    } /* more communications (either send or receive) needed */

    /* Block till all the non-blocking sends return. Application buffers
       can be safely released or reused */
    MPI_Waitall(isendcount, isendreqs, isendstats);
    free(isendreqs);
    free(isendstats);

    return;
}



/***************************/
/* Math routines           */
/***************************/

/**
 * math_gcd: compute the greatest common divisor of two numbers.
 *
 */
static int32_t
math_gcd(int32_t a, int32_t b)
{
    if (b == 0)
        return a;
    else {
        a = a - b * (a / b);
        return math_gcd(b, a);
    }
}


/**
 * octor_zcompare: Compare the Z-value of two points (p1, p2).
 *                Return -1, 0, 1 if Z(p1) is less than, equal to or
 *                great than Z(p2), respectively.
 */
extern int32_t
octor_zcompare(const void *ptr1, const void *ptr2)
{
    int8_t x1exp, x2exp, y1exp, y2exp, z1exp, z2exp;
    int8_t xmax, ymax, zmax;
    const point_t *p1 = (point_t *)ptr1;
    const point_t *p2 = (point_t *)ptr2;

    if ((p1->x == p2->x) &&
        (p1->y == p2->y) &&
        (p1->z == p2->z)) {
        /* Equal */
        return 0;
    }

    if (p1->x == p2->x) {
        /* The X components won't contribute to the difference of the two
           Z values */
        xmax = x1exp = x2exp = -1;
    } else {
        x1exp = LOG2_32b(p1->x);
        x2exp = LOG2_32b(p2->x);
        xmax = (x1exp > x2exp) ? x1exp : x2exp;
    }

    if (p1->y == p2->y) {
        /* The Y components won't contribute to the difference of the two
           Z values */
        ymax = y1exp = y2exp = -1;
    } else {
        y1exp = LOG2_32b(p1->y);
        y2exp = LOG2_32b(p2->y);
        ymax = (y1exp > y2exp) ? y1exp : y2exp;
    }

    if (p1->z == p2->z) {
        /* The Z components won't contribute to the difference of the two
           Z values */
        zmax = z1exp = z2exp = -1;
    } else {
        z1exp = LOG2_32b(p1->z);
        z2exp = LOG2_32b(p2->z);
        zmax = (z1exp > z2exp) ? z1exp : z2exp;
    }

    /* Repeat the following until one pair exponents clearly dominates.
       This loop iterates at most four times */

    while (1) {

        if ((zmax >= ymax) && (zmax >= xmax)) {
            /* The Z components dominates */

            if (z1exp > z2exp)
               return 1;
            else if (z1exp < z2exp)
                return -1;
            else {
                /* Cancel out the same bits and reset zmax */

                zmax = LOG2_32b(p1->z ^ p2->z);
                if (p1->z > p2->z) {
                    z1exp = zmax;
                    z2exp = -1;
                } else {
                    z1exp = -1;
                    z2exp = zmax;
                }
            }

            /* Go through the loop again. zmax might or might not
               dominate in the next iteration */
            continue;
        }

        if (ymax >= xmax) {
            /* The Y components dominates */

            if (y1exp > y2exp)
                return 1;
            else if (y1exp < y2exp)
                return -1;
            else {
                ymax = LOG2_32b((p1->y) ^ (p2->y));

                if (p1->y > p2->y) {
                    y1exp = ymax;
                    y2exp = -1;
                } else {
                    y1exp = -1;
                    y2exp = ymax;
                }
            }
            continue;
        }

        /* (xmax > ymax) && (xmax > zmax) */

        if (x1exp > x2exp)
            return 1;
        else if (x1exp < x2exp)
            return -1;
        else {
            xmax = LOG2_32b((p1->x) ^ (p2->x));

            if (p1->x > p2->x) {
                x1exp = xmax;
                x2exp = -1;
            } else {
                x1exp = -1;
                x2exp = xmax;
            }
        }

        continue;
    } /* while */
}


/**
 * math_zsearch: binary search in a point array to find the entry whose
 *           z-value is the largest among those entries that have smaller
 *           z-values than the search point.
 *
 * - Return the hit entry index.
 *
 */
static int32_t
math_zsearch(const point_t *interval, int32_t size, const point_t *key)
{
    int32_t start, end, offset;

    start = 0;
    end = size - 1;

    while (end >= start) {
        offset = (start + end)  / 2;
        const point_t *point = interval + offset;

        switch (octor_zcompare(key, point)) {
        case 0:
            /* equal */
            return offset;
        case 1:
            /* key > point */
            start = offset + 1;
            break;
        case -1:
            /* key < point */
            end = offset - 1;
            break;
        }
    }

    return end;
}


/**
 * math_bsearch: binary search in a offset array to find the entry whose
 *               value is the largest among those entries that have smaller
 *               value than the seach key. Almost identical to zsearch.
 *
 * - Return the hit entry index.
 *
 */
static int32_t
math_bsearch(const int64_t *table, int32_t size, int64_t key)
{
    int32_t start, end, offset;

    start = 0;
    end = size - 1;

    while (end >= start) {
        offset = (start + end) / 2;
        const int64_t pivot = table[offset];

        if (key == pivot)
            return offset;
        else if (key > pivot) {
            start = offset + 1;
        } else {
            end = offset - 1;
        }
    }

    return end;
}

/**
 * Hash function credit: crafted by Bob Jenkins, December 1996,
 *                       Public Domain. You can use this free for any
 *                       purpose.  It has no warranty.
 */

#define mix(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>13); \
  b -= c; b -= a; b ^= (a<<8); \
  c -= a; c -= b; c ^= (b>>13); \
  a -= b; a -= c; a ^= (c>>12);  \
  b -= c; b -= a; b ^= (a<<16); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>3);  \
  b -= c; b -= a; b ^= (a<<10); \
  c -= a; c -= b; c ^= (b>>15); \
}


/**
 * math_hashuint32: Hash 'count' number of uint32's to an offset value
 *
 */
static uint32_t
math_hashuint32(const void *start, int32_t length)
{
    const uint32_t *k = (uint32_t *)start;
    register uint32_t a, b, c, len;

    /* Set up the internal state */
    len = length;
    a = b = 0x9e3779b9;    /* the golden ratio; an arbitrary value */
    c = 0;                 /* A random number */

    /* Handle most of the key */
    while (len >= 3) {
        a += k[0];
        b += k[1];
        c += k[2];

        mix(a,b,c);
        k += 3; len -= 3;
    }

    /* Handle the last 2 uint32_t's */
    c += length;
    switch(len) {
        /* all the case statements fall through */

        /* c is reserved for the length */
    case 2 : b+=k[1];
    case 1 : a+=k[0];
        /* case 0: nothing left to add */
    }

    mix(a,b,c);

   return c;
}


/***************************/
/* Mesh manipulation       */
/***************************/

/**
 * node_setproperty: Set the property and whereabout of a vertex.
 *
 */
static int32_t
node_setproperty ( tree_t             *tree,
                   vertex_t           *vertex,
                   unsigned char      *pproperty,
                   bldgs_nodesearch_t  bldgs_nodesearch )
{
    tick_t   nx, ny, nz;
    int32_t  where, onX, onY, onZ, wrtSurface;
    int32_t  modX, modY, modZ;
    tick_t   masterMask;
    int8_t   masterLevel;
    double   z_meters;
    int      inBldgPos;

    /* Save local touches variable */
    int8_t touches = vertex->touches;

    /*
     * Touched by 8
     * ------------
     * It is an internal node and is always anchored
     *
     */
    if ( touches == 8 ) {
        *pproperty = 0X80;
        return 0;
    }

    /*
     * Information needed for the following cases
     * ------------------------------------------
     *
     * - Vertex coordinates
     */

    nx = vertex->x;
    ny = vertex->y;
    nz = vertex->z;

    /*
     * Touched by 7
     * ------------
     * This node can only occur for the non-rectangular buildings or in the
     * future case topography. It has to be above the 'surface' and is always
     * anchored.  Since such cases are not active yet it should return error.
     *
     */
    if ( touches == 7 ) {

        /* Temporary default */
        return -71;

        /* For future activation */
        if ( theSurfaceShift == 0 ) {
            /* A vertex can't have 7 touches if */
            return -72;
        }

        *pproperty = 0X80;
        return 0;
    }

    /*
     * Information needed for the following cases
     * ------------------------------------------
     *
     * - where:      single value that tells how many of the domain boundary
     *               faces does the vertex touch.
     *
     *                0: none,
     *                1: on a surface,
     *                2: on an edge,
     *                3: on a corner.
     *
     * - wrtSurface: single value that tells where the vertex is with respect
     *               to the surface.
     *
     *               -1: is above the surface,
     *                0: is on the surface,
     *                1: beneath the surface.
     *
     * - inBldgPos:  position of a vertex in a building.
     *
     *                0: not in a building,
     *                1: on the faces of a building,
     *                2: in the interior of a building.
     */

    /* Classical info without buildings */

    onX = 0;
    onY = 0;
    onZ = 0;
    inBldgPos = 0;

    if ( (nx == tree->nearendp[0]) || (nx == tree->farendp[0]) ) {
        onX = 1;
    }

    if ( (ny == tree->nearendp[1]) || (ny == tree->farendp[1] ) ) {
        onY = 1;
    }

    if ( (nz == tree->nearendp[2]) || (nz == tree->farendp[2] ) ) {
        onZ = 1;
    }

    /* Additional info for buildings */

    if ( theSurfaceShift > 0 ) {

        z_meters = nz * tree->ticksize;

        /* Where is the node with respect to the surface? */
        if ( z_meters < theSurfaceShift ) {
            wrtSurface = -1;
        } else if ( z_meters == theSurfaceShift ) {
            wrtSurface = 0;
        } else if ( z_meters > theSurfaceShift ) {
            wrtSurface = 1;
        } else {
            return -901;
        }

        /* Where is the node in a building? */
        if ( wrtSurface < 1 ) {
            inBldgPos = bldgs_nodesearch(nx, ny, nz, tree->ticksize );
            if ( ( inBldgPos == 0 ) && ( wrtSurface == -1 ) ) {
                /* This vertex should not exist. At this point all vertices
                 * on or above the surface must belong to a building */
                return -902;
            }
        }

    } else {
        wrtSurface =  1;
        inBldgPos  = -1;
    }


    /* Aggregate values for 'where' */

    where = onX + onY + onZ;

    /*
     * Touched by 6
     * ------------
     * This node has two main options. (1) If it is an internal node beneath
     * the surface, then it has to be a dangling node on the edge of the larger
     * element that does not 'touch' it. (2) If it is on the surface, it must
     * be at the foot of a building.  (A third option would be an internal node
     * above the surface --- not considered now for regular buildings.  This
     * would have to be reviewed for future topography and non-regular building
     * cases.)
     *
     */
    if ( touches == 6 ) {

        /* An interior node with 6 touches is always a dangling node.  To be
         * interior has to be interior in a building or beneath the surface
         * and not on a boundary.
         */
        if ( ( ( wrtSurface ==  1 ) && ( where     ==  0 ) ) /* dmain int */ ||
             ( inBldgPos == -1 ) /* bldgs int */ ||
             ( ( wrtSurface ==  0 ) && ( inBldgPos == 1 ) ) /* bldgs int */ )
        {
            masterLevel = vertex->level - 1;
            masterMask = (((tick_t)1 << (PIXELLEVEL - masterLevel))) - 1;

            modX = (nx & masterMask) ? 1 : 0;
            modY = (ny & masterMask) ? 1 : 0;
            modZ = (nz & masterMask) ? 1 : 0;

            if ( modX + modY + modZ != 1 ) {
                return -61;
            }

            if ( modX ) {
                *pproperty = XEDGE;
            } else if ( modY ) {
                *pproperty = YEDGE;
            } else {
                *pproperty = ZEDGE;
            }

            return 0;
        }

        /* At the foot's face of a building, the node is anchored.  Here the
         * node has to be on the surface and it has to touch either only one
         * lateral face or a lateral face and the bottom face (only if the
         * bottom face is also the surface) */
        if ( ( wrtSurface == 0 ) &&
             ( ( inBldgPos == 2 ) || ( inBldgPos == 3 ) ) ) {
            *pproperty = 0X80;
            return 0;
        }

        /* On or above the surface and not in the interior of a building */
        if ( ( wrtSurface != 1 ) && ( inBldgPos == 0 ) ) {
            return -62;
        }

        /* On a domain boundary error */
        if ( where != 0 ) {
            return -63;
        }

        /* Unexpected case */
        return -64;
    }

    /*
     * Touched by 5
     * ------------
     * This case can only occurs in the case of buildings or topography and
     * is always anchored. In the case of buildings considered here the node
     * has to be at the corner of a building and on the surface.
     *
     */
    if ( touches == 5 ) {

        /* No buildings error */
        if ( theSurfaceShift == 0 ) {
            return -51;
        }

        /* Not on the surface error */
        if ( wrtSurface != 0 ) {
            return -52;
        }

        /* It has to touch the two lateral faces or the two lateral faces and
         * the bottom face if equal to the surface, i.e. inBldgPos = 4 or 5 */
        if ( ( inBldgPos < 4 ) || ( inBldgPos > 5 ) ) {
            return -53;
        }

        *pproperty = 0X80;
        return 0;
    }

    /*
     * Touched by 4
     * ------------
     * This case can only occurs in the case of buildings or topography and
     * is always anchored.
     *
     */
    if ( touches == 4 ) {

        /* The node is on one of the lateral faces or on the top of a building
         * and is has to be anchored */
        if ( ( theSurfaceShift >   0 ) /* there are buildings  */ &&
             ( wrtSurface      == -1 ) /* is above the surface */ &&
             ( inBldgPos       ==  2 ) /* is on only one face  */ )
        {
            *pproperty = 0X80;
            return 0;
        }

        /* In more than one domain face error */
        if ( ( where == 2 ) || ( where == 3 ) ) {
            return -41;
        }

        /* The node is on a face and is anchored */
        if ( (where == 1) || ( (where == 0) && (wrtSurface == 0) ) ) {
            *pproperty = 0X80;
            return 0;
        }

        /* It is an internal dangling node on a face or an edge */
        if ( ( ( wrtSurface == -1 ) && ( inBldgPos == -1 ) ) /* bldgs int */ ||
             ( ( wrtSurface ==  1 ) && ( where     ==  0 ) ) /* dmain int */ )
        {
            masterLevel = vertex->level - 1;
            masterMask = (((tick_t)1) << (PIXELLEVEL - masterLevel)) - 1;

            modX = (nx & masterMask) ? 1 : 0;
            modY = (ny & masterMask) ? 1 : 0;
            modZ = (nz & masterMask) ? 1 : 0;

            switch (modX + modY + modZ) {

                case (0):
                    return -42;

                case (1) :
                    /* Dangling on an edge*/
                    if (modX) {
                        *pproperty = XEDGE;
                    } else if (modY) {
                        *pproperty = YEDGE;
                    } else {
                        *pproperty = ZEDGE;
                    }
                    break;

                case(2) :
                    /* Dangling on a face*/
                    if (modX == 0) {
                        *pproperty = XFACE;
                    } else if (modY == 0) {
                        *pproperty = YFACE;
                    } else {
                        *pproperty = ZFACE;
                    }
                    break;

                case (3):
                    return -43;

                default:
                    return -44;
            }
            return 0;
        }

        /* unexpected location error */
        return -45;
    }

    /*
     * Touched by 3
     * ------------
     * This case can only occur in the case of buildings or topography. For
     * the case considered here it is always a dangling node but it could be
     * anchored for irregular buildings or topography.
     *
     */
    if ( touches == 3 ) {

        /* No buildings or internal node error */
        if ( ( theSurfaceShift == 0 ) || ( wrtSurface == 1 ) ) {
            return -31;
        }

        /* A non-prismatic building or rear entry topography */
        if ( wrtSurface == -1 ) {

            /* This case is not covered yet */
            return -32;
        }

        /* Dangling node on the surface and at a building's foot corner.
         * It has to touch the two lateral faces (4) or the two lateral
         * faces and the bottom one if it is also the surface (5) */
        if ( ( wrtSurface == 0 ) &&
             ( ( inBldgPos == 4 ) || ( inBldgPos == 5 ) ) ) {

            masterLevel = vertex->level - 1;
            masterMask = (((tick_t)1) << (PIXELLEVEL - masterLevel)) - 1;

            modX = (nx & masterMask) ? 1 : 0;
            modY = (ny & masterMask) ? 1 : 0;
            modZ = (nz & masterMask) ? 1 : 0;

            switch ( modX + modY + modZ ) {

                case ( 0 ) :
                    return -33;

                case ( 1 ) :
                    /* Dangling on an edge*/
                    if (modX) {
                        *pproperty = XEDGE;
                    } else if (modY) {
                        *pproperty = YEDGE;
                    } else {
                        /* on the vertical edge of a foot-corner element */
                        *pproperty = ZEDGE;
                    }
                    break;

                case ( 2 ):
                    /* It can't be dangling on a face*/
                    return -34;

                default:
                    return -35;
            }

            return 0;
        }

        return -36;
    }


    /*
     * Touched by 2
     * ------------
     * In this case the node has to be on the edge of a building or in one
     * of the edges of the domain.  This case will have to be revisited for
     * topography. The same rules won't apply.
     *
     */
    if ( touches == 2 ) {

        /* The node is on the edge of a building and is anchored.
         * It must be on two faces of a building (4) */
        if ( ( wrtSurface == -1 ) && ( inBldgPos == 4 ) ) {
            *pproperty = 0X80;
            return 0;
        }

        /* Corner node error */
        if ( ( where == 3 ) || ( inBldgPos > 4 ) ) {
            inBldgPos = bldgs_nodesearch(nx, ny, nz, tree->ticksize );
            return -21;
        }

        /* On a boundary edge, it's an anchored node */
        if ( ( where == 2 ) || ( ( where == 1 ) && ( wrtSurface == 0 ) ) ) {
            *pproperty = 0X80;
            return 0;
        }

	/* RICARDO: DOUBLE CHECK THIS!!!
	 * Seems to be working but make sure again */

        if ( /* dangling in an edge on a domain boundary face */
             ( where == 1 ) ||

             /* dangling in an edge on a building face */
             ( ( wrtSurface == -1 ) && ( inBldgPos == 2 ) ) ||

             /* dangling in an edge on the surface OR (pending)...
              * (dangling in a face on the surface at a building's foot) */
             ( ( where == 0 ) && ( wrtSurface == 0 ) ) ||

             /* internal dangling node */
             ( ( where == 0 ) && ( wrtSurface == 1 ) ) )
        {
            masterLevel = vertex->level - 1;
            masterMask = (((tick_t)1) << (PIXELLEVEL - masterLevel)) - 1;
 
            modX = (nx & masterMask) ? 1 : 0;
            modY = (ny & masterMask) ? 1 : 0;
            modZ = (nz & masterMask) ? 1 : 0;

            switch (modX + modY + modZ) {

                case (0):
                    return -22;

                case (1) :
                    /* Dangling on an edge*/
                    if (modX) {
                        *pproperty = XEDGE;
                    } else if (modY) {
                        *pproperty = YEDGE;
                    } else {
                        *pproperty = ZEDGE;
                    }
                    break;

                case(2) :
					return -26;
                    break;

                case (3):
                    return -23;

                default:
                    return -24;
            }
            return 0;

        }

        return -25;
    }

    /*
     * Touched by 1
     * ------------
     * The node is either on a building corner or in one of the four corners
     * of the domain.
     *
     */
    if ( touches == 1 ) {

        /* The node is above the surface and should be in one of the four
         * top corners of a building, therefore it is an anchored node */
        if ( ( wrtSurface == -1 ) && ( inBldgPos == 6 ) ) {
            *pproperty = 0X80;
            return 0;
        }

        /* The node is in one of the eight corners of the domain */
        if ( ( where == 3 ) || ( ( where == 2 ) && ( wrtSurface == 0 ) ) ) {
            *pproperty = 0X80;
            return 0;
        }

        /* Dangling at the foot-corner of a building without foundation */
        if ( ( where == 0 ) && ( wrtSurface == 0 ) && ( inBldgPos == 5 ) ) {

            masterLevel = vertex->level - 1;
            masterMask = (((tick_t)1) << (PIXELLEVEL - masterLevel)) - 1;

            modX = (nx & masterMask) ? 1 : 0;
            modY = (ny & masterMask) ? 1 : 0;
            modZ = (nz & masterMask) ? 1 : 0;
  
            switch (modX + modY + modZ) {

                case (0):
                    return -11;

                case (1) :
                    /* Dangling on an edge*/
                    if (modX) {
                        *pproperty = XEDGE;
                    } else if (modY) {
                        *pproperty = YEDGE;
                    } else {
                        /* It can't be on a Z-edge */
                        return -12;
                    }
                    break;

                case(2) :
                    /* Dangling on a face*/
                    if (modZ == 0) {
                        *pproperty = ZFACE;
                    } else {
                        /* It can't be on a X- or Y-face */
                        return -13;
                    }
                    break;

                case (3):
                    return -14;

                default:
                    return -15;
            }

            return 0;
        }

        /* Not in a corner error */
        if ( ( where < 2 ) || ( ( where == 2 ) && ( wrtSurface != 0 ) ) ) {
            return -16;
        }

        /* Unexpected case */
        return -17;
    }

    /*
     * Illegal number of touches
     * -------------------------
     *
     */
    if ( ( touches <= 0 ) || ( touches > 8 ) ) {
        return -903;
    }

    /* Unexpected case */
    return -904;
}



/**
 * dnode_correlate: Correlate a dangling node to its anchored nodes. Only
 *                  deal with local node ids.
 *
 */
static void
dnode_correlate(tree_t *tree, mess_t *mess, link_t **vertexHashTable,
                int64_t ecount, dnode_t *dnodeTable, int32_t dnindex,
                point_t pt)
{
    int32_t hashentry;
    link_t *link;
    vertex_t *vertex;

    hashentry = math_hashuint32(&pt, 3) % ecount;
    link = vertexHashTable[hashentry];

    while (link != NULL) {
        vertex = (vertex_t *)link->record;
        if ((vertex->x == pt.x) &&
            (vertex->y == pt.y) &&
            (vertex->z == pt.z)) {

            int32link_t *lanid;

            lanid = (int32link_t *)mem_newobj(mess->int32linkpool);
            if (lanid == NULL) {
                fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                        tree->procid, __FILE__, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                exit(1);
            }

            lanid->id = vertex->lnid;
            lanid->next = dnodeTable[dnindex].lanid;
            dnodeTable[dnindex].lanid = lanid;

            break;
        } else {
            link = link->next;
        }
    }

    if (link == NULL) {
        fprintf(stderr, "Thread %d: %s %d: dnode_correlate internal error\n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    }

    return;
}


/**
 * node_harboranchored: look for an anchored vertex. Harbor it if I haven't
 *                      yet and create a message destined for its owner
 *                      processor.
 */
static void
node_harboranchored(tree_t *tree, link_t **vertexHashTable,
                    int64_t ecount, mem_t *linkpool, mem_t *vertexpool,
                    com_t *allcom, point_t pt, int64_t *hbrcount)
{
    int32_t hashentry;
    link_t *link;
    vertex_t *vertex;

    hashentry = math_hashuint32(&pt, 3) % ecount;
    link = vertexHashTable[hashentry];

    while (link != NULL) {
        vertex = (vertex_t *)link->record;
        if ((vertex->x == pt.x) &&
            (vertex->y == pt.y) &&
            (vertex->z == pt.z)) {
            /* I have harbored it already */
            break;

        } else {
            link = link->next;
        }
    }

    if (link == NULL) {
        point_t adjustedpt;
        point_t *anchored;
        pctl_t *topctl;

        /* Harbor this derived anchored vertex. */

        link = (link_t *)mem_newobj(linkpool);
        vertex = (vertex_t *)mem_newobj(vertexpool);

        if ((link == NULL) || (vertex == NULL)) {
            fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        }

        /* Initialize the fields */
        vertex->x = pt.x;
        vertex->y = pt.y;
        vertex->z = pt.z;

        /* To find the owner of a node, adjust the
           coordinate of a node if it's on the
           boundary */
        adjustedpt.x = (pt.x == tree->farendp[0]) ?
            tree->farbound[0] : pt.x;
        adjustedpt.y = (pt.y == tree->farendp[1]) ?
            tree->farbound[1] : pt.y;
        adjustedpt.z = (pt.z == tree->farendp[2]) ?
            tree->farbound[2] : pt.z;

        vertex->owner = math_zsearch(tree->com->interval, tree->com->groupsize,
                                     &adjustedpt);

/* RICARDO: Improve this error message */
        if (vertex->owner == tree->procid) {
            fprintf(stderr, "Thread %d: %s %d: internal error\n",
                    tree->procid, __FILE__, __LINE__);
	    fprintf( stderr, "coords = [ %f %f %f ]\n",
		     tree->ticksize*adjustedpt.x,
		     tree->ticksize*(unsigned)adjustedpt.y,
		     tree->ticksize*adjustedpt.z );
	    fprintf( stderr, "adjustedpt = [ 0x%X 0x%X 0x%X ]\n",
		     adjustedpt.x, adjustedpt.y, adjustedpt.z );
	    //	    fprintf( stderr, "vertex.level = %d\noctsize=0x%x\n",
	    //	     vertex->level, ((tick_t)1 << (PIXELLEVEL - vertex->level)) );
//	    wait_for_debugger( "foo", tree->procid, 1 );

            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }

        vertex->lnid = -1;

        /* Newly harbored anchored vertex may be inserted into
           unprocessed part of the hash table. Set touches to 8 to
           force it be marked as anchored */
        vertex->touches = 8;
        vertex->level = -1;   /* Not used. */

        vertex->property = 0x80; /* Mark as anchored */
        vertex->share = NULL;


        /* Link into the hash table */
        link->record = vertex;
        link->next = vertexHashTable[hashentry];
        vertexHashTable[hashentry] = link;

        /* Update the statistics */
        *hbrcount += 1;

        /* Create a message to signal the owner of the vertex */
        topctl = allcom->pctltab[vertex->owner];
        if (topctl == NULL) {
            /* This should not  happen */
            fprintf(stderr, "Thread %d: %s %d: internal error\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }

        anchored = (point_t *)mem_newobj(topctl->sndmem);

        if (anchored == NULL) {
            fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }


        anchored->x = vertex->x;
        anchored->y = vertex->y;
        anchored->z = vertex->z;
    } /* First encoutner of an anchored vertex */

    return;
}


/***************************/
/* Library routines        */
/***************************/

/**
 * octor_newtree: Return a pointer to a newly created and initialized octree.
 *                NULL on error.
 *
 */
extern octree_t *
octor_newtree(double x, double y, double z, int32_t recsize,
              int32_t myid, int32_t groupsize, MPI_Comm comm_solver_arg,
              double surface_shift)
{
    tree_t *tree;
    int32_t u32x, u32y, u32z, GCD, max, pow;
    int8_t initlevel;


    /* Ricardo:
     * Assign the cutoff depth for meshing sculpting
     */
    theSurfaceShift = surface_shift;
    theTotalDepth   = z;

    /* Allocate global static memory */
    theRecordPool = mem_new(recsize, 1 << 15);
    if (theRecordPool == NULL) {
        return NULL;
    } else {
        theRecordPoolRefs = 1;
    }

    /* Allocate memory for the tree control structure */
    tree= (tree_t *)malloc(sizeof(tree_t));
    if (!tree) {
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        return NULL;
    } else {
        theAllocatedMemSum += sizeof(tree_t);
    }

    /* Allocate tree root */
    tree->root = (oct_t *)malloc(sizeof(oct_t));
    if (!tree->root) {
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        return NULL;
    } else {
        theAllocatedMemSum += sizeof(oct_t);
    }

    /* Initialize the root */
    tree->root->type = LEAF;
    tree->root->which = 0;
    tree->root->level = 0;
    tree->root->where = LOCAL;
    tree->root->lx = 0;
    tree->root->ly = 0;
    tree->root->lz = 0;
    tree->root->appdata = NULL;

    tree->root->parent = NULL;
    tree->root->payload.leaf = NULL;

    /* Set the end points */
    tree->nearendp[0] = 0;
    tree->nearendp[1] = 0;
    tree->nearendp[2] = 0;

/*    u32x = (int32_t)(x * 10000); */
/*    u32y = (int32_t)(y * 10000); */
/*    u32z = (int32_t)(z * 10000); */

    u32x = (int32_t)(x);
    u32y = (int32_t)(y);
    u32z = (int32_t)(z);


    GCD = math_gcd(u32x, math_gcd(u32y, u32z));

    u32x /= GCD;
    u32y /= GCD;
    u32z /= GCD;

    max = (u32x > u32y ) ? u32x : u32y;
    max = (max > u32z) ? max : u32z;

    pow = LOG2_32b(max);

    tree->farendp[0] = u32x * ((tick_t)1 << (PIXELLEVEL - pow));
    tree->farendp[1] = u32y * ((tick_t)1 << (PIXELLEVEL - pow));
    tree->farendp[2] = u32z * ((tick_t)1 << (PIXELLEVEL - pow));

    tree->farbound[0] = tree->farendp[0] - 1;
    tree->farbound[1] = tree->farendp[1] - 1;
    tree->farbound[2] = tree->farendp[2] - 1;

    tree->ticksize = x / tree->farendp[0];
    tree->recsize = recsize;

    /* RICARDO */
    tree->surfacep = (tick_t)(surface_shift / tree->ticksize);

    /* Persistent static fields */
    tree->comm_tree = comm_solver_arg;
    tree->procid = myid;
    tree->groupsize = groupsize;

    tree->octmem = mem_new(sizeof(oct_t), 1 << 15);
    tree->leafmem = mem_new(sizeof(leaf_t), 1 << 15);
    tree->interiormem = mem_new(sizeof(interior_t), 1 << 15);

    if ((tree->octmem == NULL)  || (tree->leafmem == NULL) ||
        (tree->interiormem == NULL))
        return NULL;

    /* Persistent dynamical field */
    tree->firstleaf = tree->root;
    memset(tree->leafcount, 0, sizeof(int32_t) * TOTALLEVEL);
    tree->leafcount[0] = 1; /* count the root oct */

    /* Bootstrap for parallel processing */
    if (groupsize > 1) {
        int32_t threshold, tasks, taskid, mytask_low, mytask_high;
        oct_t *oct;
        int32_t *inited; /* indicate wheter an interval has been inited */

        /* Create enough tasks for every processor */
        tasks = 1;
        threshold = 10 * groupsize;

        while (tasks < threshold) {
            tasks = 0;

            oct = tree->firstleaf;

            while (oct != NULL) {
                /* Push down one more level */
                int32_t newtasks;

                newtasks = oct_sprout(oct, tree);
                tasks += newtasks;

                oct = oct_getnextleaf(oct);
            }
        }

        /* Record the level of the task octants */
        initlevel = tree->firstleaf->level;

        /* Associate the tasks to processors */
        taskid = 0;

        mytask_low = BLOCK_LOW(myid, groupsize, tasks);
        mytask_high = BLOCK_HIGH(myid, groupsize, tasks);

        oct = tree->firstleaf;
        tree->firstleaf = NULL;

        /* The way the com->interval table is initialized here blurs
           the interface between tree_ and com_. (Bad software
           engineering practice.) However, by directly assigning the
           point coordinates to the interval table, we avoid communcation. */

        /* Create a communication manager */
        if ((tree->com = com_new(myid, groupsize)) == NULL)
            return NULL;

        if ((inited = (int32_t *)calloc(groupsize, sizeof(int32_t))) == NULL) {
            fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                    (int)(theAllocatedMemSum));
            return NULL;
        } else {
            theAllocatedMemSum += groupsize * sizeof(int32_t);
        }

        while (oct != NULL) {
            int32_t procid;

            procid = BLOCK_OWNER(taskid, groupsize, tasks);

            if (!inited[procid]) {
                tree->com->interval[procid].x = oct->lx;
                tree->com->interval[procid].y = oct->ly;
                tree->com->interval[procid].z = oct->lz;

                inited[procid] = 1;
            }


            if ((taskid >= mytask_low) && (taskid <= mytask_high)) {
                oct->where = LOCAL;
                tree->firstleaf = (tree->firstleaf == NULL) ?
                    oct : tree->firstleaf;
            }
            else {
                oct->where = REMOTE;
            }

            oct = (oct_t *)octor_getnextleaf((octant_t *)oct);
            taskid++;
        }

        /* Release the interval inited table */
        free(inited);

        /* Shrink REMOTE leaf octs */
        if (oct_shrink(tree->root, tree, REMOTE, NULL, NULL) == NULL) {
            /* error */
            return NULL;
        }
        /* Overwrite how many LOCAL leaf octants (tasks) I have */
        tree->leafcount[0] = 0;
        tree->leafcount[(int32_t)initlevel] =
            BLOCK_SIZE(myid, groupsize, tasks);

        /* Check neighbors for each LOCAL leaf oct. This routine
         * assume that the distributed octree conforms to the 2-to-1 *
         * constraint (balanced in mesh generation term).
         *
         * Also notice at this point we have not pushed the surface and
         * pass NULL for the buildings plug-in.
         */
        if (com_allocpctl(tree->com, 0, tree->firstleaf,
                          tree->nearendp, tree->farendp, 0,
                          tree->ticksize, NULL) != 0)
            return NULL;

    } else {
        /* Single processor mode */
        tree->com = NULL;
    }

    /* Transient fields is not initialized until being used.  */
    memset(tree->toleaf, 0, sizeof(oct_t *) * TOTALLEVEL);

#ifdef TREE_VERBOSE
    if (tree->procid == 0) {
        printf("\nTREE_INFO: octor_newtree: initial level = %d\n", initlevel);
    }
#endif /* TREE_VERBOSE */

    return (octree_t *)tree;
}


/**
 * octor_deletetree: Release the memory used by an octree.
 *
 */
extern void
octor_deletetree(octree_t *octree)
{
    tree_t *tree = (tree_t *)octree;

    if (octree == NULL)
        return;

    /* Free theRecordPool only if the reference count drops to 0 */
    theRecordPoolRefs--;
    if (theRecordPoolRefs == 0) {
        mem_delete(theRecordPool);
    }

    free(tree->root);
    mem_delete(tree->octmem);
    mem_delete(tree->leafmem);
    mem_delete(tree->interiormem);

    /* Release memory for parallel processing */
    if (tree->groupsize > 1) {
        if (tree->com != NULL)
            com_delete(tree->com);
    }

    free(tree);

    return;
}


/**
 * octor_refinetree: Refine the subtree collections on local processor.
 *
 * - Return 0 if OK, -1 on error.
 *
 */
extern int32_t
octor_refinetree(octree_t *octree, toexpand_t *toexpand, setrec_t *setrec)
{
    tree_t *tree = (tree_t *)octree;
    oct_t *oct;

    oct = tree->firstleaf;

    while ((oct != NULL) && (oct->where == LOCAL)) {
        if (oct_expand(oct, tree, toexpand, setrec) != 0) {
            fprintf(stderr,
                    "Proc %d (Failed): created %d leaves. min = %d max = %d\n",
                    tree->procid,
                    (int32_t)tree_countleaves(tree),
                    tree_getminleaflevel(tree),
                    tree_getmaxleaflevel(tree));
            return -1;
        }

        oct = oct_getnextleaf(oct);
    }

#ifdef TREE_VERBOSE
    tree_showstat(tree, DETAILS, "octor_refinetree");
#endif /* TREE_VERBOSE */

    return 0;
}


/**
 * octor_coarsentree: Make to tree coarsened as required by the application.
 *                    Return 0 if OK, -1 on error.
 *
 */
extern int32_t
octor_coarsentree(octree_t *octree, toshrink_t *toshrink, setrec_t *setrec)
{
    oct_t *root = (oct_t *)octree->root;
    tree_t *tree = (tree_t *)octree;

    if (oct_shrink(root, tree, LOCAL, toshrink, setrec) == NULL) {
        /* error */
        return -1;
    }

#ifdef TREE_VERBOSE
    tree_showstat(tree, DETAILS, "octor_coarsentree");
#endif /* TREE_VERBOSE */

    return 0;
}


/**
 * octor_balancetree: Enforce the 2-to-1 constraint on the entire octree.
 *
 * - The communication manager is kept around after the
 * - Return 0 if OK, -1 on error.
 *
 */
extern int32_t
octor_balancetree(octree_t *octree, setrec_t *setrec, int theStepMeshingFactor)
{
    int32_t lmax, gmax, lmin, gmin, level, threshold;
    oct_t *oct, *nbr;
    dir_t dir;
    tree_t *tree = (tree_t *)octree;
    oct_stack_t stack;
    descent_t *descent;

#ifdef TREE_VERBOSE
    /* Print analysis data */
    int64_t localcount;
    localcount = tree_countleaves(tree);
    fprintf(stderr, "Thread %d: before balancetree: %qd\n", tree->procid,
            localcount);
#endif /* TREE_VERBOSE */

    /* Link LOCAL leaf octs at the same level together */
    memset(tree->toleaf, 0, sizeof(oct_t *) * TOTALLEVEL);
    oct = tree->firstleaf;
    while ((oct != NULL) && (oct->where == LOCAL)) {
        oct_linkleaf(oct, tree->toleaf);
        oct = oct_getnextleaf(oct);
    }

    /* Set bounds for Prioritized Ripple Propagation. */
    lmin = tree_getminleaflevel(tree);
    lmax = tree_getmaxleaflevel(tree);

    if (tree->groupsize == 1) {
        gmin = lmin;
        gmax = lmax;
    } else {
        MPI_Allreduce(&lmin, &gmin, 1, MPI_INT, MPI_MIN, tree->comm_tree);
        MPI_Allreduce(&lmax, &gmax, 1, MPI_INT, MPI_MAX, tree->comm_tree);
    }

    /* Work from bottom up. Prioritized ripple propagation */
    threshold = gmin + 1;
    for (level = gmax; level > threshold; level--) {

        /* Reset the message buffer of pctls if groupsize > 1. This
           needs to be done for each iteration for a different level */

        if (tree->groupsize > 1)
            com_resetpctl(tree->com, sizeof(descent_t));

        /* Traverse small octs at the same level */
        oct = tree->toleaf[level];
        while (oct != NULL) {

            for (dir = L; dir <= UF; dir = (dir_t) ((int)dir + 1)) {

                /* stack must be cleaned before calling oct_findneighbor */
                stack.top = 0;
                nbr = oct_findneighbor(oct, dir, &stack);

                /* Discard out of bounds */ //yigit

                if ( nbr != NULL ) {
                	if ( ( nbr->lz < tree->nearendp[2] ) ||
                			( nbr->lz >= tree->farendp[2] ) ) {
                		continue;
                	}

                	/* Discard out of bounds */
                	if ( ( nbr->ly < tree->nearendp[1] ) ||
                			( nbr->ly >= tree->farendp[1] ) ) {
                		continue;
                	}


                	/* Discard out of bounds */
                	if ( ( nbr->lx < tree->nearendp[0] ) ||
                			( nbr->lx >= tree->farendp[0] ) ) {
                		continue;
                	}
                }

                /* Should never return NULL */
                if ((nbr == NULL) || (nbr->level > oct->level - 2)) {
                    /* Ignore a non-existent neighbor or a neighbor
                       that is small enough already */
                    continue;
                }

                if (nbr->where == LOCAL) {
                    if (tree_pushdown(nbr, tree, &stack, setrec) != 0) {
                        fprintf(stderr, "Thread %d: %s %d: ",
                                tree->procid, __FILE__, __LINE__);
                        fprintf(stderr, "Cannot pushdown local neighbor\n");
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    } else {
                        continue;
                    }

                } else if (nbr->where != LOCAL) {
					/* Note by Yigit+Ricardo: 
					 * This condition used to be == REMOTE 
					 * The change was made in the search for fixing 
					 * progressive meshing issues */

                    /* This only happens if groupsize > 1. We have
                       maintained the validity of tree->com throughout. */

                    int32_t procid;
                    pctl_t *topctl;
                    point_t nbrLDB;
                    uint32_t dirbits;
                    tick_t edgesize;

                    /* Find the procid that accomodate the LDB pixel
                       of an equal-sized neighbor */

                    edgesize = (tick_t)1 << (PIXELLEVEL - oct->level);

                    dirbits = theDirBitRep[dir];

                    /* Assign the coordinate of oct to its neighbor */
                    nbrLDB = *(point_t *)&oct->lx;

                    /* Adjust the x, y, z coordinate as necessary */

                    if (dirbits & LEFT)
                        nbrLDB.x = oct->lx - edgesize;

                    if (dirbits & RIGHT)
                        nbrLDB.x = oct->lx + edgesize;

                    if (dirbits & DOWN)
                        nbrLDB.y = oct->ly - edgesize;

                    if (dirbits & UP)
                        nbrLDB.y = oct->ly + edgesize;

                    if (dirbits & BACK)
                        nbrLDB.z = oct->lz - edgesize;

                    if (dirbits & FRONT)
                        nbrLDB.z = oct->lz + edgesize;

                    if ((nbrLDB.x < tree->nearendp[0]) ||
                        (nbrLDB.y < tree->nearendp[1]) ||
                        (nbrLDB.z < tree->nearendp[2]) ||
                        (nbrLDB.x >= tree->farendp[0]) ||
                        (nbrLDB.y >= tree->farendp[1]) ||
                        (nbrLDB.z >= tree->farendp[2])) {
                        /* Ignore the out-of-bound neighbor oct */
                        continue;
                    }

                    /* Find out who possesses the pixel */
                    procid = math_zsearch(tree->com->interval,
                                          tree->com->groupsize,
                                          &nbrLDB);

                    if (procid == tree->procid) {
                        fprintf(stderr, "Thread %d: %s %d: internal error\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, UNEXPECTED_ERR);
                        exit(1);
                    }

                    /* Create a message destined to procid */
                    topctl = tree->com->pctltab[procid];

                    if (topctl == NULL) {
                        /* This should not happen */
                        fprintf(stderr, "Thread %d: %s %d: internal error\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                    descent = (descent_t *)mem_newobj(topctl->sndmem);
                    if (descent == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                        exit(1);

                    }

                    /* Marshal a DESCENT message */
                    descent->lx = nbr->lx;
                    descent->ly = nbr->ly;
                    descent->lz = nbr->lz;
                    descent->level = nbr->level;
                    descent->stack = stack;
                }
            } /* for all the directions */

            oct = oct->payload.leaf->next;

        } /* while */

        /* Exchange descend instructions */
        if (tree->groupsize > 1) {
            pctl_t *frompctl;

            /* Exchange DESCENT_MSG (indicated by the level of interest)
               between adjacent processors. May result in abort and exit. */
            com_OrchestrateExchange(tree->com, DESCENT_MSG, tree->comm_tree);

            /* Take care of neighbor processors' requests */
            frompctl = tree->com->firstpctl;

            while (frompctl != NULL) {
                mem_initcursor(frompctl->rcvmem);

                while ((descent =
                        (descent_t *)mem_getcursor(frompctl->rcvmem))
                       != NULL) {

                    tick_t lx, ly, lz;
                    int8_t level;

                    lx = descent->lx;
                    ly = descent->ly;
                    lz = descent->lz;
                    level = descent->level;
                    stack = descent->stack;

                    nbr = tree_searchoct(tree, lx, ly, lz, level,
                                         EXACT_SEARCH);
                    if (nbr == NULL) {
                        /* This should not happen */
                        fprintf(stderr, "Thread %d: %s %d: ",
                                tree->procid, __FILE__, __LINE__);
                        fprintf(stderr, "Cannot find the anchor oct\n");
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                    if (tree_pushdown(nbr, tree, &stack, setrec) != 0) {
                        fprintf(stderr, "Thread %d: %s %d: ",
                                tree->procid, __FILE__, __LINE__);
                        fprintf(stderr, "Cannot pushdown from anchor oct\n");
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                    /* move the next descend instruction */
                    mem_advcursor(frompctl->rcvmem);

                } /* while descent != NULL */

                frompctl = frompctl->next;

            } /* while fromctl */

            /* Synchronzie globally */
            MPI_Barrier(tree->comm_tree);

        } /* if tree->groupsize > 1 */

    } /* for all levels */

    /* Discard residual communication data, set the message size to 0 */
    if (tree->groupsize > 1) {
        com_resetpctl(tree->com, 0);
    }

#ifdef TREE_VERBOSE
    tree_showstat(tree, DETAILS, "octor_balancetree");
#endif /* TREE_VERBOSE */

    return 0;
}


/*
 * RICARDO:
 * for eliminating leaves with payload Vp < 0
 */

void backfire_parent(oct_t *parent) {

    int8_t which;

    for ( which = 0; which < 8; which++ ) {
        if ( parent->payload.interior->child[which] != NULL ) {
            return;
        }
    }

    /* all my children are null, I need to become a NULL child of my parent */

    int8_t whoami;
    oct_t *grandpa;

    whoami = parent->which;
    grandpa = parent->parent;

    /* TODO: I need to think more about this */
    if ( grandpa == NULL ) {
        /* I am at the root */
        return;
    }

    grandpa->payload.interior->child[whoami] = NULL;

    /* and now I need to check my grandpa has other valid children */
    backfire_parent(grandpa);

    return;
}


extern void
octor_carvebuildings(octree_t *octree, int flag,
                     bldgs_nodesearch_t *bldgs_nodesearch)
{
    tree_t  *tree = (tree_t *)octree;
    oct_t   *oct, *nextOct;
    int64_t  ecount;
    int64_t  myCount = 0;
    edata_t *edata;

    ecount = tree_countleaves(tree);

    oct = oct_getleftmost(tree->root);

    /* A sanity check */
    if ( tree->firstleaf != oct_getleftmost(tree->root) ) {
        fprintf(stderr,
                "Thread %d: %s %d: error in the first leaf \n",
                tree->procid, __FILE__, __LINE__);
    }

    if ( tree->firstleaf == NULL ) {
        fprintf(stderr,
                "Thread %d: %s %d: NULL first leaf \n",
                tree->procid, __FILE__, __LINE__);
    }


    while ( oct != NULL ) {

    	double Vp;

    	myCount++;

    	edata = (edata_t *)oct->payload.leaf->data;
    	Vp = edata->Vp;

    	nextOct = oct_getnextleaf(oct);

    	if ( Vp < 0 ) {//yigit

    		/* Now I will eliminate this leaf */

    		/* making sure the first leaf is updated */
    		if ( oct == tree->firstleaf ) {
    			tree->firstleaf = nextOct;
    		}

    		/* unlinking the octant and making sure the corresponding
    		 * tree->toleaf list is updated */
    		oct_unlinkleaf(oct, tree->toleaf);

    		/* Modify the statistics */
    		tree->leafcount[(int32_t)oct->level]--;

    		/* Mark this leaf as remote */
    		oct->where = REMOTE;
    	}

    	/* moving on to next octant */
    	oct = nextOct;
    }

    /* Release and recover memory */
    oct_shrink(tree->root, tree, REMOTE, NULL, NULL);

    /* A sanity check */
    if ( tree->firstleaf != oct_getleftmost(tree->root) ) {
        fprintf(stderr,
                "Thread %d: %s %d: error in the first leaf \n",
                tree->procid, __FILE__, __LINE__);
    }


#ifdef TREE_VERBOSE
    tree_showstat(tree, DETAILS, "octor_balancetree");
#endif /* TREE_VERBOSE */

    return;
}


/**
 * octor_partitiontree: distribute octants evenly among processors.
 *
 */
extern int32_t
octor_partitiontree(octree_t *octree, bldgs_nodesearch_t *bldgs_nodesearch)
{
    tree_t *tree = (tree_t *)octree;
    int32_t start_procid, end_procid, bin_procid;
    int64_t *counttable, *starttable, localcount, totalcount;
    oct_t *cutleaf;
    int64_t cutindex;
    int64_t target_low, target_high, mylow, myhigh;
    int64_t bin_low, bin_high;
    int64_t maxlow, minhigh, intersectcount;
    int64_t due, share;
    MPI_Request *irecvreqs = NULL;
    MPI_Status *irecvstats = NULL;
    void **rcvleafoct_pool_list = NULL;
    int32_t irecvcount, irecvnum;
    int32_t irecv;


    /* leafoctsize is the size of a serialized leaf octant */
    int32_t addresssize = 3 * sizeof(tick_t) + sizeof(int8_t);
    int32_t leafoctsize = addresssize + tree->recsize;

    if (tree->groupsize == 1)
        return 0;

    /* Count how many leaf octants I have */
    localcount = tree_countleaves(tree);

    /* Get the count distribution among the processors */
    tree_setdistribution(tree, &counttable, &starttable, localcount);

    totalcount = starttable[tree->groupsize - 1] +
        counttable[tree->groupsize - 1];

    /* My target range of elements */
    target_low = BLOCK_LOW(tree->procid, tree->groupsize, totalcount);
    target_high = BLOCK_HIGH(tree->procid, tree->groupsize, totalcount);
    due = target_high - target_low + 1;

    block_validate( target_low, target_high, due, totalcount, tree->groupsize );

    /* Find procids who own me data */
    start_procid = math_bsearch(starttable, tree->groupsize, target_low);
    end_procid = math_bsearch(starttable, tree->groupsize, target_high);
    if ((start_procid > tree->procid) || (end_procid < tree->procid)) {
        /* my current elements won't contribute */
        irecvcount = end_procid - start_procid + 1;
    } else {
        irecvcount = end_procid - start_procid;
    }

    /* Eliminate any procids with 0 element.(after bldgs module)--yigit*/
    for (irecv = start_procid; irecv < end_procid + 1 ; irecv++) {
    	if (counttable[irecv] == 0 && irecv != tree->procid ) {
    		irecvcount--;
    	}
    }

    /* Create MPI request and stat objects */
    if (irecvcount != 0) {
        irecvreqs = (MPI_Request *)malloc(irecvcount * sizeof(MPI_Request));
        irecvstats = (MPI_Status *)malloc(irecvcount * sizeof(MPI_Status));
        rcvleafoct_pool_list = (void **)malloc(irecvcount * sizeof(void *));

        if ((irecvreqs == NULL) || (irecvstats == NULL) ||
            (rcvleafoct_pool_list == NULL )) {
            fprintf(stderr, "Thread %d: %s %d : out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                    (int)(theAllocatedMemSum));
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        } else {
            theAllocatedMemSum += irecvcount *
                (sizeof(MPI_Request) + sizeof(MPI_Status) + sizeof(void *));
        }
    }

    /* Go through the count table and start table to post receives */

    bin_procid = start_procid;
    irecvnum = 0;
    while (due > 0) {
        /* get the bounds of the current bin */
        bin_low = starttable[bin_procid];
        bin_high = starttable[bin_procid] + counttable[bin_procid] - 1;

        maxlow = (bin_low < target_low) ? target_low : bin_low;
        minhigh = (bin_high > target_high) ? target_high : bin_high;
        intersectcount = minhigh - maxlow + 1;

        //after bldgs module-- yigit
        if (bin_procid != tree->procid && counttable[bin_procid] != 0) {
        	void *rcvleafoct_pool;
        	int32_t rcvbytesize;

        	/* allocate receive buffer */
        	rcvbytesize = leafoctsize * intersectcount;
        	rcvleafoct_pool_list[irecvnum] = malloc(rcvbytesize);
        	rcvleafoct_pool = rcvleafoct_pool_list[irecvnum];

        	if (rcvleafoct_pool == NULL) {
        		/* Out of memory */
        		fprintf( stderr,
                         "PE %d: %s %d: out of memory\n"
                         "Allocating %d bytes for %lld elements\n"
                         "Total memory allocated by Octor : %d bytes\n",
                         tree->procid, __FILE__, __LINE__,
                         rcvbytesize, intersectcount,
                         ((int)theAllocatedMemSum) );
                MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                exit(1);
            } else {
                theAllocatedMemSum += rcvbytesize;
            }

            /* Post receive */
            MPI_Irecv(rcvleafoct_pool, rcvbytesize, MPI_CHAR, bin_procid,
                      OCT_MSG, tree->comm_tree, &irecvreqs[irecvnum]);

            /* Increment the irecv count */
            irecvnum++;
        } /* else, I myself have the intersectcount elements, skip! */

        /* Update the due I need to collect */
        due -= intersectcount;

        /* Move to the next bin */
        bin_procid++;
    }

#ifdef DEBUG
    if (irecvnum != irecvcount) {
        fprintf(stderr, "Thread %d: %s %d: internal error: unmatched irecvs\n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
        exit(1);
    }
#endif /* DEBUG */

    /* Wait till everyone has posted their receives */
    MPI_Barrier(tree->comm_tree);

    /* Get range of octants I currently have */
    mylow = starttable[tree->procid];
    myhigh = starttable[tree->procid] + counttable[tree->procid] - 1;
    share = counttable[tree->procid];

    /* Set cutleaf to my local first (0th) octant */
    cutleaf = tree->firstleaf;
    cutindex = starttable[tree->procid];

    /* Send data to my peers, either standard, synchronous, or ready send
       will work */

    bin_procid = BLOCK_OWNER(mylow, tree->groupsize, totalcount);

    while (share > 0) {
        target_low = BLOCK_LOW(bin_procid, tree->groupsize, totalcount);
        target_high = BLOCK_HIGH(bin_procid, tree->groupsize, totalcount);

        maxlow = (target_low > mylow) ? target_low : mylow;
        minhigh = (target_high < myhigh) ? target_high : myhigh;
        intersectcount = minhigh - maxlow + 1;

        if (bin_procid != tree->procid) {
            int32_t sndbytesize, index;
            void *sndleafoct_pool;
            char *ptr;

            /* allocate send buffer */
            sndbytesize = leafoctsize * intersectcount;
            sndleafoct_pool = malloc(sndbytesize);

            if (sndleafoct_pool == NULL) {
                /* Out of memory */
                fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                        tree->procid, __FILE__, __LINE__);
                fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                        (int)(theAllocatedMemSum));
                MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                exit(1);
            } else {
                theAllocatedMemSum += sndbytesize;
            }

            /* Advance cutleaf if necessary */
            while (cutindex < maxlow) {
                if (cutleaf == NULL) {
                    fprintf(stderr, "Thread %d: %s %d: cutleaf should",
                            tree->procid, __FILE__, __LINE__);
                    fprintf(stderr, " not be NULL.\n");
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);
                }

                cutleaf = oct_getnextleaf(cutleaf);
                cutindex++;
            }

            /* cutleaf points to the next leaf that should be sent */

            /* Preorder traverse my local tree from cutleaf to
               assemble the outgoing message in the sndleafoct_pool */

            ptr = (char *)sndleafoct_pool;

            for (index = 0 ; index < intersectcount; index++) {
                if (cutleaf == NULL) {
                    fprintf(stderr, "Thread %d: %s %d: cutleaf should",
                            tree->procid, __FILE__, __LINE__);
                    fprintf(stderr," not be NULL.\n");
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);
                }

                /* Serialize an outgoing leaf octant */
                memcpy(ptr, &cutleaf->level, addresssize);
                ptr += addresssize;

                memcpy(ptr, cutleaf->payload.leaf->data, tree->recsize);
                ptr += tree->recsize;

                /* Modify the statistics */
                tree->leafcount[(int32_t)cutleaf->level]--;

                /* Mark this leaf as remote */
                cutleaf->where = REMOTE;

                /* move to the next leaf oct */
                cutleaf = oct_getnextleaf(cutleaf);
                cutindex++;
            }

            /* Send the octants to the processor who is receiving */
            MPI_Send(sndleafoct_pool, sndbytesize, MPI_CHAR, bin_procid,
                     OCT_MSG, tree->comm_tree);

            /* Release and recover memory */
            free(sndleafoct_pool);
            oct_shrink(tree->root, tree, REMOTE, NULL, NULL);
        } /* else, the interesectcount elements belong to myself */

        /* Update the element counts I need to share */
        share -= intersectcount;

        /* Move to the next bin */
        bin_procid++;
    } /* while I have more elements to share */

    /* Wait till I receive all the data I want */
    if (irecvcount != 0) {
        MPI_Waitall(irecvcount, irecvreqs, irecvstats);
    }

    for (irecvnum = 0; irecvnum < irecvcount; irecvnum++) {
        int32_t received, index;
        int32_t rcvbytesize;
        char *ptr;

        MPI_Get_count(&irecvstats[irecvnum], MPI_CHAR, &rcvbytesize);
        received = rcvbytesize / leafoctsize;

        ptr = (char *)rcvleafoct_pool_list[irecvnum];
        for (index = 0; index < received; index++) {
            tick_t lx, ly, lz;
            int8_t level;
            void *data;

            /* De-serialize a leaf octant */
            level = *(int8_t *)ptr;
            ptr += sizeof(int8_t);

            lx = *(tick_t *)ptr;
            ptr += sizeof(tick_t);

            ly = *(tick_t *)ptr;
            ptr += sizeof(tick_t);

            lz = *(tick_t *)ptr;
            ptr += sizeof(tick_t);

            data = ptr;
            ptr += tree->recsize;

            /* Install leaf oct and update the stat */
            if (oct_installleaf(lx, ly, lz, level, data, tree) != 0) {
                fprintf(stderr, "Thread %d: %s %d: oct_installleaf error\n",
                        tree->procid, __FILE__, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                exit(1);
            }

            /* Update statistics */
            tree->leafcount[(int32_t)level]++;

        } /* for all the incoming octants */

        /* Release the receive buffer */
        free(rcvleafoct_pool_list[irecvnum]);
    } /* All all the recv buffers */


    /* Adjust the firstleaf pointer */
    tree->firstleaf = oct_getleftmost(tree->root);

    /*
     * Check for misterious error of firstleaf being null.
     * This check was introduced for making sure that eliminating octants
     * for carving the buildings did not spoiled the mesh.
     */
    if ( tree->firstleaf == NULL ) {
        fprintf(stderr,
                "Thread %d: %s %d: got a null first leaf \n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, COMM_ERR);
        exit(1);
    }


    /* Release memory */
    if (irecvcount != 0) {
        free(rcvleafoct_pool_list);
        free(irecvreqs);
        free(irecvstats);
    }

    free(counttable);
    free(starttable);

    /* The neighboring relationship should be updated */

    tree_deletecom(tree);

    if (tree_setcom(tree, 0, bldgs_nodesearch) != 0) {
        fprintf(stderr,
                "Thread %d: %s %d: fail to create new communication manager\n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, COMM_ERR);
        exit(1);
    }

#ifdef TREE_VERBOSE
    /* tree_showstat(tree, BRIEF, "octor_partitiontree"); */
    tree_showstat(tree, DETAILS, "octor_partitiontree");
#endif /* TREE_VERBOSE */

    return 0;
}


/**
 * octor_extractmesh: Given a balanced (2-to-1 constraint) tree, which
 *                    may have been load balanced. Extract the mesh
 *                    structure.
 *
 */
extern mesh_t *
octor_extractmesh(octree_t *octree, bldgs_nodesearch_t *bldgs_nodesearch)
{
    tree_t *tree = (tree_t *)octree;
    int64_t *octCountTable;
    int64_t *octStartTable;
    int64_t *nodeCountTable;
    int64_t *nodeStartTable;
    int64_t ecount, ncount, harborcount, startgeid, startgnid, geid, gnid;
    int64_t ldnnum;
    int32_t eindex, nindex, dnindex;
    elem_t *elemTable;
    node_t *nodeTable, *node;
    dnode_t *dnodeTable;
    link_t **vertexHashTable, *link;
    mem_t *vertexpool, *linkpool;
    oct_t *oct;
    int32_t hashentry;
    vertex_t *vertex = NULL;
    mess_t *mess = NULL;
    com_t *allcom = NULL, *partcom = NULL;

    /*---------------- Initialize various data structure ----------------*/

    /* Increment theRecordPoolRefs by 1 */
    theRecordPoolRefs++;

    /* Allocate memory for the mesh control structure */
    mess = (mess_t *)malloc(sizeof(mess_t));
    if (mess == NULL) {
        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                tree->procid, __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        theAllocatedMemSum += sizeof(mess_t);
    }


    /* How many elements I have */
    ecount = tree_countleaves(tree);

    /* Allocate memory for temporary data structures and int32link pool */
    elemTable = (elem_t *)malloc(sizeof(elem_t) * ecount);
    vertexHashTable = (link_t **)calloc(ecount, sizeof(link_t *));

    vertexpool = (mem_t *)mem_new(sizeof(vertex_t), (int32_t)(ecount * 1.4));
    linkpool = (mem_t *)mem_new(sizeof(link_t), (int32_t)(ecount * 1.4));

    mess->int32linkpool = mem_new(sizeof(int32link_t),
                                  (int32_t)(ecount * 0.1));

    if ((elemTable == NULL) || (vertexHashTable == NULL) ||
        (vertexpool == NULL) || (linkpool == NULL) ||
        (mess->int32linkpool == NULL)) {
        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                tree->procid, __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        theAllocatedMemSum += sizeof(elem_t) * ecount +
            ecount * sizeof(link_t *) +
            sizeof(vertex_t) * (int32_t)(ecount * 1.4) +
            sizeof(link_t) * (int32_t)(ecount * 1.4);
    }

    /* Build elemTable and vertexHashTable. Some fields will not be
       initialized till later. */

    if (tree->groupsize > 1) {
        /* Get octant count distribution */
        tree_setdistribution(tree, &octCountTable, &octStartTable, ecount);
        startgeid = octStartTable[tree->procid];

    } else {
        /* Single processor */
        startgeid = 0;
    }


    /*---------- Traverse the elements to create vertices ----------------*/

    /* Initialize variables */
    oct = oct_getleftmost(tree->root);
    geid = startgeid;
    harborcount = 0;

    /* RICARDO SANITY CHECKS */
    if ( oct != tree->firstleaf ) {
        fprintf(stderr,
                "Thread %d: %s %d: error in the first leaf \n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, COMM_ERR);
        exit(1);
    }

    /* RICARDO SANITY CHECKS */
    if ( tree->firstleaf == NULL ) {
        fprintf(stderr,
                "Thread %d: %s %d: NULL first leaf \n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, COMM_ERR);
        exit(1);
    }

    /* Traverse the elements to create vertices */
    for (eindex = 0; eindex < ecount; eindex++) {
        tick_t edgesize;
        point_t pt;
        int32_t i, j, k;

        if (oct == NULL) {
            fprintf(stderr,
                    "\n\nThread %d: %s %d: internal error (too few octs)"
                    "\nI am in octant %d and I am supposed to be able to "
                    "go up to %d total octants\n\n",
                    tree->procid, __FILE__, __LINE__, (int)eindex, (int)ecount );
            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }

        /* Assign known fields. lnid[8] are undefined yet. */
        elemTable[eindex].geid = geid;
        elemTable[eindex].level = oct->level;
        elemTable[eindex].data = oct->payload.leaf->data;

        /* Produce eight mesh nodes, some of which may have been
           produced already. */

        edgesize = (tick_t)1 << (PIXELLEVEL - oct->level);

        for (k = 0; k < 2; k++) {
            pt.z = oct->lz + k * edgesize;

            for (j = 0; j < 2; j++) {
                pt.y = oct->ly + j * edgesize;

                for (i = 0; i < 2; i++) {
                    point_t adjustedpt;

                    pt.x = oct->lx + i * edgesize;

                    hashentry = math_hashuint32(&pt, 3) % ecount;
                    link = vertexHashTable[hashentry];

                    while (link != NULL) {
                        vertex = (vertex_t *)link->record;
                        if ((vertex->x == pt.x) &&
                            (vertex->y == pt.y) &&
                            (vertex->z == pt.z)) {
                            break;
                        } else {
                            link = link->next;
                        }
                    }

                    if (link == NULL) {
                        /* A newly encounter vertex */
                        vertex = (vertex_t *)mem_newobj(vertexpool);
                        link = (link_t *)mem_newobj(linkpool);

                        if ((vertex == NULL) || (link == NULL)) {
                            fprintf(stderr, "Thread %d: %s %d: out of memory\n"
                                    , tree->procid, __FILE__, __LINE__);
                            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                            exit(1);
                        }

                        /* Initialize the fields */
                        vertex->x = pt.x;
                        vertex->y = pt.y;
                        vertex->z = pt.z;

                        /* To find the owner of a node, adjust the
                           coordinate of a node if it's on the
                           boundary */
                        adjustedpt.x = (pt.x == tree->farendp[0]) ?
                            tree->farbound[0] : pt.x;
                        adjustedpt.y = (pt.y == tree->farendp[1]) ?
                            tree->farbound[1] : pt.y;
                        adjustedpt.z = (pt.z == tree->farendp[2]) ?
                            tree->farbound[2] : pt.z;

                        vertex->owner = (tree->groupsize == 1) ? 0 :
                            math_zsearch(tree->com->interval,
                                         tree->com->groupsize, &adjustedpt);

                        vertex->lnid = -1; /* undefined */
                        vertex->touches = 1;
                        vertex->level = oct->level;
                        vertex->share = NULL; /* undefined */

                        /* Link into the hash table */
                        link->record = vertex;
                        link->next = vertexHashTable[hashentry];
                        vertexHashTable[hashentry] = link;

                        /* Update the statistics */
                        harborcount++;

                    } else {
                        /* We have encountered this vertex already */

                        vertex->touches++;
                        vertex->level = (vertex->level < oct->level) ?
                            vertex->level : oct->level;
                    }
                } /* i */
            } /* j */
        } /* k */

        /* Move along the frontier to the next one in preorder traversal */
        oct = oct_getnextleaf(oct);

        /* Increment geid by 1 */
        geid++;

    } /* for all the local octants */

    /* RICARDO SANITY CHECKS */
    if ( oct != NULL ) {
        fprintf(stderr,
                "Thread %d: %s %d: error in the last leaf \n",
                tree->procid, __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, COMM_ERR);
        exit(1);
    }

    /*---Direct sharing: update the touches of vertices if necessary--------*/
    if (tree->groupsize > 1) {
        pctl_t *frompctl, *topctl;

        /* Assemble a vertex list for each of my adjacent
           processor. The vertices may not actually be present on
           their processors, for example, a dangling node */

        com_resetpctl(tree->com, sizeof(vertex_info_t));

        /* Visit all my vertices */
        for (hashentry = 0; hashentry < ecount; hashentry++) {

            link = vertexHashTable[hashentry];

            while (link != NULL) {
                tick_t nx, ny, nz;
                int32_t procid, nbrprocid[8], nbrs, xtick, ytick, ztick;
                point_t pt;
                int32_t idx, existed;

                vertex = (vertex_t *)link->record;

                if (vertex->touches == 8) {
                    /* All the octants sharing this vertex must
                       be on my processor. */
                    link = link->next;
                    continue;
                }

                nx = vertex->x;
                ny = vertex->y;
                nz = vertex->z;

                /* Mark as no neighbors at this moment */
                nbrs = 0;

                /* Find which neighbor(s) processor share this vertex */
                for (ztick = 0; ztick < 2; ztick++) {

                    pt.z = nz - ztick;

                    /* Discard out of bounds */
                    if ( ( pt.z < tree->nearendp[2] ) ||
                         ( pt.z >= tree->farendp[2] ) ) {
                        continue;
                    }

                    for (ytick = 0; ytick < 2; ytick++) {

                        pt.y = ny - ytick;

                        /* Discard out of bounds */
                        if ( ( pt.y < tree->nearendp[1] ) ||
                             ( pt.y >= tree->farendp[1] ) ) {
                            continue;
                        }

                        for (xtick = 0; xtick < 2; xtick++) {

                            pt.x = nx - xtick;

                            /* Discard out of bounds */
                            if ( ( pt.x < tree->nearendp[0] ) ||
                                 ( pt.x >= tree->farendp[0] ) ) {
                                continue;
                            }

                            /* If we pushed the surface down for bldgs... */
                            if ( tree->surfacep > 0 ) {

                                /* ...and the point is above the surface... */
                                if ( pt.z < tree->surfacep ) {

                                    /* ...and does not belong to any bldg... */
                                    int res = bldgs_nodesearch(
                                            pt.x, pt.y, pt.z, tree->ticksize);

                                    if ( res == 0 ) {

                                        /* ...then, discard 'air' nodes! */
                                        continue;
                                    }
                                }
                            }

                            /* Find who possess the pixel */
                            procid = math_zsearch(tree->com->interval,
                                                  tree->groupsize, &pt);

			    /* Sanity check introduced after the buildings
			     * options were incorporated. Should never occur */
			    if ( ( procid < 0 ) ||
			         ( procid > tree->groupsize ) ) {
	                        fprintf(stderr,
	                                "Thread %d: wrong procid from math "
	                                "search at octor_extractmesh direct "
	                                "sharing in vertex with coords "
	                                "x,y,z = %f %f %f\n",
	                                procid,
	                                pt.x*tree->ticksize,
	                                pt.y*tree->ticksize,
	                                pt.z*tree->ticksize);
			    }

                            if (procid == tree->procid) {

                                /* Discard my own nodes */
                                continue;

                            } else {

                                /* Owned by a remote processor */

                                /* Assume it's the first occurrence */
                                existed = 0;

                                for (idx = 0; idx < nbrs; idx++) {
                                    if (procid == nbrprocid[idx]) {
                                        /* Message created already */
                                        existed = 1;
                                        break;
                                    }
                                }

                                if (existed)
                                    continue;
                                else {
                                    nbrprocid[nbrs] = procid;
                                    nbrs++;
                                }
                            }

                        } /* xtick */
                    } /* ytick */
                } /* ztick */

                /* Create a message destined for the found (sharing)
                   neighbor processors */
                for (idx = 0; idx < nbrs; idx++) {
                    vertex_info_t *outvertex;
                    int32_t nbrid;

                    /* Get hold of the processor controller */
                    nbrid = nbrprocid[idx];
                    topctl = tree->com->pctltab[nbrid];
                    if (topctl == NULL) {
                        /* This should not happen */
                        fprintf(stderr, "Thread %d: %s %d: internal error\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                    /* Allocate space for outgoing modified point */
                    outvertex = (vertex_info_t *)mem_newobj(topctl->sndmem);

                    if (outvertex == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                    outvertex->x = vertex->x;
                    outvertex->y = vertex->y;
                    outvertex->z = vertex->z;
                    outvertex->owner = vertex->owner;
                    outvertex->touches = vertex->touches;
                    outvertex->level = vertex->level;
                }
                /* Move to the next vertex */
                link = link->next;

            } /* while link != NULL */
        } /* for all the entries in the hash table */


        /* Exchange vertex information between adjacent processors */
        com_OrchestrateExchange(tree->com, VERTEX_INFO_MSG, tree->comm_tree);


        /* Account of the vertex touches sent by my neighbors */
        frompctl = tree->com->firstpctl;

        while (frompctl != NULL) {
            vertex_info_t *invertex;

            mem_initcursor(frompctl->rcvmem);

            while ((invertex =
                    (vertex_info_t *)mem_getcursor(frompctl->rcvmem))
                   != NULL) {

                hashentry = math_hashuint32(&invertex->x, 3) % ecount;
                link = vertexHashTable[hashentry];

                while (link != NULL) {
                    vertex = (vertex_t *)link->record;

                    if ((vertex->x == invertex->x) &&
                        (vertex->y == invertex->y) &&
                        (vertex->z == invertex->z)) {
                        break;
                    } else
                        link = link->next;
                }

                if (link != NULL) {
                    /* I have already harbored this vertex */
                    vertex->touches += invertex->touches;

                } else {
                    /* Harbor this first-time vertex if I own it */
                    if (invertex->owner == tree->procid) {

                        link = (link_t *)mem_newobj(linkpool);
                        vertex = (vertex_t *)mem_newobj(vertexpool);

                        if ((link == NULL) || (vertex == NULL)) {
                            fprintf(stderr, "Thread %d: %s %d: out of memory\n"
                                    , tree->procid, __FILE__, __LINE__);
                            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                            exit(1);
                        }

                        /* Initialize the fields */
                        vertex->x = invertex->x;
                        vertex->y = invertex->y;
                        vertex->z = invertex->z;
                        vertex->owner = invertex->owner;
                        vertex->lnid = -1;
                        vertex->touches = invertex->touches;
                        vertex->level = invertex->level;
                        vertex->share = NULL;

                        /* Link into the hash table */
                        link->record = vertex;
                        link->next = vertexHashTable[hashentry];
                        vertexHashTable[hashentry] = link;

                        /* Update the statistics !!! */
                        harborcount++;

                    } else {
                        /* I do not use this vertex, nor do I own
                           this vertex. Ignore it */
                    }
                } /* Haven't seen this vertex before */

                if ((link != NULL) && (vertex->owner == tree->procid)) {
                    /* We need to add the sending processor to the
                       share list if this proc is the owner */

                    int32link_t *int32link;

                    int32link = (int32link_t *)mem_newobj(mess->int32linkpool);
                    if (int32link == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                        exit(1);
                    }

                    int32link->id = frompctl->procid;
                    int32link->next = vertex->share;
                    vertex->share = int32link;
                }

                mem_advcursor(frompctl->rcvmem);
            } /* while there are more incoming vertices */

            frompctl = frompctl->next;
        } /* While there are unprocessed incoming message */
    } /* Update vertex touches in a multi-processor environment */


    /*-----Indirect sharing: share via anchored vertices  -------------*/

    /* Go through the vertices I have harbored. Set the node
       property for each vertex. In case I own a dangling vertex,
       figure out who else (anchored vertices) I shall harbor. */

    if (tree->groupsize > 1) {
        int32_t nbrprocid, msgsize;

        /* Note we do not use the interval array for allcom */
        allcom = com_new(tree->procid, tree->groupsize);
        if (allcom == NULL) {
            fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        }

        /* Initialize allcom manually. Every processor is adjacent
           with every other processors */

        msgsize = sizeof(point_t);
        for (nbrprocid = tree->groupsize - 1; nbrprocid >= 0; nbrprocid--) {
            if (nbrprocid == tree->procid)
                continue;

            if ((allcom->pctltab[nbrprocid] = pctl_new(nbrprocid, msgsize))
                == NULL) {
                fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                        tree->procid, __FILE__, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                exit(1);
            }

            /* link into the pctl list */
            allcom->pctltab[nbrprocid]->next = allcom->firstpctl;
            allcom->firstpctl = allcom->pctltab[nbrprocid];
            allcom->nbrcnt++;
        }
    } /* if groupsize > 1 */


    for (hashentry = 0; hashentry < ecount; hashentry++) {
        int dep;
        tick_t smalloctsize;

        link = vertexHashTable[hashentry];

        while (link != NULL) {

            vertex = (vertex_t *)link->record;

            int32_t rp = node_setproperty(tree, vertex, &vertex->property,
                    bldgs_nodesearch);

            if ( rp != 0) {
                double x = vertex->x * tree->ticksize;
                double y = vertex->y * tree->ticksize;
                double z = vertex->z * tree->ticksize;
                fprintf( stderr,
                         "\n\nThread %d: %s %d: "
                         "node_setproperty() error %d\n"
                         "vertex coords (x,y,z): %f %f %f\n",
                         tree->procid, __FILE__, __LINE__, rp, x, y, z);
                MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                exit(1);
            }

            if (vertex->owner != tree->procid) {
                /* I do not own this vertex */
                link = link->next;
                continue;
            }

            /* I own this vertex. If it is an dangling node, I might
               need to harbor its parent anchors */

            smalloctsize = (tick_t)1 << (PIXELLEVEL - vertex->level);

            switch ((int)vertex->property) {
            case (XFACE):
                for (dep = 0; dep < 4; dep++) {
                    point_t pt;

                    pt.x = vertex->x;
                    pt.y = vertex->y +
                        ((dep & 0x1) ? smalloctsize : -smalloctsize);
                    pt.z = vertex->z +
                        ((dep & 0x2) ? smalloctsize : -smalloctsize);

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }

                break;

            case (YFACE):
                for (dep = 0; dep < 4; dep++) {
                    point_t pt;

                    pt.y = vertex->y;
                    pt.x = vertex->x +
                        ((dep & 0x1) ? smalloctsize : -smalloctsize);
                    pt.z = vertex->z +
                        ((dep & 0x2) ? smalloctsize : -smalloctsize);

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }
                break;

            case (ZFACE):
                for (dep = 0; dep < 4; dep++) {
                    point_t pt;

                    pt.z = vertex->z;
                    pt.x = vertex->x +
                        ((dep & 0x1) ? smalloctsize : -smalloctsize);
                    pt.y = vertex->y +
                        ((dep & 0x2) ? smalloctsize : -smalloctsize);

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }
                break;

            case (XEDGE):
                for (dep = 0; dep < 2; dep++) {
                    point_t pt;

                    pt.x = vertex->x +
                        ((dep == 1) ? smalloctsize : -smalloctsize);
                    pt.y = vertex->y;
                    pt.z = vertex->z;

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }
                break;

            case (YEDGE):
                for (dep = 0; dep < 2; dep++) {
                    point_t pt;

                    pt.y = vertex->y +
                        ((dep == 1) ? smalloctsize : -smalloctsize);
                    pt.x = vertex->x;
                    pt.z = vertex->z;

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }
                break;


            case (ZEDGE):
                for (dep = 0; dep < 2; dep++) {
                    point_t pt;

                    pt.z = vertex->z +
                        ((dep == 1) ? smalloctsize : -smalloctsize);
                    pt.x = vertex->x;
                    pt.y = vertex->y;

                    if (tree->groupsize > 1) {
                        node_harboranchored(tree, vertexHashTable, ecount,
                                            linkpool, vertexpool, allcom,
                                            pt, &harborcount);
                    }
                }
                break;

            default:
                /* Anchored node. Do nothing. */
                break;
            }

            /* Get to the next link */
            link = link->next;
        }

    } /* for all the hash table entries */


    if (tree->groupsize > 1) {
        pctl_t *frompctl;

        /* Exchange the ANCHORED_MSG to finish builing the sharing
           relationship */
        com_OrchestrateExchange(allcom, ANCHORED_MSG, tree->comm_tree);

        /* Some of my neighbors are telling me that they share
           anchored vertices owned by me */
        frompctl = allcom->firstpctl;

        while (frompctl != NULL) {
            point_t *anchored;

            mem_initcursor(frompctl->rcvmem);

            while ((anchored = (point_t *)mem_getcursor(frompctl->rcvmem))
                   != NULL) {

                hashentry = math_hashuint32(&anchored->x, 3) % ecount;
                link = vertexHashTable[hashentry];

                while (link != NULL) {
                    vertex = (vertex_t *)link->record;

                    if ((vertex->x == anchored->x) &&
                        (vertex->y == anchored->y) &&
                        (vertex->z == anchored->z)) {
                        break;
                    } else
                        link = link->next;
                }

                if ((link == NULL) ||
                    (vertex->owner != tree->procid)) {
                    fprintf(stderr, "Thread %d: %s %d: internal error",
                            tree->procid, __FILE__, __LINE__);
                    fprintf(stderr, "(cannot find my own anchored vertex)\n");
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);

                } else {
                    int32link_t *int32link;

                    /* Add the sending processor to the share list */
                    int32link = (int32link_t *)mem_newobj(mess->int32linkpool);
                    if (int32link == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                        exit(1);
                    }

                    int32link->id = frompctl->procid;
                    int32link->next = vertex->share;
                    vertex->share = int32link;
                }

                mem_advcursor(frompctl->rcvmem);
            } /* while there are more sharing information */

            frompctl = frompctl->next;
        }
    } /* Add indirect sharing information */

    if (tree->groupsize > 1) {
        /* Release the allcom */
        com_delete(allcom);
    }


    /*-------------Create node table ---------------------------------*/

    /* Allocate node table and initialize the fields. We cound't do
       this earlier since we don't know how many vertices this
       processor will have to harbor */

    nodeTable = (node_t *)malloc(sizeof(node_t) * harborcount);
    if (nodeTable == NULL) {
        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                tree->procid, __FILE__, __LINE__);
        fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                (int)(theAllocatedMemSum));
        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
        exit(1);
    } else {
        theAllocatedMemSum += sizeof(node_t) * harborcount;
    }

    ncount = 0; /* Number of nodes owned by this processor */
    ldnnum = 0; /* Number of dangling nodes owned by this processor */
    nindex = 0;

    /* Visit all the vertices I have harbored */
    for (hashentry = 0; hashentry < ecount; hashentry++) {

        link = vertexHashTable[hashentry];

        while (link != NULL) {
            vertex = (vertex_t *)link->record;

            /* Install the node data into the nodeTable. */

            node = &nodeTable[nindex];

            /* Adjust the node coordinate for ordering purpose. We
               shall reverse the coordiante at later stage */
            node->x = (vertex->x == tree->farendp[0]) ?
                tree->farbound[0] : vertex->x;
            node->y = (vertex->y == tree->farendp[1]) ?
                tree->farbound[1] : vertex->y;
            node->z = (vertex->z == tree->farendp[2]) ?
                tree->farbound[2] : vertex->z;

            if (vertex->owner == tree->procid) {
                ncount++;
                node->ismine = 1;
            } else {
                node->ismine = 0;
            }

            /* We don't know of the gnid yet. Will assign later. */

            if (node->ismine) {
                node->proc.share = vertex->share;
            } else {
                if (vertex->share != NULL) {
                    fprintf(stderr, "Thread %d: %s %d: internal error\n",
                            tree->procid, __FILE__, __LINE__);
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);
                }
                node->proc.ownerid = vertex->owner;
            }


            if (vertex->property & 0x80) {
                node->isanchored = 1;

            } else {

                node->isanchored = 0;

                if (node->ismine) {
                    unsigned char *ptr;

                    /* I need to allocate space for it in dnodeTable */
                    ldnnum++;

                    /* Store various dangling node property
                       temporarily in gnid */
                    ptr = (unsigned char *)&node->gnid;
                    *ptr = vertex->property;
                    ptr++;

                    /* Level of the smaller octants who share this
                       dangling node */
                    *ptr = vertex->level;
                }
            }

            /* Move to the next empty entry in the node table */
            nindex++;

            /* Get to the next link */
            link = link->next;
        }


    } /* for all the hash table entries */

    /* Sort all the nodes I've harbored in ascending Z-order */
    qsort(nodeTable, harborcount, sizeof(node_t), octor_zcompare);

    /* Set up the nodeCountTable and nodeStartTable, in a similar
       way as for the elements */
    if (tree->groupsize > 1) {

        /* Get node count distribution */
        tree_setdistribution(tree, &nodeCountTable, &nodeStartTable, ncount);
        startgnid = nodeStartTable[tree->procid];

    } else {
        /* Single processor */
        startgnid = 0;
    }

    /* Allocate space of the dnodeTable */
    if (ldnnum == 0) {
        dnodeTable = NULL;
    } else {
        dnodeTable = (dnode_t *)malloc(sizeof(dnode_t) * ldnnum);
        if (dnodeTable == NULL) {
            fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            fprintf(stderr, "Total memory allocated by Octor : %d bytes\n",
                    (int)(theAllocatedMemSum));
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        } else {
            theAllocatedMemSum += sizeof(dnode_t) * ldnnum;
        }
    }

    /* Assign the global node id to nodes owned by me and fill the
       lnid field of vertex_t in the hash table. Obtain the global
       node id from the owner processors for those not owned by me */

    gnid = startgnid;
    dnindex = 0; /* Dangling node index */

    if (tree->groupsize > 1) {
        partcom = com_new(tree->procid, tree->groupsize);
        if (partcom == NULL) {
            fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
            exit(1);
        }
    }


    for (nindex = 0; nindex < harborcount; nindex++) {

        /* Reverse the adjustments made earlier to make hashing work */
        nodeTable[nindex].x = (nodeTable[nindex].x == tree->farbound[0]) ?
            tree->farendp[0] : nodeTable[nindex].x;
        nodeTable[nindex].y = (nodeTable[nindex].y == tree->farbound[1]) ?
            tree->farendp[1] : nodeTable[nindex].y;
        nodeTable[nindex].z = (nodeTable[nindex].z == tree->farbound[2]) ?
            tree->farendp[2] : nodeTable[nindex].z;

        if (nodeTable[nindex].ismine &&
            (!nodeTable[nindex].isanchored)) {

            /* gnid is holding some temporary information regarding
               this DANGING node owned by me */

            /* Create a new record for the dangling node */
            dnodeTable[dnindex].ldnid = nindex;

            /* Copy the temperary information to deps */
            dnodeTable[dnindex].deps = *(uint32_t *)(&nodeTable[nindex].gnid);

            dnodeTable[dnindex].lanid = NULL;

            dnindex++;
        }

        /* We can safely overwrite gnid now */
        nodeTable[nindex].gnid = -1; /* indicate invalid */

        if (nodeTable[nindex].ismine) {
            int32link_t *int32link;

            /* Assign gnid to nodes owned by me. */
            nodeTable[nindex].gnid = gnid;
            gnid++;

            /* Create gnid_info for each processor who shares the node
               with me */

            int32link = nodeTable[nindex].proc.share;

            while (int32link != NULL) {
                int32_t procid;
                gnid_info_t *out_gnid_info;
                pctl_t *topctl;

                procid = int32link->id;
                topctl = partcom->pctltab[procid];
                if (topctl == NULL) {
                    /* First occurrence */
                    topctl = pctl_new(procid, sizeof(gnid_info_t));
                    if (topctl == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                        exit(1);
                    }

                    /* link into the pctl list */
                    partcom->pctltab[procid] = topctl;
                    partcom->pctltab[procid]->next = partcom->firstpctl;
                    partcom->firstpctl = partcom->pctltab[procid];

                    partcom->nbrcnt++;
                }

                /* Allocate space for outgoing gnid_info */
                out_gnid_info =
                    (gnid_info_t *)mem_newobj(topctl->sndmem);

                if (out_gnid_info == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                out_gnid_info->x = nodeTable[nindex].x;
                out_gnid_info->y = nodeTable[nindex].y;
                out_gnid_info->z = nodeTable[nindex].z;
                out_gnid_info->gnid = nodeTable[nindex].gnid;

                /* Get to the next sharing processor */
                int32link = int32link->next;
            }

        } else {
            /* This vertex is not owned by me. Let me expect to
               receive the information from its owner */
            int32_t procid;
            pctl_t *frompctl;

            procid = nodeTable[nindex].proc.ownerid;
            frompctl = partcom->pctltab[procid];

            if (frompctl == NULL) {
                /* First occurrence */
                frompctl = pctl_new(procid, sizeof(gnid_info_t));
                if (frompctl == NULL) {
                    fprintf(stderr, "Thread %d: %s %d: out of memory\n",
                            tree->procid, __FILE__, __LINE__);
                    MPI_Abort(MPI_COMM_WORLD, OUTOFMEM_ERR);
                    exit(1);
                }

                /* link into the pctl list */
                partcom->pctltab[procid] = frompctl;
                partcom->pctltab[procid]->next = partcom->firstpctl;
                partcom->firstpctl = partcom->pctltab[procid];

                partcom->nbrcnt++;

            } else {
                /* I already know I need to receive from procid */
            }
        } /* I don't own this node */

        hashentry = math_hashuint32(&nodeTable[nindex], 3) % ecount;
        link = vertexHashTable[hashentry];

        while (link != NULL) {
            vertex = (vertex_t *)link->record;

            if ((vertex->x == nodeTable[nindex].x) &&
                (vertex->y == nodeTable[nindex].y) &&
                (vertex->z == nodeTable[nindex].z)) {
                /* Hit */
                vertex->lnid = nindex;
                break;
            }
            link = link->next;
        }

        if (link == NULL) {
            fprintf(stderr, "Thread %d: %s %d: internal error\n",
                    tree->procid, __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }
    } /* for all the harbored nodes */


    /* Fill in the gnid of those vertices I harbor but don't own */
    if (tree->groupsize > 1) {
        pctl_t *frompctl;

        com_OrchestrateExchange(partcom, GNID_MSG, tree->comm_tree);

        /* Retreive global node ids sent to me */
        frompctl = partcom->firstpctl;

        while (frompctl != NULL) {
            gnid_info_t *in_gnid_info;

            mem_initcursor(frompctl->rcvmem);

            while ((in_gnid_info =
                    (gnid_info_t *)mem_getcursor(frompctl->rcvmem))
                   != NULL) {

                int32_t lnid;

                /* Find the vertex in my hashtable */
                hashentry = math_hashuint32(&in_gnid_info->x, 3) % ecount;
                link = vertexHashTable[hashentry];

                while (link != NULL) {
                    vertex = (vertex_t *)link->record;

                    if ((vertex->x == in_gnid_info->x) &&
                        (vertex->y == in_gnid_info->y) &&
                        (vertex->z == in_gnid_info->z)) {
                        lnid = vertex->lnid;
                        break;
                    } else
                        link = link->next;
                }

                if (link == NULL) {
                    fprintf(stderr, "Thread %d: %s %d: ",
                            tree->procid, __FILE__, __LINE__);
                    fprintf(stderr,"received unrelated vertex info.\n");
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);
                }

                /* Fill in the global node id properly */
                if (nodeTable[lnid].gnid != -1) {
                    fprintf(stderr, "Thread %d: %s %d: ",
                            tree->procid, __FILE__, __LINE__);
                    fprintf(stderr,"receive gnid that I already have.\n");
                    MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                    exit(1);
                }

                nodeTable[lnid].gnid = in_gnid_info->gnid;

                mem_advcursor(frompctl->rcvmem);
            } /* while there are more incoming vertices */

            frompctl = frompctl->next;
        } /* While there are unprocessed incoming message */

        com_delete(partcom);
    }


#ifdef DEBUG
    /* Debug: sanity check. */
    for (nindex = 0; nindex < harborcount; nindex++) {
        if (nodeTable[nindex].gnid == -1) {
            fprintf(stderr, "Thread %d: %s %d: unassigned gnid\n",
                    tree->procid, __FILE__, __LINE__);
            fprintf(stderr, "nindex = %d\n", (int32_t)nindex);
            MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
            exit(1);
        }
    }
#endif /* DEBUG */

    /* Correlate the element table to the node table */
    oct = oct_getleftmost(tree->root);

    for (eindex = 0; eindex < ecount; eindex++) {
        tick_t edgesize;
        point_t pt;
        int32_t i, j, k;
        int32_t whichchild;

        edgesize = (tick_t)1 << (PIXELLEVEL - oct->level);

        whichchild = 0;
        for (k = 0; k < 2; k++) {
            pt.z = oct->lz + k * edgesize;

            for (j = 0; j < 2; j++) {
                pt.y = oct->ly + j * edgesize;

                for (i = 0; i < 2; i++) {

                    pt.x = oct->lx + i * edgesize;

                    hashentry = math_hashuint32(&pt, 3) % ecount;
                    link = vertexHashTable[hashentry];

                    while (link != NULL) {
                        vertex = (vertex_t *)link->record;
                        if ((vertex->x == pt.x) &&
                            (vertex->y == pt.y) &&
                            (vertex->z == pt.z)) {

                            elemTable[eindex].lnid[whichchild] =
                                vertex->lnid;
                            whichchild++;
                            break;
                        }

                        link = link->next;
                    }

                    if (link == NULL) {
                        fprintf(stderr, "Thread %d: %s %d: internal error\n",
                                tree->procid, __FILE__, __LINE__);
                        MPI_Abort(MPI_COMM_WORLD, INTERNAL_ERR);
                        exit(1);
                    }

                } /* i */
            } /* j */
        } /* k */

        oct = oct_getnextleaf(oct);

    } /* for all the local octants */

    /* Correlate the dangling node to anchored nodes */
    for (dnindex = 0; dnindex < ldnnum; dnindex++) {
        unsigned char *ptr;
        int8_t level;
        unsigned char property;
        tick_t smalloctsize;
        int32_t dep;
        int32_t ldnid;

        ptr = (unsigned char *)&dnodeTable[dnindex].deps;
        property = *ptr;

        ptr++;
        level = *(int8_t *)ptr;

        smalloctsize = (tick_t)1 << (PIXELLEVEL - level);
        ldnid = dnodeTable[dnindex].ldnid;

        switch ((int)property) {
        case (XFACE):
            dnodeTable[dnindex].deps = 4;

            for (dep = 0; dep < 4; dep++) {
                point_t pt;

                pt.x = nodeTable[ldnid].x;
                pt.y = nodeTable[ldnid].y +
                    ((dep & 0x1) ? smalloctsize : -smalloctsize);
                pt.z = nodeTable[ldnid].z +
                    ((dep & 0x2) ? smalloctsize : -smalloctsize);

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;

        case (YFACE):
            dnodeTable[dnindex].deps = 4;

            for (dep = 0; dep < 4; dep++) {
                point_t pt;

                pt.y = nodeTable[ldnid].y;
                pt.x = nodeTable[ldnid].x +
                    ((dep & 0x1) ? smalloctsize : -smalloctsize);
                pt.z = nodeTable[ldnid].z +
                    ((dep & 0x2) ? smalloctsize : -smalloctsize);

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;

        case (ZFACE):
            dnodeTable[dnindex].deps = 4;

            for (dep = 0; dep < 4; dep++) {
                point_t pt;

                pt.z = nodeTable[ldnid].z;
                pt.x = nodeTable[ldnid].x +
                    ((dep & 0x1) ? smalloctsize : -smalloctsize);
                pt.y = nodeTable[ldnid].y +
                    ((dep & 0x2) ? smalloctsize : -smalloctsize);

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;

        case (XEDGE):
            dnodeTable[dnindex].deps = 2;

            for (dep = 0; dep < 2; dep++) {
                point_t pt;

                pt.x = nodeTable[ldnid].x +
                    ((dep == 1) ? smalloctsize : -smalloctsize);
                pt.y = nodeTable[ldnid].y;
                pt.z = nodeTable[ldnid].z;

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;

        case (YEDGE):
            dnodeTable[dnindex].deps = 2;

            for (dep = 0; dep < 2; dep++) {
                point_t pt;

                pt.y = nodeTable[ldnid].y +
                    ((dep == 1) ? smalloctsize : -smalloctsize);
                pt.x = nodeTable[ldnid].x;
                pt.z = nodeTable[ldnid].z;

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;


        case (ZEDGE):
            dnodeTable[dnindex].deps = 2;

            for (dep = 0; dep < 2; dep++) {
                point_t pt;

                pt.z = nodeTable[ldnid].z +
                    ((dep == 1) ? smalloctsize : -smalloctsize);
                pt.x = nodeTable[ldnid].x;
                pt.y = nodeTable[ldnid].y;

                dnode_correlate(tree, mess, vertexHashTable, ecount,
                                dnodeTable, dnindex, pt);
            }
            break;
        }
    } /* for all DANGLING nodes owned by me */


    /* Free temporary data structures */
    if (tree->groupsize > 1) {
        free(octCountTable);
        free(octStartTable);
        free(nodeCountTable);
        free(nodeStartTable);
    }
    free(vertexHashTable);
    mem_delete(vertexpool);
    mem_delete(linkpool);

    /* Discard memory used by the communication manager */
    if (tree->groupsize > 1) {
        com_resetpctl(tree->com, 0);
    }

    /* Fill in the fields for the return structure */
    mess->lenum = ecount;
    mess->lnnum = ncount;
    mess->ldnnum = ldnnum;
    mess->nharbored = harborcount;
    mess->ticksize = tree->ticksize;
    mess->elemTable = elemTable;
    mess->nodeTable = nodeTable;
    mess->dnodeTable = dnodeTable;

#ifdef MESH_VERBOSE
    mess_showstat(mess, DETAILS, "octor_extractmesh");
#endif /* TREE_VERBOSE */

    return (mesh_t *)mess;
}


/**
 * octor_deletemesh: Free memory associated with the mesh.
 *
 */
extern void
octor_deletemesh(mesh_t *mesh)
{
    mess_t *mess = (mess_t *)mesh;

    if (mesh == NULL)
        return;

    theRecordPoolRefs--;
    if (theRecordPoolRefs == 0) {
        mem_delete(theRecordPool);
    }

    if (mess == NULL)
        return;

    mem_delete(mess->int32linkpool);

    free(mess->elemTable);
    free(mess->nodeTable);
    free(mess->dnodeTable);

    free(mess);
}


/**
 * octor_showstat: Show various data type sizes, total memory usage,
 *                 total memory allocation, average memory (actual/nominal)
 *                 usage per element.
 *
 */
extern void octor_showstat(octree_t *octree, mesh_t *mesh)
{
    int32_t oct_t_size, leaf_t_size, internal_t_size,
        elem_t_size, node_t_size, dnode_t_size;

    printf("sizeof(void *)     = %zu\n", sizeof(void *));
    printf("sizeof(oct_t)      = %d\n", oct_t_size = sizeof(oct_t));
    printf("sizeof(leaf_t)     = %d\n", leaf_t_size = sizeof(leaf_t));
    printf("sizeof(interior_t) = %d\n", internal_t_size = sizeof(interior_t));
    printf("sizeof(elem_t)     = %d\n", elem_t_size = sizeof(elem_t));
    printf("sizeof(node_t)     = %d\n", node_t_size = sizeof(node_t));
    printf("sizeof(dnode_t)    = %d\n", dnode_t_size = sizeof(dnode_t));

    return;
}


/**
 * octor_getintervaltable: Return the pointer to the interval table held by
 *                         by communication manager
 *
 */
extern const point_t * octor_getintervaltable(octree_t *octree)
{
    tree_t *tree = (tree_t *)octree;

    if (tree->com == NULL)
        return NULL;
    else
        return (const point_t *)tree->com->interval;
}


#ifdef MESS_VERBOSE

/**
 * mess_showstat: TODO: move mesh_printstat here
 *
 */
static void
mess_showstat(mess_t *mess, int32_t mode, const char *comment)
{

}

#endif /* MESS_VERBOSE */
