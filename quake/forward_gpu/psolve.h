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

/*
 *
 * psolve.h: Generate an unstructured mesh and solve the linear system
 *           thereof derived and output the results.
 *
 * Input:    material database (cvm etree), physics.in, numerical.in.
 * Output:   mesh database (mesh.e) and 4D output.
 *
 *
 * Notes:
 *   For a history of changes see the ChangeLog file.  Also, if you
 *   perform any changes to the source code, please document the changes
 *   by adding the appropriate entry in the ChangeLog file and a descriptive
 *   message during CVS commit.  Thanks!
 */

#ifndef PSOLVE_H
#define PSOLVE_H

#include <inttypes.h>
#include <stdio.h>
#include <mpi.h>

#include "octor.h"

#define ERROR       HERC_ERROR

#ifndef DO_DEBUG
# ifdef  DEBUG
#  define DO_DEBUG        1
# else
#  define DO_DEBUG        0
# endif /* DEBUG */
#endif /* DO_DEBUG */

/* GPU specification */
typedef struct {
  int32_t device;
  int32_t max_threads;
  int32_t max_block_dim[3];
  int32_t max_grid_dim[3];
} gpu_spec_t;

/* Reverse node->element table lookup entry */
typedef struct rev_elem_t {
  int32_t elemid;
  int32_t offset;
} rev_elem_t;

typedef struct rev_entry_t {
  int32_t count;
  rev_elem_t elements[8];
} rev_entry_t;


extern MPI_Comm comm_solver;
extern MPI_Comm comm_IO;

/* Sets solver to use single or double precision in main variables */
#ifdef  SINGLE_PRECISION_SOLVER
   typedef float solver_float;
#else
   typedef double solver_float;
#endif

typedef int32_t     local_id_t;		/**< Local element and node id type  */
typedef local_id_t  lnid_t;		/**< Local node id type		     */
typedef local_id_t  leid_t;		/**< Local element id type	     */
typedef int64_t     global_id_t;	/**< Global element and node id type */
typedef global_id_t gnid_t;		/**< Global node id type	     */
typedef global_id_t geid_t;		/**< Global element id type	     */


struct corner_ref_t {
    lnid_t corner_lnid[8];              /**< Corners local node ids */
    double min_value, max_value;
};

typedef struct corner_ref_t corner_ref_t;

/**
 * Mesh database (element) record payload.
 */
struct mdata_t {
    int64_t nid[8];
    float edgesize, Vp, Vs, rho;
};

typedef struct mdata_t mdata_t;


/**
 * Mesh element data fields
 */
typedef struct edata_t {
    float edgesize, Vp, Vs, rho, a0_shear, a1_shear, b_shear, g0_shear, g1_shear, a0_kappa, a1_kappa, b_kappa, g0_kappa, g1_kappa;
} edata_t;



/** 3-ary double vector */
struct fvector_t {
    solver_float f[3];
};

typedef struct fvector_t fvector_t;


union df3_t {
    struct {
        double x, y, z;
    };
    double f[3];
};


/**
 * out_hdr_t: 4D output file header.
 */
struct out_hdr_t {
    /** file type string identifier: "Hercules 4D output vnnn"  */
    char    file_type_str[29];
    int8_t  format_version;	/**< File format version		*/
    int8_t  endiannes;		/**< File endianess: 0=little, 1=big	*/

    /** Identifier of the platform where the file was generated */
    int8_t   platform_id;
    unsigned char ufid[16];	/**< "Unique" file identifier		*/
    int64_t  total_nodes;	/**< Node count.			*/
    int32_t  output_steps;	/**< Number of output time steps.	*/

    /** Number of components per (node) record, e.g., 1 vs. 3 */
    int32_t  scalar_count;
    int8_t   scalar_size;	/**< size (in bytes) of each scalar value  */

    /**<
     * Type of scalar, it can take one of the following values
     * - INVALID:		 0
     * - FLOAT32 (float):	 1
     * - FLOAT64 (double):	 2
     * - FLOAT128 (long double): 3
     * - INT8:			 4
     * - UINT8:			 5
     * - INT16:			 6
     * - UINT16:		 7
     * - INT32:			 8
     * - UINT32:		 9
     * - INT64:			 10
     * - UINT64:		 11
     */
    int8_t  scalar_type;

    /**<
     * scalar class, it can take one of the following values:
     * - INVALID:		 0
     * - FLOAT_CLASS		 1
     * - INT_CLASS		 2
     */
    int8_t  scalar_class;

    int8_t  quantity_type;	/* 0: unknown, 1: displacement, 2: velocity */


    /** Mesh parameters: extent of the simulated region in meters */
    double  domain_x, domain_y, domain_z;

    /**
     * Mesh parameter: tick size, factor for converting from domain units (m)
     * to etree units.
     */
    double  mesh_ticksize;

    /** Simulation parameter: delta t (time) */
    double  delta_t;

    /** Mesh parameter: total number of elements */
    int64_t total_elements;

    /** 4D Output parameter: how often is an output time step written out */
    int32_t output_rate;

    /** Simulation parameter: total number of simulation time steps */
    int32_t total_time_steps;

    int64_t generation_date;	/**< Time in seconds since the epoch	*/
};

typedef struct out_hdr_t out_hdr_t;


/*---------------- Solver data structures -----------------------------*/

/**
 * Constants initialized element structure.
 */
struct e_t {
    double c1, c2, c3, c4;
};

typedef struct e_t e_t;


/**
 * Constants initialized node structure.
 *
 * This structure has been changed as a result of the new algorithm
 * for the terms involved with the damping in the solution for the
 * next time step.
 */
struct n_t {
    solver_float mass_simple;
    solver_float mass2_minusaM[3];
    solver_float mass_minusaM[3];
};

typedef struct n_t n_t;

/**
 * Data type to represent stiffness matrices
 */
struct fmatrix_t {
    double f[3][3];
};

typedef struct fmatrix_t fmatrix_t;


/* Solver computation and communication routines */

typedef struct messenger_t messenger_t;

/**
 * A messenger keeps track of the data exchange.
 */
struct messenger_t {
    int32_t  procid;        /**< Remote PE id.     */

    int32_t  outsize;       /**< Outgoing record size. */
    int32_t  insize;        /**< Incoming record size. */

    void*    outdata;
    void*    indata;

    int32_t  nodecount;     /**< Number of mesh nodes involved.         */
    int32_t  nidx;          /**< Current index into the mapping table.  */
    int32_t* mapping;       /**< Array of local ids for involved nodes. */

    messenger_t* next;
};


/**
 * Communication schedule structure.
 */
struct schedule_t {
    /**
     * Number of PEs this PE contributes to or retrieves data from because
     * this PE harbors their nodes.
     */
    int32_t       c_count;
    messenger_t*  first_c;
    messenger_t** messenger_c;  /**< Fast lookup table to build c-list. */
    MPI_Request*  irecvreqs_c;  /**< control for non-blocking receives. */
    MPI_Status*   irecvstats_c;

    /** Number of processors who share nodes owned by me. */
    int32_t       s_count;
    messenger_t*  first_s;
    messenger_t** messenger_s;  /* Fast lookup table to build s-list.      */
    MPI_Request*  irecvreqs_s;  /* controls for non-blocking MPI receives. */
    MPI_Status*   irecvstats_s;
};

typedef struct schedule_t schedule_t;


/**
 * Solver data structure.
 */
struct solver_t {
    e_t *eTable;            /* Element computation-invariant table */
    n_t *nTable;            /* Node computation-invariant table */

    fvector_t *tm1;         /* Displacements at timestep t - 1 */
    fvector_t *tm2;         /* Displacements at timestep t - 2 */
    fvector_t *force;       /* Force accumulation at timestep t */
};

typedef struct solver_t solver_t;


/**
 * \todo Document this struct.
 */
struct mysolver_t {
    e_t* eTable;            /* Element computation-invariant table  */
    n_t* nTable;            /* Node computation-invariant table     */

    /* \TODO Explain better what is tm1,2,3 */
    fvector_t* tm1;         /* Displacements at timestep t - 1      */
    fvector_t* tm2;         /* Displacements at timestep t - 2      */
    fvector_t* tm3;         /* Displacements at timestep t - 3      */
    fvector_t* force;       /* Force accumulation at timestep t     */

    schedule_t* dn_sched;   /* Dangling node communication schedule */
    schedule_t* an_sched;   /* Anchored node communication schedule */

    fvector_t* conv_shear_1;      /* Approximate Convolution Calculation */
    fvector_t* conv_shear_2;
    fvector_t* conv_kappa_1;
    fvector_t* conv_kappa_2;

    /* GPU device data structures */
    gpu_spec_t* gpu_spec;

    elem_t*     elemTableDevice; // this should be declared in mesh_t struct
    e_t*        eTableDevice;
    fvector_t*  tm1Device;
    fvector_t*  forceDevice;
};

typedef struct mysolver_t mysolver_t;


/* This structures was copied from geometrics.h
 * \todo We need to unify this structure because is used in several
 *       parts of the code and with different names, see above, for
 *       example, fvector_t
 */
struct vector3D_t {
    double x[3];
};

typedef struct vector3D_t vector3D_t;


/* ------------------------------------------------------------------------- *
 *               Output stations data structures
 * ------------------------------------------------------------------------- */

struct station_t {
    int32_t    id, nodestointerpolate[8];
    double*    displacementsX;
    double*    displacementsY;
    double*    displacementsZ;

    vector3D_t coords;        /* cartesian */
    vector3D_t localcoords;   /* csi, eta, dzeta in (-1,1) x (-1,1) x (-1,1) */
    FILE*      fpoutputfile;
};

typedef struct station_t station_t;



#ifdef __cplusplus
extern "C" {
#endif

extern int  solver_abort (const char* function_name, const char* error_msg,
			  const char* format, ...);
extern void solver_output_seq (void);
extern int  parsetext (FILE* fp, const char* querystring, const char type,
		       void* result);



/**
 * mu_and_lambda: Calculates mu and lambda according to the element values
 *                of Vp, Vs, and Rho and verifies/applies some rules defined
 *                by Jacobo, Leonardo and Ricardo.  It was originally within
 *                solver_init but was moved out because it needs to be used
 *                in other places as well (nonlinear)
 */
void mu_and_lambda(double *theMu, double *theLambda, edata_t *edata, int32_t eindex);

/**
 * Search a point in the domain of the local mesh.
 *
 *   input: coordinates
 *  output: 0 fail 1 success
 */
int32_t search_point( vector3D_t point, octant_t **octant );


#ifdef __cplusplus
}
#endif
#endif /* PSOLVE_H */
