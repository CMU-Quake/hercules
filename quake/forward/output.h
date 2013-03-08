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
 * Description: 4D parallel output routines.
 */
#ifndef QUAKE_OUTPUT_H
#define QUAKE_OUTPUT_H

#include <stdio.h>
#include <sys/types.h>

#include <octor.h>
#include "psolve.h"


/**
 * Structure to collect simulation output parameters.
 */
struct output_parameters_t {
    /** Flag indicating whether or not to perform output */
    int do_output;

    /**
     * Flag indicating whether the output should be peformed in parallel
     * or sequentially (i.e., through PE 0).
     */
    int parallel_output;

    /** Flag indicating whether or not to output node displacement. */
    int output_displacement;

    /** Flag indicating whether or not to output node velocity. */
    int output_velocity;


    char* displacement_filename;

    char* velocity_filename;

    /** Expected file output size including the header size */
    off_t output_size;

    mesh_t*   mesh;
    solver_t* solver;

    /* These fields are used in the output header, other fields needed
     * in the header are either fixed, or obtained at run-time
     */
    int64_t total_nodes;	/**< Node count.			*/
    int64_t total_elements;	/**< Element count.			*/
    double  delta_t;		/**< Simulation delta t			*/

    /** Number of simulation time steps between solver output steps     */
    int32_t output_rate;

    int32_t total_time_steps;
    int32_t pe_id;		/**< Processing Element (PE) id / rank.	*/
    int32_t pe_count;		/**< Number of PEs (group size).	*/

    double  domain_x, domain_y, domain_z;

    /** File name for the output summary stats */
    char*       stats_filename;
    char*       debug_filename;

    /**
     * Flag indicating whether or not to to print debug statements for
     * the output code
     */
    int         output_debug;
};

typedef struct output_parameters_t output_parameters_t;


/**
 * Output stats.
 */
struct output_stats_t {
    float	 lat_avg;	/**< Average output latency.		*/
    float	 lat_var;	/**< Output latency variance.		*/
    float	 tput_avg;	/**< Average value.			*/
    float	 tput_var;	/**< Average Variance.			*/
    float	 tput_min;	/**< Minimum value.			*/
    float	 tput_max;	/**< Maximum value.			*/
    float	 iops;		/**< I/O operations per second.		*/
    float	 iop_count;	/**< Number of I/O ops performed.	*/
    unsigned int count;		/**< Number of measurements.		*/
};

typedef struct output_stats_t output_stats_t;


/**
 * Output related macros.
 */
#ifndef	NO_OUTPUT
#define DO_OUTPUT	1
#else	/* NO_OUTPUT */
#define DO_OUTPUT	0
#endif /* NO_OUTPUT */


#ifdef __cplusplus
extern "C" {
#endif

extern int do_solver_output( void );
extern int output_init_state (output_parameters_t* params);
extern int output_fini();
extern int output_collect_io_stats (const char* filename, output_stats_t* disp_stats, output_stats_t* vel_stats, double e2e_elapsed_time);

#ifdef __cplusplus
}
#endif


#endif /* QUAKE_OUTPUT_H */
