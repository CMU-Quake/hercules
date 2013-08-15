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

/**
 * \file
 *
 * Assumptions:
 * - Three components of the desired property are saved, i.e., x,y,z.
 * - Each output file has an output header: \see out_header_t.
 * - Output is a double float per component.
 *
 * \todo Save job/run metadata as a text file.
 * \todo Add the following field to the output metadata:
 *       simulation code version.
 * \todo Add print out function for (output) metadata header.
 *
 * \section perf-stats Performance stats
 * Notes on overall performance stats for the parallel output code.
 *
 * - Avg throughput per PE serves as a rough estimate of the overall
 *   throughput (avg_local_throughput * #PEs) as well as the realized
 *   throughput per PE.
 *
 * - Min and Max throughput are the peak high and low throughput values
 *   observed during an execution.  This provides a high-level idea of the
 *   throughput variation among PEs and output time steps throught the
 *   execution.
 *
 * - Min avg, max avg, var(avg) throughtput characterize how the throughput
 *   varies across PEs, if the [min-max] range is close and var is low, this
 *   indicates that PEs realize about the same local throughput throuout the
 *   execution.
 *
 * - Min var, max var, avg (var) throughput shows what the PE local
 * throughput variance is accross PEs.  A PE's variance of the local
 * throughput shows how much the observed throughput varies for a PE
 * during the execution.  Min (var), Avg (var), Max (var) shows how the
 * variance behaves across PEs.  Do some PEs observe lower variance than others?
 *
 * Create an output stats file with the following contents:
 *
 * - Number of PEs.
 * - Number of output time steps.
 * - Expected output size.
 * - High-level stats as described above for displacement and velocity output.
 * - Detailed matrix with a row per PE with the following entries:
 *   - PE id
 *   - Number of nodes written
 *   - Avg throughput.
 *   - Stdev throughput.
 *   - Min throughput.
 *   - Max throughput.
 *   - CV throughput.
 *   - Avg latency.
 *   - Stdev latency.
 *   - CV latency.
 *   - IOP count.
 *   - IOPS (during actual I/O).
 *   - IOPS (amortized, i.e., including I/O + compute + communication time).
 * - Summary rows with min, max, median (with corresponding rank) and avg, std.
 */

#include <mpi.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <alloca.h>
#include <stdarg.h>

#include "output.h"
#include "psolve.h"
#include "util.h"


/* ------------------------------------------------------------------------- *
 * Macro definitions
 * ------------------------------------------------------------------------- */
#define HERCULES_4D_FORMAT_VERSION	0


/* unfortunately vararg macros are not well supported by many compilers or
 * on some platforms.
 */
#define DEBUG_MSG0(MSG)					\
    do {						\
        if (DO_DEBUG && po_debug_output_) {		\
	   po_debug_msg(MSG);              		\
        }				  		\
    } while (0)


#define DEBUG_MSG1(MSG,p1)				\
    do {						\
        if (DO_DEBUG && po_debug_output_) {		\
	   po_debug_msg(MSG,p1);              		\
        }				         	\
    } while (0)


#define DEBUG_MSG2(MSG,p1,p2)				\
    do {						\
        if (DO_DEBUG && po_debug_output_) {		\
            po_debug_msg(MSG, p1, p2);                  \
        }				                \
    } while (0)

#define DEBUG_MSG3(MSG,p1,p2,p3)			\
    do {						\
        if (DO_DEBUG && po_debug_output_) {		\
            po_debug_msg(MSG, p1, p2, p3);              \
        }				                \
    } while (0)



/* ------------------------------------------------------------------------- *
 * Structure and type definitions
 * ------------------------------------------------------------------------- */

typedef int64_t global_node_id_t;
typedef int32_t local_node_id_t;


/**
 * Output stat running counters.
 */
struct po_stat_counters_t {
    double min;		/**< Minimum value seen so far */
    double max;		/**< Maximum value seen so far */
    double sum;		/**< Running sum	       */
    double sum2;	/**< Running square sum	       */
    double sum_inv;	/**< Sum (1/x)		       */
    /**< Sum (1/x^2) needed to compute the variance of 1/x */
    double sum_inv2;
    int    count;	/**< Number of measurements    */
    /* Number of I/O operations per time step per node */
    unsigned int iop_per_out_step;
};

typedef struct po_stat_counters_t po_stat_counters_t;


/**
 * Information and parameters about the parallel output.
 */
struct par_output_t {
    /** Flag indicating whether this struct has been initialized. */
    int initialized;

    /** Index of the first harbored noded that is locally owned */
    int first_owned_idx;

    int64_t local_node_count;
    int64_t total_node_count;
    int64_t local_base_node_id;

    /** Relative offset seeked by a process (PE: processing element) */
    off_t relative_offset;

    /** Stride between timesteps, i.e., size of a given time step data */
    off_t stride;

    /** Base offset where to seek from, i.e., space for a file header */
    off_t base_offset;

    off_t local_time_step_chunk_size;

    off_t expected_output_file_size;

    /** File handle for the displacement output */
    FILE* disp_fp;

    /** File handle for the velocity output */
    FILE* vel_fp;

    /** Flag indicating whether or not the displacement should be saved */
    int output_displacement;

    /** Flag indicating whether or not the velocity field should be saved */
    int output_velocity;

    int output_steps;
    int cur_output_time_step;

    double delta_t;

    /** Current offset */
    off_t current_offset;

    /** Pointer to the local mesh structure (in mesh.c) */
    mesh_t* mesh;

    /** Pointer to the local solver structure (in psolve.c) */
    solver_t* solver;

    /** Displacement output time stats */
    output_stats_t disp_time;

    /** Velocity output time stats */
    output_stats_t vel_time;

    int pe_id;		/**< Processing Element (PE) id / rank.	*/
    int pe_count;	/**< Number of PEs (group size).	*/

    po_stat_counters_t disp_stats;
    po_stat_counters_t vel_stats;

    off_t bytes_written;

    /**
     * Wall clock time from output open to close, it should be very close
     * to the solver run time.  This is really an estimate, since we don't
     * perform any synchronization to figure out the global time.
     */
    double output_elapsed_time;
};

typedef struct par_output_t par_output_t;


/* ------------------------------------------------------------------------- *
 * Global variables
 * ------------------------------------------------------------------------- */

/* These are static anyway, so only visible within this file's scope */

static par_output_t po_;


/** Flag indicating whether or not to perform parallel output */
static int do_parallel_output_;


/**
 * Flag indicating whether or not to produce debug output.  This is global,
 * so it can be easily accessed within this file/module.
 */
static int po_debug_output_ = 0;


/**
 * File handle to the debug output stream.  This is global,
 * so it can be easily accessed within this file/module.
 */
static FILE* po_debug_fp_ = NULL;


/* ------------------------------------------------------------------------- *
 * Function definitions
 * ------------------------------------------------------------------------- */

static void
output_stats_init (po_stat_counters_t* stats)
{
    assert (NULL != stats);

    stats->min      = DBL_MAX;
    stats->max      = -DBL_MAX;
    stats->sum      = 0;
    stats->sum2     = 0;
    stats->sum_inv  = 0;
    stats->sum_inv2 = 0;
    stats->count    = 0;
}


/**
 * Compute the sample variance as follows:
 *
 * E[x]   = avg(x)   = sum(xi)/n
 * E[x^2] = avg(x^2) = sum(xi^2)/n
 *
 * Sample variance:
 * S^2[x] = var(x)   = sum(xi^2)/(n-1) - n*(E[x]*E[x])/(n-1)
 *
 * \param sum	Sum of sample values.
 * \param sum2  Sum of the square of sample values, i.e, Sum (x^2).
 * \param n	Number of samples.
 *
 * \return the sample variance if n > 1, 0 otherwise.
 */
static double
sample_variance (double sum, double sum2, unsigned int n)
{
    double var = 0;

    if (n > 1) {
	/* mean = sum / n ;
	 * n * mean * mean == sum * sum / n 
	 */

	var = ( sum2 / (n - 1) ) - ( (sum * sum) / n / (n - 1) );
    }

    return var;
}


/**
 * Compute population variance as:
 *
 * s^2[x] = var(x)   = sum(xi^2)/n - (sum(xi)/n)^2 = E[x^2] - E[x]^2
 * E[x]   = avg(x)   = sum(xi)/n
 * E[x^2] = avg(x^2) = sum(xi^2)/n
 */
static inline double
population_variance (double mean, double sum2, unsigned int n)
{
    return (sum2 / n) - (mean * mean);
}


/**
 * \return 0: on success
 *	  -1: general error (fstat)
 *	  -2: file is a fifo (cannot determine file size).
 *        -3: not a regular file
 */
static int
get_file_size (int fd, off_t* p_filesize)
{
    int ret = -1;
    struct stat buf;

    assert (fd >= 0);
    assert (p_filesize);

    ret = fstat (fd, &buf);

    if (ret < 0){
	perror ("fstat");
	fprintf (stderr, "Error in po_get_file_size fstat (fd = %d) returned %d\n",
		 fd, ret);
	return -1;
    }

    if (S_ISREG (buf.st_mode)) {
	*p_filesize = buf.st_size;
	ret = 0;
    }

    else if (S_ISFIFO (buf.st_mode)) {
	ret = -2;
    }

    else {
	ret = -3;
    }

    return ret;
}


static void
output_stats_update (po_stat_counters_t* stats, double val)
{
    double val2;

    assert (NULL != stats);

    if (val < stats->min) {
	stats->min = val;
    }

    if (val > stats->max) {
	stats->max = val;
    }

    stats->sum      += val;
    stats->sum_inv  += 1/val;
    val2             = val * val;
    stats->sum2     += val2;
    stats->sum_inv2 += 1/val2;
    stats->count++;
}


static void inline
po_debug_msg (const char* msg, ...)
{
    int ret;

    if (po_debug_output_ && po_debug_fp_ != NULL) {
        va_list ap;

        va_start (ap, msg);
        ret =  vfprintf (po_debug_fp_, msg, ap);
	va_end (ap);
    }

    return;
}


/**
 * Ramdomly generate a 128-bit identifier for a file.
 *
 * \note This function is not portable in the sense that it does not
 *	 produce the same result in different platforms with the same seed,
 *       but that's OK.
 */
static void
generate_fileid (unsigned char ufid[16])
{
    static int initialized = 0;

    int i;
    int* ufid_ptr = (int*)ufid;


    if (!initialized) {
	time_t t;

	srandom (time (&t));
    }

    for (i = 0; i < 4; i++) {
	*ufid_ptr = random();
	ufid_ptr++;
    }
}


/**
 * Check whether the parallel output parameters have been initialized.
 */
static int
po_check_init (const par_output_t* po)
{
    if (NULL == po) {
	/* programming error! */
	solver_abort ("po_check_init()", NULL, "po is NULL\n");
	return -1;
    }

    if (po->initialized == 0) {
	/* programming error! */
	solver_abort ("po_check_init()", NULL, "po has NOT been initialized\n");
	return -1;
    }

    return 0;
}


static FILE*
po_open_file (const char* filename, const char* mode)
{
    int ret = 0;

    FILE* fhnd = fopen (filename, mode);

    if (NULL == fhnd) {
	solver_abort ("po_open_file", "fopen() failed", "filename = %s",
		      filename);
    }

    /* fhnd must be unbuffered, otherwise we are in for big trouble, i.e.,
     * output file corruption.  Using unbuffered output essentially converts
     * stdio calls to wrappers around the system calls.
     */
    ret = setvbuf (fhnd, NULL, _IONBF, 0);
    
    if (0 != ret) {
	fprintf (stderr, "\n\nWarning!: could not disable buffering on file handle (0x%p)\n", fhnd);
	fclose (fhnd);
	fhnd = NULL;
    }

    return fhnd;
}


static unsigned int
get_output_time_step_count (int total_time_steps, int output_rate)
{
    return (total_time_steps - 1) / output_rate + 1;
}


/**
 * Fill in metadata header.
 *
 * \todo Write a valid value for the following fields: endianness, platform_id.
 * \todo Add a debug procedure to print the metadata header.
 */
static int
po_init_output_header (out_hdr_t* out_hdr, const output_parameters_t* params)
{
    int ret;
    time_t t;

    if (NULL == out_hdr || NULL == params) {
	/* programming error! */
	solver_abort ("po_init_output_header()", NULL, "NULL parameter\n");
	return -1;
    }

    memset (out_hdr, 0, sizeof (out_hdr_t));

    ret = snprintf (out_hdr->file_type_str, sizeof (out_hdr->file_type_str),
		    "Hercules 4D output v%03u", HERCULES_4D_FORMAT_VERSION);

    if (ret <= 0 || ret >= sizeof (out_hdr->file_type_str)) {
	solver_abort ("po_init_output_header()", NULL,
		      "Could not generate 4D output header, string is too short\n");
    }

    out_hdr->format_version   = HERCULES_4D_FORMAT_VERSION;
    out_hdr->endiannes	      = -1;
    out_hdr->platform_id      = -1;
    out_hdr->total_nodes      = params->total_nodes;
    out_hdr->output_steps     = get_output_time_step_count (
	    params->total_time_steps, params->output_rate);
    out_hdr->scalar_count     = 3;
    out_hdr->scalar_size      = sizeof (double);
    out_hdr->scalar_type      = 2;	/* FLOAT64_T   */
    out_hdr->scalar_class     = 1;	/* FLOAT_CLASS */
    out_hdr->quantity_type    = 0;
    out_hdr->domain_x         = params->domain_x;
    out_hdr->domain_y         = params->domain_y;
    out_hdr->domain_z         = params->domain_z;
    out_hdr->mesh_ticksize    = params->mesh->ticksize;
    out_hdr->delta_t	      = params->delta_t;
    out_hdr->total_elements   = params->total_elements;
    out_hdr->output_rate      = params->output_rate;
    out_hdr->total_time_steps = params->total_time_steps;
    out_hdr->generation_date  = time (&t);

    generate_fileid (out_hdr->ufid);

    return 0;
}


/**
 * This function creates the 4D output file and writes the metadata header
 * to it.
 *
 * \pre This function should be called only by the PE with rank 0.
 * \pre This function reads the following global variables, thus they should
 * be properly initialized:
 * - myId
 * - theDomainX
 * - theDomainY
 * - theDomainZ
 * - theNTotal
 * - theETotal
 * - theDNTotal
 * - myMesh->ticksize
 * - theTotalSteps
 *
 * \return a FILE* handle for the newly created file.  On error the solver
 *	execution is aborted by calling \c solver_abort().
 */
static FILE*
po_create_file (const char* filename, out_hdr_t* out_hdr)
{
    /* Proc 0 creates the output file */
    FILE*     fhnd;
    int	      ret;
    int	      my_rank;

    /* this should only be executed by process 0 */
    MPI_Comm_rank (comm_solver, &my_rank);
    assert (my_rank == 0);

    fhnd = po_open_file (filename, "w+");	/* this aborts on failure */

    ret = fwrite (out_hdr, sizeof(out_hdr_t), 1, fhnd);

    if (1 != ret) {
	solver_abort ("po_create_file()", "fwrite() failed",
		      "Failed to write 4D metadata header");
	fclose (fhnd);  /* this won't get executed, but anyway! */
	fhnd = NULL;
    }

    return fhnd;
}


static FILE*
po_open_4d_file (const char* filename, out_hdr_t* out_hdr, int my_rank)
{
    char  tmp;
    FILE* fhnd = NULL;

    if (0 == my_rank) {		/* PE 0 creates the file */
	fhnd = po_create_file (filename, out_hdr);
    }

    tmp = 'H';	/* a random value */

    /* all PEs should wait for PE 0 to create the file before openning it */
    MPI_Bcast (&tmp, 1, MPI_CHAR, 0, comm_solver);

    if (my_rank != 0) {            /* all other processors open the file */
	fhnd = po_open_file (filename, "r+");
    }

    return fhnd;
}


/**
 * Find each PE's relative base offset for the output.
 *
 * \pre theGroupSize must be initialized.
 * \pre theNTotal must be initialized.
 *
 * \todo update po structure
 */
static int
po_gather_base_offset (par_output_t* po, const output_parameters_t* params)
{
    int32_t  procid;
    int32_t* count_table;
    int64_t  total_count;
    mesh_t*  mesh;

    if (NULL == po) {
	solver_abort ("po_gather_base_offset", NULL, "po parameter is NULL");
	return -1;
    }

    if (NULL == params) {
	solver_abort ("po_gather_base_offset", NULL,"params parameter is NULL");
	return -1;
    }

    mesh = params->mesh;

    count_table = (int32_t*)malloc (sizeof(int32_t) * params->pe_count);

    if (NULL == count_table) {		/* Out of memory */
	solver_abort ("po_gather_base_offset", "count_table allocation",
		      NULL);
	return -1;
    }

    MPI_Allgather (&mesh->lnnum, 1, MPI_INT, count_table, 1, MPI_INT,
		   comm_solver);

    /* find my relative offset */
    po->local_base_node_id = 0;

    for (procid = 0; procid < params->pe_id; procid++) {
	po->local_base_node_id += count_table[procid];
    }

    po->relative_offset = po->local_base_node_id * sizeof (fvector_t);
    po->current_offset  = po->base_offset + po->relative_offset;

    /* this is a consistency check, always enabled, done only once */
    total_count = po->local_base_node_id;

    for (; procid < params->pe_count; procid++) {
	total_count += count_table[procid];
    }

    DEBUG_MSG2 ("po_gather_base_offset (...): rel_off=%" OFFT_FMT ", cur_off=%" OFFT_FMT,
		po->relative_offset, po->current_offset);
    DEBUG_MSG2 (" total_count=%" OFFT_FMT ", base_off=%" OFFT_FMT "\n",
		total_count, po->base_offset);

    po->total_node_count = total_count;

    free (count_table);

    if (total_count != params->total_nodes) {
	solver_abort ("po_gather_base_offset", NULL, "count_table is corrupted"
		      "\ntotal_count = %d, params->total_nodes = %d \n",
		      total_count, params->total_nodes);
	return -1;
    }

    return 0;
}


/**
 * check whether the range of locally owned nodes are in a contiguous
 * global node id range.
 */
static int
po_check_local_node_id_range (
	struct par_output_t* po,
	const output_parameters_t* params
	)
{
    global_node_id_t current_gid;
    int i;
    int is_contiguous = 1;
    int own_count;
    const mesh_t* mesh = params->mesh;

    current_gid = -1;

    po->first_owned_idx = -1;

    /* look for the first node owned by this PE */
    for (i = 0; i < mesh->nharbored && ! mesh->nodeTable[i].ismine; i++);

    if (i < mesh->nharbored && mesh->nodeTable[i].ismine) {
	current_gid = mesh->nodeTable[i].gnid;
	po->first_owned_idx = i;
    }

    own_count = 0;

    /* now check whether the range of owned nodes is contiguous */
    while (i < mesh->nharbored) {

	if (mesh->nodeTable[i].ismine) {
	    if (mesh->nodeTable[i].gnid != current_gid) {
		is_contiguous = 0;
		/* report error */
		fprintf (stderr, "Rank %d: Warning, uncontinuous global "
			 "node id range\n"
			 "expected global node id: %" INT64_FMT
			 "actual global node id: %" INT64_FMT
			 "node index: %d", params->pe_id, current_gid,
			 mesh->nodeTable[i].gnid, i);
		/* break out of loop and routine */
		return -1;
	    }

	    current_gid++;	/* update the expected global node id */
	    own_count++;
	}

	i++;
    }

    if (is_contiguous) {

	if (own_count != mesh->lnnum) {
	    solver_abort ("po_check_continuous_node_id_range()", NULL,
			  "Number of local nodes does not match "
			  "own_count = %d myMesh->lnnum = %d\n",
			  own_count, mesh->lnnum);
	    return -1;
	}

	if (mesh->nharbored != i) {
	    solver_abort ("po_check_continuous_node_id_range()", NULL,
			  "Number of harbored nodes does not match "
			  "i = %d myMesh->nharbored = %d\n",
			  i, mesh->nharbored);
	    return -1;
	}

	po->local_node_count = own_count;
    }

    return (is_contiguous == 1) ? 0 : -1;
}


static char*
generate_local_string (const char* tpl, int pe_id)
{
    static const int extra_length_default = 5;

    int    digits;
    char*  my_template;
    char*  string;
    int    extra_len;
    size_t string_len;

    assert (pe_id >= 0);

    extra_len = extra_length_default;
    digits    = log10 (pe_id);

    /* adjust for pe_ids larger or equal to 100000 */
    if (digits > extra_len) {
	extra_len = digits;
    }

    /* 6 extra chars for "-%05d" + \0 if needed */
    my_template = (char*)alloca (strlen (tpl) + 6);
    strcpy (my_template, tpl);

    /* does the template already have a "%d" */
    if (strstr (tpl, "%d") == NULL) { /* it does not */
	strcat (my_template, "-%05d");
    }

    string_len = strlen (my_template) - 2 + extra_len + 1;

    string = (char*)malloc (string_len);

    if (NULL != string) {     /* check memory allocation */
	snprintf (string, string_len, my_template, pe_id);
    }

    return string;
}


static int
po_init_debug (output_parameters_t* params)
{
    int ret = 0;
    char* local_file_name;

    assert (NULL != params);

    po_debug_fp_     = NULL;
    po_debug_output_ = 0;

    if (params->output_debug) {
	ret = -1;
	/* generate per PE filenames */
	local_file_name = generate_local_string (params->debug_filename, params->pe_id);

	if (local_file_name != NULL) {
	   po_debug_fp_ = po_open_file (local_file_name, "w");
	   xfree_char( &local_file_name );

	   if (NULL != po_debug_fp_) {
	       ret = 0;
	       po_debug_output_ = 1;
	   }
	}
    }

    return ret;
}


static int
po_print (par_output_t* po, FILE* f)
{
    assert (NULL != po);
    assert (NULL != f);

    fputs ("\n------------------------------\n", f);
    fprintf (f, "par_output_t p=%p\n", po);
    fprintf (f, "  pe_id = %d, pe_count = %d\n", po->pe_id, po->pe_count);
    fprintf (f, "  initialized=%d\n", po->initialized);
    fprintf (f, "  first_owned_idx=%d\n", po->first_owned_idx);
    fprintf (f, "  local_node_count=%" INT64_FMT "\n", po->local_node_count);
    fprintf (f, "  total_node_count=%" INT64_FMT "\n", po->total_node_count);
    fprintf (f, "  local_base_node_id=%" INT64_FMT "\n", po->total_node_count);
    fprintf (f, "  relative_offset=%" OFFT_FMT "\n", po->relative_offset);
    fprintf (f, "  stride=%" OFFT_FMT "\n", po->stride);
    fprintf (f, "  base_offset=%" OFFT_FMT "\n", po->base_offset);
    fprintf (f, "  local_time_step_chunk_size=%" OFFT_FMT "\n",
	     po->local_time_step_chunk_size);
    fprintf (f, "  expected_output_file_size=%" OFFT_FMT "\n",
	     po->expected_output_file_size);
    fprintf (f, "  output_displacement=%d\n", po->output_displacement);
    fprintf (f, "  output_velocity=%d\n", po->output_velocity);
    fprintf (f, "  output_steps=%d\n", po->output_steps);
    fprintf (f, "  cur_output_time_step=%d\n", po->cur_output_time_step);
    fprintf (f, "  current_offset=%" OFFT_FMT "\n", po->current_offset);
    fprintf (f, "  bytes_written=%" OFFT_FMT "\n", po->bytes_written);
    fprintf (f, "  delta_t=%f\n", po->delta_t);
    fprintf (f, "  disp_fp=%p\n", po->disp_fp);
    fprintf (f, "  vel_fp=%p\n", po->vel_fp);
    fprintf (f, "  mesh=%p\n", po->mesh);
    fprintf (f, "  solver=%p\n", po->solver);
    fprintf (f, "  output_elapsed_time=%f\n", po->output_elapsed_time);
    fputs ("------------------------------\n", f);

    return 0;
}


static int inline
po_print_debug (par_output_t* po)
{
    if (DO_DEBUG && po_debug_output_) {
	return po_print (po, po_debug_fp_);
    }

    return 0;
}


/**
 * Initialize a parallel output structure.
 *
 * High-level initialization steps:
 * - gather i/o parameters
 * - check whether we are actually performing I/O.
 *   - what properties are to be saved.
 *   - get output file names.
 * - check range continuity.
 * - obtain offsets.
 *
 */
static int
po_init (par_output_t* po, output_parameters_t* params)
{
    if (NULL == po) {
	/* programming error! */
	solver_abort ("po_init()", NULL, "po is NULL\n");
	return -1;
    }

    if (po->initialized != 0) {
	/* programming error! */
	solver_abort ("po_init()", NULL, "po has already been initialized\n");
	return -1;
    }

    if (NULL == params) {
	solver_abort ("po_init()", NULL, "params is NULL\n");
	return -1;
    }

    memset (po, 0, sizeof(*po));

    DEBUG_MSG1 ("params->total_nodes=%d\n", params->total_nodes);

    po->output_elapsed_time  = -MPI_Wtime();
    po->pe_id                = params->pe_id;
    po->pe_count	     = params->pe_count;
    po->relative_offset      = 0;
    po->base_offset          = sizeof (out_hdr_t);
    po->disp_fp		     = NULL;
    po->vel_fp		     = NULL;
    po->output_velocity      = 0;
    po->output_displacement  = 0;
    po->first_owned_idx      = -1;
    po->cur_output_time_step = -1;
    po->stride		     = sizeof (fvector_t) * params->total_nodes;
    po->mesh		     = params->mesh;
    po->solver		     = params->solver;
    po->delta_t		     = params->delta_t;
    po->bytes_written	     = 0;


    output_stats_init (&po->disp_stats);
    output_stats_init (&po->vel_stats);

    if (po_check_local_node_id_range (po, params) != 0) {
	solver_abort ("po_init()", NULL, "global id range for locally owned "
		      "nodes is not contiguous");
	return -1;
    }

    /* gather i/o parameters */
    if (po_gather_base_offset (po, params) != 0) {
	return -1;
    }

    po->output_steps = get_output_time_step_count (params->total_time_steps,
						   params->output_rate);

    DEBUG_MSG2 ("po->output_steps=%d\nparams->total_time_steps=%d\n",
		po->output_steps, params->total_time_steps);
    DEBUG_MSG1 ("params->output_size=%" OFFT_FMT "\n", params->output_size);


    po->expected_output_file_size  = params->output_size;
    po->local_time_step_chunk_size = po->local_node_count * sizeof (fvector_t);

    po->disp_stats.iop_per_out_step = 1;
    po->vel_stats.iop_per_out_step = po->local_node_count;
    po->total_node_count = params->total_nodes;

    /* open files depending on whether we are saving displacement and/or
     * velocity data
     */
    if (0 != params->output_displacement
	&& NULL != params->displacement_filename)
    {
	out_hdr_t disp_hdr;

	po_init_output_header (&disp_hdr, params);
	disp_hdr.quantity_type = 1;
	po->disp_fp = po_open_4d_file (params->displacement_filename,
				       &disp_hdr, params->pe_id);

	if (NULL == po->disp_fp) {
	    return -1;
	}

	po->output_displacement = 1;
    }

    if (0 != params->output_velocity
	&& NULL != params->velocity_filename)
    {
	out_hdr_t vel_hdr;

	po_init_output_header (&vel_hdr, params);
	vel_hdr.quantity_type = 2;
	po->vel_fp = po_open_4d_file (params->velocity_filename,
				      &vel_hdr, params->pe_id);

	if (NULL == po->vel_fp) {
	    return -1;
	}

	po->output_velocity = 1;
    }

    po->initialized = 1;

    po_print_debug (po);

    return 0;
}


static int
po_check_file_size (FILE* file, off_t expected_size)
{
    int ret = -1;
    int fd;
    off_t actual_size;

    assert (NULL != file);

    fd = fileno (file);

    if (-1 == fd) {
	perror ("po_check_file_size: fileno(file)");
	return -1;
    }

    ret = get_file_size (fd, &actual_size);

    if (0 != ret) {
	fprintf (stderr, "Could not get file size for file (d): FILE* (fd):"
		 "0x%p (%d)\n",	file, fd);

	return -3;
    }

    if (actual_size != expected_size) {
	fprintf (stderr, "\n\nWarning!: File does not have expected size\n"
		 "  FILE* (fd): 0x%p (%d)\n"
		 "  Expected size: %" OFFT_FMT "\n"
		 "  Actual size: %" OFFT_FMT "\n"
		 "  Difference: %" OFFT_FMT "\n\n",
		 file, fd, expected_size, actual_size,
		 (expected_size - actual_size));
	return -2;
    }

    return 0;
}


static int
po_check_file_sizes (par_output_t* po)
{
    int ret = 0;
    int ret2;

    assert (NULL != po);

    if (po->output_displacement && NULL != po->disp_fp) {
	ret2 = po_check_file_size (po->disp_fp, po->expected_output_file_size);
	if (ret2 != 0) {
	    fprintf (stderr, "Displacement file size check failed!\n");
	    ret = -1;
	}
    }

    if (po->output_velocity && NULL != po->vel_fp) {
	ret2 = po_check_file_size (po->vel_fp, po->expected_output_file_size);
	if (ret2 != 0) {
	    fprintf (stderr, "Velocity file size check failed!\n");
	    ret += -2;
	}
    }

    return ret;
}


static int
po_fini (par_output_t* po)
{
    int ret = 0;
    int recvval, sendval = 1;
    assert (NULL != po);

    if (!po->initialized) {
	return -1;
    }

    if (0 != po->pe_id) {
	/* delay closing files on PE 0 so we can check their file size */

        if (NULL != po->disp_fp) {
	   fflush (po->disp_fp);
	   fclose (po->disp_fp);
	   po->disp_fp = NULL;
        }

        if (NULL != po->vel_fp) {
	   fflush (po->vel_fp);
	   fclose (po->vel_fp);
	   po->vel_fp = NULL;
        }
    }

    /* PE0: wait until all processes have closed the files and then
     * check the file size.
     */
    MPI_Reduce (&sendval, &recvval, 1, MPI_INT, MPI_SUM, 0, comm_solver);

    po->output_elapsed_time  += MPI_Wtime();

    if (0 == po->pe_id) {

	if (NULL != po->disp_fp) {
	   fflush (po->disp_fp);
	}

	if (NULL != po->vel_fp) {
	    fflush (po->vel_fp);
	}

	ret = po_check_file_sizes (po);

	/* close files on PE 0 now */
	if (NULL != po->disp_fp) {
	    fclose (po->disp_fp);
	    po->disp_fp = NULL;
	}

	if (NULL != po->vel_fp) {
	    fclose (po->vel_fp);
	    po->vel_fp = NULL;
	}
    }

    po->initialized = 0;

    return ret;
}


/**
 * Output initialization (Entry point function).
 *
 * \post Global variable do_parallel_output_c is initialized.
 */
int
output_init_state (output_parameters_t* params)
{
    assert (NULL != params);

    do_parallel_output_ = params->parallel_output;

    po_init_debug (params);

    if (0 != params->parallel_output) {
	return po_init (&po_, params);
    }

    return 0;
}


static int
po_fini_debug()
{
    int ret = 0;

    /* close all local debug files */
    if (NULL != po_debug_fp_) {
	fflush (po_debug_fp_);
	ret = fclose (po_debug_fp_);
	po_debug_fp_ = NULL;
    }

    return ret;
}


/**
 * Output finalization and cleanup (Entry point function).
 */
int
output_fini()
{
    if (do_parallel_output_) {
	return po_fini (&po_);
    }

    po_fini_debug();

    return 0;
}


static off_t inline
compute_current_offset (par_output_t* po)
{
    return po->base_offset + po->stride * po->cur_output_time_step
	    + po->relative_offset;
}


static int
write_displacement (par_output_t* po)
{
    size_t     ret;
    off_t      offset;
    fvector_t* disp_ptr;
    double     disp_out_time;

    /* Position for my output for the current outputstep */
    offset = compute_current_offset (po);


    DEBUG_MSG1 ("write_displacement(): ts=%d", po->cur_output_time_step + 1);
    DEBUG_MSG3 (" range=%" OFFT_FMT ", +%" OFFT_FMT ", %" OFFT_FMT,
		offset, po->local_time_step_chunk_size,
		offset + po->local_time_step_chunk_size);

    DEBUG_MSG2 (" stride=%" OFFT_FMT " relative_offset=%" OFFT_FMT,
	        po->stride, po->relative_offset);

    DEBUG_MSG2 (" base_offset=%" OFFT_FMT " l_base_nid=%d",
		po->base_offset, po->local_base_node_id);
    DEBUG_MSG1 (" wrsz=%zu", sizeof(fvector_t) * po->local_node_count);

    disp_out_time = - MPI_Wtime();	    /* for timing */
    if (fseeko (po->disp_fp, offset, SEEK_SET) != 0) {
	solver_abort ("write_displacement", "fseeko failed",
		      "offset=%" OFFT_FMT ", file handle = 0x%p", offset,
		      po->disp_fp);
	return -1;
    }

    /* address of first owned node data */
    disp_ptr = po->solver->tm1 + po->first_owned_idx;

    ret	= fwrite (disp_ptr, sizeof(fvector_t), po->local_node_count,
		  po->disp_fp);

    if (ret != po->local_node_count) {
	solver_abort ("write_displacement", "fwrite failed",
		      "disp_fp=0x%p, po=0x%p, offset=%" OFFT_FMT ", ret=%ld",
		      po->disp_fp, po, offset, ret);
	return -1;
    }

    /* There are cases where we really want to flush everything before the
     * next step. e.g., a parallel machine where there is no client-side
     * write-back cache.  However, in cases where the system does provide a
     * a cache and memory is not a limiting resource it's probably OK to
     * let the system complete the I/O in the background.
     */
    fflush (po->disp_fp);

    if (fdatasync (fileno (po->disp_fp)) != 0) {
	solver_abort ("write_displacement", "fdatasync failed", "ret=%ld",ret);
    }

    disp_out_time += MPI_Wtime();

    DEBUG_MSG2 (" l_node_count=%d time=%f\n",
		po->local_node_count, disp_out_time);
    /* update timing statistics */
    output_stats_update (&po->disp_stats, disp_out_time);

    return 0;
}


/**
 * Write out this PE's node velocities
 */
static int
write_velocity (par_output_t* po)
{
    int count;
    fvector_t vel;
    off_t     offset;
    double    vel_out_time = - MPI_Wtime();	    /* Timing */

    const fvector_t* tm1;
    const fvector_t* tm2;

    assert (NULL != po);

    offset = compute_current_offset (po);

    if (fseeko (po->vel_fp, offset, SEEK_SET) != 0) {
	solver_abort ("write_velocity", "fseeko failed",
		      "offset=%" OFFT_FMT ", file handle = 0x%p", offset,
		      po->vel_fp);
	return -1;
    }

    tm1 = po->solver->tm1 + po->first_owned_idx;
    tm2 = po->solver->tm2 + po->first_owned_idx;

    for (count=0; count < po->local_node_count; count++) {
	vel.f[0] = (tm1->f[0] - tm2->f[0]) / po->delta_t;
	vel.f[1] = (tm1->f[1] - tm2->f[1]) / po->delta_t;
	vel.f[2] = (tm1->f[2] - tm2->f[2]) / po->delta_t;

	if (fwrite (&vel, sizeof(fvector_t), 1, po->vel_fp) != 1) {
	    solver_abort ("write_velocity()", "fwrite failed", NULL);
	    return -1;
	}

	tm1++;
	tm2++;
    }

    fflush (po->vel_fp);

    if (fdatasync (fileno (po->vel_fp)) != 0) {
	solver_abort ("write_velocity", "fdatasync failed", NULL);
    }


    vel_out_time += MPI_Wtime();
    output_stats_update (&po->vel_stats, vel_out_time);

    return 0;
}


/**
 * Output the velocity of the mesh nodes in parallel.
 */
static int
po_do_output (par_output_t* po)
{
    int ret_d = 0, ret_v = 0;
    off_t offset;

    po_check_init (po);

    if (! po->output_displacement && ! po->output_velocity) {
	return 0;	/* nothing to do */
    }

    po->cur_output_time_step++;

    offset = compute_current_offset (po);

    /* is the current offset up to date? */
    if (offset != po->current_offset) {
        fprintf (stderr, "\nWarning!: PE %d, unexpected offset %"OFFT_FMT
		 " != %"OFFT_FMT"\n", po->pe_id, po->current_offset, offset);
        DEBUG_MSG2 ("Unexpected offset=%"OFFT_FMT", po->current_offset=%"OFFT_FMT"\n",
		    offset, po->current_offset);

	po->current_offset = offset;
    }

    DEBUG_MSG1("po_do_output(): offset=%"OFFT_FMT"\n", offset);

    if (po->output_displacement) {
	ret_d = write_displacement (po);
    }

    if (po->output_velocity) {
	ret_v = write_velocity (po);
    }


    po->current_offset += po->stride;

    return ret_v + ret_d;
}


/**
 * Perform solver's output (Entry point function).
 */
int
do_solver_output( void )
{
    if (do_parallel_output_) {
	return po_do_output (&po_);
    }

    else {
	solver_output_seq();
    }

    return 0;
}


/**
 * Compute stats in the counters structure and set the appropriate values
 * for the throughput (stats structure).
 *
 * \param[out]    stats Computed stats for latency and throughput.
 * \param[in,out] counters Collected running counters, the collected variable
 *	(X) is elapsed time for each output step (i.e., latency).
 */
static void
po_compute_stats (
	output_stats_t*     stats,
	po_stat_counters_t* counters,
	off_t		    chunk_size
	)
{
    unsigned int n;

    assert (NULL != stats);
    assert (NULL != counters);
    assert (0 < chunk_size);

    /* minimum throughtput (what took the longest time) */
    stats->tput_min = chunk_size / counters->max;

    /* maximum throughtput (what took the shortest time) */
    stats->tput_max = chunk_size / counters->min;

    n		    = counters->count;
    stats->count    = n;
    stats->lat_avg  = counters->sum / n;
    stats->lat_var  = sample_variance (counters->sum, counters->sum2, n);
    stats->tput_avg = counters->sum_inv * chunk_size / n;
    stats->tput_var = sample_variance (counters->sum_inv,counters->sum_inv2,n)
	* chunk_size * chunk_size;
    stats->iops     = counters->sum_inv * counters->iop_per_out_step / n;
    stats->iop_count = counters->iop_per_out_step * counters->count;
}


#define STAT_DATA_LEN	9
static int
po_print_detailed_stats (
	unsigned int    pe_count,
	const int32_t*  count_table,
	const float*	data_table,
	FILE*	        fp
	)
{
    int pid;	/* PE id */
    int fid;	/* field id */

    if (NULL == fp) { /* we can't do anything */
	return 0;
    }

    fputs ("#\n# peid ncount avg(lat) std(lat) avg(tput) std(tput) "
	   "min(tput) max(tput) iops iop_count wtime am_iops cv(tput)\n\n", fp);

    for (pid = 0; pid < pe_count; pid++) {
	fprintf (fp, "%d %d", pid, count_table[pid]);

	for (fid = 0; fid < STAT_DATA_LEN; fid++) {
	    fprintf (fp, " %f", data_table [pid*STAT_DATA_LEN + fid]);
	}

	/* field 7: iop_count, field 8: elapsed wall time */
	fprintf (fp, " %f", data_table [pid*STAT_DATA_LEN + 7]
		 / data_table [pid*STAT_DATA_LEN + 8]);

	/* print coefficient of variation: stdev / avg. */
	fprintf (fp, " %f", data_table[pid * STAT_DATA_LEN + 3]
		 / data_table[pid * STAT_DATA_LEN + 2]);

	fputc ('\n', fp);
    }

    fputs ("\n\n", fp);

    return 0;
}


static int
po_collect_detailed_stats (
	par_output_t*	      po,
	const output_stats_t* stats,
	FILE*		      fp,
	double		      e2e_elapsed_time
	)
{
    int32_t* count_table = NULL;
    float*   data_table  = NULL;
    int32_t  count;
    float    data[STAT_DATA_LEN];

    assert (NULL != po);
    assert (NULL != stats);

    if (0 == po->pe_id) {
	count_table = (int32_t*)malloc (sizeof(int32_t) * po->pe_count);
	data_table = (float*)malloc (sizeof(float)*po->pe_count*STAT_DATA_LEN);
	assert (NULL != count_table);
	assert (NULL != data_table);
    }

    count = po->local_node_count;
    MPI_Gather (&count, 1, MPI_INT, count_table, 1, MPI_INT, 0, comm_solver);

    data[0] = stats->lat_avg;
    data[1] = sqrt (stats->lat_var);
    data[2] = stats->tput_avg;
    data[3] = sqrt (stats->tput_var);
    data[4] = stats->tput_min;
    data[5] = stats->tput_max;
    data[6] = stats->iops;
    data[7] = stats->iop_count;
    data[8] = MPI_Wtime() + e2e_elapsed_time; /* local elapsed time */

    MPI_Gather (data, STAT_DATA_LEN, MPI_FLOAT, data_table, STAT_DATA_LEN,
		MPI_FLOAT, 0, comm_solver);

    if (0 == po->pe_id) {
	if (NULL != fp) {
	    po_print_detailed_stats (po->pe_count, count_table, data_table, fp);
	}

	free (count_table);
	free (data_table);
    }

    return 0;
}
#undef STAT_DATA_LEN


static int
po_collect_stats (
	par_output_t*	    po,
	po_stat_counters_t* counters,
	output_stats_t*     st,
	FILE*		    fp,
	const char*	    out_type,
	double	            e2e_elapsed_time
	)
{
    float min, max, sum, osum2, sum2, mina, maxa, avgv, minv, maxv, avg_iops;
    /* mina: minimum average
     * maxa: maximum average
     * vara: average's variance
     * avgv: variance's average
     * minv: minimum variance
     * maxv: maximum variance
     */

    assert (NULL != st);

    po_compute_stats (st, counters, po->local_time_step_chunk_size);

    /* reduce min, max, avg tput, stdev (avg tput) */
    osum2 = st->tput_avg * st->tput_avg;

    MPI_Reduce (&st->tput_min, &min,  1, MPI_FLOAT, MPI_MIN, 0, comm_solver);
    MPI_Reduce (&st->tput_max, &max,  1, MPI_FLOAT, MPI_MAX, 0, comm_solver);
    MPI_Reduce (&st->tput_avg, &mina, 1, MPI_FLOAT, MPI_MIN, 0, comm_solver);
    MPI_Reduce (&st->tput_avg, &maxa, 1, MPI_FLOAT, MPI_MAX, 0, comm_solver);
    MPI_Reduce (&st->tput_avg, &sum,  1, MPI_FLOAT, MPI_SUM, 0, comm_solver);
    MPI_Reduce (&osum2,	       &sum2, 1, MPI_FLOAT, MPI_SUM, 0, comm_solver);
    MPI_Reduce (&st->tput_var, &minv, 1, MPI_FLOAT, MPI_MIN, 0, comm_solver);
    MPI_Reduce (&st->tput_var, &maxv, 1, MPI_FLOAT, MPI_MAX, 0, comm_solver);
    MPI_Reduce (&st->tput_var, &avgv, 1, MPI_FLOAT, MPI_SUM, 0, comm_solver);
    MPI_Reduce (&st->iops, &avg_iops, 1, MPI_FLOAT, MPI_SUM, 0, comm_solver);

    avgv     /= po->pe_count;
    avg_iops /= po->pe_count;


    if (0 == po->pe_id && NULL != fp) {
	float avg = sum / po->pe_count;
	uint64_t iop_count;

	/* hack: deduce the iop_count from counters->iop_per_out_step as
	 * follows:
	 * when dealing with displacement output iop_per_out_step is 1,
	 * when dealing with velocity output iop_per_out_step is the
	 * local node count.
	 */
	if (1 == counters->iop_per_out_step) { /* displacement output */
	    iop_count = po->output_steps * po->pe_count;
	} else { /* velocity output */
	    iop_count = po->output_steps * po->total_node_count;
	}

	fputs ("# Output type: ", fp);
	fputs (out_type, fp);

	fprintf (fp, "\n#\n# Global iop count: %"UINT64_FMT"\n", iop_count);
	fprintf (fp, "# Global avg (iops) amortized: %.2f\n",
		 iop_count / po->output_elapsed_time);

	fputs ("#\n# Average stats, i.e., per PE:\n#\n", fp);
	fprintf (fp, "# min throughput: %.2f\n", min);
	fprintf (fp, "# max throughput: %.2f\n", max);
	fprintf (fp, "# avg (avg throughput): %.2f\n", avg);
	fprintf (fp, "# std (avg throughput): %.2f\n",
		 sqrt (population_variance (avg, sum2, po->pe_count)));
	fprintf (fp, "# min (avg throughput): %.2f\n", mina);
	fprintf (fp, "# max (avg throughput): %.2f\n#\n", maxa);
	fprintf (fp, "# avg (std throughput): %.2f\n", sqrt (avgv));
	fprintf (fp, "# min (std throughput): %.2f\n", sqrt (minv));
	fprintf (fp, "# max (std throughput): %.2f\n", sqrt (maxv));
	fprintf (fp, "#\n# avg (iops) during I/O: %.2f\n", avg_iops);
	fprintf (fp, "# avg (iops) amortized: %.2f\n",
		 ((float)iop_count) / (po->output_elapsed_time * po->pe_count));
    }

    /* gather details */
    po_collect_detailed_stats (po, st, fp, e2e_elapsed_time);

    return 0;
}


static int
po_collect_io_stats (
	par_output_t*   po,
	const char*     filename,
	output_stats_t* disp_stats,
	output_stats_t* vel_stats,
	double	        e2e_elapsed_time
	)
{
    int   ret = 0;
    FILE* fp  = NULL;	/* safeguard */

    if (do_parallel_output_) {

	if (0 == po->pe_id && NULL != filename) {
	    fp = po_open_file (filename, "w");

	    if (NULL != fp) {
		fputs ("# Output summary stats\n#\n", fp);
		fprintf (fp, "# Number of PEs: %d\n", po->pe_count);
		fprintf (fp, "# Number of output time steps: %d\n",
			 po->output_steps);
		fprintf (fp, "# TS Stride: %"INT64_FMT"\n", po->stride);
		fprintf (fp, "# Expected output file size: %" INT64_FMT "\n",
			 po->expected_output_file_size);
		fprintf (fp, "# Elapsed 4D ouput time: %.2f\n#\n#\n",
			 po->output_elapsed_time);
	    }
	}

	if (po_.output_displacement) {
		ret = po_collect_stats (po, &po->disp_stats, disp_stats, fp,
					"displacement", e2e_elapsed_time);
	}

	if (po_.output_velocity) {
	    ret = po_collect_stats (po, &po->vel_stats, vel_stats, fp,
				    "velocity", e2e_elapsed_time);
	}

	if (0 == po->pe_id && NULL != fp) {
	    int flag = 0, wtime_is_global = 0, ret;

	    e2e_elapsed_time += MPI_Wtime();
	    ret = MPI_Attr_get (comm_solver, MPI_WTIME_IS_GLOBAL,
				&wtime_is_global, &flag);

	    if (!flag) {
		wtime_is_global = 0;
	    }
	    fprintf (fp, "\n#\n# Approximate end-to-end time : %.2f seconds\n"
		     "# MPI_WTIME_IS_GLOBAL: 0x%x\n#\n", e2e_elapsed_time,
		     wtime_is_global);
	    fprintf (fp, "# flag: %d, ret = %d\n", flag, ret);
	    fclose (fp);
	}
    }

    return ret;
}


int
output_collect_io_stats (
	const char*     filename,
	output_stats_t* disp_stats,
	output_stats_t* vel_stats,
	double	        e2e_elapsed_time
	)
{
    /* compute local throughput stats and then send them to PE0 */
    if (do_parallel_output_) {
	po_collect_io_stats (&po_, filename, disp_stats, vel_stats,
			     e2e_elapsed_time);
    }

    return 0;
}
