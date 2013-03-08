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
 * Description: 4D output for Quake's numerical solver.
 *
 * Copyright (C) 2006 Julio Lopez. All Rights Reserved.
 *
 *
 * Package:     Hercules: Quake's ground motion simulation numerical solver.
 * Name:        $RCSfile: q4showmeta.c,v $
 * Language:    C
 */
#include "output.h"
#include "psolve.h"
#include "util.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>


static int
q4_read_output_header (int fd, out_hdr_t* out_hdr)
{
    ssize_t rdlen;

    assert (NULL != out_hdr);

    rdlen = read (fd, out_hdr, sizeof (out_hdr_t));

    if (sizeof (out_hdr_t) != rdlen) {
	perror ("Error reading metadata header");
	return -1;
    }

    return 0;
}


/** Type of scalar text names */
static char* scalar_type_str[] = {
    "INVALID",
    "FLOAT32",
    "FLOAT64",
    "FLOAT128",
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    "INT64",
    "UINT64"
};

/** Scalar class names */
static char* scalar_class_str[] = { "INVALID", "FLOAT_CLASS", "INT_CLASS" };
static char* scalar_name_str[]  = { "Unknown", "Displacement", "Velocity" };


/**
 * Print the metadata header contents to stdout in "WaveDB" format.
 */
static int
q4_print_output_header_w (out_hdr_t* out_hdr, FILE* f)
{
    int    i;
    char   date_str[30];
    time_t sim_time;

    assert (NULL != out_hdr);

    fprintf (f, "file_type = \"%.29s\"\n", out_hdr->file_type_str);
    fprintf (f, "format_version = %hhu\n", out_hdr->format_version);
    fprintf (f, "endiannes = %hhu\n",      out_hdr->endiannes);
    fprintf (f, "platform_id = %hhu\n",    out_hdr->platform_id);
    fprintf (f, "ufid = 0x");

    for (i = 0; i < sizeof (out_hdr->ufid); i++) {
	fprintf (f, "%hhx", out_hdr->ufid[i]);
    }

    sim_time = out_hdr->generation_date;
    ctime_r(&sim_time, date_str);
    date_str[strlen(date_str) - 1] = '\0';

    fprintf (f, "\ngeneration_date = \"%s\"\n", date_str);
    fprintf (f, "node_count = %" INT64_FMT "\n", out_hdr->total_nodes);
    fprintf (f, "element_count = %" INT64_FMT "\n", out_hdr->total_elements);
    fprintf (f, "simulation_time_steps = %d\n", out_hdr->total_time_steps);
    fprintf (f, "simulation_delta_t = %f\n", out_hdr->delta_t);
    fprintf (f, "output_rate = %d\n", out_hdr->output_rate);
    fprintf (f, "output_time_step_count = %d\n", out_hdr->output_steps);
    fprintf (f, "output_dt = %f\n", out_hdr->delta_t*out_hdr->output_rate);
    fprintf (f, "number_of_scalars = %d\n", out_hdr->scalar_count);
    fprintf (f, "scalar_size = %hhd\n", out_hdr->scalar_size);
    fprintf (f, "number_of_components = %d\n", out_hdr->scalar_count);
    fprintf (f, "number_of_columns = %" INT64_FMT "\n", out_hdr->total_nodes);
    fprintf (f, "number_of_rows = %d\n", out_hdr->output_steps);



    if (out_hdr->scalar_type > 11) {
	fprintf (stderr, "Warning, found an unknown scalar type in metadata"
		 "header: %hhd\n", out_hdr->scalar_type);
	out_hdr->scalar_type = 0;
    }

    fprintf (f, "scalar_type = %d\nscalar_type_name = \"%s\"\n",
	     out_hdr->scalar_type, scalar_type_str[out_hdr->scalar_type]);

    if (out_hdr->scalar_class > 2) {
	fprintf (stderr, "Warning, found an unknown scalar class in metadata"
		 "header: %hhd\n", out_hdr->scalar_class);
	out_hdr->scalar_class = 0;
    }

    fprintf (f, "scalar_class = %d\nscalar_class_name = \"%s\"\n",
	     out_hdr->scalar_class, scalar_class_str[out_hdr->scalar_class]);

    if (out_hdr->quantity_type > 2) {
	fprintf (stderr, "Warning, found an unknown scalar class in metadata"
		 "header: %hhd\n", out_hdr->quantity_type);
	out_hdr->quantity_type = 0;
    }

    fprintf (f, "value_type = %d\nvalue_type_name = \"%s\"\n",
	     out_hdr->quantity_type, 
	     scalar_name_str[out_hdr->quantity_type]);

    fprintf (f, "domain_extent_x_meters = %f\n", out_hdr->domain_x);
    fprintf (f, "domain_extent_y_meters = %f\n", out_hdr->domain_y);
    fprintf (f, "domain_extent_z_meters = %f\n", out_hdr->domain_z);
    fprintf (f, "mesh_tick_size = %f\n",         out_hdr->mesh_ticksize);
    fprintf (f, "header_size = %zu\n",	 sizeof (out_hdr_t));

    return 0;
}


/**
 * Print the metadata header contents to stdout.
 */
static int
q4_print_output_header (out_hdr_t* out_hdr, FILE* f)
{
    int    i;
    char   date_str[30];
    time_t sim_time;

    assert (NULL != out_hdr);

    fprintf (f, "File type:\t\t%.29s\n", out_hdr->file_type_str);
    fprintf (f, "Format version:\t\t%hhu\n", out_hdr->format_version);
    fprintf (f, "Endiannes:\t\t%hhu\n", out_hdr->endiannes);
    fprintf (f, "Platform ID:\t\t%hhu\n", out_hdr->platform_id);
    fprintf (f, "UFID:\t\t\t0x");

    for (i = 0; i < sizeof (out_hdr->ufid); i++) {
	fprintf (f, "%hhx", out_hdr->ufid[i]);
    }

    sim_time = out_hdr->generation_date;
    printf ("\nGeneration date:\t%s",
	    ctime_r(&sim_time, date_str));

    fprintf (f, "\nNode count:\t\t%" INT64_FMT "\n", out_hdr->total_nodes);
    fprintf (f, "Element count:\t\t%" INT64_FMT "\n", out_hdr->total_elements);
    fprintf (f, "Simulation time steps:\t%d\n", out_hdr->total_time_steps);
    fprintf (f, "Simulation delta t:\t%f\n", out_hdr->delta_t);
    fprintf (f, "Output rate:\t\t%d\n", out_hdr->output_rate);
    fprintf (f, "Output time step count:\t%d\n", out_hdr->output_steps);
    fprintf (f, "Number of scalars:\t%d\n", out_hdr->scalar_count);
    fprintf (f, "Scalar size:\t\t%hhd\n", out_hdr->scalar_size);

    if (out_hdr->scalar_type > 11) {
	fprintf (stderr, "Warning, found an unknown scalar type in metadata"
		 "header: %hhd\n", out_hdr->scalar_type);
	out_hdr->scalar_type = 0;
    }

    fprintf (f, "Scalar type:\t\t%d (%s)\n", out_hdr->scalar_type,
	     scalar_type_str[out_hdr->scalar_type]);

    if (out_hdr->scalar_class > 2) {
	fprintf (stderr, "Warning, found an unknown scalar class in metadata"
		 "header: %hhd\n", out_hdr->scalar_class);
	out_hdr->scalar_class = 0;
    }

    fprintf (f, "Scalar class:\t\t%d (%s)\n", out_hdr->scalar_class,
	     scalar_class_str[out_hdr->scalar_class]);

    if (out_hdr->quantity_type > 2) {
	fprintf (stderr, "Warning, found an unknown scalar class in metadata"
		 "header: %hhd\n", out_hdr->quantity_type);
	out_hdr->quantity_type = 0;
    }

    fprintf (f, "Quantity type:\t\t%d (%s)\n", out_hdr->quantity_type, 
	     scalar_name_str[out_hdr->quantity_type]);

    fputs ("\n--- Mesh parameters ---\n\n", stdout);

    fprintf (f, "Domain extent (x,y,z) meters:\t%f, %f, %f\n", out_hdr->domain_x,
	    out_hdr->domain_y, out_hdr->domain_z);
    fprintf (f, "Mesh tick size:\t\t%f\n", out_hdr->mesh_ticksize);


    return 0;
}


static int
q4_read_and_print_metadata (int fd, const char* filename, int format)
{
    int ret;
    out_hdr_t out_hdr;

    memset(&out_hdr, 0, sizeof(out_hdr_t));

    ret = q4_read_output_header (fd, &out_hdr);

    if (ret != 0) {
	return -1;
    }

    fprintf (stdout, "input_file = \"%s\"\n", filename);

    if (format) {
	ret = q4_print_output_header_w (&out_hdr, stdout);
    } else {
	ret = q4_print_output_header (&out_hdr, stdout);
    }

    return ret;
}


static void
print_usage (const char* progname)
{
    fputs ("Usage: ", stderr);
    fputs (progname, stderr);
    fputs (" [-w] <q4d_filename>\n", stderr);
    exit (255);
}


/**
 * Extremely basic and primitive command line argument parsing.
 */
static int
get_command_line_args (int argc, const char* argv[], const char** filename,
		       int* format)
{
    *format = 0;
    *filename = NULL;

    if (argc == 3) {
	/* is the second argument -w ? */
	if (strcmp (argv[1], "-w") != 0) {
	    print_usage (argv[0]);
	    return -1;
	}
	*format = 1;
	*filename = argv[2];
    }

    else if (argc == 2) {
	*filename = argv[1];
    }

    else {
	print_usage (argv[0]);
	return -1;
    }

    return 0;
}


/** open input file */
static int
q4_open_file (const char* filename)
{
    int fd = -1;

    assert (NULL != filename);

    fd = open (filename, O_RDONLY);

    if (-1 == fd) {
	perror (filename);
	fputs ("Failed to open input file, bailing out!\n", stderr);
    }

    return fd;
}


int
main (int argc, const char* argv[])
{
    const char* filename;
    int fd, format;

    format = 0;

    if (get_command_line_args (argc, argv, &filename, &format) != 0) {
	return 255;
    }

    if ((fd = q4_open_file (filename)) == -1) {
	return 254;
    }

    q4_read_and_print_metadata (fd, filename, format);

    close (fd);

    return 0;
}
