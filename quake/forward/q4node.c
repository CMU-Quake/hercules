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
 * Description: Retrieve time series for a single mesh node.
 * Package:     Hercules: Quake's ground motion simulation numerical solver.
 * Language:    C
 */
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#include "psolve.h"
#include "util.h"

static int
q4series (
	int		fd,
	unsigned long	node_id,
	int64_t		node_count,
	unsigned int	out_steps
	)
{
    static const size_t vector_size = sizeof (double) * 3;
    int ostep;
    int ret = 0;
    off_t off_skip = (node_count - 1) * vector_size;

    /* seek to the offset of the first data value for the node */
    off_t node_offset = node_id * vector_size + sizeof(out_hdr_t);
    
    if (lseek (fd, node_offset, SEEK_SET) == -1) {
	perror ("Seek to first data value failed");
	return -2;
    }

    for (ostep = 0; ostep < out_steps && ret == 0; ostep++) {
	double val[3];
	double mag;

	/* read node data */
	ssize_t rret = read (fd, val, vector_size);

	if (rret != vector_size) {
	    perror ("Could not read data value from file");
	    fprintf (stderr, "time step = %d\n", ostep);
	    ret = -3;
	    return -3;
	}

	/* print node data */
	mag = sqrt (val[0] * val[0] + val[1] * val[1] + val[2] * val[2]);

	printf ("%d %g %g %g %g\n", ostep, val[0], val[1], val[2], mag);

	/* seek to next value */
	if (lseek (fd, off_skip, SEEK_CUR) == (off_t)-1) {
	    perror ("Seek to next data value failed!");
	    fprintf (stderr, "time step = %d\n", ostep);
	    ret = -4;
	    return -4;
	}
    }

    return ret;
}


static int
q4node (const char* filename, unsigned long node_id)
{
    int fd;
    out_hdr_t out_hdr;
    ssize_t rret;
    int ret;

    /* open 4D output */
    fd = open (filename, O_RDONLY);
    if (fd < 0) {
	perror ("Cannot open file");
        fprintf (stderr, "Filename: %s\n", filename);
	return -1;
    }

    /* read metadata */
    rret = read (fd, &out_hdr, sizeof(out_hdr_t));

    if (rret != sizeof (out_hdr_t)) {
	perror ("read() failed reading file header");
	close (fd);
	return -2;
    }

    /* validate point */
    if (node_id >= out_hdr.total_nodes) {
	fprintf (stderr, "Invalid node id = %lu, the total number of nodes is %"
		 INT64_FMT "\n", node_id, out_hdr.total_nodes);
	close (fd);
	return -3;
    }

    ret = q4series (fd, node_id, out_hdr.total_nodes, out_hdr.output_steps);
    close (fd);

    return ret;
}




static void
usage (const char* program_name)
{
    fputs ("Usage: ", stderr);
    fputs (program_name, stderr);
    fputs (" filename node_id\n", stderr);

    exit (1);
}


int
main (int argc, char** argv)
{
    char*	  endptr;
    unsigned long node_id;
    const char*   filename;
    int   ret;

    if (argc != 3) {
	usage (argv[0]);
        return 2;
    }

    filename = argv[1];
    
    node_id = strtoul (argv[2], &endptr, 10);

    if (endptr == argv[2] || *endptr != '\0') {
	fprintf (stderr, "Invalid node id: %s\n", argv[2]);
	usage (argv[0]);
	return 3;
    }

    ret = q4node (filename, node_id);

    if (ret != 0) {
	return 1;
    }

    return 0;
}
