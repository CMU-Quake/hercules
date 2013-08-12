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
 * Description: Hercules common / communication support functions, e.g.,
 *		ones that have MPI dependencies
 */
#ifndef H_COMM_UTIL_H
#define H_COMM_UTIL_H

#include <mpi.h>


int
broadcast_string( char** string, int root_rank, MPI_Comm comm );

int
broadcast_char_array( char string[], size_t len, int root_rank, MPI_Comm comm );


#endif /* H_COMM_UTIL_H */
