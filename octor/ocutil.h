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

#ifndef OCUTIL_H
#define OCUTIL_H

#include <sys/types.h>

/**
 * Macros with the format string for int64_t and uint64_t types.
 */
#if (defined __WORDSIZE) && (__WORDSIZE == 64)
#  define UINT64_FMT        "lu"
#  define INT64_FMT     "ld"
#  define MPI_INT64     MPI_LONG
#else /*  __WORDSIZE && __WORDSIZE == 64 */
#  define UINT64_FMT        "llu"
#  define INT64_FMT     "lld"
#  define MPI_INT64     MPI_LONG_LONG_INT
#endif

#endif /* OCUTIL_H */
