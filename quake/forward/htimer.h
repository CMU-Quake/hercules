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
 * Copyright (C) 2007 Julio Lopez. All Rights Reserved.
 * For copyright and disclaimer details see the file COPYING
 * included in the package distribution.
 *
 * Package:     Quake project's Hercules numerical solver.
 * Description: Timing utility functions.
 *
 * Author:      Julio Lopez (jclopez at ece dot cmu dot edu)
 *
 * File info:   $RCSfile: htimer.h,v $
 *              $Revision: 1.3 $
 *              $Date: 2009/08/08 20:57:08 $
 *              $Author: jclopez $
 */
#ifndef HTIMER_H
#define HTIMER_H

#include <stdint.h>
#include <sys/time.h>
#include <stdio.h>


#ifndef PERF_TIME_T_DEFINED
# define PERF_TIME_T_DEFINED
typedef uint64_t perf_time_t;
#endif /* PERF_TIME_T_DEFINED */


#ifndef MICROS_PER_SECOND
#define MICROS_PER_SECOND       1000000
#endif /* MICROS_PER_SECOND */


struct htimerv_t {
    perf_time_t    total;	/** Cummulative value	      */
    struct timeval tv;		/** For calls to gettimeofday */
};


typedef struct htimerv_t htimerv_t;

#define HTIMERV_ZERO	        { 0, { 0, 0 } }

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


static inline void     tv_reset (struct timeval* tv);
static inline int      tv_print (const struct timeval* tv, FILE* stream);
static inline double   tv_seconds (const struct timeval* tv);
static inline uint64_t tv_usec (const struct timeval* tv);

static inline double
usec_to_sec( uint64_t usec )
{
    return ((double)usec) / MICROS_PER_SECOND;
}


static inline void
tv_reset (struct timeval* tv)
{
    tv->tv_sec  = 0;
    tv->tv_usec = 0;
}


static inline int
tv_print (const struct timeval* tv, FILE* stream)
{
    return fprintf (stream, "%ld.%ld", tv->tv_sec, tv->tv_usec);
}


static inline double
tv_seconds (const struct timeval* tv)
{
    return (double)tv->tv_sec + ((double)tv->tv_usec) / MICROS_PER_SECOND;
}


static inline uint64_t
tv_usec (const struct timeval* tv)
{
    return ((uint64_t)tv->tv_sec) * MICROS_PER_SECOND + tv->tv_usec;
}


static inline void
tv_start(struct timeval* tv)
{
    struct timeval t;

    gettimeofday (&t, NULL);
    tv->tv_sec  = -t.tv_sec;
    tv->tv_usec = -t.tv_usec;
}


static inline void
tv_stop( struct timeval* tv )
{
    struct timeval t;

    gettimeofday (&t, NULL);
    tv->tv_sec  += t.tv_sec;
    tv->tv_usec += t.tv_usec;
}


/**
 * Compute the time difference in seconds between two timeval structures
 * \c a and \c b.  Assume a > b, i.e., a is more recent than b.
 */
static inline double
tv_difftime( const struct timeval* a, const struct timeval* b)
{
    return ((double)(a->tv_sec - b->tv_sec))
	+ (a->tv_usec - b->tv_usec) / 1000000.0;
}


static inline void
htimerv_reset( htimerv_t* t )
{
    t->total = 0;
    tv_reset( &t->tv );
}


static inline void
htimerv_start( htimerv_t* t )
{
    tv_start( &t->tv );
}


static inline void
htimerv_stop( htimerv_t* t )
{
    tv_stop( &t->tv );
    t->total += tv_usec( &t->tv );
}


static inline uint64_t
htimerv_get_total_usec( htimerv_t* t )
{
    return t->total;
}


static inline double
htimerv_get_total_sec( htimerv_t* t )
{
    return usec_to_sec( t->total );
}

#ifdef __cplusplus
} /* extern "C" { */
#endif /* __cplusplus */

#endif /* HTIMER_H */
