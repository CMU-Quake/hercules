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

#ifndef QUAKE_UTIL_H_
#define QUAKE_UTIL_H_


/* -------------------------------------------------------------------------- */
/*                        Structures and definitions                          */
/* -------------------------------------------------------------------------- */

#define LINESIZE        512

typedef enum {
    NO = 0, YES
} noyesflag_t;

/* -------------------------------------------------------------------------- */
/*                                  Methods                                   */
/* -------------------------------------------------------------------------- */

int32_t  imin(int32_t x, int32_t y);
//int  vector_is_zero( const fvector_t* v );
int  vector_is_all_zero( const fvector_t* v );

void MultAddMatVec( fmatrix_t* M, fvector_t* V1, double c, fvector_t* V2 );

int  Search_Quality_Table(double Q, double *theQTABLE, int QTable_Size);

int  parsedarray( FILE *fp, const char *querystring,int size, double *array );

int  vsrule( edata_t  *edata, double theFactor );

#endif /* QUAKE_UTIL_H_ */
