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
 * xplatform.h - portability support for various platforms
 *
 *
 */

#ifndef XPLATFORM_H
#define XPLATFORM_H

#ifdef ALPHA
#include "etree_inttypes.h"
#else
#include <inttypes.h>
#endif 

#include "schema.h"

/*
 * endian_t - the endianness of the system 
 *
 */
#ifndef ENDIAN_T
#define ENDIAN_T
typedef enum endian_t {unknown_endianness = -1, little, big} endian_t;
#endif


/*
 * member_t - describes the member size and alignment of a structure 
 *            corresponding to a schema on a particular platform
 *
 * name of the member is the same as that in a schema, thus omitted here
 */
typedef struct member_t {
    int32_t offset;             /* offset in the structure               */
    int32_t size;               /* size of the member = field_t.size     */
} member_t;
    

/*
 * scb_t - "structure control block"; 
 *          records the layout of a structure corresponding to a schema
 *          for a particular platform
 *
 */
typedef struct scb_t {
    endian_t endian;
    int32_t membernum;
    member_t *member;
} scb_t;


/*
 * xplatform_testendian - determine the endianness of the system 
 *
 */
endian_t xplatform_testendian();


/*
 * xplatform_swapbytes - convert between different endian format 
 *
 */
void xplatform_swapbytes(void *to, const void *from, int32_t size);


/*
 * xplatform_createstruct - create a structure instance for a schema
 *                       for a given schema
 *
 */
scb_t *xplatform_createscb(schema_t schema);


/* 
 * xplatform_destroystruct - release memory held by the structure control
 *                           block
 *
 */
void xplatform_destroyscb(scb_t *scb);






#endif /* XPLATFORM_H */

