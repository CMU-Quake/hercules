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
 * xplatform.c - portability support 
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>


#include "xplatform.h"

/*
 * platform_t - a platform records the specification of the endianness of
 *              the hardware, and the structure alignment requirement for
 *              the hardware/compiler configuration.
 *
 * the alignement array contains the alignment requirement for 1-, 2-, 4-, 8-
 * byte 
 */
typedef struct platform_t {
    endian_t endian;
    int32_t  alignment[4]; 
} platform_t;


static platform_t testplatform();

/*
 * xplatform_testendian - determine the endianess of the system
 *
 * - set an interger variable to 1 
 * - return little if the lower order byte is nonzero, big otherwise
 *
 */
endian_t xplatform_testendian()
{
    uint32_t i = 1;
    uint8_t *loworderbyte;

    loworderbyte = (uint8_t *)&i;
    if (*loworderbyte != 0) return little;
    else return big;
}

    
/*
 * xplatform_swapbytes - swap the bytes between two different byte ordering
 *
 * No boundary overflow checking, always return
 *
 */
void xplatform_swapbytes(void *to, const void *from, int32_t size)
{
    int32_t i;

    const uint8_t *frombyteptr = (const uint8_t *)from;
    uint8_t *tobyteptr = (uint8_t *)to + (size - 1);

    for (i = 0; i < size; i++) {
        *tobyteptr = *frombyteptr;

        frombyteptr++;
        tobyteptr--;
    }
    return;
}


/*
 * xplatform_createstruct - create a scb for a schema on the current 
 *                          platform (hardware/compiler)
 *
 * return pointer the a scb_t if OK, NULL on error
 *
 */
scb_t *xplatform_createscb(schema_t schema)
{
    scb_t *scb;
    platform_t ptfm;
    int32_t memberind, startoffset;

    if ((scb = (scb_t *)malloc(sizeof(scb_t))) == NULL) {
        perror("xplatform_createscb: malloc scb_t");
        return NULL;
    }

    scb->membernum = schema.fieldnum;
    if ((scb->member = (member_t *)malloc(scb->membernum * sizeof(member_t)))
        == NULL) {
        perror("xplatform_createscb: malloc member array");
        free(scb);
        return NULL;
    }

    ptfm = testplatform();
    scb->endian = ptfm.endian;

    startoffset = 0;
    for (memberind = 0; memberind < scb->membernum; memberind++) {
        int32_t alignment; /* current member's alignment requirement */
        int32_t size;

        size = schema.field[memberind].size;
        switch (size) {
        case (1): alignment = ptfm.alignment[0]; break;
        case (2): alignment = ptfm.alignment[1]; break;
        case (4): alignment = ptfm.alignment[2]; break;
        case (8): alignment = ptfm.alignment[3]; break;
        default:
            fprintf(stderr, 
                    "xplatform_createscb: unknown data type size %d\n", size);
            xplatform_destroyscb(scb);
            return NULL;
        }
            
        if (startoffset % alignment != 0) 
            /* force alignement */
            startoffset = ((startoffset / alignment) + 1) * alignment;

        scb->member[memberind].offset = startoffset;
        scb->member[memberind].size = size;
        startoffset += size;
    }

    return scb;
}


/*
 * xplatform_destroyscb - release memory held by the structure control
 *                        block
 *
 */
void xplatform_destroyscb(scb_t *scb)
{
    free(scb->member);
    free(scb);
    return;
}


/* 
 * testplatform - determine the current platform 
 *
 */
platform_t testplatform()
{
    platform_t ptfm;
    struct {int8_t a; int8_t b;} align1;
    struct {int8_t a; int16_t b;} align2;
    struct {int8_t a; int32_t b;} align4;
    struct {int8_t a; int64_t b;} align8;

    ptfm.endian = xplatform_testendian();
    ptfm.alignment[0] = (int32_t)((char *)&align1.b - (char *)&align1);
    ptfm.alignment[1] = (int32_t)((char *)&align2.b - (char *)&align2);
    ptfm.alignment[2] = (int32_t)((char *)&align4.b - (char *)&align4);
    ptfm.alignment[3] = (int32_t)((char *)&align8.b - (char *)&align8);

    return ptfm;
}



