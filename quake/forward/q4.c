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

/**
 * q4.c - Query the time series of an observation point.
 *
 *
 */

#include "q4.h"


int q4_point(double x, double y, double z, etree_t *mep, FILE *result_fp,
             out_hdr_t result_hdr, FILE *outfp, int out_format)
{
    double edgesize, ldb[3], center[3], distance[3], phi[8];
    mdata_t mdata;
    etree_addr_t searchAddr, elemAddr;
    int out_step, which;
    off_t offset_base;

    searchAddr.x = (etree_tick_t)(x / result_hdr.mesh_ticksize);
    searchAddr.y = (etree_tick_t)(y / result_hdr.mesh_ticksize);
    searchAddr.z = (etree_tick_t)(z / result_hdr.mesh_ticksize);
    searchAddr.level = ETREE_MAXLEVEL;

    /* Search the mesh etree for the element that contains the point*/
    if (etree_search(mep, searchAddr, &elemAddr, NULL, &mdata) != 0) {
        fprintf(stderr, "%s\n", etree_strerror(etree_errno(mep)));
        return -1;
    }

    /* Calcuate the left-down-back corner's coordinate */
    ldb[0] = elemAddr.x * result_hdr.mesh_ticksize;
    ldb[1] = elemAddr.y * result_hdr.mesh_ticksize;
    ldb[2] = elemAddr.z * result_hdr.mesh_ticksize;

    /* Calcuate how far away is the query point from the center */
    edgesize = mdata.edgesize;
    center[0] = ldb[0] + edgesize / 2;
    center[1] = ldb[1] + edgesize / 2;
    center[2] = ldb[2] + edgesize / 2;

    distance[0] = x - center[0];
    distance[1] = y - center[1];
    distance[2] = z - center[2];

    /* Calculate the phi's */
    phi[0] = (1 - 2 * distance[0] / edgesize) 
        * (1 - 2 * distance[1] / edgesize)
        * (1 - 2 * distance[2] / edgesize) / 8;

    phi[1] = (1 + 2 * distance[0] / edgesize) 
        * (1 - 2 * distance[1] / edgesize)
        * (1 - 2 * distance[2] / edgesize) / 8;
    
    phi[2] = (1 - 2 * distance[0] / edgesize) 
        * (1 + 2 * distance[1] / edgesize)
        * (1 - 2 * distance[2] / edgesize) / 8;

    phi[3] = (1 + 2 * distance[0] / edgesize) 
        * (1 + 2 * distance[1] / edgesize)
        * (1 - 2 * distance[2] / edgesize) / 8;

    phi[4] = (1 - 2 * distance[0] / edgesize) 
        * (1 - 2 * distance[1] / edgesize)
        * (1 + 2 * distance[2] / edgesize) / 8;

    phi[5] = (1 + 2 * distance[0] / edgesize) 
        * (1 - 2 * distance[1] / edgesize)
        * (1 + 2 * distance[2] / edgesize) / 8;
    
    phi[6] = (1 - 2 * distance[0] / edgesize) 
        * (1 + 2 * distance[1] / edgesize)
        * (1 + 2 * distance[2] / edgesize) / 8;

    phi[7] = (1 + 2 * distance[0] / edgesize) 
        * (1 + 2 * distance[1] / edgesize)
        * (1 + 2 * distance[2] / edgesize) / 8;
    
    
    /* Retrieve the data for each step */

    offset_base = sizeof(out_hdr_t);

    for (out_step = 0; out_step < result_hdr.output_steps; out_step++) {
        fvector_t qres;

        qres.f[0] = qres.f[1] = qres.f[2] = 0;

        for (which = 0; which < 8; which++) {
            fvector_t nodeValue;
            int64_t gnid;
            off_t offset;

            gnid = mdata.nid[which];
            offset = offset_base + gnid * sizeof(fvector_t);

#ifdef BIGBEN
            if (fseek(result_fp, offset, SEEK_SET) == -1) {
                fprintf(stderr, "q4_point:");
                perror("fseeko");
                return -1;
            }
#else
            if (fseeko(result_fp, offset, SEEK_SET) == -1) {
                fprintf(stderr, "q4_point:");
                perror("fseeko");
                return -1;
            }
#endif
            
            if (fread(&nodeValue, sizeof(fvector_t), 1, result_fp) != 1) {
                fprintf(stderr, "q4_point:");
                perror("fread");
                return -1;
            }

            qres.f[0] += phi[which] * nodeValue.f[0];
            qres.f[1] += phi[which] * nodeValue.f[1];
            qres.f[2] += phi[which] * nodeValue.f[2];
        } /* for which */

        if (out_format == 1) {
            /* ASCII */
            fprintf(outfp, "%.24g %.24g %.24g\n", 
                    qres.f[0], qres.f[1], qres.f[2]);
        } else {
            /* binary */
            if (fwrite(&qres, sizeof(fvector_t), 1, outfp) != 1 ) {
                fprintf(stderr, "q4_point:");
                perror("fwrite");
                return -1;
            }
        }
            
        /* Move to the next output timestep results' beginning */
        offset_base += result_hdr.total_nodes * sizeof(fvector_t);

    } /* for ts */


    return 0;
    
}
    
