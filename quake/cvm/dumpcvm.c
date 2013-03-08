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
 * dumpcvm.c: Dump the content of a cvm database to a binary flat file in
 *            ascending locational code order. 
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "etree.h"
#include "xplatform.h"
#include "cvm.h"
#include "code.h"

#define KEYSIZE 13

int main(int argc, char **argv)
{
    char *cvmetree, *outputfile, *outformat;
    FILE *outfp;
    etree_t *cvmEp;
    cvmpayload_t rawElem;
    int64_t totalcount;
    struct timeval starttime, endtime;
    int scantime;
    endian_t myformat, targetformat;
    float outVp, outVs, outrho;
    int outi, outj, outk;

    if (argc != 4) {
        printf("\nusage: dumpcvm cvmetree output format\n");
        printf("cvmetree: pathname to the CVM etree\n");
        printf("output: pathname to the flat output file\n");
        printf("format: little or big\n");
        exit(1);
    }

    cvmetree = argv[1];
    cvmEp = etree_open(cvmetree, O_RDONLY, 0, 0, 0);
    if (!cvmEp) {
        fprintf(stderr, "Cannot open CVM material database %s\n", cvmetree);
        exit(1);
    }


    outputfile = argv[2];
    outfp = fopen(outputfile, "w+");
    if (outfp == NULL) {
        perror("fopen");
        exit(1);
    }
        
    outformat = argv[3];
    if (strcmp(outformat, "little") == 0) {
        targetformat = little;
    } else if (strcmp(outformat, "big") == 0) {
        targetformat = big;
    } else {
        fprintf(stderr, "Unknown target data format\n");
        exit(1);
    }

    myformat = xplatform_testendian();

    /* go through all the records stored in the underlying btree */
    gettimeofday(&starttime, NULL);

    totalcount = 0;
    memset(cvmEp->key, 0, KEYSIZE);

    if (btree_initcursor(cvmEp->bp, cvmEp->key) != 0) {
        fprintf(stderr, "Cannot set cursor in the underlying database\n");
        exit(1);
    }

    do {
        etree_tick_t i, j, k;

        if (btree_getcursor(cvmEp->bp, cvmEp->hitkey, "*", &rawElem) != 0) {
            fprintf(stderr, "Read cursor error\n");
            exit(1);
        } 

        code_morton2coord(ETREE_MAXLEVEL + 1, (char *)cvmEp->hitkey + 1,
                          &i, &j, &k);
                

        /* Write to the output file, do format conversion if necessary */
        if (myformat == targetformat) {
            outi = i; 
            outj = j;
            outk = k;
            outVp = rawElem.Vp;
            outVs = rawElem.Vs;
            outrho = rawElem.rho;
        } else {
            xplatform_swapbytes(&outi, &i, 4);
            xplatform_swapbytes(&outj, &j, 4);
            xplatform_swapbytes(&outk, &k, 4);
            xplatform_swapbytes(&outVp, &rawElem.Vp, 4);
            xplatform_swapbytes(&outVs, &rawElem.Vs, 4);
            xplatform_swapbytes(&outrho, &rawElem.rho, 4);
        }

        if ((fwrite(&outi, 4, 1, outfp) != 1) ||
            (fwrite(&outj, 4, 1, outfp) != 1) ||
            (fwrite(&outk, 4, 1, outfp) != 1) ||
            (fwrite(&outVp, 4, 1, outfp) != 1) ||
            (fwrite(&outVs, 4, 1, outfp) != 1) ||
            (fwrite(&outrho, 4, 1, outfp) != 1)) {
            fprintf(stderr, "Error writing CVM record\n");
            perror("fwrite");
            exit(1);
        }

        
        totalcount++;
    } while (btree_advcursor(cvmEp->bp) == 0);

    if (fclose(outfp) != 0) {
        perror("fclose");
        exit(1);
    }
    etree_close(cvmEp);


    gettimeofday(&endtime, NULL);
    scantime = (endtime.tv_sec - starttime.tv_sec);

    printf("Dump the CVM database in %d seconds\n", scantime);
    printf("Dump %qd octants\n", totalcount);
    printf("Output format is %s-endian", outformat);
    printf("Output file is %s\n", outputfile);
 
    return 0;
}

            
    

