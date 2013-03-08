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
 * pickrecord.c: 
 *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>

#include "xplatform.h"

int main(int argc, char **argv)
{
    FILE *cvmfp;
    off_t offset;
    int ini, inj, ink;
    int i, j, k;
    endian_t targetformat, myformat;
    
    if (argc != 4) {
        printf("\nusage: pickrecord cvmfile format offset\n");
        exit(1);
    }

    cvmfp = fopen(argv[1], "r");
    if (cvmfp == NULL) {
        perror("fopen");
        exit(1);
    }

    if (strcmp(argv[2], "little") == 0) {
        targetformat = little;
    } else if (strcmp(argv[2], "big") == 0) {
        targetformat = big;
    } else {
        fprintf(stderr, "Unknown target data format\n");
        exit(1);
    }


    sscanf(argv[3], "%qd", &offset);


#ifdef BIGBEN
    if (fseek(cvmfp, offset, SEEK_SET) != 0) {
        perror("fseeko");
        exit(1);
    }
#else 
    if (fseeko(cvmfp, offset, SEEK_SET) != 0) {
        perror("fseeko");
        exit(1);
    }
#endif

    if ((fread(&ini, 4, 1, cvmfp) != 1) ||
        (fread(&inj, 4, 1, cvmfp) != 1) ||
        (fread(&ink, 4, 1, cvmfp) != 1)) {
        perror("fread");
        exit(1);
    }

    myformat = xplatform_testendian();
    
    if (myformat == targetformat) {
        i = ini;
        j = inj;
        k = ink;
    } else {
        xplatform_swapbytes(&i, &ini, 4);
        xplatform_swapbytes(&j, &inj, 4);
        xplatform_swapbytes(&k, &ink, 4);
    }

    printf("(i, j, k) = {%d, %d, %d}\n", i, j, k);
    
    fclose(cvmfp);
    return 0;
}
