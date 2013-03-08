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
 * showdbctl.c - print material database control/meta data.
 * 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "etree.h"
#include "cvm.h"

int main(int argc, char ** argv)
{
    char *etreefile;
    dbctl_t *myctl;
    etree_t *ep;

    /* read command line argument */
    if (argc != 2) {
        fprintf(stderr, "Usage: showdbctl etreename\n");
        exit(1);
    }

    etreefile = argv[1];

    if ((ep = etree_open(etreefile, O_RDONLY, 0, 0, 0)) == NULL) {
        fprintf(stderr, "Fail to open the etree\n");
        exit(1);
    }

    myctl = cvm_newdbctl();
    if (myctl == NULL) {
        perror("cvm_newdbctl");
        exit(1);
    }

    if ((myctl = cvm_getdbctl(ep)) == NULL) {
        fprintf(stderr, "Cannot get the material database control data\n");
        exit(1);
    }

    printf("\n");
    printf("create_db_name:               %s\n", ep->pathname);
    printf("create_model_name:            %s\n", myctl->create_model_name);
    printf("create_author:                %s\n", myctl->create_author);
    printf("create_date:                  %s\n", myctl->create_date);
    printf("create_field_count:           %s\n", myctl->create_field_count);
    printf("create_field_names:           %s\n", myctl->create_field_names);
    printf("\n");

    printf("region_origin_latitude_deg:   %f\n", 
           myctl->region_origin_latitude_deg);
    printf("region_origin_longitude_deg:  %f\n", 
           myctl->region_origin_longitude_deg);
    printf("region_length_east_m:         %f\n", myctl->region_length_east_m);
    printf("region_length_north_m:        %f\n", myctl->region_length_north_m);
    printf("region_depth_shallow_m:       %f\n", 
           myctl->region_depth_shallow_m);
    printf("region_depth_deep_m:          %f\n", myctl->region_depth_deep_m);
    printf("\n");

    printf("domain_endpoint_x:            %u\n", myctl->domain_endpoint_x);
    printf("domain_endpoint_y:            %u\n", myctl->domain_endpoint_y);
    printf("domain_endpoint_z:            %u\n", myctl->domain_endpoint_z);

    cvm_freedbctl(myctl);
    
    etree_close(ep);
    
    return 0;
}

        
