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
 * cvm.h - Data types and routines for manipulating material databases 
 *         (in etree format)
 *
 *
 */

#ifndef CVM_H
#define CVM_H

#include "etree.h"

#define DIST1LAT 110922.0    /* distance(meters) of 1 degree latitude   */
#define DIST1LON 92382.0     /* distance(meters) of 1 degree longitude  */

typedef struct cvmpayload_t {
    float Vp, Vs, rho;
} cvmpayload_t;

//typedef struct cvmpayload_t cvmpayload_t;


/**
 * material database control/meta data 
 *
 */
typedef struct dbctl_t {
    char *create_model_name;
    char *create_author;
    char *create_date;
    char *create_field_count;
    char *create_field_names;

    double region_origin_latitude_deg;   
    double region_origin_longitude_deg;  
    double region_length_east_m;     
    double region_length_north_m; 
    double region_depth_shallow_m;  
    double region_depth_deep_m; 

    etree_tick_t domain_endpoint_x;  /* east  */
    etree_tick_t domain_endpoint_y;  /* north */
    etree_tick_t domain_endpoint_z;  /* depth */

} dbctl_t;

dbctl_t *cvm_newdbctl();
void cvm_freedbctl(dbctl_t *dbctlPtr);

int cvm_setdbctl(etree_t *cvmEp, dbctl_t *dbctlPtr);
dbctl_t *cvm_getdbctl(etree_t *cvmEp);

int cvm_query(etree_t *cvmEp, double east_m, double north_m, double depth_m, 
              cvmpayload_t* payload);

#endif /* CVM_H */
