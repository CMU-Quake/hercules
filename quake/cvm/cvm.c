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
 * cvm.c - Data types and routines for manipulating
 *         material databases (in etree format)
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cvm.h"

/* Asssume the string length is less than 4K */
#define MAXMETALEN 4096


#ifndef DO_DEBUG
#ifdef  DEBUG
#define DO_DEBUG	1
#else
#define DO_DEBUG	0
#endif  /* DEBUG */
#endif  /* DO_DEBUG */


/**
 * cvm_setdbctl
 *
 * - return 0 if OK, -1 on error
 */
int cvm_setdbctl(etree_t *ep, dbctl_t *dbctlPtr)
{
    char appmeta[MAXMETALEN];

    sprintf(appmeta, "%s %s %s %s %s %f %f %f %f %f %f %u %u %u",
            dbctlPtr->create_model_name,
            dbctlPtr->create_author,
            dbctlPtr->create_date,
            dbctlPtr->create_field_count,
            dbctlPtr->create_field_names,
            
            dbctlPtr->region_origin_latitude_deg,
            dbctlPtr->region_origin_longitude_deg,
            dbctlPtr->region_length_east_m,
            dbctlPtr->region_length_north_m,
            dbctlPtr->region_depth_shallow_m,
            dbctlPtr->region_depth_deep_m,

            dbctlPtr->domain_endpoint_x,
            dbctlPtr->domain_endpoint_y,
            dbctlPtr->domain_endpoint_z
            );

    assert(strlen(appmeta) < MAXMETALEN);

    if (etree_setappmeta(ep, appmeta) != 0) {
        fprintf(stderr, "%s\n", etree_strerror(etree_errno(ep)));
        return -1;
    }

    return 0;
}
           
/**
 * cvm_getdbctl:
 *
 * - retrieve the cvm database control infomation
 * - return a pointer to a dbctl_t if OK, NULL3 on error
 *
 */
dbctl_t *cvm_getdbctl(etree_t *ep)
{
    char *appmeta, *token, *saveptr;
    dbctl_t *dbctlPtr;

    if ((dbctlPtr =cvm_newdbctl()) == NULL) {
        return NULL;
    }

    /* read the appl. meta data from the etree */
    if (ep == NULL) {
        return NULL;
    }

    appmeta = etree_getappmeta(ep);
    if (!appmeta) {
        fprintf(stderr, "cvm_getdbctl: cvm database %s has no control data!\n",
                ep->pathname);
        return NULL;
    }

    /* parse the text string and allocate space to hold the control data */
    token = (char *)strtok_r(appmeta, " ", &saveptr);
    if ((dbctlPtr->create_model_name = strdup(token)) == NULL ) {
        perror("cvm_getdbctl:strdup create_model_name");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if ((dbctlPtr->create_author = strdup(token)) == NULL) {
        perror("cvm_getdbctl:strdup create_author");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if ((dbctlPtr->create_date = strdup(token)) == NULL) {
        perror("cvm_getdbctl:strdup create_date");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if ((dbctlPtr->create_field_count = strdup(token)) == NULL) {
        perror("cvm_getdbctl:strdup create_field_count");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if ((dbctlPtr->create_field_names = strdup(token)) == NULL) {
        perror("cvm_getdbctl:strdup create_field_names");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_origin_latitude_deg) != 1) {
        fprintf(stderr, 
                "cvm_getdbctl: cannot sscanf region_origin_latitude_deg\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_origin_longitude_deg) != 1) {
        fprintf(stderr,
                "cvm_getdbctl:cannot sscanf region_origin_longitude_deg\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_length_east_m) != 1) {
        fprintf(stderr, 
                "cvm_getdbctl: cannot sscanf region_length_east_m\n");
        return NULL;
    }
    
    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_length_north_m) != 1) {
        fprintf(stderr, 
                "cvm_getdbctl: cannot sscanf region_length_north_m\n");
        return NULL;
    }
    
    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_depth_shallow_m) != 1) {
        fprintf(stderr, 
                "cvm_getdbctl: cannot sscanf region_depth_shallow_m\n");
        return NULL;
    }
    
    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%lf", &dbctlPtr->region_depth_deep_m) != 1) {
        fprintf(stderr, 
                "cvm_getdbctl: cannot sscanf region_depth_deep_m\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%u", &dbctlPtr->domain_endpoint_x) != 1) {
        fprintf(stderr,
                "cvm_getdbctl: cannot sscanf domain_endpoint_x\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%u", &dbctlPtr->domain_endpoint_y) != 1) {
        fprintf(stderr,
                "cvm_getdbctl: cannot sscanf domain_endpoint_y\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    if (sscanf(token, "%u", &dbctlPtr->domain_endpoint_z) != 1) {
        fprintf(stderr,
                "cvm_getdbctl: cannot sscanf domain_endpoint_z\n");
        return NULL;
    }

    token = (char *)strtok_r(NULL, " ", &saveptr);
    assert(token == NULL);

    /* remember to free the application metadata text string */
    free(appmeta);

    return dbctlPtr;
}
        
/**
 * cvm_newdbctl
 *
 * - return NULL on error
 *
 */
dbctl_t* cvm_newdbctl()
{
    dbctl_t* dbctl = (dbctl_t*)malloc( sizeof(dbctl_t) );

    return dbctl;
}
            

/**
 * cvm_freedbctl:
 *
 * - release memory held by the text strings
 *
 */
void cvm_freedbctl(dbctl_t *dbctlPtr)
{
    assert(dbctlPtr != NULL);

    assert(dbctlPtr->create_model_name != NULL);
    free(dbctlPtr->create_model_name);

    assert(dbctlPtr->create_author != NULL);
    free(dbctlPtr->create_author);

    assert(dbctlPtr->create_date != NULL);
    free(dbctlPtr->create_date);

    assert(dbctlPtr->create_field_count != NULL);
    free(dbctlPtr->create_field_count);

    assert(dbctlPtr->create_field_names != NULL);
    free(dbctlPtr->create_field_names);
    
    free(dbctlPtr);
    
    return;
}

/**
 * cvm_query:
 *
 * \return 0 if OK, -1 on error
 */
int
cvm_query( etree_t *ep, double east_m, double north_m, double depth_m,
	   cvmpayload_t* payload )
{
    static int ready = 0;
    static double tickSize;

    etree_addr_t queryAddr;

    if (!ready) {
        /* prepare for the query */
        
        dbctl_t *myctl;

        myctl = cvm_getdbctl(ep);
        if (myctl == NULL) {
            fprintf(stderr, "cvm_query: cannot get databae control data\n");
            return -1;
        }
        
        tickSize = myctl->region_length_east_m / myctl->domain_endpoint_x;

        cvm_freedbctl(myctl);
        ready = 1;
    }
    
    queryAddr.x = (etree_tick_t)(east_m  / tickSize);
    queryAddr.y = (etree_tick_t)(north_m / tickSize);
    queryAddr.z = (etree_tick_t)(depth_m / tickSize);
    queryAddr.level = ETREE_MAXLEVEL;

    if (etree_search( ep, queryAddr, NULL, "*", payload ) != 0) {

	if (DO_DEBUG) {
	    fprintf( stderr, "cvm_query: %s\n"
		     "  query addr: %08x %08x %08x %2d\n"
		     "  coords:     %8.0f %8.0f %8.0f %10.10f\n",
		     etree_strerror( etree_errno( ep ) ),
		     queryAddr.x, queryAddr.y, queryAddr.z, queryAddr.level,
		     east_m, north_m, depth_m, tickSize );
	}

        return -1;
    }

    return 0;
}


    


