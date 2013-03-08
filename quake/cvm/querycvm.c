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
 * querycvm.c
 *
 *
 */
#include "stdio.h"
#include "stdlib.h"

#include "etree.h"
#include "cvm.h"

#define CVMBUFFERSIZE 100 /* performance knob */

int main( int argc, char** argv )
{
    char*    cvmetree;
    etree_t* cvmEp;
    double east_m, north_m, depth_m;
    cvmpayload_t rawElem;
    int res;

    if (argc != 5) {
	fprintf( stderr, "\nusage: %s cvm_etree_file east_m north_m depth_m\n\n",
		 argv[0] );
	return 1;
    }

    cvmetree = argv[0];
    sscanf( argv[2], "%lf", &east_m  );
    sscanf( argv[3], "%lf", &north_m );
    sscanf( argv[4], "%lf", &depth_m );


    cvmEp = etree_open( cvmetree, O_RDONLY, CVMBUFFERSIZE, 0, 0 );
    if (!cvmEp) {
	perror( "etree_open(...)" );
	fprintf( stderr, "Cannot open CVM material database %s\n", cvmetree );
	return 2;
    }

    res = cvm_query( cvmEp, east_m, north_m, depth_m, &rawElem );

    if (res != 0) {
	fprintf( stderr, "Cannot find the query point\n" );
    } else {
	printf( "\nMaterial property for\n(%f East, %f North, %f Depth)\n",
		east_m, north_m, depth_m );
	printf( "Vp =      %.4f\n", rawElem.Vp );
	printf( "Vs =      %.4f\n", rawElem.Vs );
	printf( "density = %.4f\n", rawElem.rho );
	printf( "\n" );
    }

    etree_close( cvmEp );

    return (res == 0) ? 0 : 3;
}
