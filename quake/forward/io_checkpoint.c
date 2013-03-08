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

#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <stdio.h>
#include "timers.h"
#include "io_checkpoint.h"
#include "psolve.h"

/* Writes header and tm1 and tm2 arrays */
void checkpoint_write(int step, int myID, mesh_t* myMesh, char* theCheckPointingDirOut,
		      int theGroupSize, mysolver_t* mySolver, MPI_Comm comm_solver ){

    char       filename[256];
    FILE*      fp;
    int        nharboredmax, localnharbored;
    off_t      offset;
    size_t     written;
    static int CheckpointNumber = 0;
    int        GroupStart;

    static int WriteGroupSize = 512; /* This determines how many PEs write at once */

    Timer_Start("Checkpoint");

    /* compute max nharbored for stripe size */
    localnharbored = myMesh->nharbored;
    MPI_Reduce( &localnharbored, &nharboredmax, 1, MPI_INT, MPI_MAX, 0,
                comm_solver );
    MPI_Bcast( &nharboredmax, 1, MPI_INT, 0, comm_solver );

    /* Sanity check for nharbored */
    if ((myMesh->nharbored > nharboredmax) || (localnharbored > nharboredmax)){
        fprintf(stderr, "Checkpoint has nharbored greater than nharboredmax!");
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(1);
    }

    /* Create file name */
    sprintf( filename, "%s%s%d", theCheckPointingDirOut, "/checkpoint.out",
             CheckpointNumber );

    /* PE 0 creates file and writes the header */
    if (myID == 0) {
	fp = fopen( filename, "wb" );
	if(fp == NULL) {
	    fprintf(stderr, "PE0 Can't Create Checkpoint File!");
	    MPI_Abort(MPI_COMM_WORLD, -1);
	    exit(1);
	}
        fwrite( &theGroupSize, sizeof(int), 1, fp );
        fwrite( &step,         sizeof(int), 1, fp );
        fwrite( &nharboredmax, sizeof(int), 1 ,fp );
	fclose( fp );
    }

    MPI_Barrier( comm_solver );	

    /* Go through all PEs in chunks of WriteGroupSize */
    for (GroupStart = 0; GroupStart < theGroupSize; GroupStart += WriteGroupSize){

	if ( (myID >= GroupStart) && (myID < GroupStart + WriteGroupSize) ){

	    fp = fopen( filename, "rb+" );  /*Open Existing File*/
	    if(fp == NULL) {
		fprintf(stderr, "Can't Open Checkpoint File!");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }
	    
	    MPI_Barrier( comm_solver );  /* Needless, but may(?) help lustre */

	    /* Determine offset to write tm1 field */
	    offset = 
		( 3 * sizeof(int) )  +                               /* Header Size */
		( 2 * myID * nharboredmax * sizeof(fvector_t) );     /* Processor Offset */
	    
	    fseeko( fp, offset, SEEK_SET );
	    
	    /* write tm1 - due to some prior switch, this is tm2 here */
	    written = fwrite( mySolver->tm2, sizeof(fvector_t), myMesh->nharbored, fp );
	    if(written != myMesh->nharbored) {
		fprintf(stderr, "Error writing Checkpoint File!");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }
	
	    /* Add fieldsize to get tm2 offset */
	    offset += myMesh->nharbored * sizeof(fvector_t);
	
	    fseeko( fp, offset, SEEK_SET );
	
	    /* write tm2 - due to some prior switch, this is tm1 here */
	    written = fwrite( mySolver->tm1, sizeof(fvector_t), myMesh->nharbored, fp );
	    if(written != myMesh->nharbored) {
		fprintf(stderr, "Error writing Checkpoint File!");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }
	
	    fclose( fp );
	}

	MPI_Barrier( comm_solver );	
    }

    /* Alternate checkpointing files */
    CheckpointNumber ? (CheckpointNumber=0):(CheckpointNumber=1);

    MPI_Barrier( comm_solver );	/*Just for Timer's sake*/
    Timer_Stop("Checkpoint");

}



/* Reads tm1 and tm2 arrays and returns step */
extern int  checkpoint_read ( int myID, mesh_t* myMesh, char* theCheckPointingDirOut,
			      int theGroupSize, mysolver_t* mySolver, MPI_Comm comm_solver ){

    char     filename[256];
    FILE*    fp;
    int      nharboredmax, PESize, step;
    off_t    offset;
    size_t   num_read;
    int      GroupStart;

    static int ReadGroupSize = 512; /* This determines how many PEs read at once */

    Timer_Start("Checkpoint");

    /* Create checkpointing file name */
    sprintf( filename, "%s%s", theCheckPointingDirOut, "/checkpoint.in" );

    /* PE 0 reads the header for all */
    if (myID == 0) {
        fp = fopen( filename, "rb" );
	if(fp == NULL) {
	    fprintf(stderr, "PE0 Can't Open Checkpoint File To Read Header!");
	    MPI_Abort(MPI_COMM_WORLD, -1);
	    exit(1);
	}
        fread( &PESize,       sizeof(int), 1 , fp );
        fread( &step,         sizeof(int), 1 , fp );
        fread( &nharboredmax, sizeof(int), 1 , fp );
        fclose( fp );
        /* check number of processors is right */
        if (PESize != theGroupSize) {
	    fprintf(stderr, "Checkpoint file PE count (%d) does not"
		    "match current PE count (%d)!\n", PESize, theGroupSize);
	    MPI_Abort(MPI_COMM_WORLD, -1);
	    exit(1);
	}
    }

    /* broadcast header data */
    MPI_Bcast( &PESize,       1, MPI_INT, 0, comm_solver );
    MPI_Bcast( &step,         1, MPI_INT, 0, comm_solver );
    MPI_Bcast( &nharboredmax, 1, MPI_INT, 0, comm_solver );

    /* sanity check nhabored */
    if (myMesh->nharbored > nharboredmax) {
	fprintf(stderr, "Local nhabored (%d) greater than checkpoint file nharboredmax! (%d)!\n",
		myMesh->nharbored, nharboredmax);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }

    MPI_Barrier( comm_solver );

    /* Go through all PEs in chunks of ReadGroupSize */
    for (GroupStart = 0; GroupStart < theGroupSize; GroupStart += ReadGroupSize){
	
	if ( (myID >= GroupStart) && (myID < GroupStart + ReadGroupSize) ){

	    /* open checkpoint file */
	    fp = fopen( filename, "rb" );
	    if(fp == NULL) {
		fprintf(stderr, "Can't Open Checkpoint File!");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }

	    MPI_Barrier( comm_solver );  /* Needless, but may(?) help lustre */

	    /* Determine offset to find tm1 field */
	    offset = 
		( 3 * sizeof(int) )  +                               /* Header Size */
		( 2 * myID * nharboredmax * sizeof(fvector_t) );     /* Processor Offset */

	    fseeko( fp, offset, SEEK_SET );

	    num_read = fread( mySolver->tm1, sizeof(fvector_t), myMesh->nharbored, fp );
	    if (num_read != myMesh->nharbored) {
		fprintf(stderr, "Problem reading checkpoint file tm1 field!\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }

	    offset += myMesh->nharbored * sizeof(fvector_t);

	    fseeko( fp, offset, SEEK_SET );

	    num_read = fread( mySolver->tm2, sizeof(fvector_t), myMesh->nharbored, fp );
	    if (num_read != myMesh->nharbored) {
		fprintf(stderr, "Problem reading checkpoint file tm2 field!\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
		exit(1);
	    }

	    fclose( fp );
	}
    }

    MPI_Barrier( comm_solver ); /*Just for Timer's sake*/
    Timer_Stop("Checkpoint");

    return step;
}
