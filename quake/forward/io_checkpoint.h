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
#include <sys/types.h>
#include "psolve.h"

extern void checkpoint_write( int step, int myID, mesh_t* myMesh, char* theCheckPointingDirOut,
			       int theGroupSize, mysolver_t* mySolver, MPI_Comm comm_solver );

extern int  checkpoint_read ( int myID, mesh_t* myMesh, char* theCheckPointingDirOut,
                               int theGroupSize, mysolver_t* mySolver, MPI_Comm comm_solver );
