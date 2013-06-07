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

#include "psolve.h"

extern int planes_print(int32_t myID, mysolver_t* mySolver);
extern void planes_setup(int32_t myID, const char *numericalin,
                         double surfaceShift);
extern void planes_close(int32_t myID);
extern void planes_IO_PES_main(int32_t myID);


