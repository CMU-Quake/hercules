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

#ifndef MESHFORMATLAB_H_
#define MESHFORMATLAB_H_

void saveMeshCoordinatesForMatlab( mesh_t      *myMesh,       int32_t myID,
                                   const char  *parametersin, double  ticksize,
                                   damping_type_t theTypeOfDamping, double xoriginm, double yoriginm,
                                   double zoriginm, noyesflag_t includeBuildings );

#endif /* MESHFORMATLAB_H_ */

