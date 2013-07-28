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
 *
 *  Created on: Mar 23, 2012
 *      Author: yigit
 */

#ifndef SOFTSOIL_H_
#define SOFTSOIL_H_


void softsoil_init ( int32_t myID, const char *parametersin );
int32_t softsoil_initparameters ( const char *parametersin );
double get_modulus_factor(double x_m,double y_m,double z_m, double XMeshOrigin,double YMeshOrigin,double ZMeshOrigin);
double get_damping_ratio(double x_m,double y_m,double z_m, double XMeshOrigin,double YMeshOrigin,double ZMeshOrigin);
double get_lower_modulus_ratio(int i, double shearstrain);
double get_upper_modulus_ratio(int i, double shearstrain);
void construct_strain_table ( const char *parametersin );


#endif /* SOFTSOIL_H_ */

