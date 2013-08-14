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

#ifndef STIFFNESS_H_
#define STIFFNESS_H_

typedef enum {
    CONVENTIONAL = 0, EFFECTIVE
} stiffness_type_t;

void linear_elements_count   (int32_t myID, mesh_t *myMesh);
void linear_elements_mapping (int32_t myID, mesh_t *myMesh);
void stiffness_init          (int32_t myID, mesh_t *myMesh, mysolver_t *mySolver);
void stiffness_delete        (int32_t myID);

void compute_addforce_conventional( mesh_t* myMesh, mysolver_t* mySolver, fmatrix_t (*theK1)[8],
				    fmatrix_t (*theK2)[8] );
void compute_addforce_effective_cpu( mesh_t* myMesh, mysolver_t* mySolver );
void compute_addforce_effective_gpu( int32_t myID, mesh_t* myMesh, mysolver_t* mySolver );

void firstVector_kappa( const double* atu, double* finalVector, double kappa);
void firstVector_mu( const double* atu, double* finalVector, double b);

#endif /* STIFFNESS_H_ */
