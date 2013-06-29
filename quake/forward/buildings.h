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

#ifndef BUILDINGS_H_
#define BUILDINGS_H_

#include "quake_util.h"

/* -------------------------------------------------------------------------- */
/*                          Public Method Prototypes                          */
/* -------------------------------------------------------------------------- */

double get_surface_shift();
noyesflag_t get_fixedbase_flag();
noyesflag_t get_constrained_slab_flag();

/** 3-ary int vector */
struct intvector_t {
    int32_t f[3];
};
typedef struct intvector_t intvector_t;


int bldgs_nodesearch ( tick_t x, tick_t y, tick_t z, double ticksize );

int bldgs_nodesearch_com ( tick_t x, tick_t y, tick_t z, double ticksize );

int pushdowns_nodesearch ( tick_t x, tick_t y, tick_t z, double ticksize );

int pushdowns_search ( tick_t x, tick_t y, tick_t z, double ticksize);

int bldgs_setrec   ( octant_t *leaf,
                     double    ticksize,
                     edata_t  *edata,
                     etree_t  *cvm,
                     double xoriginm,
                     double yoriginm,
                     double zoriginm);

int bldgs_toexpand ( octant_t *leaf,
                     double    ticksize,
                     edata_t  *edata,
                     double    theFactor );

int bldgs_correctproperties ( mesh_t *myMesh, edata_t *edata, int32_t lnid0 );

void bldgs_fixedbase_init ( mesh_t *myMesh, double simTime );

void constrained_slabs_init ( mesh_t *myMesh, double simTime, int32_t group_number, int32_t myID);

void bldgs_load_fixedbase_disps ( mysolver_t* mySolver, double simDT, int step );

void bldgs_update_constrainedslabs_disps (  mysolver_t* mySolver, double simDT, int step, int32_t myID);

void bldgs_init ( int32_t myID, const char *parametersin );

void bldgs_finalize();

void solver_constrained_buildings_close ();
/* -------------------------------------------------------------------------- */

#endif /* BUILDINGS_H_ */
