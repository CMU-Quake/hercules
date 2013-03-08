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

/*
 * code.h -  conversion between different octant address and locational code
 *
 *
 */

#ifndef CODE_H
#define CODE_H

#ifdef ALPHA
#include "etree_inttypes.h"
#else
#include <inttypes.h>
#endif


#include "etree.h"

int code_addr2key(etree_t *ep, etree_addr_t addr, void *key);
int code_key2addr(etree_t *ep, void *key, etree_addr_t *paddr);

int code_isancestorkey(const void *ancestorkey, const void *childkey);
int code_derivechildkey(const void *key, void *childkey, int branch);
int code_extractbranch(const void *morton, int level);
void code_setbranch(void *morton, int level, int branch);
void code_setlevel(void *key, int level, etree_type_t type);

int code_comparekey(const void *key1, const void *key2, int size);

void code_morton2coord(int bits, void *morton, etree_tick_t *px, 
                       etree_tick_t *py, etree_tick_t *pz);

void code_coord2morton(int bits, etree_tick_t x, etree_tick_t y, 
                       etree_tick_t z, void *morton);

#endif /* CODE_H */




