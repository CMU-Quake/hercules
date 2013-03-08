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
 * dlink.h - auxilliary data structure for doubly linked list operations
 *
 *
 */
#ifndef DLINK_H
#define DLINK_H

typedef struct dlink_t {
    struct dlink_t *prev;
    struct dlink_t *next;
} dlink_t;


void dlink_init(dlink_t *sentinel);
void dlink_insert(dlink_t *sentinel, dlink_t *newlink);
void dlink_delete(dlink_t *oldlink);
void dlink_addstub(dlink_t *sentinel, dlink_t *firstlink);

#endif
