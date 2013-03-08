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
 * dlink.c - implementation of doubly linked list
 *
 *
 */

#include <stdio.h>
#include "dlink.h"


/*
 * dlink_init - initialize a doubly linked list
 *
 */
void dlink_init(dlink_t *sentinel)
{
    sentinel->next = sentinel;
    sentinel->prev = sentinel;
    return;
}


/*
 * dlink_insert - insert a new entry at the head of the d-list
 *
 */
void dlink_insert(dlink_t *sentinel, dlink_t *newlink)
{
    /* link in the new comer */
    newlink->next = sentinel->next;
    newlink->prev = sentinel;
    
    /* modify neighbors' pointers*/
    sentinel->next->prev = newlink; 
    sentinel->next = newlink;
    return;
}


/*
 * dlink_delete - delete an entry pointed by oldlink
 *
 */
void dlink_delete(dlink_t *oldlink)
{
    oldlink->prev->next = oldlink->next;
    oldlink->next->prev = oldlink->prev;
    oldlink->next = oldlink->prev = NULL;
    return;
}


/*
 * dlink_addstub - create a stub for the dlink 
 *
 */
void dlink_addstub(dlink_t *sentinel, dlink_t *firstlink)
{
    sentinel->next = firstlink;
    sentinel->prev = firstlink->prev;

    sentinel->prev->next = sentinel;
    sentinel->next->prev = sentinel;
    return;
}
