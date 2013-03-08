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

/**
 * buffer.h: single-threaded buffer manager implementing LRU replacement policy
 *
 *
 */

#ifndef BUFFER_H
#define BUFFER_H


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ALPHA
#include "etree_inttypes.h"
#else
#include <inttypes.h>
#endif  


#include "dlink.h"

#define O_INCORE 020000000000

#ifndef PAGENUM_T
typedef off_t pagenum_t;
#define PAGENUM_T
#endif

/*
 * bcb_t - buffer control block for each page in the buffer pool
 *
 */
typedef struct bcb_t {
    pagenum_t pagenum;
    void *pageaddr;
    dlink_t lruln; 
    dlink_t hashln; /* overload hashln for free list use */
    uint32_t refcount;
    char modified;
} bcb_t;


/*
 * buffer_t -buffer pool manager that contains the following information
 *
 * - the file name being cached and the file desciptor
 * - the pointer to the buffer pool, the bcb for each frame
 *   and the number of frames allocated
 * - the current free available frames in the buffer pool
 * - the LRU links list to find victim (cached) pages 
 * - the hash tabel to locate a cached page 
 *
 */
typedef struct buffer_t {
    char *filename;
    int fd;
    int flags;

    void *pool;
    bcb_t *bcbtable;
    size_t framecount;
    uint32_t pagesize;
    size_t freecount;

    dlink_t freebcblist;
    
    dlink_t bcblru;

    uint32_t bcbhtsize;
    dlink_t *bcbhashtable;

    uint64_t reqs, hits, hitlookups, misslookups;

} buffer_t;

    
buffer_t *buffer_init(const char *filename, int flags, size_t framecount, 
                      uint32_t pagesize);
int buffer_destroy(buffer_t *buf);

void *buffer_emptyfix(buffer_t *buf, pagenum_t pagenum);
void *buffer_fix(buffer_t *buf, pagenum_t pagenum);

int buffer_ref(buffer_t *buf, void *pageaddr);
int buffer_unref(buffer_t *buf, void *pageaddr);
void buffer_mark(buffer_t *buf, void *pageaddr);
pagenum_t buffer_pagenum(buffer_t *buf, void *pageaddr);

int buffer_isdirty(buffer_t *buf, void *pageaddr);

void buffer_showusage(buffer_t *buf, FILE *fp);

#endif /* BUFFER_H */

