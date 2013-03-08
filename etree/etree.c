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
 * etree.c - A library for manipulating large octrees on disk
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>
#include <errno.h>

#include "etree.h"
#include "buffer.h"
#include "code.h"

#ifndef ROOTPAGE
#define ROOTPAGE 1
#endif /* ROOTPAGE */

#ifndef DEFAULTBUFSIZE
#define DEFAULTBUFSIZE 20
#endif

#ifndef PATH_MAX
#define PATH_MAX 2048
#endif

static const 
int HEADERSIZE = 1 + 4 * 4 + 2 * sizeof(BIGINT) * (ETREE_MAXLEVEL + 1);

static const char msg_NOERROR[] = "No error occurs";
static const char msg_UNKNOWN[] = "Unknown error code";
static const char msg_LEVEL_OOB[] = "Octant level out of bounds";
static const char msg_LEVEL_OOB2[] = "Returned octant level out of bounds";
static const char msg_LEVEL_CHILD_OOB[] = "Child octant level out of bounds";
static const char msg_OP_CONFLICT[] = "Operation conflicts with ongoing append or cursor";
static const char msg_EMPTY_TREE[] = "The etree is empty";
static const char msg_DUPLICATE[] = "Octant already exists in the etree";
static const char msg_NOT_FOUND[] = "Can't find the octant";
static const char msg_NO_MEMORY[] = "Out of memory";
static const char msg_NOT_LEAF_SPROUT[] = "Sprouting point is not a leaf";
static const char msg_NO_ANCHOR[] = "Cannot find sprouting octant in etree";
static const char msg_NO_CURSOR[] = "No cursor in effect";
static const char msg_NOT_APPENDING[] = "Etree not in appending mode";
static const char msg_ILLEGAL_FILL[] = "Invalid fill ratio";
static const char msg_APPEND_OOO[] = "Attempted to append an octant out of order";
static const char msg_END_OF_TREE[] = "Already at the end of the tree";
static const char msg_NOT_NEWTREE[] = "Etree not opened with O_CREAT|O_TRUNC";
static const char msg_NOT_3D[] = "Sprouting not supported for 4D etrees";
static const char msg_CREATE_FAILURE[] = "Unable to create boundary etree";
/* static const char msg_OCTREE_FAILURE[] = "Unable to rebuild incore octree image"; */
static const char msg_BOUNDARY_ERROR[] = "Unable to record boundary octants";
static const char msg_INVALID_NEIGHBOR[] = "Searching for corner neighbor not supported";
static const char msg_IO_ERROR[] = "Low level IO error";
static const char msg_NOT_WRITABLE[] = "Etree is not opened as writable";
static const char msg_APPMETA_ERROR[] = "Unable to access application-specific meta data";
static const char msg_NO_SCHEMA[] = "No schema defined but request a particular filed";
static const char msg_NO_FIELD[] = "The reuqested field is not defined in the schema registered";
static const char msg_DISALLOW_SCHEMA[] = "A schema must be registered after an etree is newly created or truncated and before any insertion/appending operation";
static const char msg_CREATE_SCHEMA[] = "Attempt to register a schema failed";
/* static const char msg_CONTAIN_INTERIOR[] = "Contain interior nodes"; */
static const char msg_TOO_BIG[] = "Domain larger than the etree address space";
static const char msg_NOT_ALIGNED[] = "Left-lower corner not aligned";

/* Statistics routine */
static void updatestat(etree_t * ep, etree_addr_t addr, int mode);
int writemeta(etree_t *ep, off_t endoffset);

static int writeheader(etree_t *ep);
static int readheader(etree_t *ep);
static int storeappmeta(etree_t *ep, off_t endoffset);
static int loadappmeta(etree_t *ep);

/*
 * etree_straddr - Format a string representation of an octant address
 */
char *etree_straddr(etree_t *ep, char *buf, etree_addr_t addr)
{
    if (ep->dimensions == 3) { /* 3d case */
	sprintf(buf, "(%u %u %u %d)%c",
		addr.x, addr.y, addr.z, addr.level,
		addr.type == ETREE_INTERIOR ? 'I' : 'L');
    }
    else { /* 4d case */
	sprintf(buf, "[%u %u %u %u %d]%c",
		addr.x, addr.y, addr.z, addr.t, addr.level,
		addr.type == ETREE_INTERIOR ? 'I' : 'L');
    }
    return buf;
}

/*
 * etree_errno - Return the result error of the last operation
 *
 */
etree_error_t etree_errno(etree_t *ep)
{
    return ep->error;
}


/*
 * etree_strerror - Return a string describing the error of the latest 
 *                  operation
 *
 */
const char * etree_strerror(etree_error_t error)
{
    switch (error) {

    case(ET_NOERROR):
        return msg_NOERROR;

    case(ET_LEVEL_OOB):
        return msg_LEVEL_OOB;

    case(ET_LEVEL_OOB2):
        return msg_LEVEL_OOB2;

    case(ET_LEVEL_CHILD_OOB):
        return msg_LEVEL_CHILD_OOB;

    case(ET_OP_CONFLICT):
        return msg_OP_CONFLICT;

    case(ET_DUPLICATE):
        return msg_DUPLICATE;

    case(ET_NOT_FOUND):
        return msg_NOT_FOUND;

    case(ET_NO_MEMORY):
        return msg_NO_MEMORY;

    case(ET_NOT_LEAF_SPROUT):
        return msg_NOT_LEAF_SPROUT;

    case(ET_EMPTY_TREE):
        return msg_EMPTY_TREE;

    case(ET_NO_ANCHOR):
        return msg_NO_ANCHOR;
        
    case(ET_NO_CURSOR):
        return msg_NO_CURSOR;

    case(ET_NOT_APPENDING):
        return msg_NOT_APPENDING;

    case(ET_ILLEGAL_FILL):
        return msg_ILLEGAL_FILL;

    case(ET_APPEND_OOO):
        return msg_APPEND_OOO;

    case(ET_END_OF_TREE):
        return msg_END_OF_TREE;

    case(ET_NOT_NEWTREE):
        return msg_NOT_NEWTREE;

    case(ET_NOT_3D):
        return msg_NOT_3D;

    case(ET_CREATE_FAILURE):
        return msg_CREATE_FAILURE;

    case(ET_BOUNDARY_ERROR):
        return msg_BOUNDARY_ERROR;

    case(ET_INVALID_NEIGHBOR):
        return msg_INVALID_NEIGHBOR;

    case(ET_IO_ERROR):
        return msg_IO_ERROR;

    case(ET_NOT_WRITABLE):
        return msg_NOT_WRITABLE;

    case(ET_APPMETA_ERROR):
        return msg_APPMETA_ERROR;

    case(ET_NO_SCHEMA):
        return msg_NO_SCHEMA;

    case(ET_NO_FIELD):
        return msg_NO_FIELD;

    case(ET_CREATE_SCHEMA):
        return msg_CREATE_SCHEMA;

    case(ET_DISALLOW_SCHEMA):
        return msg_DISALLOW_SCHEMA;
        
    case(ET_TOO_BIG) :
        return msg_TOO_BIG;
        
    case (ET_NOT_ALIGNED):
        return msg_NOT_ALIGNED;

    default:
        return msg_UNKNOWN;
    }
}


/*
 * etree_open - Open or create an etree for operation
 * 
 * - "flags" is one of O_RDONLY or O_RDWR. flags may also be bitwise-or'd
 *    with O_CREAT or O_TRUNC. The semantics is the same as that in UNIX
 * - "bufsize" is the internal buffer space being allocated, specified in 
 *   terms of megabytes.
 * - Allocate and initialize control structure for manipulating the etree.
 * - Create and initialize I/O buffer
 * - Return pointer to etree_t if OK, NULL on error. Applications should
 *   invoke perror() to check the details for the error.
 *
 */
etree_t *etree_open(const char *pathname, int32_t flags, int32_t bufsize, 
                    int32_t payloadsize, int32_t dimensions)
{
    etree_t *ep;
    int32_t level;
    struct stat buf;
    int existed;
    uint32_t pagesize;
    const char *fullpathname;
    char pathbuf[PATH_MAX * 2];

    /* check the existence of the pathname */
    if (stat(pathname, &buf) == 0) 
        existed = 1;
    else{
        existed = 0;

        /* errno is freshly set by stat() */
        if (errno != ENOENT) 
            return NULL;
    }

    if (((flags & O_CREAT) == 0) && !existed) {
        /* O_CREAT is not specified but the file does not exist */
        fprintf(stderr, "etree_open: O_CREAT must be specified to open ");
        fprintf(stderr, "a non-existent etree\n");
        return NULL;
    }

    if (flags & O_TRUNC) {
        if (!((flags & O_WRONLY) || (flags & O_RDWR))) {
            fprintf(stderr, "etree_open: O_TRUNC must be specified with ");
            fprintf(stderr, "the open mode that allows writing (i.e., is ");
            fprintf(stderr, "O_WRONLY or O_RDWR\n");
            return NULL;
        }
    }

    /* allocate the control structure */
    if ((ep = (etree_t *)malloc(sizeof(etree_t))) == NULL) {
        /* perror("etree_open: malloc etree_t structure");*/
        return NULL;
    }
    memset(ep, 0, sizeof(etree_t));

    /* record the full path name of the etree */
    if (*pathname == '/') 
        fullpathname = pathname;
    else {
        int length;
        char *charptr;

        if (getcwd(pathbuf, PATH_MAX) == NULL) {
            /* perror("etree_open:getcwd"); */
            return NULL;
        }
        length = strlen(pathbuf);
        charptr = pathbuf + length;
        *(charptr) = '/';

        if (strlen(pathname) > PATH_MAX ) {
            fprintf(stderr, "etree_open: pathname too long\n");
            return NULL;
        }

        charptr++;
        strcpy(charptr, pathname);
        
        fullpathname = pathbuf;
    }
            
    /* absolute path to the tree name, strdup it */
    if ((ep->pathname = strdup(fullpathname)) == NULL) {
        /* perror("etree_open: strdup"); */
        return NULL;
    }

    ep->flags = flags;


    if (((flags & O_TRUNC) != 0) ||
        (((flags & O_CREAT) != 0) && (!existed))) {
        /* 
           Either O_TRUNC is specified or create a brand new file 
           initialize the Etree meta data 
        */        

        ep->endian = xplatform_testendian();
        ep->version = ETREE_VERSION;
        
        if ((dimensions < 1) || (dimensions > 4)) {
            fprintf(stderr, "etree_open: invalid dimension parameter.\n");
            fprintf(stderr, "etree_open: the dimension should be either 1, 2");
            fprintf(stderr, ", 3, or 4.\n");
            return NULL;
        }
        ep->dimensions = dimensions;

        ep->rootlevel = 0;   /* default value */

        for (level = 0; level <= (int)ETREE_MAXLEVEL; level++) 
            ep->leafcount[level] = ep->indexcount[level] = 0;

        ep->appmetasize = 0; /* default: no app meta defined */ 
        ep->appmetadata = NULL; 
    }
    else {
        /* open an existing etree */
        if (readheader(ep) != 0) {
            fprintf(stderr, "etree_open: corrupted meta data.\n");
            return NULL;
        }
    }


    /* init dynamic control fields */
    ep->keysize = ep->dimensions * sizeof(etree_tick_t) + 1;
    if (((ep->key = malloc(ep->keysize)) == NULL) ||
        ((ep->hitkey = malloc(ep->keysize)) == NULL)) {
        /* perror("etree_open: malloc temp key"); */
        return NULL;
    }
    

    /* init the pagesize of the underlying Btree if it's new/truncated*/
    pagesize = getpagesize();
    bufsize = (bufsize <= 0) ? DEFAULTBUFSIZE : bufsize;
    ep->bp = btree_open(fullpathname, flags, ep->keysize, "byte string",
                        payloadsize, pagesize, bufsize, code_comparekey,
                        HEADERSIZE);

    if (ep->bp == NULL) {
        /* perror("etree_open"); */
        fprintf(stderr, "etree_open: Fail to open B-tree\n"); 
        return NULL;
    }

    /* store the application meta data in memory if one is defined; the
       Btree might grow into the appmeta data region */
    if (ep->appmetasize > 0) {
        if (loadappmeta(ep) != 0) {
            fprintf(stderr, "etree_open: Fail to load appl. meta data.\n");
            return NULL;
        }
    }
       

    ep->error = ET_NOERROR;

    ep->searchcount = ep->insertcount = 0;
    ep->appendcount = ep->sproutcount = ep->deletecount = 0;
    ep->cursorcount = 0;

    return ep;
}


/*
 * etree_registerschema - register schema with the underlying Btree 
 *
 * - ERROR:
 *   ET_DISALLOW_SCHEMA
 *   ET_CREATE_SCHEMA
 */
int etree_registerschema(etree_t *ep, const char *defstring)
{
    int res;

    res = btree_registerschema(ep->bp, defstring);
    switch (res){
    case(-11) :
        ep->error = ET_DISALLOW_SCHEMA;
        return -1;
    case(-12) :
    case(-15) :
        ep->error = ET_CREATE_SCHEMA;
        return -1;
    default:
        return 0;
    }
}


/* 
 * etree_getschema - return the original ASCII schema definition string
 *
 */
char* etree_getschema(etree_t *ep)
{
    return btree_getschema(ep->bp);
}
    



/*
 * etree_close - close the etree handle
 *
 * - this is a more destructive function than others
 * - release resource no matter what error may occur
 * - write metadata if the etree was opened for write
 * - return 0 if OK, -1 on error; application programs should
 *   invoke perror() to identify the problem reported by the
 *   underlying system
 *
 */
int etree_close(etree_t *ep)
{
    off_t endoffset;

    free(ep->key);
    free(ep->hitkey);

    /* record the end of the etree */
    endoffset = btree_getendoffset(ep->bp);
    
    /* done with the underlying btree */
    if (btree_close(ep->bp) != 0) 
        return -1;

    if (((ep->flags & O_INCORE) == 0) &&
        ((ep->flags & O_RDWR) || (ep->flags & O_WRONLY)))
        if (writemeta(ep, endoffset) != 0) 
            return -1;
    
    free(ep->pathname); /*strdup'ed */
    if (ep->appmetadata != NULL)
        free(ep->appmetadata);
        
    free(ep);

    return 0;
}



/*
 * etree_insert - Insert an octant into etree
 *
 * - Convert the octant address to the locational key
 * - Insert the octant into the underlying B-tree
 * - Ignore duplicate and set the errno 
 * - "duplicate" refers to 
 *    an octant cannot be inserted twice
 *    an octant cannot be inserted as both ETREE_LEAF and ETREE_INTERIOR
 * - Return 0 if inserted, -1 otherwise; 
 * - ERROR:
 *   
 *    ET_LEVEL_OOB
 *    ET_OP_CONFLICT
 *    ET_DUPLICATE
 *    ET_IO_ERROR
 *    ET_NOT_WRITABLE
 *
 */
int etree_insert(etree_t *ep, etree_addr_t addr, const void *payload)
{
    int insed, res;

    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)) {
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }
                                         
    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    res = btree_insert(ep->bp, ep->key, payload, &insed);
    if (res != 0) {
        switch (res) {
        case(-1) : ep->error = ET_OP_CONFLICT; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

    if (insed == 0) {
        ep->error = ET_DUPLICATE;
        return -1;
    }

    updatestat(ep, addr, 1);

    ep->error = ET_NOERROR;
    ep->insertcount++;

    return 0;
}
    


/*
 * etree_search - Search an octant in the etree database
 *
 * - The search octant address "addr" does not need to specify the "type"
 * - Convert the search address to the locational key
 * - Search the key in the B-tree
 * - Convert the hit locational key back to hit octant address, which now
 *   contains the hit octant's type info
 * - Return 0 if found,  -1 if not found; 
 * - "found" is defined as 
 *   1) an octant with the exact same address is located
 *   2) an octant (at a higher octree level) that encloses the extent
 *      of the search octant is located; in this case, we call it
 *      ancestor hit.
 * - ERROR:
 *
 *    ET_LEVEL_OOB 
 *    ET_LEVEL_OOB2
 *    ET_EMPTY_TREE 
 *    ET_NOT_FOUND 
 *    ET_IO_ERROR
 *    ET_NO_SCHEMA
 *    ET_NO_FIELD
 *
 */
int etree_search(etree_t *ep, etree_addr_t addr, etree_addr_t *hitaddr, 
                 const char *fieldname, void *payload)
{
    etree_addr_t probeaddr, leafaddr;
    int res;

    ep->searchcount++;

    leafaddr = addr;
    leafaddr.type = ETREE_LEAF;

    if (code_addr2key(ep, leafaddr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    res = btree_search(ep->bp, ep->key, ep->hitkey, fieldname, payload);
    if (res != 0) {
        switch (res) {
        case(-2) : ep->error = ET_EMPTY_TREE; break;
        case(-3) : ep->error = ET_NOT_FOUND; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        case(-13) : ep->error = ET_NO_SCHEMA; break;
        case(-14) : ep->error = ET_NO_FIELD; break;
        }
        return -1;
    }

    if (ep->dimensions == 3) {
        if (!code_isancestorkey(ep->hitkey, ep->key)) {
            ep->error = ET_NOT_FOUND;
            return -1;
        } 
    } else {
        if (memcmp(ep->hitkey, ep->key, ep->keysize) != 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
    }
            
    if (code_key2addr(ep, ep->hitkey, &probeaddr) != 0) {
        ep->error = ET_LEVEL_OOB2; 
        return -1;
    }

    if (hitaddr != NULL) 
        *hitaddr = probeaddr;

    ep->error = ET_NOERROR;

    return 0;
}



/*
 * etree_findneighbor - Search for a neighbor in the etree database
 *
 * - Valid only for 3D
 * - Derive the neighbor at dir direction of current code 
 *    ( we are not interested in corner neighbors )
 * - The neighbor may be out of the etree address space. In this case,
 *   return not found as result
 * - Search the etree for the octant
 * - Return 0 if found, -1 if failed; 
 * - ERROR:
 *    
 *    ET_NOT_3D
 *    ET_INVALID_NEIGHBOR
 *    ET_LEVEL_OOB 
 *    ET_LEVEL_OOB2
 *    ET_EMPTY_TREE 
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 *    ET_NO_SCHEMA
 *    ET_NO_FIELD
 *
 */
int etree_findneighbor(etree_t *ep, etree_addr_t addr, etree_dir_t dir,
                       etree_addr_t *nbaddr, const char *fieldname, 
                       void *payload)
{
    etree_addr_t probeaddr;
    etree_tick_t size;

    if (ep->dimensions != 3) {
        ep->error = ET_NOT_3D;
        return -1;
    }

    if (addr.level > (int)ETREE_MAXLEVEL) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }
        

    size = (etree_tick_t)1 << (ETREE_MAXLEVEL - addr.level);
    probeaddr = addr;
    probeaddr.level = ETREE_MAXLEVEL;

    switch (dir){
    case(d_L) : 
        if (probeaddr.x == 0) {
            /* out of the domain */
            ep->error = ET_NOT_FOUND;
            return -1;
        } 

        probeaddr.x--; 
        break;

    case(d_R) : 
        probeaddr.x += size; 
        break;

    case(d_D) : 
        if (probeaddr.y == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }

        probeaddr.y--; 
        break;

    case(d_U) : 
        probeaddr.y += size; 
        break;

    case(d_B) : 
        if (probeaddr.z == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.z--; 
        break;
        
    case(d_F) :
        probeaddr.z += size;
        break;

    case(d_LB):
        if ((probeaddr.x == 0) || (probeaddr.z == 0)) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.x--;
        probeaddr.z--;
        break;

    case(d_UB) :
        if (probeaddr.z == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.y += size;
        probeaddr.z--;
        break;

    case(d_RB) :
        if (probeaddr.z == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }

        probeaddr.x += size;
        probeaddr.z--;
        break;

    case(d_DB) :
        if ((probeaddr.y == 0) || (probeaddr.z == 0)) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.y--;
        probeaddr.z--;
        break;
        
    case(d_LF) :
        if (probeaddr.x == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.x--;
        probeaddr.z += size;
        break;
        
    case(d_UF):
        probeaddr.y += size;
        probeaddr.z += size;
        break;

    case(d_RF):
        probeaddr.x += size;
        probeaddr.z += size;
        break;
        
    case(d_DF):
        if (probeaddr.y == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.y--;
        probeaddr.z += size;
        break;

    case(d_LD):
        if ((probeaddr.x == 0) || (probeaddr.y == 0)) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.x--;
        probeaddr.y--;
        break;

    case(d_LU):
        if (probeaddr.x == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.x--;
        probeaddr.y += size;
        break;
        
    case(d_RU):
        probeaddr.x += size;
        probeaddr.y += size;
        break;
        
    case(d_RD):
        if (probeaddr.y == 0) {
            ep->error = ET_NOT_FOUND;
            return -1;
        }
        probeaddr.y--;
        probeaddr.x += size;
        break;

    default:
        ep->error = ET_INVALID_NEIGHBOR;
        return -1;
    }

    return etree_search(ep, probeaddr, nbaddr, fieldname, payload);
}



/*
 * etree_sprout - Sprout a leaf octant into eight children
 *
 * - Only valid for 3D
 * - Check the sprouting addr is a leaf octant  
 * - Remove the sprouting addr from the etree;
 * - Derive the children's address from the sprouting octant
 * - The childpayload array is assumed to hold the payload for the eight 
 *   children in Z-order
 * - Return 0 if OK, -1 if failed; 
 * - ERRORS:
 * 
 *    ET_NOT_3D
 *    ET_NOT_LEAF_SPROUT
 *    ET_LEVEL_OOB
 *    ET_NO_MEMORY
 *    ET_LEVEL_CHILD_OOB
 *    ET_NO_ANCHOR
 *    ET_EMPTY_TREE
 *    ET_OP_CONFLICT
 *    ET_IO_ERROR
 *    ET_NOT_WRITABLE
 *
 */
int etree_sprout(etree_t *ep, etree_addr_t addr, const void *childpayload[8])
{
    int index, res;
    void *keyset[8];

    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)){
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }

    if (ep->dimensions != 3) {
        ep->error = ET_NOT_3D;
        return -1;
    }

    if (addr.type != ETREE_LEAF) {
        ep->error = ET_NOT_LEAF_SPROUT;
        return -1;
    }

    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    for (index = 0; index < 8; index++) {
        if ((keyset[index] = malloc(ep->keysize)) == NULL) {
            ep->error = ET_NO_MEMORY;
            return -1;
        }
        if (code_derivechildkey(ep->key, keyset[index], index) != 0) {
            ep->error = ET_LEVEL_CHILD_OOB;
            return -1;
        }
    }

    res = btree_bulkupdate(ep->bp, ep->key, 8, (const void **)keyset, 
                           childpayload);

    for (index = 0; index < 8; index++) 
        free(keyset[index]);

    if (res != 0) {
        switch(res) {
        case(-1): ep->error = ET_OP_CONFLICT; break;
        case(-2): ep->error = ET_EMPTY_TREE; break;
        case(-3): ep->error = ET_NO_ANCHOR; break;
        case(-4): /* This should never happen */ break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }
    
    updatestat(ep, addr, 2);
    ep->error = ET_NOERROR;
    ep->sproutcount++;

    return 0;
}



/*
 * etree_delete - Delete an octant from the etree
 *
 * - Lazy method: NO merge of underflow B-tree nodes is done
 * - Return 0 if successfully deleted, -1 otherwise
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_OP_CONFLICT
 *    ET_EMPTY_TREE
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 *
 */
int etree_delete(etree_t *ep, etree_addr_t addr)
{
    int res;

    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)) {
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }

    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    res = btree_delete(ep->bp, ep->key);

    if (res != 0) {
        switch(res){
        case(-1): ep->error = ET_OP_CONFLICT; break;
        case(-2): ep->error = ET_EMPTY_TREE; break;
        case(-3): ep->error = ET_NOT_FOUND; break;
        }
        return -1;
    }

    updatestat(ep, addr, -1);
    ep->error = ET_NOERROR;
    ep->deletecount++;

    return 0;
}


/*
 * etree_update - Modify the content/payload of an octant in the etree
 *
 * - Update the payload for the object with exact hit
 * - Return 0 if updated, -1 otherwise
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_EMPTY_TREE
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 *    ET_NOT_WRITABLE
 *
 */
int etree_update(etree_t *ep, etree_addr_t addr, const void *payload)
{
    int res;

    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)) {
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }

    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    
    res = btree_update(ep->bp, ep->key, payload);

    if (res != 0) {
        switch(res){
        case(-2) : ep->error = ET_EMPTY_TREE; break;
        case(-3) : ep->error = ET_NOT_FOUND; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

    ep->error = ET_NOERROR;
    return 0;
}



/*
 * etree_initcursor - Set the cursor in the etree for preorder traversal 
 *
 * - The preorder traversal starts from octant with addr
 * - To start from the first octant in the etree, set the fields of addr
 *   to 0. That is addr.x = addr.y = addr.z = addr.level = 0
 * - Return 0 if OK, -1 on error
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_EMPTY_TREE
 *    ET_IO_ERROR
 *
 */
int etree_initcursor(etree_t *ep, etree_addr_t addr)
{
    int res;

    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    res = btree_initcursor(ep->bp, ep->key);

    if (res != 0) {
        switch (res) {
        case(-2) :  ep->error = ET_EMPTY_TREE; break;
        case(-9) :  ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

    ep->error = ET_NOERROR;
    return 0;
}


/*
 * etree_getcursor - Obtain the content of the octant currently pointed to
 *                   by the cursor
 *
 * - retrieve the cursor from B-tree
 * - convert the locational key properly
 * - return 0 if OK, -1 otherwise
 * - ERROR:
 *
 *    ET_NO_CURSOR:
 *    ET_LEVEL_OOB2
 *    ET_NO_SCHEMA
 *    ET_NO_FIELD
 *
 */
int etree_getcursor(etree_t *ep, etree_addr_t *addr, const char *fieldname,
                    void *payload)
{
    int res;

    res = btree_getcursor(ep->bp, ep->key, fieldname, payload);

    if (res != 0) {
        switch (res) {
        case(-5) : ep->error = ET_NO_CURSOR; break;
        case(-13) : ep->error = ET_NO_SCHEMA; break;
        case(-14) : ep->error = ET_NO_FIELD; break;
        }
        
        return -1;
    }

    if (code_key2addr(ep, ep->key, addr) != 0) {
        ep->error = ET_LEVEL_OOB2;
        return -1;
    }


    ep->error = ET_NOERROR;
    ep->cursorcount++;
    return 0;
}
    
    
/*
 * etree_advcursor - Move the cursor to the next octant in pre-order 
 *
 * - wrapper function to call advcursor in the underlying B-tree
 * - return 0 if OK, -1 otherwise
 * - ERROR:
 *
 *    ET_NO_CURSOR
 *    ET_END_OF_TREE
 *    ET_IO_ERROR
 *
 */
int etree_advcursor(etree_t *ep)
{
    int res;

    res = btree_advcursor(ep->bp);
    
    if (res != 0) {
        switch(res){
        case(-5) : ep->error = ET_NO_CURSOR; break;
        case(1) : ep->error = ET_END_OF_TREE; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

    ep->error = ET_NOERROR;    
    return 0;
}


/*
 * etree_stopcursor - Stop the cursor operation
 *
 * - wrapper function
 * - return 0 if OK, -1 otherwise
 * - ERROR:
 *
 *    ET_NO_CURSOR
 *
 */
int etree_stopcursor(etree_t *ep)
{
    if (btree_stopcursor(ep->bp) != 0) {
        ep->error = ET_NO_CURSOR;
        return -1;
    }

    ep->error = ET_NOERROR;
    return 0;  
}


/*
 * etree_beginappend - Start a transcation of appending octant in preorder
 *
 * - wrapper function 
 * - return 0 if OK,  -1 otherwise
 * - ERROR:
 *   
 *    ET_ILLEGAL_FILL
 *    ET_IO_ERROR
 *
 */
int etree_beginappend(etree_t *ep, double fillratio)
{
    int res;

    res = btree_beginappend(ep->bp, fillratio);

    if (res != 0) {
        switch (res) {
        case(-6) : ep->error = ET_ILLEGAL_FILL; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

    ep->error = ET_NOERROR;
    return 0;
}
    

/*
 * etree_append - Append an octant to the end of the etree
 *
 * - wrapper function 
 * - return 0 if OK, -1 otherwise
 * - ERROR:
 *
 *    ET_LEVEL_OOB:
 *    ET_APPEND_OOO:
 *    ET_NOT_WRITABLE
 *    ET_IO_ERROR
 *
 */
int etree_append(etree_t *ep, etree_addr_t addr, const void *payload)
{
    int res;

    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)) {
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }

    if (code_addr2key(ep, addr, ep->key) != 0) {
        ep->error = ET_LEVEL_OOB;
        return -1;
    }

    res = btree_append(ep->bp, ep->key, payload);

    if (res != 0) {
        switch (res) {
        case(-8) : ep->error = ET_APPEND_OOO; break;
        case(-9) : ep->error = ET_IO_ERROR; break;
        }
        return -1;
    }

     
    updatestat(ep, addr, 1);
    ep->error = ET_NOERROR;
    ep->appendcount++;

    return 0;
}
    

/*
 * etree_endappend - Terminate the appending transaction
 *
 * - return 0 if OK, -1 on error
 * - ERROR:
 *
 *    ET_NOT_APPENDING:
 *
 */
int etree_endappend(etree_t *ep)
{
    if (btree_endappend(ep->bp) != 0) {
        ep->error = ET_NOT_APPENDING;
        return -1;
    }

    ep->error = ET_NOERROR;
    return 0;
}


/*
 * etree_getmaxleaflevel - Return the max leaf level in the etree
 *
 */
int etree_getmaxleaflevel(etree_t *ep)
{
    int level;

    for (level = ETREE_MAXLEVEL; level >= 0; level--) {
        if (ep->leafcount[level] != 0)
            return level;
    }
    
    return -1;
}


/*
 * etree_getminleaflevel - Return the min leaf level in the etree
 *
 */
int etree_getminleaflevel(etree_t *ep)
{
    int level;

    for (level = 0; level <= (int)ETREE_MAXLEVEL; level++) {
        if (ep->leafcount[level] != 0) 
            return level;
    }
    return -1;
}


/*
 * etree_getavgleaflevel - Return the average leaf level in the etree
 *
 */
float etree_getavgleaflevel(etree_t *ep)
{
    double totallevels;
    uint64_t totalcount;
    int level;

    totalcount = 0;
    totallevels = 0;

    for (level = 0; level <= (int)ETREE_MAXLEVEL; level++ ) {
        totallevels += level * ep->leafcount[level];
        totalcount += ep->leafcount[level];
    }
    return (float)(totallevels / totalcount);
}

/**
 * etree_gettotalcount 
 *
 */
uint64_t etree_gettotalcount(etree_t *ep)
{
    int level;
    uint64_t totalcount;

    totalcount = 0;
    for (level = 0; level <= (int)ETREE_MAXLEVEL; level++ ) {
        totalcount += ep->leafcount[level];
        totalcount += ep->indexcount[level];
    }

    return totalcount;
}
    

/**
 * etree_isempty
 *
 * return 1 if true, 0 if false
 */
int etree_isempty(etree_t *ep)
{
    return btree_isempty(ep->bp);
}


/*
 * etree_hasleafonly
 *
 */
int etree_hasleafonly(etree_t *ep)
{
    int level;

    for (level = 0; level <= (int)ETREE_MAXLEVEL; level++) {
        if (ep->indexcount[level] != 0) 
            return 0;
    }
    
    for (level = 0; level <= (int)ETREE_MAXLEVEL; level++) {
        if (ep->leafcount[level] != 0) 
            return 1;
    }
    
    return 0; /* empty */
}

/*
 * etree_getpayloadsize
 *
 */
int etree_getpayloadsize(etree_t *ep)
{
    return btree_getvaluesize(ep->bp);
}

/* 
 * etree_getkeysize
 *
 */
int etree_getkeysize(etree_t *ep)
{
    return ep->keysize;
}

   



/*
 * etree_getappmeta - get the application meta data string
 *
 * return a pointer to (an allocated) meta data string if application meta
 * data is defined, NULL if no application meta data is defined
 *
 */
char *etree_getappmeta(etree_t *ep)
{
    char *appmetadata;

    if (ep->appmetasize == 0) {
        ep->error = ET_APPMETA_ERROR;
        return NULL;
    } 
    else {
        appmetadata = strdup(ep->appmetadata);
        if (appmetadata == NULL) {
            ep->error = ET_NO_MEMORY;
            return NULL;
        } else 
            return appmetadata;
    }
}

/* 
 * etree_setappmeta - set the application meta data 
 *
 * return 0 if OK, -1 on error
 *
 */
int etree_setappmeta(etree_t *ep, const char *appmetadata)
{
    
    if (((ep->flags & O_RDWR) == 0) &&
        ((ep->flags & O_WRONLY) == 0)) {
        ep->error = ET_NOT_WRITABLE;
        return -1;
    }

    if (ep->appmetadata != NULL) {
        free(ep->appmetadata);
        ep->appmetasize = 0;
    }

    ep->appmetadata = strdup(appmetadata);
    if (ep->appmetadata == NULL) {
        ep->error = ET_NO_MEMORY;
        return -1;
    } else {
        /* application meta data successfully strdup'ed 
           +1 for the trailing NULL 
        */
        ep->appmetasize = strlen(appmetadata) + 1;
        return 0;
    }
}
        
        
    

        
/*-------------------
 *
 *  Local routines 
 *
 *-------------------
 */

/*
 * updatestat - Update the etree statistics according the mode
 *
 * - Mode:
 *  1: insert or append addr to the etree
 *  2: sprout octant
 *  -1: delete octant
 *
 * - Assume addr has a valid level range, which should have been checked 
 *   earlier
 *
 */
void updatestat(etree_t * ep, etree_addr_t addr, int mode)
{
    switch (mode) {
    case(1) :
        if (addr.type == 0) 
            ep->indexcount[addr.level]++;
        else 
            ep->leafcount[addr.level]++;
        break;
    case(2) : /* Address type must be 1 (leaf ) */
        ep->leafcount[addr.level]--;
        ep->leafcount[addr.level + 1] += 8;
        break;

    case(-1):
        if (addr.type == 0) 
            ep->indexcount[addr.level]--;
        else 
            ep->leafcount[addr.level]--;
        break;

    default:
        /* this should never occur */
        break;
    }
    return;
}



/*
 * writeheader - write the meta data to the etree header 
 *
 * - return 0 if OK, -1 on error
 */
int writeheader(etree_t *ep)
{
    int etreefd;
    uint32_t version, dimensions, rootlevel, appmetasize;
    BIGINT level;
    BIGINT leafcount[ETREE_MAXLEVEL + 1];
    BIGINT indexcount[ETREE_MAXLEVEL + 1];

    /* convert the header data if byte swapping is necessary */
    if (xplatform_testendian() != ep->endian) {
        xplatform_swapbytes(&version, &ep->version, 4);
        xplatform_swapbytes(&dimensions, &ep->dimensions, 4);
        xplatform_swapbytes(&rootlevel, &ep->rootlevel, 4);
        xplatform_swapbytes(&appmetasize, &ep->appmetasize, 4);

        for (level = 0; level <=  ETREE_MAXLEVEL; level++) {
            xplatform_swapbytes(&leafcount[level], &ep->leafcount[level],
                                sizeof(BIGINT));
            xplatform_swapbytes(&indexcount[level], &ep->indexcount[level],
                                sizeof(BIGINT));
        }
    } else {
        version = ep->version;
        dimensions = ep->dimensions;
        rootlevel = ep->rootlevel;
        appmetasize = ep->appmetasize;
        
        for (level = 0; level <= ETREE_MAXLEVEL; level++) {
            leafcount[level] = ep->leafcount[level];
            indexcount[level] = ep->indexcount[level];
        }
    }

    /* write meta data to the etree header */
    etreefd = open(ep->pathname, O_WRONLY);
    if (etreefd == -1) {
        fprintf(stderr, "writeheader: open etree file\n");
        return -1;
    }
 
    if ((write(etreefd, (ep->endian == little) ? "L" : "B", 1) != 1) ||
        (write(etreefd, &version, 4) != 4) ||
        (write(etreefd, &dimensions, 4) != 4) ||
        (write(etreefd, &rootlevel, 4) != 4) ||
        (write(etreefd, &appmetasize, 4) != 4)) {
        fprintf(stderr, "writeheader: write\n");
        return -1;
    }
    
    
    for (level = 0; level <= ETREE_MAXLEVEL; level++) {
        if (write(etreefd, &leafcount[level], sizeof(BIGINT)) 
            != sizeof(BIGINT)) {
            fprintf(stderr, "writeheader: write\n");
            return -1;
        }
        if (write(etreefd, &indexcount[level], sizeof(BIGINT))
            != sizeof(BIGINT)) {
            fprintf(stderr, "writeheader: write\n");
            return -1;
        }
    }
    
    if (close(etreefd) != 0) {
        fprintf(stderr, "writeheader: close\n");
        return -1;
    }
    
    return 0;
}



              
/*
 * readheader - read the etree header from the beginning of the etree file
 *   
 * - return 0 if OK, -1 on error
 */
int readheader(etree_t *ep)
{
    int etreefd;
    char endianchar;
    uint32_t version, dimensions, rootlevel, appmetasize;
    BIGINT level;
    BIGINT leafcount[ETREE_MAXLEVEL + 1];
    BIGINT indexcount[ETREE_MAXLEVEL + 1];

    int bytesread;

    etreefd = open(ep->pathname, O_RDONLY);
    if (etreefd == -1) {
        fprintf(stderr, "readheader: open etree file\n");
        return -1;
    }
    

    bytesread = read(etreefd, &endianchar, 1);
    if (bytesread != 1) {
        perror("readheader: read endianchar");
        fprintf(stderr, "bytesread = %d\n", bytesread);
        fprintf(stderr,"readheader: read header (endian)\n");
        return -1;
    }
        
    /*
    if (read(etreefd, &endianchar, 1) != 1) {
        fprintf(stderr,"readheader: read header (endian)\n");
        return -1;
        }*/


    if (read(etreefd, &version, 4) != 4) {
        fprintf(stderr,"readheader: read header (version)\n");
        return -1;
    }

    if (read(etreefd, &dimensions, 4) != 4) {
        fprintf(stderr,"readheader: read header (dimension)\n");
        return -1;
    }

    if (read(etreefd, &rootlevel, 4) != 4) {
        fprintf(stderr,"readheader: read header (rootlevel)\n");
        return -1;
    }
    if (read(etreefd, &appmetasize, 4) != 4) {
        fprintf(stderr,"readheader: read header (appmetasize)\n");
        return -1;
    }
    
    for (level = 0; level <= ETREE_MAXLEVEL; level++) {
        if (read(etreefd, &leafcount[level], sizeof(BIGINT)) 
            != sizeof(BIGINT)) {
            fprintf(stderr, "readheader: read header (leafcount[%d])", level);
            return -1;
        }
        if (read(etreefd, &indexcount[level], sizeof(BIGINT))
            != sizeof(BIGINT)) {
            fprintf(stderr, "readheader: read header (indexcount[%d])", level);
            return -1;
        }
    }
    
    /* load the data into runtime control structure */
    if (endianchar == 'L') 
        ep->endian = little;
    else if (endianchar == 'B')
        ep->endian = big;
    else {
        fprintf(stderr, "readheader: corrupted meta(endian) : %c\n", 
                endianchar);
        return -1;
    }

    if (xplatform_testendian() != ep->endian) {
        xplatform_swapbytes(&ep->version, &version, 4);
        xplatform_swapbytes(&ep->dimensions, &dimensions, 4);
        xplatform_swapbytes(&ep->rootlevel, &rootlevel, 4);
        xplatform_swapbytes(&ep->appmetasize, &appmetasize, 4);

        for (level = 0; level <= ETREE_MAXLEVEL; level++) {
            xplatform_swapbytes(&ep->leafcount[level],&leafcount[level], 
                                sizeof(BIGINT));
            xplatform_swapbytes(&ep->indexcount[level],&indexcount[level], 
                                sizeof(BIGINT));
        }

    } else {
        ep->version = version;
        ep->dimensions = dimensions;
        ep->rootlevel = rootlevel;
        ep->appmetasize = appmetasize;

        for (level = 0; level <= ETREE_MAXLEVEL; level++) {
            ep->leafcount[level] = leafcount[level];
            ep->indexcount[level] = indexcount[level];
        }

    }

    if (ep->version != ETREE_VERSION) {
        fprintf(stderr, "readheader: incompatible library version\n");
        return -1;
    }

    if (close(etreefd) != 0) {
        fprintf(stderr, "readheader: close etree file\n");
        return -1;
    }

    return 0;
}


/*
 * writemeta - write the meta data to the etree header and trailer 
 *             appropriately
 *
 * - return 0 if OK, -1 on error
 */
int writemeta(etree_t *ep, off_t endoffset)
{

    if ((writeheader(ep) == 0) &&
        (storeappmeta(ep, endoffset) == 0)) 
        return 0;
    else
        return -1;
}
        
/*
 * storeappmeta - store application meta data to etree trailer
 *
 * - return 0 if OK, -1 on error
 *
 */
int storeappmeta(etree_t *ep, off_t endoffset)
{
    int etreefd;
    
    /* Note: use the host/platform variable */
    if (ep->appmetasize == 0) 
        /* no application meta data defined */
        return 0;
    
    etreefd = open(ep->pathname, O_WRONLY);
    if (etreefd == -1) {
        fprintf(stderr, "storeappmeta: open etree file\n");
        return -1;
    }

    if (lseek(etreefd, endoffset, SEEK_SET) != endoffset) {
        fprintf(stderr, "storeappmeta: lseek\n");
        return -1;
    }

    if (write(etreefd, ep->appmetadata, ep->appmetasize) != 
        (int)ep->appmetasize) {
        fprintf(stderr, "storeappmeta: write\n");
        return -1;
    }
    
    if (close(etreefd) != 0) {
        fprintf(stderr, "storeappmeta: close\n");
        return -1;
    }

    return 0;
}
    
     


/*
 * loadappmeta - load application meta data from etree into memory
 *
 * - return 0 if OK, -1 on error
 *
 */
int loadappmeta(etree_t *ep)
{
    int etreefd;
    off_t endoffset;

    ep->appmetadata = (char *)malloc(ep->appmetasize);
    if (ep->appmetadata == NULL) {
        fprintf(stderr, "loadappmeta: malloc ascii schema\n");
        return -1;
    }

    etreefd = open(ep->pathname, O_RDONLY);
    if (etreefd == -1) {
        fprintf(stderr, "loadappmeta: open etree file\n");
        return -1;
    }

    endoffset = btree_getendoffset(ep->bp);
    if (lseek(etreefd, endoffset, SEEK_SET) != endoffset) {
        fprintf(stderr, "loadappmeta: lseek etree file\n");
        return -1;
    }

    if (read(etreefd, ep->appmetadata, ep->appmetasize) != 
        (int)ep->appmetasize) {
        fprintf(stderr, "loadappmeta: read application meta data\n");
        return -1;
    }
    
    if (close(etreefd) != 0) {
        fprintf(stderr, "loadappmeta: close etree file\n");
        return -1;
    }

    return 0;
}
    

    
