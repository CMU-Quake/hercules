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
 * etree.h - Header file for the etree.c library
 *
 */

#ifndef ETREE_H
#define ETREE_H

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "xplatform.h"
#include "btree.h"


#define ETREE_VERSION	1

/**
 * ETREE_MAXLEVEL - Maximum tree level (zero origin) supported by
 * this version of the etree library
 */
#define ETREE_MAXLEVEL	(sizeof(etree_tick_t) * 8 - 1)


/**
 * ETREE_MAXBUF - Maximum size (in bytes) for a buffer
 * passed to the etree_straddr function.
 */
#define ETREE_MAXBUF 8192

/**
 * etree_tick_t - Etree address space coordinate value
 * 
 * The etree address space is a 2^31 x 2^31 x 2^31 uniform grid if
 * etree_tick_t is defined as uint32_t (32 bits).  The etree address
 * space is a 2^63 x 2 ^ 63 x 2 ^ 63 uniform grid if etree_tick_t is
 * defined as uint64_t (64 bits).
 *
 */
typedef uint32_t etree_tick_t;


/**
 * BIGINT - Data type to store stats of octants at different levels
 * Always set to be the same as etree_tick_t
 */
typedef etree_tick_t BIGINT;

/**
 * etree_dir_t: Different directions for neighbor finding and other ops
 *
 * DON'T change this
 */
typedef enum etree_dir_t {  

    /* corners */
    d_LDB = 0, d_RDB, 
    d_LUB, d_RUB,
    d_LDF, d_RDF,
    d_LUF, d_RUF,

    /* faces */
    d_L, d_R,              /* Left and right along X axis              */
    d_D, d_U,              /* Down and up along Y axis                 */
    d_B, d_F,              /* Back and front along Z axis              */

    /* edges */
    d_LD, d_LU, d_LB, d_LF,
    d_RD, d_RU, d_RB, d_RF, 
    d_DB, d_DF, d_UB, d_UF, 
 
    /* internal */
    d_IN,

    /* external */
    d_OUT,

    /* illegal case */
    d_OMEGA, 

} etree_dir_t;


/**
 * etree_type_t - Octants are either leaf or interior nodes
 */
/* $begin etreeaddrt */
typedef enum { ETREE_INTERIOR = 0, ETREE_LEAF } etree_type_t;

/* $end etreeaddrt */
/*
 * etree_addr_t: Octant address structure for application programs
 * 
 * Each octant is identified by its location in the etree address space, 
 * its size, and its type. 
 *  - The location is identified by the coordindate of the octant's left 
 *    lower corner, that is, the corner whose coordinate has the minimum 
 *    value along each dimension among the 8 corners. 
 *  - The size of the octant is specified by its level in the octree.
 *  - the type of the octant is either interior or leaf
 *
 * The timestep is valid only when working with 4d dataset.
 */
/* $begin etreeaddrt */
typedef struct etree_addr_t {
    etree_tick_t x, y, z;   /* (x, y, z, t) is lower left corner */
    etree_tick_t t;         /* Time dimension for 4D etrees */
    int level;              /* Octant level  */
    etree_type_t type;      /* ETREE_LEAF or ETREE_INTERIOR */
} etree_addr_t;
/* $end etreeaddrt */

/**
 * etree_error_t - Error number enumeration
 */
typedef enum {
    ET_NOERROR,              /* No error occurs                      */
    ET_LEVEL_OOB,            /* Octant level out of bound            */
    ET_LEVEL_OOB2,           /* Returns an octant from etree oob     */
    ET_LEVEL_CHILD_OOB,      /* Sprout causes the child's level oob  */
    ET_OP_CONFLICT,          /* In conflict with append or curosr ops*/
    ET_EMPTY_TREE,           /* The etree is empty                   */
    ET_DUPLICATE,            /* Attempt to insert an existent octant */
    ET_NOT_FOUND,            /* Fail to find the octant              */
    ET_NO_MEMORY,            /* Out of memory                        */
    ET_NOT_LEAF_SPROUT,      /* Sprouting point is not a leaf octant */
    ET_NO_ANCHOR,            /* Cannot find the spouting loc in etree*/
    ET_NO_CURSOR,            /* No curosr in effect                  */
    ET_NOT_APPENDING,        /* Etree not in appending mode          */
    ET_ILLEGAL_FILL,         /* Invalid fill ratio                   */
    ET_APPEND_OOO,           /* Appending an octant out of order     */
    ET_END_OF_TREE,          /* Already at the end of the tree       */
    ET_NOT_3D,               /* The etree is not storing an octree   */
    ET_INVALID_NEIGHBOR,     /* Neighbor in the direction not supported */
    ET_IO_ERROR,             /* Low level IO error                   */
    ET_NOT_WRITABLE,         /* Etree not opened as writable         */
    ET_NOT_NEWTREE,          /* The etree is not open for O_CREAT    */
    ET_CREATE_FAILURE,       /* Cannot create boundary etree         */
    ET_OCTREE_FAILURE,       /* Cannot rebuild incore octree image   */
    ET_BOUNDARY_ERROR,       /* Cannot record boundary octants       */
    ET_APPMETA_ERROR,        /* Unable to access app. meta data      */
    ET_NO_SCHEMA,            /* No schema defined                    */
    ET_NO_FIELD,             /* No such field in the schema defined  */
    ET_DISALLOW_SCHEMA,      /* Cannot register schema for this etree*/
    ET_CREATE_SCHEMA,        /* Error when register a schema         */
    ET_CONTAIN_INTERIOR,     /* Contain interior nodes               */
    ET_TOO_BIG,              /* Larger than etree address space      */
    ET_NOT_ALIGNED,          /* Left-lower corner not aligned        */

} etree_error_t;


/**
 * etree_t - Runtime etree handle
 * 
 * The etree_t structure encapsulates the specification of the octree
 * being manipulated and the underlying storage facilities. Application 
 * programmers should be careful to maintain the etree_t abstraction. Use
 * access functions to get the content of etree_t. Avoid directly 
 * referencing the fields of etree_t because they are subject to change
 * in the future. 
 */
typedef struct etree_t {
    char *pathname;          /* Name of the etree database            */
    int flags;               /* Open mode (flag) for the etree        */

    /************************************************/
    /* Persistent fields that enter the meta header */
    /************************************************/
    endian_t endian;         /* Format of the etree                      */

    uint32_t version;        /* For compatibility with etree library     */
    uint32_t dimensions;     /* Dimensionality of the etree   (NEW)      */
    uint32_t rootlevel;      /* The level of the octree root             */
    uint32_t appmetasize;    /* size of the application meta data (ASCII)*/

    BIGINT leafcount[ETREE_MAXLEVEL + 1]; 
                             /* Number of leaf octants at each level     */
    BIGINT indexcount[ETREE_MAXLEVEL + 1];   
                             /* Number of index octants at each level    */


    /********************************************/
    /* Dynamic fields instantiated at runntime  */
    /********************************************/
    char *appmetadata;       /* pointer to the application meta data     */

    int32_t keysize;         /* size of the locational key            */
    void *key;               /* Locational key for all operations     */
    etree_addr_t addr;       /* The correspoding octant address       */
    void *hitkey;            /* Store the key returned from search    */
    btree_t *bp;             /* Handle to the underlying B-tree       */

    etree_error_t error;     /* Status of the lastest operation       */
    
    uint64_t searchcount;    /* Number of searches in this session    */
    uint64_t insertcount;    /* Number of inserts in this session     */
    uint64_t appendcount;    /* Number of appends in this session     */
    uint64_t sproutcount;    /* Number of sprouts in this session     */
    uint64_t deletecount;    /* Number of deletes in this session     */
    uint64_t cursorcount;    /* Number of cursor octant retrieved     */

    
} etree_t;


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/*
 * Error reporting functions
 */

/** 
 * etree_errno - get the error code of the last failed operation.
 *
 * @param ep handle to the etree for which to retrieve the error code.
 *
 * @return error code of the last failed operation.
 */
etree_error_t etree_errno(etree_t* ep);

/**
 * etree_strerror - a text string describing the error corresponding
 * to a given error code.
 *
 * @param error code returned by the etree_errno() function.
 *
 * @return a text message corresding to the given error code.
 */
const char* etree_strerror(etree_error_t error);


/*
 * Initialization and cleanup 
 */

/**
 * etree_open - Open or create an etree for operation.
 *
 * Allocates and initializes a control structure to manipulate an etree.
 * Creates and initializes the I/O buffer.
 *
 * @param path  The name of the etree file in the file system.
 * @param flags specifies the mode in which to open an etree file.
 *     It is one of O_RDONLY or O_RDWR.
 *     Flags may also be bitwise-or'd with O_CREAT or O_TRUNC. The
 *     semantics are the same as that in UNIX
 * @param bufsize specifies the size of the internal buffer allocated to cache
 *     etree pages.  The size is specified in megabytes.
 * @param payloadsize: The size of the associated octant data (i.e.,
 *     record).  This parameter is only used to created a new etree
 *     database (i.e., O_CREAT, O_TRUNC was specified), otherwise it is
 *     ignored.
 * @param dimensions:  The dimensionality for the new etree database. 
 *     This parameter is only used to created a new etree database (i.e.,
 *     O_CREAT, O_TRUNC was specified), otherwise it is ignored.
 *
 * @return a pointer to an etree_t if OK, NULL on error. Applications should
 *     invoke perror("etree_open") to check the details for the error.
 */
etree_t *etree_open(const char *path, int32_t flags, int32_t bufsize, 
                    int32_t payloadsize, int32_t dimensions);

/**
 * etree_registerschema - register schema with an etree
 *
 * Schema can only be registerd once when the etree is either newly created
 * or truncated
 *
 * return 0 if OK, -1 on error
 *
 * - ERROR:
 *   ET_DISALLOW_SCHEMA
 *   ET_CREATE_SCHEMA
 * 
 */
int etree_registerschema(etree_t *ep, const char *defstring);


/**
 * etree_getschema - get the ASCII schema definition string
 *
 * return a pointer to the (allocated) schema string, NULL if no schema
 * defined
 *
 */
char *etree_getschema(etree_t *ep);
 


/**
 * etree_close - close an etree file.
 *
 * Release the resources used to access the etree file.
 *
 * @param ep etree handle to close.
 *
 * @return 0 on success, -1 on error.  Application programs should call
 *     etree_errno() and etree_strerror() to identify the nature of the error.
 */
int etree_close(etree_t *ep);


/*
 * Direct octant access methods
 */

/**
 * etree_insert - Insert an octant into an etree.
 *
 * - Convert the octant address to the locational key
 * - Insert the octant into the underlying B-tree
 * - Ignore duplicate and set the errno 
 * - "duplicate" refers to 
 *     an octant cannot be inserted twice
 *     an octant cannot be inserted as both ETREE_LEAF and ETREE_INTERIOR
 *
 * @param ep handle to the etree into where the octant is to be inserted.
 * @param addr octant address structure containing the address of the
 *     octant to insert.
 * @param value address of the data to store in the etree database.
 * 
 * @return 0 on success, -1 on error.  Application programs should call
 *     etree_errno() and etree_strerror() to identify the nature of the error.
 *
 * - ERROR:
 *    ET_LEVEL_OOB
 *    ET_OP_CONFLICT
 *    ET_DUPLICATE
 *    ET_IO_ERROR
 *    ET_NOT_WRITABLE
 */
int etree_insert(etree_t *ep, etree_addr_t addr, const void *payload);

/**
 * etree_delete - Delete an octant from the etree
 *
 * - Lazy method: NO merge of underflow B-tree nodes is done
 *
 * @param ep handle to the etree from where the octant is to be deleted.
 * @param addr octant address structure containing the address of the
 *      octant to be deleted.
 *
 * @return 0 if successfully deleted, -1 otherwise.
 *
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_OP_CONFLICT
 *    ET_EMPTY_TREE
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 */
int etree_delete(etree_t *ep, etree_addr_t addr);

/**
 * etree_update - Modify the content/payload of an octant in the etree
 *
 * - Update the payload for the object with exact hit
 *
 * @param ep handle to the etree where the octant is to be modified.
 * @param addr octant address structure containing the address of the
 *     octant to be modified.  The data associated with the octant is
 *     modified only if the specified octant address matches exactly
 *     the address of the octant found in the database.
 * @param payload location of the new data associated with the octant.
 *
 * @return 0 if updated, -1 otherwise
 *
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_EMPTY_TREE
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 *    ET_NOT_WRITABLE
 *
 */
int etree_update(etree_t *ep, etree_addr_t addr, const void *payload);

/**
 * etree_sprout - Sprout a leaf octant into eight children
 *
 * - Only valid for 3D
 * - Check the sprouting addr is a leaf octant  
 * - Remove the sprouting addr from the etree;
 * - Derive the children's address from the sprouting octant
 * - The childpayload array is assumed to hold the payload for the eight children
 *   in Z-order
 *
 * @param ep handle to the etree where the new octant are to be inserted.
 * @param addr octant address structure containing the address of the
 *      octant to be subdivided.  The octant is subdivided only if it
 *      exists in the database and is a leaf octant.
 * @param childrenpayload[8] array with the location of the data payloads
 *      associated with the new children octants.  The payloads for the
 *      new children are expected to be in Z-order. E.g., \{ (0,0,0); (0,0,1);
 *      (0,1,0); (0,1,1); (1,0,0); (1,0,1); (1,1,0); (1,1,1); \}.
 *
 * @return 0 if OK, -1 if failed; 
 *
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
 */
int etree_sprout(etree_t *ep, etree_addr_t addr, const void *childpayload[8]);

/*
 * Appending octants
 */

/**
 * etree_beginappend - Start an append transcation.
 *
 * In an append transaction, an application can only append octants
 * into the database.  During an append transaction etree_append calls
 * execute faster than in stand-alone appends (i.e., etree_append calls
 * outside an append transaction).
 * This function is useful to perform ``bulk'' appends into the
 * database in preorder traversal of the octree (Z-order).
 *
 * @param ep etree handle of the database where the append operations
 *      will be performed.
 * @param fillratio  Specifies the fill ratio at which an append
 *      operation should cause a split of the pages in the etree database.
 *
 * @return 0 on success,  -1 otherwise.
 *
 * - ERROR:
 *   
 *    ET_ILLEGAL_FILL
 *    ET_IO_ERROR
 */
int etree_beginappend(etree_t *ep, double fillratio);

/**
 * etree_append - Append an octant to the end of the etree.  This
 * function fails if the octant address specified does not preserve
 * the locational code order in the database.
 *
 * @param ep}: handle to the etree into where the octant is to be appended.
 * @param addr}: octant address structure containing the address of the
 *     octant to append.
 * @param payload}: address of the octant's associated data to store in the
 *     etree database.
 *
 * @return 0 if OK, -1 otherwise
 *
 * - ERROR:
 *
 *    ET_LEVEL_OOB:
 *    ET_APPEND_OOO:
 *    ET_NOT_WRITABLE
 *    ET_IO_ERROR
 */
int etree_append(etree_t *ep, etree_addr_t addr, const void *payload);

/**
 * etree_endappend - Terminate an append transaction.
 *
 * @param ep etree handle of the database where the append transaction
 *     is being performed.
 * @return 0 if OK, -1 on error.
 *
 * - ERROR:
 *
 *    ET_NOT_APPENDING:
 */
int etree_endappend(etree_t *ep);


/*
 * Searching for octants
 */

/**
 * etree_search - Search an octant in the etree database.
 *
 * An octant is "found" if either:
 *   1) an octant with the exact same address is located
 *   2) an octant (at a higher octree level) that encloses the extent
 *      of the search octant is located; in this case, we call it
 *      ancestor hit.
 *
 * - The search octant address "addr" does not need to specify the "type"
 * - Convert the search address to the locational key
 * - Search the key in the B-tree
 * - Convert the hit locational key back to hit octant address, which now
 *   contains the hit octant's type info
 *
 * @param ep handle to the etree where the octant is to be searched.
 * @param addr octant address structure containing the address of the
 *      octant for which to search.  This octant address does not need
 *      to specify the octant type (ETREE\_LEAF vs. ETREE\_INTERIOR).
 * @param hitaddr is an output parameter and points to an octant address
 *      structure.  If the search succeeds the address of the found octant is
 *      stored in the location pointed by {\em hitaddr}.
 *      The hit address also contains the octant's type information (ETREE\_LEAF
 *      vs. ETREE\_INTERIOR).
 * @param fieldname}: name of the field of interest
 * @param payload is an output parameter.  On success, the data associated
 *      with the found octant is stored in the memory pointed by payload.
 *
 * @return 0 if found,  -1 if not found.
 *
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
                 const char *fieldname, void *payload);

/**
 * etree_findneigbhor - Search for a neighbor in the etree database
 *
 * - Valid only for 3D
 * - Derive the neighbor at dir direction of current code
 *  ( we are not interested in corner neighbors )
 * - Search the etree for the octant
 *
 * @param ep handle to the etree where the neighbor is to be searched.
 * @param addr octant address structure containing the address of the
 *     octant for which a neighbor is to be searched.
 * @param dir}: specifies the direction of the search.
 * @param nbaddr}: is an output parameter and points to an octant address
 *     structure where the address of the found neighbor is stored if the
 *     search succeeds.
 * @param fieldname}: name of the field of interest
 * @param payload}: is an output parameter.  On success, the data associated
 *     with the found neighbor octant is stored in the memory location pointed
 *     by payload.
 *
 * @return 0 if found, -1 if failed.
 *
 * - ERROR:
 *    ET_NOT_3D
 *    ET_INVALID_NEIGHBOR
 *    ET_LEVEL_OOB 
 *    ET_LEVEL_OOB2
 *    ET_EMPTY_TREE 
 *    ET_NOT_FOUND
 *    ET_IO_ERROR
 *    ET_NO_SCHEMA
 *    ET_NO_FIELD
 */
int etree_findneighbor(etree_t *ep, etree_addr_t addr, etree_dir_t dir,
                       etree_addr_t *nbaddr, const char *fieldname, 
                       void *payload);

/* 
 * Z-order (preorder) traversal 
 */

/**
 * etree_initcursor - Set the cursor in the etree for preorder traversal 
 *
 * - The preorder traversal starts from octant with addr
 * - To start from the first octant in the etree, set the fields of addr
 *   to 0. That is addr.x = addr.y = addr.z = addr.level = 0
 *
 * @param ep handle to the etree where the traversal is to be performed.
 * @param addr octant address structure specifying the address of the
 *      octant where the traversal operation starts.
 *
 * @return 0 if OK, -1 on error.
 *
 * - ERRORS:
 *
 *    ET_LEVEL_OOB
 *    ET_EMPTY_TREE
 *    ET_IO_ERROR
 */
int etree_initcursor(etree_t *ep, etree_addr_t addr);


/**
 * etree_getcursor - Obtain the content of the octant currently pointed to
 *                   by the cursor
 *
 * - retrieve the cursor from B-tree
 * - convert the locational key properly
 *
 * @param ep}: handle to the etree where the traversal is performed.
 * @param addr}: address of the current octant pointed to by the cursor.
 *     This is an output parameter.
 * @param fieldname}: name of the field of interest
 * @param payload}: is an output parameter.  The octant's data payload is
 *     stored at the memory location specified in payload.
 *
 * @return 0 if OK, -1 otherwise.
 *
 * - ERROR:
 *
 *    ET_NO_CURSOR:
 *    ET_LEVEL_OOB2
 *    ET_NO_SCHEMA
 *    ET_NO_FIELD
 *
 */
int etree_getcursor(etree_t *ep, etree_addr_t *addr, const char *fieldname, 
                    void *payload);

/**
 * etree_advcursor - Move the cursor to the next octant in pre-order 
 *
 * - wrapper function to call advcursor in the underlying B-tree
 *
 * @param ep handle to the etree where the traversal is performed.
 *
 * @return 0 if OK, -1 otherwise.
 *
 * - ERROR:
 *
 *    ET_NO_CURSOR
 *    ET_END_OF_TREE
 *    ET_IO_ERROR
 */
int etree_advcursor(etree_t *ep);

/**
 * etree_stopcursor - Stop the cursor operation
 *
 * @param ep handle to the etree where the traversal is performed.
 *
 * @return 0 if OK, -1 otherwise.
 *
 * - ERROR:
 *
 *    ET_NO_CURSOR
 */
int etree_stopcursor(etree_t *ep);

/*
 * Miscelaneous helper and access functions
 */

/**
 * etree_straddr - create a human-readable text representation of an octant
 * address.
 *
 * @param ep handle of the etree the octant belongs to.
 * @param buf character array where to store the text representation of the
 *     octant address.
 * @param addr octant address to represent as text.
 *
 * @return a pointer to where the octant address text representation is
 *     stored (i.e., buf).
 */
char* etree_straddr(etree_t *ep, char *buf, etree_addr_t);


/**
 * etree_getmaxleaflevel get the highest level in the etree where there is
 * at least one leaf octant.
 *
 * @param ep etree handle (i.e., etree\_t struct) where to obtain the
 *     information from.
 * @return the maximum leaf level; -1 if there are no leaf octants stored
 *     in the etree.
 */
int etree_getmaxleaflevel(etree_t *ep);

/**
 * etree_getminleaflevel get the lowest level in the etree where there is
 * at least one leaf octant.
 *
 * @param ep etree handle (i.e., etree\_t struct) where to obtain the
 *     information from.
 * @return the minimum leaf level; -1 if there are no leaf octants in the 
 *  etree
 */
int etree_getminleaflevel(etree_t *ep);


/**
 * etree_getavgleaflevel: Get the average level in the etree where there is
 * at least one leaf octant.
 *
 * @param ep etree handle (i.e., etree\_t struct) where to obtain the
 *     information from.
 * @return the av leaf level, < 0; -1 if there are no leaf octants stored
 *     in the etree.
 */
float etree_getavgleaflevel(etree_t *ep);


/*
 * Fetching and retrieving metadata strings
 */

/**
 * Get the metadata string stored in the etree database.
 *
 * @param ep    etree handle (i.e., pointer to a etree struct).
 * @param meta  a text string containing the application metadata.  This is
 *		an output parameter.  The memory needed to store the metadata
 *		is allocated by this function using malloc().  The caller
 *		should release the memory with free().
 *
 * @return	a pointer to the meta data string if one is defined, NULL if 
 *      no application meta data is defined 
 *
 */
char* etree_getappmeta(etree_t* ep);

/**
 * Set the application metadata.
 *
 * @param ep    etree handle (i.e., pointer to an etree_t struct).
 * @param meta  a text string containing the application metadata.
 * @param len	length of the metadata string.
 *
 * @return	0 on success, -1 on error.
 */
int etree_setappmeta(etree_t* ep, const char* appmetadata);


/**
 * etree_isempty
 *
 * return 1 if true, 0 if false
 */
int etree_isempty(etree_t *ep);


/**
 * etree_hasleafonly
 *
 * return 1 if true, 0 if false
 */
int etree_hasleafonly(etree_t *ep);

/**
 * etree_getpayloadsize
 *
 */
int etree_getpayloadsize(etree_t *ep);

/**
 * etree_getkeysize
 *
 */
int etree_getkeysize(etree_t *ep);

/**
 * etree_gettotalcount
 *
 */
uint64_t etree_gettotalcount(etree_t *ep);


#ifdef __cplusplus
} /* extern "C" { */
#endif /* __cplusplus */


#endif /* ETREE_H */

