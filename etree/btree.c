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
 * btree.c - Btree with bulk (limited) insert and update support
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include <errno.h>

#include "btree.h"
#include "buffer.h"
#include "xplatform.h"
#include "schema.h"


/*
 * mybtree_t - internal control structure of a btree
 *
 */

typedef struct mybtree_t {     /* internal control block for the btree      */
  
    /************************************************************************/
    /*                 Persistent B-tree meta data                          */
    /************************************************************************/

    endian_t endian;           /* tree format when it was created           */
    uint32_t pagesize;         /* btree page size                           */

    pagenum_t pagecount;       /* number of pages in the btree              */
    pagenum_t rootpagenum;     /* root page number                          */

    uint32_t keysize;          /* key size of the btree                     */
    uint32_t valuesize;        /* payload size of each record               */

    uint32_t asciischemasize;  /* set when open'n btree or register'n schema */

    /************************************************************************/
    /*      Control fields initialized when opening a btree                 */
    /************************************************************************/
    char *pathname;            /* the path name of the btree                */
    int flags;                 /* flags for opening the btree               */
    
    off_t startoffset;         /* where to write the meta data and schema   */

    int allowschema;           /* whether a schema can be imposed           */
    schema_t *schema;          /* schema of the payload                     */
    char *asciischema;         /* set when registering schema, NULL otherwise*/
    scb_t *scb;                /* structure control block if schema != NULL */

    char *fieldname;           /* pointer to the last accessed field name   */
    int32_t fieldind;          /* field index of the last accessed field    */

    btree_compare_t *compare;  /* handler to application comparison function*/
    int32_t leafentrysize;     /* leaf node entry size                      */
    int32_t leafcapacity;      /* maximum number of entries in a leaf node  */
    int32_t indexentrysize;    /* index node entry size                     */
    int32_t indexfanout;       /* maximum number of entries in an index node*/
    pagenum_t nextpage;        /* next free page number                     */
    
    buffer_t *buf;             /* handler to the buffer manager             */

    /************************************************************************/
    /*      Control fields initialized for cursor operations                */
    /************************************************************************/

    void *cursorpage;          /* pointer to the cursor page in the buffer  */
    int32_t cursoroffset;      /* entry offset in the current cursorpage    */
    void *cursorptr;           /* pointer to the current cursor             */

    /************************************************************************/
    /*      Control fields initialized for append operations                */
    /************************************************************************/

    int enableappend;          /* append cannot interleave with insert      */
    void *appendpage;          /* pointer to the current append leaf page   */
    int32_t appendleafmax;     /* maximum number of appends on leaf page    */
    int32_t appendindexmax;    /* maximum number of appends on index page   */
    
}mybtree_t;


/*
 * metahdrsize - compact representation of the btree meta data stored 
 *               before the root page; the size does NOT include the
 *               size for the ASCII schema if such a schema exists.
 *
 * sizeof(endian) = 1; either "L" or "B"
 * sizeof(pagesize) = 4 ; 
 * sizeof(rootpagenum) = sizeof(pagecount) = 8; 
 * sizeof(keysize) = sizeof(valuesize) = 4;
 * sizeof(asciischemasize) = 4; 
 *
 */
static int32_t metahdrsize = 1 + 4 + 8 + 8 + 4 + 4 + 4;

/*
 * noswap - we need a static variable to indicate whether we need to
 *          swap the count/rightsibnum; 
 */
static int noswap = 1;  

/* 
 * noswapkey - decide whether we need to swap the key for comparison/store
 *
 * - only useful when keys are numerals
 */
static int noswapkey = 1;


/*
 * keytype - record the key type if the key is numeral; 
 *
 */
static char keytype[16];

/*
 * platformkey - convenience variable to hold the key in platform-specific
 *               format
 *
 */
static char platformkey[8];
static char *platformkeyptr = (char *)&platformkey;

/*
 * numeral_compare - default numeral comparison function
 *
 */
static int 
numeral_compare(const void *key1, const void *key2, int size);


/* 
 * schema related routines 
 *
 */
static int writeheader(mybtree_t *mybp);

static int readheader(mybtree_t *mybp);

static int
whichfield(mybtree_t *mybp, const char *fieldname);

static void 
extractfield(mybtree_t *mybp, void *value, const void *src, int32_t fieldind);

static void 
populatefield(mybtree_t *mybp, void *dest, const void *value, 
              int32_t fieldind);


/* 
 * hdr_t: runtime structure holding header info for each B-tree page
 *
 * - in the original implementation, hdr_t holds the fields of the header
 * - however, different platforms have different alignment requirement 
 *   for a structure
 * - therefore, I changed it to hold pointers to the different fields
 *
 */
typedef struct hdr_t {
    int32_t *countptr;          /* number of entry in this page              */
    pagenum_t *rightsibnumptr;  /* right neighbor's page number              */
    char *typeptr;              /* 'l': leaf; 'i': index                     */

    /* these two entries need to be initialized when the 
       page is first read in; runtime variables , a hack */
    void **ppageaddrptr;        /* parent page's location in the buffer      */
    int32_t *pentryptr;        /* which entry in the parent page points to me*/
} hdr_t;


/*
 * hdrsize - the size of the compact header, we force the pointer type to be
 *           8 bytes (for 64 bit machine)
 * 
 */
static const int32_t hdrsize = 4 + 8 + 1 + 8 + 4;

void setheader(hdr_t *hdrptr, const void *pageaddr);

/*
 * search routines 
 *
 */
static int32_t 
findentrypoint(mybtree_t *mybp, const void *key, void **pageaddrptr);

static void * 
locateleaf(mybtree_t *mybp, void *pageaddr, const void *key);

static void 
cascadeunref(mybtree_t *mybp, void *pageaddr);

static int 
binarysearch(mybtree_t *mybp, const void *pageaddr, const void *key);

static void *
sink(mybtree_t *mybp, void *pageaddr, int where);


/*
 * insert/split routines 
 *
 */
static int
insert(mybtree_t *mybp, void *pageaddr, int32_t entry,
       int32_t newcount, const void *keys[], const void *values[]);

static int
simpleinsert(mybtree_t *mybp, void *pageaddr, int32_t entry, 
             int32_t newcount, const void *keys[], const void *values[]);

static void 
plugin(mybtree_t *mybp, void *pageaddr, int32_t entry, int32_t newcount,
       const void *keys[], const void *values[]);


static int
splitinsert(mybtree_t *mybp, void *pageaddr, int32_t entry, 
            int32_t newcount, const void *keys[], const void *values[]);

static int
splitroot(mybtree_t *mybp, void *pageaddr, int32_t cnt1, int32_t cnt2,
          void **newaddr1ptr, void **newaddr2ptr);

static int 
splitpage(mybtree_t *mybp, void *pageaddr, int32_t cnt1, int32_t cnt2,
          void **newaddr1ptr, void **newaddr2ptr);

static int 
inorder(mybtree_t *mybp, void *pageaddr, int32_t entry, int32_t count, 
        const void *keys[]);


/*
 * delete routines 
 *
 */
static 
void unplug(mybtree_t *mybp, void *pageaddr, int32_t entry);


/*
 * append routines 
 *
 */
static void *
append(mybtree_t *mybp, void *pageaddr, const void *key, 
       const void *value, int *pcode);

static void *
splitappend(mybtree_t *mybp, void *pageaddr, const void *key, 
            const void *value, int *pcode);



/*
 * return value conventions:
 *
 * for function calls that return integer values to indicate success or
 * failure, the following convention is followed.
 *
 * 0: success
 * -1: conflict operations (insert in conflict with append and cursor)
 * -2: empty B-tree
 * -3: cannot find the octant (search, update, bulkupdate)
 * -4: bulk loading (bulk update or insert) value out of order
 * -5: no cursor in effect
 * -6: illegal fill ratio for append operation
 * -7: not in append mode 
 * -8: append a key out of order
 * -9: low-level IO error. higher-level application programs should invoke
 *     perror() to identify the low level error reported
 * -10: too many records for bulk insert 
 * -11: schema definition disallowed
 * -12: schema create fails
 * -13: no schema defined
 * -14: unknown member name
 * -15: structure control block create fails
 * 
 */



/*
 * btree_open - open or create a btree 
 *
 * - allocate and initialize the control structure
 * - create and initialize I/O buffer
 * - register the function provided by the application for comparison 
 * - return pointer to btree_t if OK, NULL on error
 *
 */
btree_t *btree_open(const char *pathname, int flags, uint32_t ksize, 
                    const char *ktype,
                    uint32_t vsize, uint32_t pagesize, int32_t bufsize,
                    btree_compare_t *compare, off_t startoffset)
{
    mybtree_t *mybp;
    uint32_t payloadsize;
    struct stat statbuf;
    int32_t existed;
    size_t framecount;
    off_t rootstart;
    double blksizeK;

    /* make sure that current platform support large file system, i.e.
       sizeof(off_t) == 8 */
    if (sizeof(pagenum_t) != 8) {
        fprintf(stderr, "The package is not compiled with LFS support.\n");
        fprintf(stderr, "Specify -D_LARGEFILE64_SOURCE ");
        fprintf(stderr, "-D_FILE_OFFSET_BITS=64 to gcc\n");
        return NULL;
    }
        
    /* check the existence of the pathname */
    if (stat(pathname, &statbuf) == 0)
        existed = 1;
    else {
        existed = 0;
        
        if (errno != ENOENT) {
            perror("btree_open (stat):");
            return NULL;
        }
    }

    if (((flags & O_CREAT) == 0) && !existed) {
        fprintf(stderr, "btree_open: O_CREAT must be specified to open ");
        fprintf(stderr, "the non-existent btree %s.\n", pathname);
        return NULL;
    }
        

    if (flags & O_TRUNC) {
        if (!((flags & O_WRONLY) || (flags & O_RDWR))) {
            fprintf(stderr, "btree_open: O_TRUNC must be specified with ");
            fprintf(stderr, "the open mode that allows writing (i.e., is ");
            fprintf(stderr, "O_WRONLY or O_RDWR).\n");
            return NULL;
        }
    }

    /* allocate the control structure */
    if ((mybp = (mybtree_t *)malloc(sizeof(mybtree_t))) == NULL) {
        /* application should invoke perror() to identify the error */
        return NULL; 
    }
    memset(mybp, 0, sizeof(mybtree_t));    

    /* the next two fields are used by readheader */
    if ((mybp->pathname = strdup(pathname)) == NULL) {
        perror("btree_open: strdup");
        return NULL;
    }

    /* record the startoffset */
    mybp->startoffset = startoffset;

    /* set flags to avoid writing a read-only file while closing the btree*/
    mybp->flags = flags;


    if (((flags & O_TRUNC) != 0) ||
        (((flags & O_CREAT) != 0) && (!existed))) {
        /* 
           Either O_TRUNC is specified or create a brand new btree
           initialize the Btree meta data 
        */        

        mybp->endian = xplatform_testendian();
        mybp->pagesize = pagesize;

        rootstart = startoffset + metahdrsize;

        if ((rootstart % mybp->pagesize) == 0)
            mybp->rootpagenum = (pagenum_t)(rootstart / mybp->pagesize);
        else
            mybp->rootpagenum = (pagenum_t)(rootstart / mybp->pagesize) + 1;

        mybp->pagecount = 0; /* empty btree */

        mybp->keysize = ksize;
        mybp->valuesize = vsize;

        mybp->asciischemasize = 0;  

        /* allow applicatin to define a schmea */
        mybp->schema = NULL;
        mybp->asciischema = NULL;
        mybp->scb = NULL;
        mybp->allowschema = 1;
        
    } else {
        /* open an existing btree */
        if (readheader(mybp) != 0) {
            fprintf(stderr, "btree_open: corrupted meta data\n");
            return NULL;
        }
    }

    /* no ascii schema is created yet */
    mybp->asciischema = NULL;

    /* determine whether the system endianness is compatible with that of
       the btree */
    if (mybp->endian == xplatform_testendian()) 
        noswap = 1;
    else
        noswap = 0;

    /* install comparison function */
    if (compare == NULL) {
        /* this error would never occur in etree library */
        if ((strcmp(ktype, "int32_t") && (ksize == 4)) ||
            (strcmp(ktype, "int64_t") && (ksize == 8)) ||
            (strcmp(ktype, "uint32_t") && (ksize == 4)) ||
            (strcmp(ktype, "uint64_t") && (ksize == 8)) ||
            (strcmp(ktype, "float") && (ksize == 4)) || 
            (strcmp(ktype, "double") && (ksize == 8)) ||
            (strcmp(ktype, "float32_t") && (ksize == 4)) || 
            (strcmp(ktype, "float64_t") && (ksize == 8))) {

            /* install the numeral comparison function */
            mybp->compare = numeral_compare;
            strcpy(keytype, ktype);

            /* we only need to swap key if it's of numeral type;
               set noswapkey to be the same flag as that for value */
            noswapkey = noswap;
        }
        else {
            fprintf(stderr, "btree_open: unknown numerical key type\n");
            fprintf(stderr, "btree_open: check the data type and its size\n");
            return NULL;
        }
    } else
        mybp->compare = compare;
    
    /* init index structure information */
    payloadsize = mybp->pagesize - hdrsize;
    mybp->leafentrysize = mybp->keysize + mybp->valuesize;
    mybp->leafcapacity = payloadsize / mybp->leafentrysize;
    mybp->indexentrysize = mybp->keysize + sizeof(pagenum_t);
    mybp->indexfanout = payloadsize / mybp->indexentrysize;
    mybp->nextpage = mybp->rootpagenum + mybp->pagecount;

    /* buffer_init open the file for I/O */
    blksizeK = 1.0 * mybp->pagesize / 1024;
    framecount = (size_t)(bufsize * 1024 / blksizeK);

    if ((mybp->buf = buffer_init(pathname, flags, framecount, mybp->pagesize)) 
        == NULL) {
        /* cannot allocate buffer space */
        return NULL;
    }

    if (((flags & O_TRUNC) != 0) ||
        (((flags & O_CREAT) != 0) && (!existed))) {
        void *zeropage;
        pagenum_t pageid;

        zeropage = calloc(pagesize, 1);
        if (zeropage == NULL) {
            fprintf(stderr, "btree_open: out of memory\n");
            return NULL;
        }

        /* Zero out the preceding pages */
        for (pageid = 0; pageid < mybp->rootpagenum; pageid++) {
            if (write(mybp->buf->fd, zeropage, pagesize) != (int)pagesize) {
                fprintf(stderr, "btree_open: init error\n");
                return NULL;
            }
        }

        free(zeropage);
    }


    /* no cursor in effect */
    mybp->cursoroffset = -1; 

    /* not in appending mode */
    mybp->enableappend = 0;  
    mybp->appendpage = NULL;

    return (btree_t *)mybp;
}


/*
 * btree_registerschema - register schema for a TRUNC'ed or newly CREAT'ed
 *                        btree
 *
 * - load the binary schema control structure and store the ascii schema too
 * - return 0 if OK, -11 if attempting to define a schema on a btree that 
 *   disallow schema; -12 if schema create fails; -15 if scb create fails
 * 
 */
int btree_registerschema(btree_t *bp, const char *defstring)
{
    mybtree_t *mybp;
    int32_t fieldind, compactsize;
    uint32_t payloadsize;
    off_t rootstart;

    mybp = (mybtree_t *)bp;
    if (mybp->allowschema == 0) 
        return -11;
    
    mybp->schema = schema_create(defstring);
    if (mybp->schema == NULL) 
        return -12;

    /* build the structure control block for this platform */
    mybp->scb = xplatform_createscb(*mybp->schema);
    if (mybp->scb == NULL)
        return -15;

    /* what's exactly the payload size */
    compactsize = 0;
    for (fieldind = 0; fieldind < mybp->schema->fieldnum; fieldind++) 
        compactsize += mybp->schema->field[fieldind].size;
    mybp->valuesize = compactsize;
    
    /* adjust fields that are affected by valuesize */
    payloadsize = mybp->pagesize - hdrsize;
    mybp->leafentrysize = mybp->keysize + mybp->valuesize;
    mybp->leafcapacity = payloadsize / mybp->leafentrysize;

    /* ascii schema only needs to be produced once, when the schema is
       registered for the first time */
    mybp->asciischema = schema_toascii(mybp->schema, 
                                       &mybp->asciischemasize);

    /* adjust rootpagenum */
    rootstart = mybp->startoffset + metahdrsize + mybp->asciischemasize;
    if (rootstart % mybp->pagesize == 0) 
        mybp->rootpagenum = (pagenum_t)(rootstart / mybp->pagesize);
    else
        mybp->rootpagenum = (pagenum_t)(rootstart / mybp->pagesize) + 1;

    mybp->nextpage = mybp->rootpagenum + mybp->pagecount;

    /* disallow registering schema again */
    mybp->allowschema = 0;

    return 0;
}
    

/*
 * btree_printschema - output the schema in ascii format that can be 
 *                     interpreted by application programmer
 *
 * - return 0 if OK, -13 if no schema defined
 */
int btree_printschema(btree_t *bp, FILE *fp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    int fieldind;

    if (mybp->schema == NULL) 
        /* no schema defined */
        return -13;
    
    fprintf(fp, "Schema for %s:\n\n", mybp->pathname);
    
    fprintf(fp, "%-9s%-16s%-12s%-8s\n", "Field", "Name", "Type", "Size");
    for (fieldind = 0; fieldind < mybp->schema->fieldnum; fieldind++) 
        fprintf(fp, "%-9d%-16s%-12s%-8d\n", fieldind,
                mybp->schema->field[fieldind].name,
                mybp->schema->field[fieldind].type,
                mybp->schema->field[fieldind].size);
    return 0;
}
    

/*
 * btree_getschema - return strdup'ed schema ascii string if defined,
 * otherwise, return NULL
 *
 */
char *btree_getschema(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    
    if (mybp->schema == NULL)
        return NULL;
    else 
        return schema_getdefstring(mybp->schema);
}
        
/*
 * btree_close  - release resource
 *
 * - return 0 if OK, -9 if low level IO error
 *
 */

int btree_close(btree_t *bp)
{
    int res = 0;

    mybtree_t *mybp = (mybtree_t *)bp;

    if (buffer_destroy(mybp->buf) != 0)
        res = -9;

    if (((mybp->flags & O_INCORE) == 0) &&
        ((mybp->flags & O_RDWR) || (mybp->flags & O_WRONLY))) 
        if (writeheader(mybp) != 0)
            res = -9;

    /* free memory */
    free(mybp->pathname); /* strdup'ed */

    if (mybp->schema != NULL) 
        schema_destroy(mybp->schema);

    if (mybp->asciischema != NULL) 
        free(mybp->asciischema); /* malloc'ed by schema_toascii() */

    if (mybp->scb != NULL)
        xplatform_destroyscb(mybp->scb);

    free(mybp);

    return res;
}

/*
 * writeheader: write the meta data to the startoffset 
 *
 * return 0 if OK, -1 on error
 *
 */
int writeheader(mybtree_t *mybp)
{
    int btreefd;
    uint32_t pagesize, keysize, valuesize, asciischemasize;
    pagenum_t pagecount, rootpagenum;

    /* update the pagecount */
    mybp->pagecount = mybp->nextpage - mybp->rootpagenum;

    /* convert the meta data if byte swapping is necessary */
    if (!noswap) {
        xplatform_swapbytes(&pagesize, &mybp->pagesize, 4);
        xplatform_swapbytes(&pagecount, &mybp->pagecount, 8);
        xplatform_swapbytes(&rootpagenum, &mybp->rootpagenum, 8);
        xplatform_swapbytes(&keysize, &mybp->keysize, 4);
        xplatform_swapbytes(&valuesize, &mybp->valuesize, 4);
        xplatform_swapbytes(&asciischemasize, &mybp->asciischemasize, 4);
    } else {
        pagesize = mybp->pagesize;
        pagecount = mybp->pagecount;
        rootpagenum = mybp->rootpagenum;
        keysize = mybp->keysize;
        valuesize = mybp->valuesize;
        asciischemasize = mybp->asciischemasize;
    }
    
    /*
      write meta header to the btree, the btree has been closed
       when we destroy the buffer associated with it 
    */
    btreefd = open(mybp->pathname, O_WRONLY);
    if (btreefd == -1) {
        perror("writeheader: open");
        return -1;
    }
    
    if (lseek(btreefd, mybp->startoffset, SEEK_SET) != mybp->startoffset) {
        perror("writeheader: lseek");
        return -1;
    }

    if (write(btreefd, (mybp->endian == little) ? "L" : "B", 1) != 1) {
        perror("writeheader: write meta(endian)");
        return -1;
    }

    if (write(btreefd, &pagesize, 4) != 4) {
        perror("writeheader: write meta(pagesize)");
        return -1;
    }
        
    if (write(btreefd, &pagecount, 8) != 8) {
        perror("writeheader: write meta(pagecount)");
        return -1;
    }
    if (write(btreefd, &rootpagenum, 8) != 8) {
        perror("writeheader: write meta(rootpagenum)");
        return -1;
    }
    if (write(btreefd, &keysize, 4) != 4) {
        perror("writeheader: write meta(keysize)");
        return -1;
    }
    if (write(btreefd, &valuesize, 4) != 4) {
        perror("writeheader: write meta(valuesize)");
        return -1;
    }

    if (mybp->asciischema != NULL) {
        /* 
           only newly registered schema set asciischema to non-NULL, 
           we need to flush it to disk;
           otherwise, don't temper with the asciischemasize and the
           asciischema
        */

        if (write(btreefd, &asciischemasize, 4) != 4) {
            perror("writeheader: write meta(asciischemasize)");
            return -1;
        }

        if (write(btreefd, mybp->asciischema, mybp->asciischemasize) != 
            (int)mybp->asciischemasize) {
            perror("writeheader: write ASCII schema");
            return -1;
        }
    }

    if ((fsync(btreefd) != 0) || (close(btreefd) != 0)) {
        perror("writeheader: close btree file");
        return -1;
    }

    return 0;
}


/*
 * readheader - read the meta data from startoffset
 *
 * also load schema if one is defined
 * 
 * - return 0 if OK , -1 on error
 */
int readheader(mybtree_t *mybp)
{
    int btreefd;
    char endianchar;
    uint32_t pagesize, keysize, valuesize, asciischemasize;
    pagenum_t pagecount, rootpagenum;

    btreefd = open(mybp->pathname, O_RDONLY);
    if (btreefd == -1) {
        perror("readheader: open");
        return -1;
    }
    
    if (lseek(btreefd, mybp->startoffset, SEEK_SET) != mybp->startoffset) {
        perror("readheader: lseek");
        return -1;
    }
    
    if (read(btreefd, &endianchar, 1) != 1) {
        perror("readheader: read meta(endian)");
        return -1;
    }
    if (read(btreefd, &pagesize, 4) != 4) {
        perror("readheader: read meta(pagesize)");
        return -1;
    }
    if (read(btreefd, &pagecount, 8) != 8) {
        perror("readheader: read meta(pagecount)");
        return -1;
    }
    if (read(btreefd, &rootpagenum, 8) != 8) {
        perror("readheader: read meta(rootpagenum)");
        return -1;
    }
    if (read(btreefd, &keysize, 4) != 4) {
        perror("readheader: read meta(keysize)");
        return -1;
    }
    if (read(btreefd, &valuesize, 4) != 4) {
        perror("readheader: read meta(valuesize)");
        return -1;
    }
    if (read(btreefd, &asciischemasize, 4) != 4) {
        perror("readheader: read meta(asciischemasize)");
        return -1;
    }

    /* load the data into the runtime control structure */
    if (endianchar == 'L') 
        mybp->endian = little;
    else if (endianchar == 'B')
        mybp->endian = big;
    else {
        fprintf(stderr, "readheader: corrupted meta(endian) : %c\n", endianchar);
        return -1;
    }
    
    if (xplatform_testendian() != mybp->endian) {
        /* we have to convert */
        xplatform_swapbytes(&mybp->pagesize, &pagesize, 4);
        xplatform_swapbytes(&mybp->pagecount, &pagecount, 8);
        xplatform_swapbytes(&mybp->rootpagenum, &rootpagenum, 8);
        xplatform_swapbytes(&mybp->keysize, &keysize, 4);
        xplatform_swapbytes(&mybp->valuesize, &valuesize, 4);
        xplatform_swapbytes(&mybp->asciischemasize, &asciischemasize, 4);
    } else {
        mybp->pagesize = pagesize;
        mybp->pagecount = pagecount;
        mybp->rootpagenum = rootpagenum;
        mybp->keysize = keysize;
        mybp->valuesize = valuesize;
        mybp->asciischemasize = asciischemasize;
    }

    if (mybp->asciischemasize != 0) {
        /* schema defined */
        char *asc_schema;

        asc_schema = (char *)malloc(mybp->asciischemasize);
        if (asc_schema == NULL) {
            perror("readheader: malloc ascii schema");
            return -1;
        }
        if (read(btreefd, asc_schema, mybp->asciischemasize) 
            != (int)mybp->asciischemasize) {
            perror("readheader: load ascii schema");
            return -1;
        }
        mybp->schema = schema_fromascii(asc_schema);
        if (mybp->schema == NULL) {
            fprintf(stderr, 
                    "readheader: cannot convert ascii schema to binary\n");
            return -1;
        }
        free(asc_schema);

        mybp->scb = xplatform_createscb(*mybp->schema);
        if (mybp->scb == NULL) {
            fprintf(stderr, 
                    "readheader: cannot create structure control block\n");
            return -1;
        }
    } else {
        /* no schema defined */
        mybp->schema = NULL;
        mybp->asciischema = NULL;
        mybp->scb = NULL;
    }

    /* Never allow schema definintion for an existing B-tree, no matter
       whether there is an existing schema in place or not */
    mybp->allowschema = 0;

    if (close(btreefd) != 0) {
        perror("readheader: close btree file");
        return -1;
    }
    return 0;
}
    



/*
 * btree_leafcapacity - return the leaf capacity of the btree
 *
 */
int btree_leafcapacity(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;

    return mybp->leafcapacity;
}


/*
 * btree_numofpages - return the number of B-tree pages used
 *
 */
pagenum_t btree_numofpages(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;

    return mybp->nextpage - mybp->rootpagenum;
}


/* 
 * btree_getendoffset - return the end offset of the btree 
 *
 */
off_t btree_getendoffset(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;

    return mybp->pagesize * mybp->nextpage;
}


/*
 * btree_insert - insert a data object with key and value
 *
 * - avoid conflict with ongoing stateful operations (append, cursor)
 * - find the page and the location where the key is to be inserted
 * - do the insertion, split if necessary
 * - whether the key is inserted or not (because of duplicate) is recorded 
 *   in *insed
 * - insert() takes care of the unref of the search path. If insert() is
 *   not invoked, unref the path explicitly.
 * - return 0 if OK, -1 if in conflict mode, -9 if lowlevel IO error occurs
 *
 */
int btree_insert(btree_t *bp, const void *key, const void *value, int *insed)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr;
    hdr_t hdr;
    int32_t entry;
    int res;

    if (mybp->enableappend == 1) {
        /* append mode in effect */
        return -1;
    }
    if (mybp->cursoroffset != -1) {
        /* curosr mode in effect */
        return -1;
    }

    mybp->allowschema = 0;

    if ((mybp->nextpage == mybp->rootpagenum) &&       /* empty B-tree */
        ((pageaddr = buffer_emptyfix(mybp->buf, mybp->rootpagenum)) != NULL)) {

        int32_t count = 0;
        pagenum_t rightsibnum = -1;

        mybp->nextpage++;

        setheader(&hdr, pageaddr);

        if (noswap) {
            *(hdr.countptr) = count;
            *(hdr.rightsibnumptr) = rightsibnum;
        } else {
            xplatform_swapbytes(hdr.countptr, &count, 4);
            xplatform_swapbytes(hdr.rightsibnumptr, &rightsibnum, 8);
        }

        /* type is one byte char; ppageaddr and pentry are both runtime
           variables, so byte ordering is not an issue */
        *(hdr.typeptr) = 'l';        
        *(hdr.ppageaddrptr) = NULL;
        *(hdr.pentryptr) = -1;

        entry = -1;
    }
    else 
        entry = findentrypoint(mybp, key, &pageaddr);
    
    if (entry == -9) 
        return -9;

    if (insed != NULL) {

        if (entry != -1) {
            /* We found an entry to compare against */
            const void *entrykey;

            entrykey = (char *)pageaddr +
                hdrsize + mybp->leafentrysize * entry;
            
            if (mybp->compare(key, entrykey, mybp->keysize) == 0) {
                /* 
                   Don't insert duplicate, but the operation is correct.
                   Don't forget to unref the search path.
                 */
                cascadeunref(mybp, pageaddr);
                res = 0;                
                *insed = 0;
            }  else {
                res = insert(mybp, pageaddr, entry, 1, &key, &value);
                *insed = 1;
            }
        } else {
            /* nothing to compare against, go ahead with the insert */
            res = insert(mybp, pageaddr, entry, 1, &key, &value);
            *insed = 1;
        }
    }  else {
        /* Don't care whether it's inserted or not */
        res = insert(mybp, pageaddr, entry, 1, &key, &value);
    }

    return res;
}



/*
 * btree_bulkinsert - bulk insert a sorted array of records
 *
 * - check the record array contains keys in ascending order
 * - find the entry after which to insert the records
 * - return if OK, -1 if in conflict mode, -4 if bulk is out of order,
 *    -9 if lowlevel IO error occurs
 *
 */
int btree_bulkinsert(btree_t *bp, int count, const void *keys[], 
                     const void *values[])
{

    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr;
    int32_t entry;
    int res;

    /* sanity check */
    if (mybp->enableappend == 1) {
        /* append mode in effect */
        return -1;
    }
    if (mybp->cursoroffset != -1) {
        /* curosr mode in effect */
        return -1;
    }

    mybp->allowschema = 0;

    entry = findentrypoint(mybp, keys[0], &pageaddr);
    if (entry == -9) 
        return -9;

    res = inorder(mybp, pageaddr, entry, count, keys);
    if (res != 1) {
        /* res may be -4 (out of order keys) or -9 (IO error) */
        /* Don't forget to unref the path */
        cascadeunref(mybp, pageaddr);
        return res;
    }
    
    res = insert(mybp, pageaddr, entry, count, keys, values);

    return res;

}



    
/*
 * btree_delete - remove the record with key from the btree
 *
 * - avoid conflict with stateful operations (append, cursor)
 * - lazy method, NO merge of underflow B-tree nodes is done
 * - return 0 if found and deleted, -1 if in conflict mode, -2 if empty btree
 *   -3 if not found ,  -9 if lowlevel IO error occurs
 *
 */
int btree_delete(btree_t *bp, const void *key)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr, *hitptr;
    int32_t entry;
    int res;

    if (mybp->enableappend == 1) {
        /* append mode in effect */
        return -1;
    }
    if (mybp->cursoroffset != -1) {
        /* curosr mode in effect */
        return -1;
    }

    if (mybp->nextpage == mybp->rootpagenum) {
        /* emptry B-tree */
        return -2;
    } 
        
    entry = findentrypoint(mybp, key, &pageaddr);
    if (entry == -9) 
        return -9;

    hitptr = (char *)pageaddr + hdrsize + entry * mybp->leafentrysize;
    if ((entry < 0) || (memcmp(key, hitptr, mybp->keysize) != 0)) 
        res = -3;
    else {
        unplug(mybp, pageaddr, entry);
        res = 0;
    }

    cascadeunref(mybp, pageaddr);
    return res;
}


/*
 * btree_update - update the content of the record with key
 *
 * - return 0 if found and updated, -2 if empty B-tree, -3 if not found
 *   -9 if lowlevel IO error occurs, 
 *
 */
int btree_update(btree_t *bp, const void *key, const void *value)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr, *hitptr;
    int32_t entry;
    int res;

    if (mybp->nextpage == mybp->rootpagenum) {
        /* empty B-tree */
        return -1;
    } 
        
    entry = findentrypoint(mybp, key, &pageaddr);
    if (entry == -9) 
        return -9;

    hitptr = (char *)pageaddr + hdrsize + entry * mybp->leafentrysize;
    if ((entry < 0) || (memcmp(key, hitptr, mybp->keysize) != 0)) 
        res = -3;
    else {
        char *dest;
        
        dest = (char *)hitptr + mybp->keysize;

        if (mybp->schema == NULL) 
            memcpy(dest, value, mybp->leafentrysize - mybp->keysize);
        else
            populatefield(mybp, dest, value, mybp->schema->fieldnum);

        buffer_mark(mybp->buf, pageaddr);        
        res = 0;
    }

    cascadeunref(mybp, pageaddr);
    return res;
}


/*
 * btree_bulkupdate - update the record at anchorkey with an array of
 *                    sorted records in bulk mode 
 *
 * - check the record array contains keys in ascending order
 * - find the anchor for the insertion
 * - overwrite the anchor record 
 * - bulk insert the record into the B-tree
 * - return if OK, -1 if in conflict mode, -2 if empty B-tree, -3 if cannot
 *   find the anchor point, -4 if the bulk is out of order, 
 *   -9 if lowlevel IO error occurs
 *
 */
int btree_bulkupdate(btree_t *bp, const void *anchorkey, int count, 
                     const void *keys[], const void *values[])
{

    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr, *dest;
    int32_t entry;

    /* sanity check */
    if (mybp->enableappend == 1) {
        /* append mode in effect */
        return -1;
    }
    if (mybp->cursoroffset != -1) {
        /* curosr mode in effect */
        return -1;
    }
    if (mybp->nextpage == mybp->rootpagenum) {
        /* empty B-tree */
        return -2;
    } 

    entry = findentrypoint(mybp, anchorkey, &pageaddr);
    if (entry == -9) 
        return -9;

    dest = (char *)pageaddr + hdrsize + entry * mybp->leafentrysize;

    if ((entry == -1) ||
        (memcmp(dest, anchorkey, mybp->keysize) != 0)) {
        /* Cannot find anchor */
        /* Don't forget to unref the search path */
        cascadeunref(mybp, pageaddr);
        return -3;
    }

    
    if (!inorder(mybp, pageaddr, entry, count, keys)) {
        /* Keys are out of order */
        /* Don't forget to unref the search path */
        cascadeunref(mybp, pageaddr);
        return -4;
    }
    
    /* overwrite the anchor */
    if (noswapkey) 
        memcpy(dest, keys[0], mybp->keysize);
    else
        xplatform_swapbytes(dest, keys[0], mybp->keysize);

    dest = (char *)dest + mybp->keysize;
    if (mybp->schema == NULL) 
        memcpy(dest, values[0], mybp->valuesize);
    else
        populatefield(mybp, dest, values[0], mybp->schema->fieldnum);
    
    /* bulk insert the remaing keys */
    if (count - 1 > 0) 
        return insert(mybp, pageaddr, entry, count - 1, &keys[1], &values[1]);
    else
        return 0;

}

/**
 * btree_isempty
 *
 * return 1 if true, 0 if false;
 */
int btree_isempty(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;

    if (mybp->nextpage == mybp->rootpagenum)
        return 1;
    else
        return 0;
}
    

/**
 * btree_getvaluesize
 *
 */
int btree_getvaluesize(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;

    return mybp->valuesize;
}


/*
 * btree_search - search for a record with key
 *
 * - member is string specifying either all of the fields (with "*"
 *   or simply NULL) or a particular field (its name must be defined 
 *   in the schema)
 * - locate the leaf page where the key may be found
 * - find the entry whose key is the maximum among all the keys less 
 *   than the search key
 * - return 0 if found,  -2 if empty B-tree, -3 if not found,
 *    -9 if lowlevel IO error occurs, 
 *    -13 if no schmea defined and request a field
 *    -14 if schema defined but no specified field 
 * - if found and value is not null, convert the result in host/platform
 *   format 
 *
 */
int btree_search(btree_t *bp, const void *key, void *hitkey, 
                 const char *fieldname, void *value)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr;
    char *base, *src;
    int32_t entry, fieldind;
    int res; 
    
    if (mybp->nextpage == mybp->rootpagenum) {
        /* empty B-tree */
        return -2;
    } 

    /* determine what to do with the payload */
    if ((fieldind = whichfield(mybp, fieldname)) < 0) 
        return fieldind;

    entry = findentrypoint(mybp, key, &pageaddr);
    if (entry == -9) return -9;

    if (entry < 0) 
        res = -3;
    else {
        res = 0;
        base = (char *)pageaddr + hdrsize;
        src = base + mybp->leafentrysize * entry ;

        if (noswapkey)
            memcpy(hitkey, src, mybp->keysize);
        else
            xplatform_swapbytes(hitkey, src, mybp->keysize);
        
        src += mybp->keysize;
        if (value != NULL) {
            if (mybp->schema == NULL) 
                memcpy(value, src, mybp->valuesize);
            else
                extractfield(mybp, value, src, fieldind);
        }
    }

    cascadeunref(mybp, pageaddr);

    return res;
}


/*
 * btree_initcursor - set the cursor at the specified key
 *
 * - if the curosr is set to the position before the first record, treat 
 *   this as a special case and set the cursor to the first record
 * - record the leaf page in the cursorpage
 * - return 0 if OK, -2 if empty btree,  -9 if lowlevel IO error occurs
 *
 */
int btree_initcursor(btree_t *bp, const void *key)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr;
    hdr_t header;
    int32_t entry;

    if (mybp->enableappend == 1) {
        /* append mode already in effect */
        return -1;
    }

    if (mybp->nextpage == mybp->rootpagenum) {
        /* emptry B-tree */
        return -2;
    }      

    if (mybp->cursoroffset != -1)  /* terminate current cursor */
        btree_stopcursor(bp);

    entry = findentrypoint(mybp, key, &pageaddr);
    if (entry == -9) return -9;

    setheader(&header, pageaddr);

    mybp->cursorpage = pageaddr;
    mybp->cursoroffset = (entry < 0) ? 0 : entry;
    mybp->cursorptr = (char *)pageaddr + hdrsize +
        mybp->cursoroffset * mybp->leafentrysize;

    cascadeunref(mybp, *(header.ppageaddrptr));
    return 0;
}


/*
 * btree_getcursor - retrieve the content of the object pointed by 
 *                   the current cursor
 *
 * - return 0 if OK, -5 if no cursor in effect, 
 *   -13 if no schema defined but request a particular field 
 *   -14 if schema is defined but cannot find the particular field
 *
 */
int btree_getcursor(btree_t *bp, void *key, const char *fieldname, void *value)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *src;
    int32_t fieldind;

    if (mybp->cursoroffset == -1) return -5;

    /* determine what to do with the payload */
    if ((fieldind = whichfield(mybp, fieldname)) < 0) 
        return fieldind;

    if (noswapkey)
        memcpy(key, mybp->cursorptr, mybp->keysize);
    else
        xplatform_swapbytes(key, mybp->cursorptr, mybp->keysize);

    if (value != NULL) {
        src = (char *)mybp->cursorptr + mybp->keysize;

        if (mybp->schema == NULL) 
            memcpy(value, src, mybp->valuesize);
        else
            extractfield(mybp, value, src, fieldind);
    }

    return 0;
}

    

/*
 * btree_advcusor - move the curosr one step forward
 *
 * - when reaching the end of the etree, invalidate the cursor 
 * - return 0 if OK, 1 if end of btree is reached,
 *   -5 if no cursor in effect, -9 if low level IO error
 *
 */
int btree_advcursor(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    hdr_t header;
    int32_t count;

    setheader(&header, mybp->cursorpage);

    if (mybp->cursoroffset == -1) 
        return -5;

    if (noswap) 
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);


    if (mybp->cursoroffset < (count - 1)) {      /* same cursor page */
        mybp->cursoroffset++;
        mybp->cursorptr = (char *)mybp->cursorptr + mybp->leafentrysize;
        return 0;
    } else {
        pagenum_t rightsibnum;

        if (noswap) 
            rightsibnum = *(header.rightsibnumptr);
        else
            xplatform_swapbytes(&rightsibnum, header.rightsibnumptr, 8);

        if (rightsibnum == -1) { /* already at the last leaf page */
            btree_stopcursor(bp);
            return 1;
        }  else {             /* cross over */
            void *nextpage;
            
            if ((nextpage = buffer_fix(mybp->buf, rightsibnum)) == NULL) {
                /* cannot fix next page */
                return -9;
            }

            /* Erase runtime trace */
            *(header.ppageaddrptr) = NULL;
            *(header.pentryptr) = -1;

            buffer_unref(mybp->buf, mybp->cursorpage);

            mybp->cursorpage = nextpage;
            mybp->cursoroffset = 0;
            mybp->cursorptr = (char *)mybp->cursorpage + hdrsize;
            return 0;
        }
    }
}


/*
 * btree_stopcursor - stop the cursor and release resources
 *
 * return 0 if OK, -5 if no cursor in effect
 */
int btree_stopcursor(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    hdr_t header;

    if (mybp->cursoroffset == -1) return -5;

    mybp->cursoroffset = -1;

    setheader(&header, mybp->cursorpage);

    /* Erase runtime trace */
    *(header.ppageaddrptr) = NULL;
    *(header.pentryptr) = -1;

    buffer_unref(mybp->buf, mybp->cursorpage);
    return 0;
}


/*
 * btree_beginappend - start a transaction to append records 
 *
 * - locate the right most leaf page and fix the rightmost path from 
 *   the B-tree root page to this leaf page
 * - return 0 if OK, -6 if illegal fillratio, -9 if low level IO error
 *   occurs, -1 if in conflict mode
 *
 */
int btree_beginappend(btree_t *bp, double fillratio)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr;
    hdr_t header;


    if (mybp->enableappend == 1) {
        /* append mode already in effect */
        return 0;
    }

    if (mybp->cursoroffset != -1) {
        /* curosr mode in effect */
        return -1;
    }

    if ((fillratio <= 0) || (fillratio > 1)) {
        /* invalid fillratio */
        return -6;
    }

    mybp->allowschema = 0;

    if ((mybp->nextpage == mybp->rootpagenum) &&     /* empty B-tree */
        ((pageaddr = buffer_emptyfix(mybp->buf, mybp->rootpagenum)) != NULL)) {
        int32_t count = 0;
        pagenum_t rightsibnum = -1;

        mybp->nextpage = mybp->rootpagenum + 1; 

        setheader(&header, pageaddr);

        if (noswap) {
            *(header.countptr) = count;
            *(header.rightsibnumptr) = rightsibnum;
        } else {
            xplatform_swapbytes(header.countptr, &count, 4);
            xplatform_swapbytes(header.rightsibnumptr, &rightsibnum, 8);
        }

        *(header.typeptr) = 'l';
        *(header.ppageaddrptr) = NULL;
        *(header.pentryptr) = -1;
    } else {
        void *ppageaddr;

        if ((ppageaddr = buffer_fix(mybp->buf, mybp->rootpagenum)) == NULL) {
            /*cannot fix root page */
            return -9;
        }

        setheader(&header, ppageaddr);
        *(header.ppageaddrptr) = NULL;
        pageaddr = sink(mybp, ppageaddr, 1);
    }

    if (pageaddr == NULL) {
        /* cannot fix the leaf page */
        return -9;
    } 

    mybp->enableappend = 1;
    mybp->appendpage = pageaddr;
    mybp->appendleafmax = (int32_t)(mybp->leafcapacity * fillratio);
    mybp->appendindexmax = (int32_t)(mybp->indexfanout * fillratio);

    return 0;
}



/*
 * btree_endappend - end the appending transaction
 *
 * - unref the path from append page to root
 * - unset the enableappend flag
 * - return 0 if OK, -7 if not in append mode 
 *
 */
int btree_endappend(btree_t *bp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    
    if (mybp->enableappend == 0) {
        /* not in append mode*/
        return -7;
    }

    cascadeunref(mybp, mybp->appendpage);
    mybp->enableappend = 0;
    return 0;
}


/*
 * btree_append - append the data object to the right most of the leaf page
 *
 * - if not inside an append transaction, make it a single-append transaction
 * - may cause split of ancestor nodes
 * - return 0 if OK, -8 if append a key out of order, -9 if low level IO
 *   error
 *
 */
int btree_append(btree_t *bp, const void *key, const void *value)
{
    int intx, res;
    mybtree_t *mybp = (mybtree_t *)bp;
    
    if (mybp->enableappend == 0) {
        intx = 0;
        btree_beginappend(bp, 1);
    } else /* part of a bulk append */
        intx = 1;

    if ((mybp->appendpage = append(mybp, mybp->appendpage, key, value, &res))
        == NULL) {
        /* either -8 (out of order key) or -9 (IO error) */
        return res;
    }

    if (intx == 0) 
        btree_endappend(bp);

    return 0;
}


/*
 * btree_stat - printout btree statistics
 *
 * - find max/min/avg fanout for index pages
 * - find max/min/avg utilization on leafpages
 * - return 0 if OK, and other error code if applicable
 * 
 */
int btree_printstat(btree_t *bp, FILE *fp)
{
    mybtree_t *mybp = (mybtree_t *)bp;
    void *pageaddr, *ppageaddr, *nextpageaddr;
    hdr_t header;
    pagenum_t leafpagecount, indexpagecount;
    int leafcapmax, leafcapmin, leafcaptotal;
    int indexcapmax, indexcapmin, indexcaptotal;
    int depth, rootcap;
    int count;
    pagenum_t rightsibnum;


    if (mybp->nextpage == mybp->rootpagenum) {        /* empty B-tree */
        return -2;
    } 

    if ((ppageaddr = buffer_fix(mybp->buf, mybp->rootpagenum)) == NULL) {
        /* cannot fix root page */
        return -9;
    }
    depth = 1;

    setheader(&header, ppageaddr);

    *(header.ppageaddrptr) = NULL;
    pageaddr = sink(mybp, ppageaddr, 0);
    if (pageaddr == NULL) {
        /* cannot fix the leaf page */
        return -9;
    } 

    /* record the parent location */
    setheader(&header, pageaddr);
    ppageaddr = *(header.ppageaddrptr);

    /* process leaf pages first */
    leafpagecount = 0;
    leafcapmax = 0; leafcapmin = mybp->leafcapacity;
    leafcaptotal = 0;
    do {

        setheader(&header, pageaddr);
        leafpagecount++;

        if (noswap) {
            count = *(header.countptr);
            rightsibnum = *(header.rightsibnumptr);
        } else {
            xplatform_swapbytes(&count, header.countptr, 4);
            xplatform_swapbytes(&rightsibnum, header.rightsibnumptr, 8);
        }


        leafcapmax = (leafcapmax > count) ? leafcapmax : count;
        leafcapmin = (leafcapmin < count) ? leafcapmin : count;
        leafcaptotal += count;
        
        if (rightsibnum != -1 ){
            if ((nextpageaddr = buffer_fix(mybp->buf, rightsibnum)) == NULL) {
                /* cannot fix next leaf page */
                return -9;
            }
        } else nextpageaddr = NULL;


        buffer_unref(mybp->buf, pageaddr);
        pageaddr = nextpageaddr;

    } while (pageaddr != NULL);

    /* recursively process index pages */
    rootcap = -1;
    indexpagecount = 0;
    indexcapmax = 0; indexcapmin = mybp->indexfanout;
    indexcaptotal = 0;

    while (ppageaddr != NULL) {
        depth++;
        pageaddr = ppageaddr;

        setheader(&header, pageaddr);
        ppageaddr = *(header.ppageaddrptr);;

        do {
            setheader(&header, pageaddr);

            indexpagecount++;
            
            if (noswap) {
                count = *(header.countptr);
                rightsibnum = *(header.rightsibnumptr);
            } else {
                xplatform_swapbytes(&count, header.countptr, 4);
                xplatform_swapbytes(&rightsibnum, header.rightsibnumptr, 8);
            }

            if (ppageaddr != NULL) { /* ignore root */
                indexcapmax = (indexcapmax > count) ? indexcapmax : count;
                indexcapmin = (indexcapmin < count) ? indexcapmin : count;
                indexcaptotal += count;
            } else {
                if (noswap) 
                    rootcap = *(header.countptr);
                else 
                    xplatform_swapbytes(&rootcap, header.countptr, 4);

                indexcaptotal += rootcap;
            }

            if (rightsibnum != -1) {
                if ((nextpageaddr = buffer_fix(mybp->buf, rightsibnum)) 
                    == NULL) {
                    /* cannot fix next index page */
                    return -9;
                }
            } else nextpageaddr = NULL;

            buffer_unref(mybp->buf, pageaddr);
            pageaddr = nextpageaddr;
        } while(pageaddr != NULL);
    }
    
    indexcapmin = (indexcapmin > indexcapmax) ? 0 : indexcapmin;

    fprintf(fp, "Btree space utilization:\n");
    fprintf(fp, "--------------------------------------------------------\n");

    if (sizeof(long int) == 8) 
        fprintf(fp, "Total pages:\t\t\t%ld\n", 
                (long int)(indexpagecount + leafpagecount));
     else 
        fprintf(fp, "Total pages:\t\t\t%lld\n",
                (long long int)(indexpagecount + leafpagecount));

    fprintf(fp, "Depth:\t\t\t\t%d\n\n", depth);

    if (sizeof(long int) == 8) 
        fprintf(fp, "Leaf pages:\t\t\t%ld\n", (long int)leafpagecount);
    else
        fprintf(fp, "Leaf pages:\t\t\t%lld\n", (long long int)leafpagecount);
    

    fprintf(fp, "Capacity:\t\t\t(%d)\n\n", mybp->leafcapacity);
    fprintf(fp, "  max utilization:\t\t%.2f%%\n", 
            leafcapmax * 100.0 / mybp->leafcapacity);
    fprintf(fp, "  min utilization:\t\t%.2f%%\n",
            leafcapmin * 100.0 / mybp->leafcapacity);
    fprintf(fp, "  avg utilization:\t\t%.2f%%\n\n",
           leafcaptotal * 100.0 / (mybp->leafcapacity * leafpagecount));

    if (sizeof(long int) == 8) 
        fprintf(fp, "Index pages:\t\t\t%ld\n", (long int)indexpagecount);
    else
        fprintf(fp, "Index pages:\t\t\t%lld\n", (long long int)indexpagecount);

    fprintf(fp, "Capacity:\t\t\t(%d)\n\n", mybp->indexfanout);

    if (rootcap != -1) {
        printf("  root utilization:\t\t%.2f%%\n", 
               rootcap * 100.0 / mybp->indexfanout);
    }
    fprintf(fp, "  max utilization:\t\t%.2f%%\n", 
            indexcapmax * 100.0 / mybp->indexfanout);
    fprintf(fp, "  min utilization:\t\t%.2f%%\n",
           indexcapmin * 100.0 / mybp->indexfanout);
    fprintf(fp, "  avg utilization:\t\t%.2f%%\n", 
           indexcaptotal * 100.0 / (mybp->indexfanout * indexpagecount));
    fprintf(fp, "--------------------------------------------------------\n\n");
    return 0;
}





/*
 *     Local routines 
 *
 */


/*
 * locateleaf - traverse down the B-tree to find the page whose key 
 *              range cover the insert key value
 *
 * - return the pointer to the leaf page, NULL on error
 *
 */
void *locateleaf(mybtree_t *mybp, void * pageaddr, const void *key)
{
    hdr_t header, childheader;
    int32_t entry;
    pagenum_t childpagenum;
    void *childpageaddr, *hitptr;

    setheader(&header, pageaddr);

    if (*(header.typeptr) == 'l') 
        return pageaddr; 

    entry = binarysearch(mybp, pageaddr, key);
    hitptr = (char *)pageaddr + hdrsize + mybp->indexentrysize * entry 
        + mybp->keysize;

    if (noswap)
        /* childpagenum = *(pagenum_t *)hitptr; */
        /* hitptr may be not properly aligned, some platform (e.g. ALPHA)
           complains about this, though it can still run; to be safe
           I memcpy the pagenum here */
           
        memcpy(&childpagenum, hitptr, 8);
    else
        xplatform_swapbytes(&childpagenum, hitptr, 8);

    if ((childpageaddr = buffer_fix(mybp->buf, childpagenum)) == NULL) {
        fprintf(stderr, "(DEBUG)locateleaf: cannot fix child page.\n");
        return NULL;
    }

    setheader(&childheader, childpageaddr);

    *(childheader.ppageaddrptr) = pageaddr;
    *(childheader.pentryptr) = entry;

    return locateleaf(mybp, childpageaddr, key);
}


/*
 * sink - return the leftmost/rightmost leaf page depending on "where",
 *        NULL on error
 *
 * where is 0 (for the leftmost) or 1 (rightmost)
 *
 */
void *sink(mybtree_t *mybp, void * pageaddr, int where)
{
    hdr_t header, childheader;
    int32_t entry, count;
    pagenum_t childpagenum;
    void *childpageaddr, *hitptr;
    

    setheader(&header, pageaddr);

    if (*(header.typeptr) == 'l') 
        return pageaddr; 

    if (where == 0) 
        entry = 0;
    else {
        if (noswap)
            count = *(header.countptr);
        else
            xplatform_swapbytes(&count, header.countptr, 4);

        entry = count - 1;
    }

    hitptr = (char *)pageaddr + hdrsize + mybp->indexentrysize * entry 
        + mybp->keysize;

    if (noswap)
        /* childpagenum = *(pagenum_t *)hitptr; */
        memcpy(&childpagenum, hitptr, 8);
    else
        xplatform_swapbytes(&childpagenum, hitptr, 8);

    if ((childpageaddr = buffer_fix(mybp->buf, childpagenum)) == NULL) {
        fprintf(stderr, "(DEBUG)sink: cannot fix child page.\n");
        return NULL;
    }

    setheader(&childheader, childpageaddr);
    *(childheader.ppageaddrptr) = pageaddr;
    *(childheader.pentryptr) = entry;

    return sink(mybp, childpageaddr, where);
}


/*
 * cascadeunref - release all the fixes (references) on the path to the root
 *
 */

void cascadeunref(mybtree_t *mybp, void *pageaddr)
{
    void *addr;
    hdr_t header;

    addr = pageaddr;
    while (addr != NULL) {
        void *ppageaddr;

        setheader(&header, addr);
        ppageaddr = *(header.ppageaddrptr);
        
        /* Erase runtime trace */
        *(header.ppageaddrptr) = NULL;
        *(header.pentryptr) = -1;

        buffer_unref(mybp->buf, addr);
        addr = ppageaddr;
    } 
    return;
}


/*
 * binarysearch - find the entry whose key is the maximum among all the
 *                keys less than the search key
 *
 * - return the offset in the array
 *
 */
int binarysearch(mybtree_t *mybp, const void *pageaddr, const void *key)
{
    int count, start, end, offset, recordsize;
    hdr_t header;
    const char *base;

    setheader(&header, pageaddr);
    
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    start = 0;
    end = count - 1;
    offset = (start + end) / 2;
    recordsize = (*(header.typeptr) == 'l') ? 
        mybp->leafentrysize : mybp->indexentrysize;

    base = (char *)pageaddr + hdrsize;
    do{
        if (end < start) return end;
        else {
            const void *pivot = base + offset * recordsize;
            switch (mybp->compare(key, pivot, mybp->keysize)) {
            case (0) : /* equal */
                return offset;
            case(1):  /* key larger than the key at pivot */
                start = offset + 1;
                offset = (start + end)/2;
                break;
            case(-1): /* key smaller than the key at pivot */
                end = offset - 1;
                offset = (start + end)/2;
                break;
            }  
        }
    } while (1);
}


/*
 * insert - insert records into the page pointed by pageaddr
 *
 * - split / create new pages if necessary
 * - return 0 if OK, -9 if low level IO error, -10 if too many bulk insert, 
 *
 */
int insert(mybtree_t *mybp, void *pageaddr, int32_t entry,
           int32_t newcount, const void *keys[], const void *values[])
{
    hdr_t header;
    int32_t maxcount, count;

    setheader(&header, pageaddr);

    maxcount = (*(header.typeptr) == 'l') ? 
                mybp->leafcapacity : mybp->indexfanout;
    
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    if ((count + newcount) > 2 * maxcount) {
        /* I cannot handle too many bulk inserts */
        return -10;
    }

    if ((count + newcount) <= maxcount) /* no split */    
        return simpleinsert(mybp, pageaddr, entry, newcount, keys, values);
    else 
        return splitinsert(mybp, pageaddr, entry, newcount, keys, values);
}
 

/*
 * simpleinsert - plugin the records at the position after entry
 *
 * - unref the path back to the root
 * - awlays return 0 if OK, 
 *
 */
int simpleinsert(mybtree_t *mybp, void *pageaddr, int32_t entry, 
                 int32_t newcount, const void *keys[], const void *values[])
{
    hdr_t header;
    int count;

    setheader(&header, pageaddr);

    plugin(mybp, pageaddr, entry, newcount, keys, values);
    
    /* increment the count */
    if (noswap)
        *(header.countptr) += newcount;
    else {
        xplatform_swapbytes(&count, header.countptr, 4);
        count += newcount;
        xplatform_swapbytes(header.countptr, &count, 4);
    }
    

    buffer_mark(mybp->buf, pageaddr);

    cascadeunref(mybp, pageaddr);
    return 0;
}


/*
 * plugin - plugin "newcount" new keys and values after position entry
 *
 * - move entries after entry "newcount" position to the right 
 * - populate the field or directly copy the value depending on whether
 *   schema is defined
 *
 */
void plugin(mybtree_t *mybp, void *pageaddr, int32_t entry, int32_t newcount,
            const void *keys[], const void *values[])
{
    int keysize, recordsize, offset, movingcount, count;
    char *dest, *src;
    hdr_t header;
    int index;

    setheader(&header, pageaddr);
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    keysize = mybp->keysize;
    recordsize = (*(header.typeptr) == 'l') ? 
        mybp->leafentrysize : mybp->indexentrysize;

    offset = entry + 1;
    movingcount = count - offset;

    src = (char *)pageaddr + hdrsize + offset * recordsize;
    dest = src + newcount * recordsize;
    memmove(dest, src, movingcount * recordsize);

    dest = src;

    for (index = 0; index < newcount; index++) {
        /* store the key */
        if (noswapkey)
            memcpy(dest, keys[index], keysize);
        else
            xplatform_swapbytes(dest, keys[index], keysize);

        if (*(header.typeptr) == 'i') {
            /* index page */

            /* treat index page separately, which does not involves schema;
               the values are pagenumbers stored in platform-specific format
            */
            if (noswap)
                memcpy(dest + keysize, values[index], sizeof(pagenum_t));
            else
                xplatform_swapbytes(dest + keysize, values[index], 
                                    sizeof(pagenum_t));
        } 
        else {
            /* leaf page */

            if (mybp->schema == NULL) 
                /* no schmea defined , treat values as binary blobs */
                memcpy(dest + keysize, values[index], mybp->valuesize);

            else 
                /* use schema to compactly store the value */
                populatefield(mybp, dest + keysize, values[index], 
                              mybp->schema->fieldnum);
        }
        dest = dest + recordsize;
    }

    return ;
}


/*
 * splitinsert - split the current page into two and inser the new entries
 *
 * - special treatment for root page split because it should always has page
 *   number mybp->rootpagenum
 * - return 0 if insertion complete successfully, -9 if low level IO error
 *
 */
int splitinsert(mybtree_t *mybp, void *pageaddr, int32_t entry, 
                 int32_t newcount, const void *keys[], const void *values[])
{
    void *newaddr1, *newaddr2;
    void /* *newbase1, */ *newbase2;
    void *ppageaddr;
    int32_t pentry, count, count1, count2, c1, c2, totalcount;
    int32_t newcount1, newcount2; 
    int32_t entry1 = -1, entry2 = -1;
    hdr_t newhd1, newhd2, header;
    pagenum_t pagenum;
    void *newvalue;

    setheader(&header, pageaddr);
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    totalcount = count + newcount;
    count2 = totalcount / 2;          /* final number of entry on page2 */
    count1 = totalcount - count2;     /* final number of entry on page1 */

    /* calculate how to split */
    if (count1 <= entry + 1) {
        c1 = count1;                  /* split count */
        newcount1 = 0;
        entry2 = entry - count1;
    } else if (count1 <= entry + 1 + newcount) {
        c1 = entry + 1;
        newcount1 = count1 - c1;
        entry1 = entry;
        entry2 = -1;
    } else /* entry + 1 + newcount < count1 */ {
        c1 = count1 - newcount;
        newcount1 = newcount;
        entry1 = entry;
    }

    c2 = count - c1;
    newcount2 = newcount - newcount1;

    if (buffer_pagenum(mybp->buf, pageaddr) == mybp->rootpagenum) {
        if (splitroot(mybp, pageaddr, c1, c2, &newaddr1, &newaddr2) != 0)
            return -9;
    }
    else {
        /* pageaddr1 is set to pageaddr */
        if (splitpage(mybp, pageaddr, c1, c2, &newaddr1, &newaddr2) != 0)
            return -9;
    }

    /* newbase1 = (char *)newaddr1 + hdrsize; */
    newbase2 = (char *)newaddr2 + hdrsize;

    setheader(&newhd1, newaddr1);
    setheader(&newhd2, newaddr2);

    /* find the right position to install the new record */
    if (newcount1 != 0) {
        plugin(mybp, newaddr1, entry1, newcount1, &keys[0], &values[0]);
        
        if (noswap)
            *(newhd1.countptr) = count1;
        else
            xplatform_swapbytes(newhd1.countptr, &count1, 4);

        buffer_mark(mybp->buf, newaddr1);
    } 
    if (newcount2 != 0) {
        plugin(mybp, newaddr2, entry2, newcount2, &keys[newcount1],
               &values[newcount1]);

        if (noswap)
            *(newhd2.countptr) = count2;
        else
            xplatform_swapbytes(newhd2.countptr, &count2, 4);

        buffer_mark(mybp->buf, newaddr2);
    }
        
    ppageaddr = *(newhd2.ppageaddrptr);
    pentry = *(newhd2.pentryptr);       
    pagenum = mybp->nextpage - 1;
    
    /* Erase runtime trace */
    *(newhd1.ppageaddrptr) = NULL;
    *(newhd1.pentryptr) = -1;

    *(newhd2.ppageaddrptr) = NULL;
    *(newhd2.pentryptr) = -1;

    buffer_unref(mybp->buf, newaddr1);
    buffer_unref(mybp->buf, newaddr2);



    /* pass the pagenum in platform format */
    newvalue = &pagenum;

    if (noswapkey)
        return insert(mybp, ppageaddr, pentry, 1, (const void **)&newbase2, 
                      (const void **)&newvalue);
    else {
        xplatform_swapbytes(platformkey, newbase2, mybp->keysize);
        return insert(mybp, ppageaddr, pentry, 1, 
                      (const void **)&platformkeyptr,
                      (const void **)&newvalue);
    }
        
}


/*
 * splitroot - create two new pages and split the content on the root
 *             page to the two new pages
 *
 * - this function works with both the leaf case and index case
 * - count1 is the number of entries to be assigned to the first page
 * - return 0 if OK, -9 if low level IO error
 *
 */
int splitroot(mybtree_t *mybp, void *pageaddr, int32_t cnt1, int32_t cnt2,
              void **newaddr1ptr, void **newaddr2ptr)
{
    hdr_t header, header1, header2;
    void *payload, *payload1, *payload2;
    int recordsize,  movingsize, dummycount = 1;
    pagenum_t pagenum1, pagenum2, dummynum = -1;
    
    setheader(&header, pageaddr);
    recordsize = (*(header.typeptr) == 'l') ? 
        mybp->leafentrysize : mybp->indexentrysize;

    pagenum1 = mybp->nextpage;
    pagenum2 = mybp->nextpage + 1;
    if (((*newaddr1ptr = buffer_emptyfix(mybp->buf, pagenum1)) == NULL) ||
        ((*newaddr2ptr = buffer_emptyfix(mybp->buf, pagenum2)) == NULL)){
        fprintf(stderr, "(DEBUG)splitroot: cannot allocate buffer frame.\n");
        return -9;
    }
    mybp->nextpage += 2;

    setheader(&header1, *newaddr1ptr);
    setheader(&header2, *newaddr2ptr);

    if (noswap) {
        *(header1.countptr) = cnt1;
        *(header2.countptr) = cnt2;
        *(header1.rightsibnumptr) = pagenum2;
        *(header2.rightsibnumptr) = dummynum;
    } else {
        xplatform_swapbytes(header1.countptr, &cnt1, 4);
        xplatform_swapbytes(header2.countptr, &cnt2, 4);

        xplatform_swapbytes(header1.rightsibnumptr, &pagenum2, 8);
        xplatform_swapbytes(header2.rightsibnumptr, &dummynum, 8);
    }

    *(header1.typeptr) = *(header2.typeptr) = *(header.typeptr);

    /* let the second child know where to hook */
    *(header2.ppageaddrptr) = pageaddr;
    *(header2.pentryptr) = 0;  /* used for insertion at higher level */

    /* also let the first child know where to hook */
    *(header1.ppageaddrptr) = pageaddr;
    *(header1.pentryptr) = 0;

    /* copy data across to the two pages */
    payload = (char *)pageaddr + hdrsize;
    payload1 = (char *)(*newaddr1ptr) + hdrsize;
    payload2 = (char *)(*newaddr2ptr) + hdrsize;

    movingsize = recordsize * cnt1;
    memcpy(payload1, payload, movingsize);
    memcpy(payload2, (char *)payload + movingsize, recordsize * cnt2);
    buffer_mark(mybp->buf, *newaddr1ptr);
    buffer_mark(mybp->buf, *newaddr2ptr);

    /* update the root page to record the fisrt child */
    *(header.typeptr) = 'i';

    if (noswap) {
        *(header.countptr) = dummycount;

        /* init/store key zero */
        memset(payload, 0, mybp->keysize);
        /* *(pagenum_t *)((char *)payload + mybp->keysize) = pagenum1; */

        /* install pagenum/value */
        memcpy((char *)payload + mybp->keysize, &pagenum1, 8);

    } else {
        xplatform_swapbytes(header.countptr, &dummycount, 4);
        memset(payload, 0, mybp->keysize);
        xplatform_swapbytes((char *)payload + mybp->keysize, &pagenum1, 8);
    }

    buffer_mark(mybp->buf, pageaddr);

    return 0;
}



/*
 * splitpage - create a new page, and move the specified number of entries
 *             to the new page 
 *
 * - this function works with both the leaf case and index case
 * - *newaddr1 is set to be pageaddr
 * - return 0 if OK, -9 if low level error
 *
 */
int splitpage(mybtree_t *mybp, void *pageaddr, int32_t cnt1, int32_t cnt2,
              void **newaddr1ptr, void **newaddr2ptr)
{
    hdr_t header, header2;
    void *payload, *payload2;
    int32_t recordsize;
    pagenum_t pagenum2;

    setheader(&header, pageaddr);

    recordsize = (*(header.typeptr) == 'l') ? 
        mybp->leafentrysize : mybp->indexentrysize;

    pagenum2 = mybp->nextpage;
    if ((*newaddr2ptr = buffer_emptyfix(mybp->buf, pagenum2)) == NULL) {
        fprintf(stderr, "(DEBUG)splitpage: cannot allocate buffer frame.\n");
        return -9;
    }
    mybp->nextpage++;


    setheader(&header2, *newaddr2ptr);
    if (noswap) {
        *(header.countptr) = cnt1;
        *(header2.countptr) = cnt2;
        *(header2.rightsibnumptr) = *(header.rightsibnumptr);
        *(header.rightsibnumptr) = pagenum2;
    } else {
        xplatform_swapbytes(header.countptr, &cnt1, 4);
        xplatform_swapbytes(header2.countptr, &cnt2, 4);

        /* *header.rightsibnumptr stores the native format */
        memcpy(header2.rightsibnumptr, header.rightsibnumptr, 8);

        xplatform_swapbytes(header.rightsibnumptr, &pagenum2, 8);
    }

    *(header2.typeptr) = *(header.typeptr);

    /* copy data across to the two pages */
    payload = (char *)pageaddr + hdrsize;
    payload2 = (char *)(*newaddr2ptr) + hdrsize;
    memcpy(payload2, (char *)payload + recordsize * cnt1, recordsize * cnt2);

    buffer_mark(mybp->buf, pageaddr);
    buffer_mark(mybp->buf, *newaddr2ptr);

    /* let the second child know where to hook to the parent */
    *(header2.ppageaddrptr) = *(header.ppageaddrptr);
    *(header2.pentryptr) = *(header.pentryptr);

    *newaddr1ptr = pageaddr; 
    return 0;
}


/*
 * unplug - remove the record at offset entry and pack the remainders
 *
 */
void unplug(mybtree_t *mybp, void *pageaddr, int32_t entry)
{
    void *src, *dest, *hitptr;
    int count, movingsize;
    hdr_t header;

    hitptr = (char *)pageaddr + hdrsize + entry * mybp->leafentrysize;
    
    setheader(&header, pageaddr);
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    movingsize = (count - (entry + 1)) * mybp->leafentrysize;
    dest = hitptr;
    src = (char *)hitptr + mybp->leafentrysize;
    memmove(dest, src, movingsize);

    count--;
    if (noswap)
        *(header.countptr) = count;
    else
        xplatform_swapbytes(header.countptr, &count, 4);

    buffer_mark(mybp->buf, pageaddr);
    
    return;
}



/*
 * append - append the data object to the right most of the current page
 *
 * - split (unevenly) if page is full
 * - return a pointer to the original page if no split occurs; otherwise, 
 *   unref the original appendpage and return a pointer to the new (split) 
 *   pageaddr
 * - error code is -8 if the append is out of order, 
 *   or -9 if low level IO error
 *
 */
void *append(mybtree_t *mybp, void *pageaddr, 
                      const void *key, const void *value, int *pcode)
{
    hdr_t header;
    int32_t maxcount, count;
    char *base;
    const void *lastkey;

    setheader(&header, pageaddr);

    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    base = (char *)pageaddr + hdrsize;
    if (*(header.typeptr) == 'l') {
        int recordsize = mybp->leafentrysize;

        maxcount = mybp->appendleafmax;
        if (count > 0) {
            lastkey = base + recordsize * (count - 1);
            if (mybp->compare(key, lastkey, mybp->keysize) < 0) {
                *pcode = -8;
                return NULL;
            }
        }
    }
    else 
        maxcount = mybp->appendindexmax;


    if (count < maxcount) {
        /* no split, append to the end, plugin after the last entry */ 

        plugin(mybp, pageaddr, count - 1, 1, &key, &value);

        count++;
        if (noswap)
            *(header.countptr) = count;
        else
            xplatform_swapbytes(header.countptr, &count, 4);

        buffer_mark(mybp->buf, pageaddr);
        return pageaddr;
    } 
    else 
        return splitappend(mybp, pageaddr, key, value, pcode);
}


/*
 * splitappend - create a new page to hold the appending record
 *
 * - uneven split, all the existing records remain on the original page 
 * - special treatment for root page split because it should always has 
 *   page number mybp->rootpagenum
 * - return a pointer to the new split page's address
 * - error code: -9 if low level IO error occurs
 */
void *splitappend(mybtree_t *mybp, void *pageaddr,
                           const void *key, const void *value,
                           int *pcode)
{
    void *newaddr1, *newaddr2;
    hdr_t newhd1, newhd2;
    void /* *newbase1, */ *newbase2;
    void *ppageaddr;
    int32_t count, count1, count2;
    hdr_t header;
    pagenum_t pagenum;

    setheader(&header, pageaddr);
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    count1 = count;
    count2 = 0;

    if (buffer_pagenum(mybp->buf, pageaddr) == mybp->rootpagenum) {
        if (splitroot(mybp, pageaddr, count1, count2, &newaddr1, &newaddr2)
            != 0) {
            *pcode = -9;
            return NULL;
        }
    }
    else {
        /* pageaddr1 is set to pageaddr */
        if (splitpage(mybp, pageaddr, count1, count2, &newaddr1, &newaddr2) 
            != 0) {
            *pcode = -9;
            return NULL;
        }
    }

    /* newbase1 = (char *)newaddr1 + hdrsize; */
    newbase2 = (char *)newaddr2 + hdrsize;

    setheader(&newhd1, newaddr1);
    setheader(&newhd2, newaddr2);

    /* plugin the appending object in the first slot of the second page */
    plugin(mybp, newaddr2, -1, 1, &key, &value);
    count2 = 1;
    if (noswap)
        *(newhd2.countptr) = count2;
    else
        xplatform_swapbytes(newhd2.countptr, &count2, 4);
    buffer_mark(mybp->buf, newaddr2);

    /* Erase the runtime trace */
    *(newhd1.ppageaddrptr) = NULL;
    *(newhd1.pentryptr) = -1;

    /* unref the passed page */
    buffer_unref(mybp->buf, newaddr1);

    /* append an index entry at higer level */
    ppageaddr = *(newhd2.ppageaddrptr);
    pagenum = mybp->nextpage - 1;

    *(newhd2.ppageaddrptr) = append(mybp, ppageaddr, newbase2, &pagenum, 
                                    pcode);
    
    return (*(newhd2.ppageaddrptr) == NULL) ? NULL : newaddr2;
}



/*
 * findentrypoint - find where on which leaf page is the key residing 
 *
 * - return the offset on the leaf page and store the address of the leaf
 *    page in pageaddr; -9 if low level error occurs
 * - NOTE: the entry could be -1, which should not be treated as an error
 *
 */
int32_t findentrypoint(mybtree_t *mybp, const void *key, void **pageaddrptr)
{
    void *ppageaddr, *pageaddr;
    hdr_t header;
    int32_t entry;

    if ((ppageaddr = buffer_fix(mybp->buf, mybp->rootpagenum)) == NULL) {
        fprintf(stderr, "(DEUBG)findentrypoint: cannot fix root page.\n");
        return -9;
    }

    setheader(&header, ppageaddr);
    *(header.ppageaddrptr) = NULL;

    pageaddr = locateleaf(mybp, ppageaddr, key);
    if (pageaddr == NULL) {
        fprintf(stderr, "(DEBUG)findentrypoint: cannot fix leaf page.\n");
        return -9;
    } 

    entry = binarysearch(mybp, pageaddr, key); 
    *pageaddrptr = pageaddr;
    
    return entry;
}



/*
 * inorder - check whether the keys being bulk inserted are consistent 
 *           with the keys at and after the anchor point 
 *
 * - return 1 if in order, -4 not inoder; -9 if low level IO error occurs
 *
 */
int inorder(mybtree_t *mybp, void *pageaddr, int32_t entry, 
            int32_t number, const void *keys[])
{
    const void *anchorkey, *nextkey = NULL;
    int i, foundnextkey, count;
    hdr_t header;
    void *nextpage = NULL;

    /* check the anchor */
    if (entry != -1) {
        anchorkey = (char *)pageaddr + hdrsize + entry * mybp->leafentrysize;
        if (mybp->compare(anchorkey, keys[0], mybp->keysize) > 0) 
            return -4;
    }

    /* check the updating keys */
    for (i = 0; i < number - 1; i++) 
        if (mybp->compare(keys[i], keys[i + 1], mybp->keysize) > 0) 
            return -4;

    /* check the key after anchor */
    setheader(&header, pageaddr);
    foundnextkey = 0;
    if (noswap)
        count = *(header.countptr);
    else
        xplatform_swapbytes(&count, header.countptr, 4);

    
    if (entry + 1 < count) { 
        nextkey = (char *)pageaddr + hdrsize + 
            (entry + 1) * mybp->leafentrysize;
        foundnextkey = 1;
        nextpage = NULL;
    } else {
        pagenum_t rightsibnum;

        if (noswap)
            rightsibnum = *(header.rightsibnumptr);
        else
            xplatform_swapbytes(&rightsibnum, header.rightsibnumptr, 8);

        while ((rightsibnum != -1) && (!foundnextkey)) {
            if ((nextpage = buffer_fix(mybp->buf, rightsibnum)) == NULL) {
                fprintf(stderr, "(DEUBG)inorder: cannot fix next page\n");
                return -9;
            }

            setheader(&header, nextpage);
            if (noswap)
                count = *(header.countptr);
            else
                xplatform_swapbytes(&count, header.countptr, 4);

            if (count > 0) {
                foundnextkey = 1;
                nextkey = (char *)nextpage + hdrsize;
            } else {
                if (noswap)
                    rightsibnum = *(header.rightsibnumptr);
                else
                    xplatform_swapbytes(&rightsibnum, header.rightsibnumptr,8);

                /* Erase runtime trace */
                *(header.ppageaddrptr) = NULL;
                *(header.pentryptr) = -1;

                buffer_unref(mybp->buf, nextpage);
                nextpage = NULL;
            }
        }
    }

    if (foundnextkey) {
        if (mybp->compare(keys[number - 1], nextkey, mybp->keysize) > 0) {
            /* last key greater than the key after anchor */
            return -4;
        }
    }

    /* release the (remote) right sibling page */
    if (nextpage != NULL) {

        setheader(&header, nextpage);

        /* Erase runtime trace */
        *(header.ppageaddrptr) = NULL;
        *(header.pentryptr) = -1;

        buffer_unref(mybp->buf, nextpage);
    }

    return 1;
}


/* 
 * setheader - install pointer to the proper position at the beginning
 *             of a page
 *
 */
void setheader(hdr_t *hdrptr, const void *pageaddr)
{
    hdrptr->rightsibnumptr = (pagenum_t *)pageaddr;
    hdrptr->ppageaddrptr = (void **)((char *)pageaddr + 8);
    hdrptr->countptr = (int32_t *)((char *)pageaddr + 16);
    hdrptr->pentryptr = (int32_t *)((char *)pageaddr + 20);
    hdrptr->typeptr = (char *)((char *)pageaddr + 24);
    return;
}
    

/*
 * extractfield - extract value from a byte string 
 *
 */
void extractfield(mybtree_t *mybp, void *value, const void *src, 
                  int32_t fieldind)
{
    const void *fieldptr;
    void *memberptr;
    int32_t size;

    if (fieldind < mybp->schema->fieldnum) {
        /* a particular field */
        fieldptr = (const char *)src + mybp->schema->field[fieldind].offset;
        size = mybp->schema->field[fieldind].size;

        if (noswap) 
            memcpy(value, fieldptr, size);
        else
            xplatform_swapbytes(value, fieldptr, size);
    } else {
        /* the whole structure is needed */
        int memberind;

        for (memberind = 0; memberind < fieldind; memberind++) {
            size = mybp->schema->field[memberind].size;

            fieldptr = (const char *)src + 
                mybp->schema->field[memberind].offset;
            memberptr = (char *)value + mybp->scb->member[memberind].offset;

            if (noswap) 
                memcpy(memberptr, fieldptr, size);
            else
                xplatform_swapbytes(memberptr, fieldptr, size);
        }
    }
    return;
}

/*
 * whichfield - determine the field index of the "fieldname" 
 *
 * - return 0 if no schema defined and request the whole field
 *          -13 if no schema defined and request a particular field
 *          -14 if schema defined but the field not found
 *          schema->fieldnum if schema defined and request the whole field
 *             fieldindex if schema defined and the fieldname found
 */
int whichfield(mybtree_t *mybp, const char *fieldname)
{
    int fieldind;

    if (mybp->schema != NULL) {
        if ((fieldname == NULL) || (strcmp(fieldname, "*") == 0)) 
            return mybp->schema->fieldnum;
        else
            /* look at the field name cache */
            if ((mybp->fieldname != NULL) &&
                (strcmp(fieldname, mybp->fieldname) == 0)) 
                return mybp->fieldind;
    
        /* check the field name one by one */
        for (fieldind = 0; fieldind < mybp->schema->fieldnum; fieldind++) 
            if (strcmp(fieldname, mybp->schema->field[fieldind].name) == 0) {
                /* update cache */
                mybp->fieldname = mybp->schema->field[fieldind].name;
                mybp->fieldind = fieldind;
                break;
            }
        if (fieldind == mybp->schema->fieldnum)
            /* the field name not found, we've completed the loop */
            return -14;
        else
            return fieldind;
    } else {
        /* no schema defined */
        if (fieldname == NULL) 
            /* the fieldind is not used if schema is not defined */
            return 999;
        else
            /* no schema defined, but request a particular field */            
            return -13;
    }
}
    

/*
 * populatefield - set value to proper position in a byte string 
 *
 */
void populatefield(mybtree_t *mybp, void *dest, const void *value, 
                   int32_t fieldind)
{
    void *fieldptr;
    const void *memberptr;
    int32_t size;

    if (fieldind < mybp->schema->fieldnum) {
        /* a particular field */
        fieldptr = (char *)dest + mybp->schema->field[fieldind].offset;
        size = mybp->schema->field[fieldind].size;

        if (noswap) 
            memcpy(fieldptr, value, size);
        else
            xplatform_swapbytes(fieldptr, value, size);
    } else {
        /* the whole structure is needed */
        int memberind;

        for (memberind = 0; memberind < fieldind; memberind++) {
            size = mybp->schema->field[memberind].size;

            fieldptr = (char *)dest + mybp->schema->field[memberind].offset;
            memberptr = (char *)value + mybp->scb->member[memberind].offset;

            if (noswap) 
                memcpy(fieldptr, memberptr, size);
            else
                xplatform_swapbytes(fieldptr, memberptr, size);
        }
    }
    return;
}


/*
 * numeral_compare - default numeral comparison function
 *
 * - key1 is in foreign/platform-specfic variable format 
 * - key2 is in native/storage-specific format
 * - return 1, 0, -1 if key1 >, =, or < key2
 *
 */
int numeral_compare(const void *key1, const void *key2, int size)
{
    unsigned char keybuf1[8], keybuf2[8];

    /* this excessive memcpy is to avoid complaits from platforms that 
       have strict alignment requirement, type casting is not enough to 
       subdue the warnings */
    memcpy(keybuf1, key1, size);
    if (noswapkey) 
        memcpy(keybuf2, key2, size);
    else
        xplatform_swapbytes(keybuf2, key2, size);
    
    /* now cast the key buffers to proper data type for comparison */
    if (strcmp(keytype, "int32_t") == 0) {
        if (*(int32_t *)keybuf1 > (*(int32_t *)keybuf2)) 
            return 1;
        else if (*(int32_t *)keybuf1 < (*(int32_t *)keybuf2)) 
            return -1;
        else
            return 0;
    }

    if (strcmp(keytype, "int64_t") == 0) {
        if (*(int64_t *)keybuf1 > (*(int64_t *)keybuf2)) 
            return 1;
        else if (*(int64_t *)keybuf1 < (*(int64_t *)keybuf2)) 
            return -1;
        else
            return 0;
    }

    
    if (strcmp(keytype, "uint32_t") == 0) {
        if (*(uint32_t *)keybuf1 > (*(uint32_t *)keybuf2)) 
            return 1;
        else if (*(uint32_t *)keybuf1 < (*(uint32_t *)keybuf2)) 
            return -1;
        else
            return 0;
    }
 
    if (strcmp(keytype, "uint64_t") == 0) {
        if (*(uint64_t *)keybuf1 > (*(uint64_t *)keybuf2)) 
            return 1;
        else if (*(uint64_t *)keybuf1 < (*(uint64_t *)keybuf2)) 
            return -1;
        else
            return 0;
    }
   
    if ((strcmp(keytype, "float") == 0) || 
        (strcmp(keytype, "float32_t") == 0)) {
        if (*(float *)keybuf1 > (*(float *)keybuf2)) 
            return 1;
        else if (*(float *)keybuf1 < (*(float *)keybuf2)) 
            return -1;
        else
            return 0;
    } 

    if ((strcmp(keytype, "double") == 0) || 
        (strcmp(keytype, "float64_t") == 0)) {
        if (*(double *)keybuf1 > (*(double *)keybuf2)) 
            return 1;
        else if (*(double *)keybuf1 < (*(double *)keybuf2)) 
            return -1;
        else
            return 0;
    }
    
    /* this point should never be reached */
    exit(1);
    return 0;
}
