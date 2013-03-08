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
 * code.c -  conversion between different octant address and locational key
 *
 *
 */

#include <stdlib.h> 
#include <string.h>
#include <assert.h>

#include "code.h"
#include "xplatform.h"
#include "expandtable.h"
#include "extracttable.h"

static const int theMaxLevelP1 = ETREE_MAXLEVEL + 1;
static const int theTimeStepOffset = (ETREE_MAXLEVEL + 1) / 8 * 3 + 1;
static const int theMortonBytes = (ETREE_MAXLEVEL + 1) / 8 * 3;
static const int theMaxOffsetP1 = (ETREE_MAXLEVEL + 1) * 3;
static endian_t  theEndianness  = unknown_endianness;

static void setprefix(etree_t *ep, unsigned int time, void *toptr);
static void getprefix(etree_t *ep, void *fromptr, unsigned int *ptimestep);


/**
 * code_addr2key - convert the octant address to the locational key
 *
 * - check the level, which is the only source of error
 * - call code_coord2morton to convert from the coordinate
 * - set the last byte of the key properly according to the type of the oct
 *   for index oct, do nothing;
 *   for leaf oct, set the most significant bit of the least significant 
 *   byte to 1
 * - return 0 if OK, -1 on error
 *
 */
int code_addr2key(etree_t *ep, etree_addr_t addr, void *key)
{
    /* 
       etree_addr_t checkaddr;
    */


    if (addr.level >= theMaxLevelP1) 
        return -1;

    /* Use the newer version */
    code_coord2morton(theMaxLevelP1, addr.x, addr.y, addr.z, (char *)key + 1); 


    /* for debug */
    /*
    code_morton2coord(theMaxLevelP1, key + 1, &checkaddr.x, &checkaddr.y,
                      &checkaddr.z);
    if ((addr.x != checkaddr.x)  ||
        (addr.y != checkaddr.y) ||
        (addr.z != checkaddr.z)) {
        fprintf(stderr, "New conversion routine not correct.\n");
        exit(-1);
    }
    */


    *(unsigned char *)key = (unsigned char)addr.level;
    if (addr.type == ETREE_LEAF) 
        *(unsigned char *)key |= 0x80;

    if (ep->dimensions == 4) {
        /* an ad-hoc solution, subject to future modification */
        setprefix(ep, addr.t, (char *)key + theTimeStepOffset);
    }
    return 0;
}


/*
 * code_key2addr - convert the locational code to the octant address
 *
 * - check the level, which is the only source of error
 * - call code_morton2coord to convert to the coordinate
 * - extract the octant level and type from the last byte
 *   the most significant bit of the least significant byte is 1 for leaf 
 *   and 0 for index
 * - return 0 if OK, -1 on error
 *
 */
int code_key2addr(etree_t *ep, void *key, etree_addr_t *paddr)
{
    unsigned char LSB;
    int level;

    LSB = *(unsigned char *)key;
    level = LSB & 0x7F;

    if (level >= theMaxLevelP1) 
        return -1;

    paddr->level = level;
    paddr->type = (LSB & 0x80) ? ETREE_LEAF : ETREE_INTERIOR;
    code_morton2coord(theMaxLevelP1, (char *)key + 1, 
                      &paddr->x, &paddr->y, &paddr->z);

    if (ep->dimensions == 4) {
        getprefix(ep, (char *)key + theTimeStepOffset, &paddr->t);
    }

    return 0;
}

/**
 * code_setlevel 
 *
 */
void code_setlevel(void *key, int level, etree_type_t type)
{
    *(unsigned char *)key = (unsigned char)(level);

    if (type == ETREE_LEAF) 
        *(unsigned char *)key |= 0x80;  
    
    return;
}

    

/*
 * code_isancestorkey - check whether ancestorkey is an ancestor ocant 
 *                      (either leaf or index) of the child octant
 *
 * - assume the keys are for 3D dataset, ignore the timestep field
 * - an octant is an ancestor of itself
 * - return 1 if true, 0 otherwise
 *
 */
int code_isancestorkey(const void *ancestorkey, const void *childkey)
{
    unsigned char LSB_an, LSB_ch;
    int level_an, level_ch;
    int offset, byte, inbyteoffset, level, i;
    unsigned mask[8] ={0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const unsigned char *byteunit_an, *byteunit_ch;

    LSB_an = *(unsigned char *)ancestorkey;
    LSB_ch = *(unsigned char *)childkey;

    level_an = LSB_an & 0x7F;
    level_ch = LSB_ch & 0x7F;

    if (level_ch < level_an) 
        /* the "child" is at higher level than the "parent" */
        return 0; 

    byteunit_an = (const unsigned char *)ancestorkey + 1;
    byteunit_ch = (const unsigned char *)childkey + 1;
    
    if (level_ch == level_an) {
        if (memcmp(byteunit_an, byteunit_ch, theMortonBytes) == 0) 
            /* exactly same octant addresses */
            return 1;
        else 
            return 0;
    }


    /* compare the prefix of the child and parent */
    offset = theMaxOffsetP1;

    for (level = 0; level <= level_an; level++) 
        for (i = 0; i < 3; i++) {
            offset--;
            byte = offset / 8;
            inbyteoffset = offset % 8;
            if ((*(byteunit_an + byte) & mask[inbyteoffset]) !=
                (*(byteunit_ch + byte) & mask[inbyteoffset])) 
                return 0;
        }

    return 1;
}


/*
 * code_derivechildkey - construct the child key in the specified branch 
 *                       from the parent key
 *
 * - assume the keys are for 3D dataset; ignore the timestep field
 * - the children are all leafs
 * - return 0 if OK, -1 on error
 *
 */
int code_derivechildkey(const void *key, void *childkey, int branch)
{
    unsigned char LSB;
    int level;

    LSB = *(unsigned char *)key;
    level = LSB & 0x7F;

    if (level >= (int)ETREE_MAXLEVEL) 
        return -1;

    /* copy the prefix of the parent's key and set the branch at the new
       level */
    memcpy((char *)childkey + 1, (const char *)key + 1,  theMortonBytes);
    code_setbranch((char *)childkey + 1, level + 1, branch);

    /* set the child's level */
    *(unsigned char *)childkey = (unsigned char)(level + 1);
    *(unsigned char *)childkey |= 0x80;  

    return 0;
}


/*
 * code_extractbranch - extract the three bits for "level" 
 *
 * - assume the key is for 3D dataset, ignore the timestep field
 * - level 0 corresponds to the leading three bits in the morton code
 * - return the branch (0 - 7) at the specified level, -1 on error
 *
 */
int code_extractbranch(const void *morton, int level)
{
    const unsigned char *byteunit;
    int offset, byte, inbyteoffset;
    unsigned mask[8] ={0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    int branch = 0;
    int IxBit, IyBit, IzBit;

    byteunit = (const unsigned char *)(morton);
    offset = theMaxOffsetP1 - 1;
    offset = offset - 3 * level;

    byte = offset / 8;
    inbyteoffset = offset % 8;
    IzBit = *(byteunit + byte) & mask[inbyteoffset];
    branch = (IzBit == 0) ? 0 : 1;
    offset--;

    byte = offset / 8;
    inbyteoffset = offset%8;
    IyBit = *(byteunit + byte) & mask[inbyteoffset];
    branch <<= 1;
    branch = (IyBit == 0) ? branch : branch | 0x1;
    offset--;
    
    byte = offset / 8;
    inbyteoffset = offset % 8;
    IxBit = *(byteunit + byte) & mask[inbyteoffset];
    branch <<= 1;
    branch = (IxBit == 0) ? branch : branch | 0x1;

    return branch;
}


/*
 * code_comparekey - compare the locational keys of two octants 
 *
 * - comparison starts from the most significant byte to the next to
 *   the least significant byte
 * - treat the least significant bytes specially to spot octants
 *   who only differ in their "type" and report them as equal
 * - return -1,0 or 1 respectively if key1 <, =, > key2
 *
 */
int code_comparekey(const void *key1, const void *key2, int size)
{
    int i;
    unsigned char v1,v2;

    for (i = size - 1; i >= 1; i--) {

        v1 = *(unsigned char *)((const char *)key1 + i);
        v2 = *(unsigned char *)((const char *)key2 + i);

        if (v1 > v2) {
	  return 1;
        } else if (v1 < v2) {
	  return -1;
        }
    }
    
    if (size % 4 == 0) {
      v1 = *(unsigned char *)key1;
      v2 = *(unsigned char *)key2;
    } else {
      v1 = *(unsigned char *)key1 & 0x7F;
      v2 = *(unsigned char *)key2 & 0x7F;
    }

    if (v1 > v2) 
        return 1;
    else if (v1 < v2) 
        return -1;
    
    return 0;
}


static void code_coord2morton_port(int bits, etree_tick_t x, etree_tick_t y, 
				   etree_tick_t	z, void* morton);

/**
 * coord2morton - new version of coord2morton  for fast conversion
 *
 */
void code_coord2morton(int bits, etree_tick_t x, etree_tick_t y, 
                       etree_tick_t z, void *morton)
{
#ifdef ALIGNMENT
    code_coord2morton_port (bits, x, y, z, morton);
#else
    unsigned int vbit0, vbit1, vbit2;
    unsigned int *part;
    int iter, totalparts;

    /* quick fix for big endian platforms */
    if (theEndianness == unknown_endianness || theEndianness < 0) {
        theEndianness = xplatform_testendian();
    }

    if (theEndianness == big) {
	/* use portable routine instead */
        code_coord2morton_port (bits, x, y, z, morton);
	return;
    }

    part = (unsigned int *)morton;

    vbit0 = x;
    vbit1 = y;
    vbit2 = z;

    totalparts = bits / 32 * 3;
    for (iter = 0; iter < totalparts; iter++) {
        unsigned int tmp0, tmp1, tmp2;
        
        *part = Expand0bit[0x7ff & vbit0] +
            Expand1bit[0x7ff & vbit1] +
            Expand2bit[0x3ff & vbit2];

        tmp0 = vbit0 >> 11;
        tmp1 = vbit1 >> 11;
        tmp2 = vbit2 >> 10;

        vbit0 = tmp2;
        vbit1 = tmp0;
        vbit2 = tmp1;
        
        part++;
    }

#endif

    return;
}
        

/**
 * code_coord2morton_port - transform X, Y, Z to morton code (portable)
 *
 * - shuffle the bits to create zyxzyx...zyx
 * - "bits" specifies the number of bits of original etree_tick_t
 * - portable function: it works both in little and big endian platforms
 * - slower than the platform specific routine
 */
static void
code_coord2morton_port(
	int		bits,
	etree_tick_t	x,
	etree_tick_t	y, 
	etree_tick_t	z,
	void*		morton
	)
{
    int offset, byte, inbyteoffset, i;
    unsigned char *byteunit;
    unsigned char mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    memset(morton, 0, 3 * sizeof(etree_tick_t));
    byteunit = (unsigned char *)morton;

    offset = 0;
    for (i = 0; i < bits; i++) {
        int IxLastBit, IyLastBit, IzLastBit;

        IxLastBit = x & 0x1;
        byte = offset / 8;
        inbyteoffset = offset % 8;
        *(byteunit + byte) = (IxLastBit == 1) ?
            *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
        x >>= 1;
        offset++;

        IyLastBit = y & 0x1;
        byte = offset / 8;
        inbyteoffset = offset % 8;
        *(byteunit + byte) = (IyLastBit == 1) ?
            *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
        y >>= 1;
        offset++;

        IzLastBit = z & 0x1;
        byte = offset / 8;
        inbyteoffset = offset % 8;
        *(byteunit + byte) = (IzLastBit == 1) ?
            *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
        z >>= 1;
        offset++;
    }
}


/*
  static void morton2coord (int bits, void* morton, etree_tick_t* px,
  etree_tick_t*	py, etree_tick_t* pz);
*/


/**
 * code_morton2coord_port - converse of code_coord2morton (portable)
 *
 * - transform morton code to *px, *py, *z
 * - "bits" specifies the size of the original etree_tick_t
 *
 */
static void
code_morton2coord_port(
	int		bits,
	void*		morton,
	etree_tick_t*	px, 
	etree_tick_t*	py,
	etree_tick_t*	pz
	)
{
    int offset, byte, inbyteoffset, i;
    unsigned mask[8] ={0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    unsigned char *byteunit;
    etree_tick_t Ix, Iy, Iz;
    /*    etree_tick_t Ix2, Iy2, Iz2; */

    byteunit = (unsigned char *)morton;
    Ix = Iy = Iz = 0;
    offset = 3 * bits - 1;

    for (i = 0; i < bits; i++){
        int IxLastBit, IyLastBit, IzLastBit;

        byte = offset / 8;
        inbyteoffset = offset % 8;
        IzLastBit = *(byteunit + byte) & mask[inbyteoffset];
        Iz <<= 1;
        Iz = (IzLastBit == 0) ? Iz : Iz | 0x1;
        offset--;

        byte = offset / 8;
        inbyteoffset = offset%8;
        IyLastBit = *(byteunit + byte) & mask[inbyteoffset];
        Iy <<= 1;
        Iy = (IyLastBit == 0) ? Iy : Iy | 0x1;
        offset--;
    
        byte = offset / 8;
        inbyteoffset = offset % 8;
        IxLastBit = *(byteunit + byte) & mask[inbyteoffset];
        Ix <<= 1;
        Ix = (IxLastBit == 0) ? Ix : Ix | 0x1;
        offset--;
    }

    *px = Ix;
    *py = Iy;
    *pz = Iz;

    /* code to debug new morton2coord routines */
    /*
    morton2coord(bits, morton, &Ix2, &Iy2, &Iz2);

    if ((Ix != Ix2) ||
        (Iy != Iy2) ||
        (Iz != Iz2)) {
        fprintf(stderr, "The new morton2coord routine is incorrect.\n");
        fprintf(stderr, "correct addr is   : %u %u %u\n", Ix, Iy, Iz);
        fprintf(stderr, "incorrect addr is : %u %u %u\n", Ix2, Iy2, Iz2);
        exit(-1);
    }
    */
}

/*
 * morton2coord - new version of morton2coord for fast convertion
 *
 *
 */
void code_morton2coord(int bits, void *morton, etree_tick_t *px, 
                       etree_tick_t *py, etree_tick_t *pz)
{
#ifdef ALIGNMENT
    code_morton2coord_port (bits, morton, px, py, pz);
#else 
    uint16_t *part;
    int iter, totalparts;
    register etree_tick_t vbit0, vbit1, vbit2, tmp0, tmp1, tmp2;
    int shifts0, shifts1, shifts2;

    /* quick fix for big endian platforms */
    if (unknown_endianness == theEndianness || theEndianness < 0) {
        theEndianness = xplatform_testendian();
    }

    if (theEndianness == big) {
	/* use portable routine instead */
	code_morton2coord_port (bits, morton, px, py, pz);
	return;
    }

    totalparts = bits / 16 * 3;
    part = (uint16_t *)morton + totalparts - 1;

    shifts0 = shifts1 = shifts2 = bits;
    tmp0 = tmp1 = tmp2 = 0;
    vbit0 = vbit1 = vbit2 = 0;
    
    for (iter = 0; iter < totalparts; iter++) {
        int tmpshifts;

        shifts0 -= 6;
        shifts1 -= 5;
        shifts2 -= 5;

        vbit0 = ((int)Extract0bit[*part]) << shifts0 | tmp0;
        vbit1 = ((int)Extract1bit[*part]) << shifts1 | tmp1;
        vbit2 = ((int)Extract2bit[*part]) << shifts2 | tmp2;

        tmpshifts = shifts0;
        shifts0 = shifts2;
        shifts2 = shifts1;
        shifts1 = tmpshifts;
        
        tmp0 = vbit2;
        tmp1 = vbit0;
        tmp2 = vbit1;

        part--;
    }

    *px = vbit0;
    *py = vbit1;
    *pz = vbit2;

#endif

    return;

}


/*
 * code_setbranch - set the three bits for "level" 
 *
 * - (theMaxOffsetP1 - 1) points to the most significant bit the morton code
 * - level 0 corresponds to the leading three bits in the morton code
 *
 */
void code_setbranch(void *morton, int level, int branch)
{
    int offset, byte, inbyteoffset;
    unsigned mask[8] ={0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    unsigned char *byteunit;
    int IxBit, IyBit, IzBit;

    byteunit = (unsigned char *)morton;
    offset = theMaxOffsetP1 - 1;
    offset = offset - 3 * level - 2; /* pointing to xbit */

    byte = offset / 8;
    inbyteoffset = offset % 8;
    IxBit = (branch & 0x1);
    *(byteunit + byte) = (IxBit == 1) ?
        *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
    branch >>= 1;
    offset++;

    byte = offset / 8;
    inbyteoffset = offset % 8;
    IyBit = (branch & 0x1);
    *(byteunit + byte) = (IyBit == 1) ?
        *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
    branch >>= 1;
    offset++;

    byte = offset / 8;
    inbyteoffset = offset % 8;
    IzBit = (branch & 0x1);
    *(byteunit + byte) = (IzBit == 1) ?
        *(byteunit + byte) | mask[inbyteoffset] : *(byteunit + byte);
    
    return;
}


/*
 * setprefix - prefix the timestep to the locational key
 *
 * Assume the key has enough memory space to accomodate the prefix.
 *
 */
void setprefix(etree_t *ep, unsigned int timestep, void *toptr)
{
    if (xplatform_testendian() == ep->endian)
        memcpy(toptr, &timestep, sizeof(unsigned int));
    else 
        xplatform_swapbytes(toptr, &timestep, sizeof(unsigned int));

    return;
}


/*
 * getprefix - extract the timestep prefix from the locational key 
 *             and store into *ptimestep
 *
 */
void getprefix(etree_t *ep, void *fromptr, unsigned int *ptimestep)
{
    if (xplatform_testendian() == ep->endian)
        memcpy(ptimestep, fromptr, sizeof(unsigned int));
    else
        xplatform_swapbytes(ptimestep, fromptr, sizeof(unsigned int));
    
    return;
}


