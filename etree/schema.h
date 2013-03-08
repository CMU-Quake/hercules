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
 * schema.h - support for schema definition
 *
 *
 */

#ifndef SCHEMA_H
#define SCHEMA_H

#ifdef ALPHA
#include "etree_inttypes.h"
#else
#include <inttypes.h>
#endif 

/*
 * endian_t - the endianness of the system 
 *
 */
#ifndef ENDIAN_T
#define ENDIAN_T
typedef enum endian_t {unknown_endianness = -1, little, big} endian_t;
#endif


/*
 * field_t : primitive type for each entry of a schema
 *
 * - supported data types are: int8_t/char, uint8_t/unsigned char, 
 *                             int16_t, uint16_t, int32_t, uint32_t,
 *                             float32_t, float64_t, int64_t, uint64_t
 *
 */
typedef struct field_t {
    char *name;                 /* the variable name of the field        */
    char *type;
    int32_t size;               /* size of the field                     */
    int32_t offset;             /* offset of in a compact representation */
} field_t;


/* 
 * schema_t: a runtime data structure containing the a collection of fields
 *           and the endianness of the current platform
 *
 */
typedef struct schema_t {
    endian_t endian; 
    int32_t fieldnum;          /* number of fields                         */
    field_t *field;            /* entries of the table are the fields      */
} schema_t;
    

/*
 * schema_create - create and initialize a schema
 *
 * - allocate memory for the schema application shall later call 
 *   schema_destroy to clean up
 * - parse the definition string
 * - empty defstring create a stub schema for schema_fromascii
 * - return pointer to a schema if OK, NULL on error
 *
 */
schema_t *schema_create(const char *defstring);


/*
 * schema_destroy - release memory held by a schema
 *
 *
 */
void schema_destroy(schema_t *schema); 


/*
 * schema_toascii - ASCII output of a schema to a character buffer 
 * 
 * - serialize the schema (binary format) for portable output
 * - allocate memory to hold the ASCII schema, application should later
 *   release the memory with free();
 * - return 0 if OK, -1 on error
 * - the size (strlen()) of the ascii schema is returned in *asciisizeptr
 * 
 */
char *schema_toascii(const schema_t *bin_schema, uint32_t *asciisizeptr);


/*
 * schema_fromascii - input a converted schema ASCII string into a schema
 *
 * - memory is allocated for the schema, application shall later call 
 *   schema_destroy to clean up
 * - return a  pointer to the initialized schema if OK, NULL on error
 *
 */
schema_t *schema_fromascii(const char *asc_schema);


/*
 * schema_getdefstring - reconstruct the definition string
 * 
 * - serialize the schema (binary format) for portable output
 * - allocate memory to hold the definition string, 
 * - application should later release the memory with free();
 * - return pointer to the def string if OK, NULL on error 
 * 
 */
char *schema_getdefstring(const schema_t *schema);

#endif /* SCHEMA_H */

