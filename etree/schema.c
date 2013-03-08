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
 * schema.c - support for schema definition
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef ALPHA
char *strtok_r(char *s1, const char *s2, char **savept);
#endif

#ifdef SOLARIS
char *strtok_r(char *s1, const char *s2, char **savept);
#endif

#include "schema.h"
#include "xplatform.h"

static int IsValidName(const char *name);

/*
 * schema_create - create and initialize a schema
 *
 * - allocate memory for the schema, application shall later call 
 *   schema_destroy to clean up
 * - parse the definition string;
 * - empty defstring create a stub schema for schema_fromascii
 * - return pointer to a schema if OK, NULL on error
 *
 */
schema_t *schema_create(const char *defstring)
{
    schema_t *schema;
    char *dupstring, *saveptr, *name, *type;
    int offset, fieldind;
    int tempsize, failed;
    field_t *newtable;
    
    /* allocate memory */
    if ((schema = (schema_t *)malloc(sizeof(schema_t))) == NULL) {
        perror("schema_create: malloc");
        return NULL;
    }
    
    /* check whether we should proceed to parse the definition string */
    if (defstring == NULL) {
        return schema;
    } else if ((dupstring = strdup(defstring)) == NULL) {
        /* remember to duplicate the string for strtok_r */
        perror("schema_create: strdup");
        return NULL;
    }
    
    schema->endian = xplatform_testendian();
    schema->fieldnum = 0;
    schema->field = NULL;
        
    /* 
       allocate field table with a tempsize (8);
       we need to further expand the table or shrink as more or less 
       fields are actually parsed 
    */
    tempsize = 8;
    schema->field = (field_t *)malloc(tempsize * sizeof(field_t));
    if (schema->field == NULL) {
        perror("schema_create: malloc field table");
        free(dupstring);
        free(schema);
    }

    /*
      special treatment for the first field's type since the dupstring
      has to be passed in as the first argument 
    */
    fieldind = 0;
    offset = 0;
    type = (char *)strtok_r(dupstring, " ", &saveptr);
    if (type == NULL) {
        fprintf(stderr, "schema_create: illegal syntax \"%s\"\n", defstring);
        free(dupstring);
        free(schema);
        return NULL;
    }
    
    /* 
       install the current field entry in the schema table and 
       parse the next one 
    */
    failed = 0;
    schema->fieldnum = 0;
    while (type != NULL) {

        /* check whether we shall expand the table */
        if (fieldind == tempsize) {
            tempsize *= 2;
            newtable = 
                (field_t *)realloc(schema->field, tempsize * sizeof(field_t));
            if (newtable == NULL) {
                perror("schema_create: reallocate field table");
                failed = 1;
                break;
            } else
                schema->field = newtable;
        }

        /* determine the size of the field */
        if ((strcmp(type, "int8_t") == 0) ||
            (strcmp(type, "uint8_t") == 0) ||
            (strcmp(type, "char") == 0)) 
            schema->field[fieldind].size = 1;

        else if ((strcmp(type, "int16_t") == 0) ||
                 (strcmp(type, "uint16_t") == 0))
            schema->field[fieldind].size = 2;

        else if ((strcmp(type, "int32_t") == 0) ||
                 (strcmp(type, "uint32_t") == 0) ||
                 (strcmp(type, "float32_t") == 0) ||
                 (strcmp(type, "float") == 0))
            schema->field[fieldind].size = 4;

        else if ((strcmp(type, "float64_t") == 0) ||
                 (strcmp(type, "int64_t") == 0) ||
                 (strcmp(type, "uint64_t") == 0) ||
                 (strcmp(type, "double") == 0))
            schema->field[fieldind].size = 8;

        else {
            fprintf(stderr, "schema_create: unknown type \"%s\"\n", type);
            failed = 1;
            break;
        }

        schema->field[fieldind].offset = offset;
        offset +=  schema->field[fieldind].size;  /* for the next field */
        
        /* proceed to read in the variable name */
        name = (char *)strtok_r(NULL, " ;", &saveptr);
        if (name == NULL) {
            fprintf(stderr, "schema_create: variable name missing \"%s\"\n",
                    defstring);
            failed = 1;
            break;
        }


        /* allocate memory for the field name */
        if (!IsValidName(name)) {
            fprintf(stderr, "schema_create: invalid field name %s\n", name);
            failed = 1;
            break;
        }

        schema->field[fieldind].name = NULL;
        schema->field[fieldind].type = NULL;
           
        if ((schema->field[fieldind].name = strdup(name)) == NULL) {
            perror("schema_create: strdup field name");
            failed = 1;
            break;
        }

        /* record the type verbatim */
        if ((schema->field[fieldind].type = strdup(type)) == NULL) {
            perror("schema_create: strdup field type");
            failed = 1;
            break;
        }
           
        /* parse the next field's type */
        fieldind++;
        type = (char *)strtok_r(NULL, " ;", &saveptr);
    }
    
    schema->fieldnum = fieldind; 
    free(dupstring); /* must be released no matter what */

    if (failed ) {
        schema_destroy(schema);
        return NULL;
    }
    
    /* shrink the table if we have over-provisioned */
    if (schema->fieldnum < tempsize) {
        newtable = (field_t *)
            realloc(schema->field, schema->fieldnum * sizeof(field_t));
        if (newtable == NULL) {
            perror("schema_create: reallocate (shrink) field table");
            schema_destroy(schema);
            return NULL;
        }
        schema->field = newtable;
    }
    
    return schema;
}


/*
 * schema_destroy - release memory held by a schema
 *
 */
void schema_destroy(schema_t *schema)
{
    int fieldind;

    for (fieldind = 0; fieldind < schema->fieldnum; fieldind++) {
        if (schema->field[fieldind].name != NULL) 
            free(schema->field[fieldind].name);
        if (schema->field[fieldind].type != NULL)
            free(schema->field[fieldind].type);
    }
    
    free(schema->field); 
    free(schema); 

    return;
}

        
/*
 * schema_toascii - ASCII output of a schema to a character buffer 
 * 
 * - serialize the schema (binary format) for portable output
 * - allocate memory to hold the ASCII schema, application should later
 *   release the memory with free();
 * - return pointer to the ascii string if OK, NULL on error
 * 
 */
char *schema_toascii(const schema_t *schema, uint32_t *asciisizeptr)
{
    int fieldind;
    int bufsize, incrsize, length, asciisize;
    char *buf, *ptr, *newbuf;
    char fieldnumcount[128], sizecount[128], offsetcount[128];

    bufsize = 1024;
    incrsize = 1024;
    
    asciisize = 0; /* the actual size of the ascii string */
    if ((buf = (char *)malloc(bufsize)) == NULL) {
        perror("schema_toascii: malloc");
        return NULL;
    }
    ptr = buf;

    /* output the endianness and number of fields first */
    if (schema->endian == little) 
        strcpy(ptr, "L ");
    else
        strcpy(ptr, "B ");
    length = 2;
    asciisize += length;

    sprintf(fieldnumcount, "%d ", schema->fieldnum);
    ptr = buf + asciisize;
    strcpy(ptr, fieldnumcount);
    length = strlen(fieldnumcount);
    asciisize += length;

    for (fieldind = 0 ; fieldind < schema->fieldnum; fieldind++) {
        int len1, len2, len3, len4;

        sprintf(sizecount, "%d ", schema->field[fieldind].size);
        sprintf(offsetcount, "%d ", schema->field[fieldind].offset);
        
        len1 = strlen(schema->field[fieldind].name);
        len2 = strlen(schema->field[fieldind].type);
        len3 = strlen(sizecount);
        len4 = strlen(offsetcount);

        
        /* pad one space behind the name (len1) and the type (len2)*/
        length = (len1 + 1) + (len2 + 1) + len3 + len4; 
        
        if (length + asciisize > bufsize) {
            bufsize += incrsize;
            newbuf = (char *)realloc(buf, bufsize);
            if (newbuf == NULL) {
                perror("schema_toascii: reallocate (expand)");
                free(buf);
                return NULL;
            } else
                buf = newbuf;
        }

        ptr = buf + asciisize;

        strcpy(ptr, schema->field[fieldind].name);
        ptr += len1;
        *ptr = ' ';
        ptr++;

        strcpy(ptr, schema->field[fieldind].type);
        ptr += len2;
        *ptr = ' ';
        ptr++;

        strcpy(ptr, sizecount);
        ptr += len3;

        strcpy(ptr, offsetcount);
        asciisize += length;
    }
            
    /* allocate extra one byte of the terminating '\0' */
    *(buf + asciisize) = '\0';
    asciisize++;

    /* adjust the buffer size */
    if (asciisize < bufsize) {
        newbuf = (char *)realloc(buf, asciisize); 
        if (newbuf == NULL) {
            perror("schema_toascii: reallocate (shrink)");
            free(buf);
            return NULL;
        } else
            buf = newbuf;
    }
        
    *asciisizeptr = asciisize;
    return buf;
}
        

/*
 * schema_fromascii - input a converted schema ASCII string into a schema
 *
 * - memory is allocated for the schema, application shall later call 
 *   schema_destroy to clean up
 * - assume the ascii string is produced by schema_toascii
 * - return a  pointer to the initialized schema if OK, NULL on error
 *
 */
schema_t *schema_fromascii(const char *asc_schema)
{
    int fieldind;
    char *buf, *saveptr;
    char *endian, *fieldnum, *name, *size, *offset, *type;
    schema_t *schema;

    if ((schema = schema_create(NULL)) == NULL) {
        fprintf(stderr, "schema_fromascii: cannot create schema stub\n");
        return NULL;
    }
    
    if ((buf = strdup(asc_schema)) == NULL) {
        perror("schema_fromascii: strdup");
        return NULL;
    }

    endian = (char *)strtok_r(buf, " ", &saveptr);
    if (strcmp(endian, "L") == 0) 
        schema->endian = little;
    else if (strcmp(endian, "B") == 0) 
        schema->endian = big;
    else {
        fprintf(stderr, "schema_fromascii: unknown endianness %s\n", endian);
        free(schema);
        return NULL;
    }
    
    fieldnum = (char *)strtok_r(NULL, " ", &saveptr);
    if (fieldnum == NULL) {
        fprintf(stderr, "schema_fromascii: missing fieldnum %s\n", asc_schema);
        free(schema);
        return NULL;
    }

    if (sscanf(fieldnum, "%d", &schema->fieldnum) != 1) {
        fprintf(stderr, "schema_fromascii: unknown fieldnum specifier %s\n",
                fieldnum);
    }
    
    /* allocate the field table */
    schema->field = (field_t *)malloc(schema->fieldnum * sizeof(field_t));
    if (schema->field == NULL) {
        perror("schema_fromascii: malloc field table");
        free(schema);
        return NULL;
    }

    for (fieldind = 0 ; fieldind < schema->fieldnum; fieldind++) {
        name = (char *)strtok_r((char *)NULL, " ", &saveptr);
        type = (char *)strtok_r((char *)NULL, " ", &saveptr); 
        size = (char *)strtok_r((char *)NULL, " ", &saveptr);
        offset = (char *)strtok_r((char *)NULL, " ", &saveptr);
        
        if ((name == NULL) || (type == NULL) ||
            (size == NULL) || (offset == NULL)) {
            fprintf(stderr, "schema_fromascii: incomplete ascii schema %s\n",
                    asc_schema);
            break;
        }
        
        if (sscanf(size, "%d", &schema->field[fieldind].size) != 1) {
            fprintf(stderr, "schema_fromascii: unknown size specifier %s\n",
                    size);
            break;
        }
        
        if (sscanf(offset, "%d", &schema->field[fieldind].offset) != 1) {
            fprintf(stderr, "schema_fromascii: unknown offset specifer %s\n",
                    offset);
            break;
        }
            

        if ((schema->field[fieldind].name = strdup(name)) == NULL) {
            perror("schema_fromascii: strdup field name");
            break;
        }
        
        if ((schema->field[fieldind].type = strdup(type)) == NULL) {
            perror("schema_fromascii: strdup field type");
            break;
        }
        
    }

    free(buf);  /* we need to free the memory */

    if (fieldind < schema->fieldnum) {
        /* 
           break out of the loop because of error 
           here is a trick to guarantee the proper behavior 
           of schema_destroy 
        */
        schema->fieldnum = fieldind; 
        schema_destroy(schema);
        return NULL;
    }


    return schema;
    
}
        

/*
 * IsValidName - check whether the field name is valid or not 
 *
 * return 1 if valid, 0 otherwise
 *
 */
int IsValidName(const char *name)
{
    const char *ptr;
    int length, i;

    ptr = name;

    if (!isalpha((int)(*ptr)))  return 0;

    length = strlen(name);
    for (ptr = name + 1, i = 1; i < length; ptr++, i++) 
        if (!(isalnum((int)(*ptr)) || *ptr == '_'))
	  return 0;

    return 1;
}
    
      
  
/*
 * schema_getdefstring - reconstruct the definition string
 * 
 * - serialize the schema (binary format) for portable output
 * - allocate memory to hold the definition string, 
 * - application should later release the memory with free();
 * - return pointer to the def string if OK, NULL on error 
 * 
 */
char *schema_getdefstring(const schema_t *schema)
{
    int fieldind;
    int bufsize, incrsize, length, asciisize;
    char *buf, *ptr, *newbuf;

    bufsize = 1024;
    incrsize = 1024;
    
    asciisize = 0; /* the actual size of the ascii string */
    if ((buf = (char *)malloc(bufsize)) == NULL) {
        perror("schema_getdefstring: malloc");
        return NULL;
    }
    ptr = buf;

    for (fieldind = 0 ; fieldind < schema->fieldnum; fieldind++) {
        int len1, len2;

        len1 = strlen(schema->field[fieldind].name);
        len2 = strlen(schema->field[fieldind].type);
        
        /* pad one space after type and semi-colon + one space after name */
        length = (len1 + 2) + (len2 + 1);
        
        if (length + asciisize > bufsize) {
            bufsize += incrsize;
            newbuf = (char *)realloc(buf, bufsize);
            if (newbuf == NULL) {
                perror("schema_getdefstring: reallocate (expand)");
                free(buf);
                return NULL;
            } else
                buf = newbuf;
        }

        ptr = buf + asciisize;

        strcpy(ptr, schema->field[fieldind].type);
        ptr += len2;
        *ptr = ' ';
        ptr++;

        strcpy(ptr, schema->field[fieldind].name);
        ptr += len1;
        *ptr = ';';
        ptr++;
        *ptr = ' ';
        ptr++;

        asciisize += length;
    }
            
    /* replace the last white space with '\0' */
    *(buf + asciisize - 1) = '\0';

    /* adjust the buffer size */
    if (asciisize < bufsize) {
        newbuf = (char *)realloc(buf, asciisize); 
        if (newbuf == NULL) {
            perror("schema_getdefstring: reallocate (shrink)");
            free(buf);
            return NULL;
        } else
            buf = newbuf;
    }
        
    return buf;
}
