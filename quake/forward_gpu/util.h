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
 * Description: Miscellaneous support functions.
 */
#ifndef HERC_UTIL_H
#define HERC_UTIL_H

#include <sys/types.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#include "funcname.h"

/** Replace fseeko with fseek */
#ifdef BIGBEN
#define fseeko	fseek
#endif


/**
 * Macros with the format string for int64_t and uint64_t types.
 */
#if (defined __WORDSIZE) && (__WORDSIZE == 64)
#  define UINT64_FMT		"lu"
#  define INT64_FMT		"ld"
#  define MPI_INT64		MPI_LONG
#else /*  __WORDSIZE && __WORDSIZE == 64 */
#  define UINT64_FMT		"llu"
#  define INT64_FMT		"lld"
#  define MPI_INT64		MPI_LONG_LONG_INT
#endif

#if (defined _FILE_OFFSET_BITS) && (_FILE_OFFSET_BITS == 64)
#  define OFFT_FMT		UINT64_FMT
#else
#  define OFFT_FMT		"d"
#endif



#ifndef STRINGIZE
#  ifdef HAVE_STRINGIZE
#    define STRINGIZE(s)          #s
#  else
#    define STRINGIZE(s)          "s"
#  endif
#endif /* STRINGIZE */



#ifndef STRINGX
#  define STRINGX(s)		STRINGIZE(s)
#endif /* ! STRINGX */


#define HU_VOID_CAST_EXP	(void)

/**
 * Execute a given statement only if the condition is true.
 * The goal of this macro is to simplify certain code constructions such as:
 *
 * if (foo) {
 *	bar();
 * }
 *
 * and replace them with
 *   SUL_COND_EXEC (foo, bar());
 *
 * It is mainly intended to be used inside other macros.
 */
#define HU_COND_EXEC(flag,statement)  \
    (HU_VOID_CAST_EXP ((flag) ? (statement) : 0 ))



#define HU_ASSERT_COND(p,c)						\
    do {                                                                \
	if ((c) && ! (p)) {                                             \
	    hu_assert_abort( __FILE__,	__FUNCTION_NAME, __LINE__,	\
			     STRINGX(p) );				\
	}                                                               \
    } while (0)



#ifdef HERCULES_DISABLE_ASSERT
# define HU_ENABLE_ASSERT	0
#else
# define HU_ENABLE_ASSERT	1
#endif /* HERCULES_DISABLE_ASSERT */

#define HU_ASSERT(p)		HU_ASSERT_COND((p),HU_ENABLE_ASSERT)
#define HU_ASSERT_PTR(p)	HU_ASSERT(NULL != (p))

/**
 * Unconditional assertions, these are not disabled at compile time.
 */
#define HU_ASSERT_ALWAYS(p)     HU_ASSERT_COND((p),1)
#define HU_ASSERT_PTR_ALWAYS(p) HU_ASSERT_ALWAYS(NULL != (p))


/**
 * Conditional global barrier.
 * Execute a \c MPI_Barrier( \c comm_solver ) when flag is set.
 */
#define HU_COND_GLOBAL_BARRIER(flag)		\
    HU_COND_EXEC( flag, MPI_Barrier( comm_solver ) )

#define HERC_ERROR		-100


#define XMALLOC_N(type,n)			\
    ((type*)hu_xmalloc( ((n) * sizeof(type)), NULL ))

#define XMALLOC(type)			XMALLOC(type,1)


/**
 * Allocate memory for an named array.  An array of the given length and type
 * is allocated and assigned to the specified pointer.  
 * If the allocation fails the error is reported with a message that includes
 * the variable name and the amount of memory to be allocated.
 *
 * \param var name of the variable in the program, i.e., pass the variable.
 * \param type the basic type of each element in the array.
 * \param n array length.
 */
#define XMALLOC_VAR_N(var,type,n)					\
    do {								\
	(var) = ((type*)hu_xmalloc(((n) * sizeof(type)), STRINGX(var))); \
    } while (0)


/**
 * Allocate memory for a string of a given length.  The amount of
 * allocated memory is 1 byte larger than the specified string length.
 */
#define XMALLOC_STRING(var,n)   XMALLOC_VAR_N((var),char,((n)+1))

#define XMALLOC_VAR(var,type)	XMALLOC_VAR((var),(type),1)

#ifndef HAVE_FDATASYNC
/* replace with fsync */
#define fdatasync(fd)	fsync(fd)
#endif /* HAVE_FDATASYNC */

enum hu_config_option_t {
    HU_REQUIRED,
    HU_REQ_WARN,
    HU_OPTIONAL,
    HU_OPTIONAL_WARN
};

typedef enum hu_config_option_t hu_config_option_t;


/* ------------------------------------------------------------------------- *
 * Function declaration. 
 * Documentation along with function definition in .c file.
 * ------------------------------------------------------------------------- */

/** Compatibility function renaming macro */
#define solver_abort	hu_solver_abort

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Type of the parsing function for configuration variables in the
 * config file.
 */
typedef int (*hu_config_parse_fn)( const char* str, void* value );


extern void* hu_xmalloc( size_t size, const char* varname );

extern void
hu_assert_abort(
	const char*     file,
	const char*     function_name,
	int             line,
	const char*     assertion_text
	);


extern FILE*  hu_fopen( const char* filename, const char* mode );
extern int    hu_fclose( FILE* fp );
extern int    hu_fclosep( FILE** fp );
extern int    hu_fseeko( FILE* fp, off_t offset, int whence );
extern off_t  hu_ftello( FILE* fp );
extern size_t hu_fwrite( void* ptr, size_t size, size_t n, FILE* fp );

extern size_t hu_fread( void* ptr, size_t size, size_t n, FILE* fp );
extern size_t hu_fread_trace( void* ptr, size_t size, size_t n, FILE* fp,
			      const char* fname );

#ifdef HU_FVERBOSE
#define hu_fread(ptr,size,n,fp)					\
    hu_fread_trace( ptr, size, n, fp, __FUNCTION_NAME )
#endif

extern int    hu_print_tee_va( FILE *fp, const char* format, va_list args );

extern int
hu_parsetext_v(	FILE* fp, const char* string, const char type, void* result );

extern int hu_solver_abort( const char* function_name, const char* error_msg,
			    const char* format, ... );


extern int hu_parse_double( const char* str, double* value );
extern int hu_parse_float( const char* str, float* value );
extern int hu_parse_int( const char* str, int base, int* value );

extern int
hu_config_read(
	FILE*              fp,
	const char*        key,
	hu_config_parse_fn parse_val,
	hu_config_option_t flag,
	void*		   val
	);


extern int
hu_config_get_string( FILE* fp, const char* key, hu_config_option_t flag,
		      char** val, size_t* len );

extern int
hu_config_get_string_unsafe( FILE* fp, const char* key, hu_config_option_t flag,
			     char* val );

extern int
hu_config_get_int(
	FILE*		   fp,
	const char*	   key,
	int*		   val,
	hu_config_option_t flag
	);

extern int
hu_config_get_uint(
	FILE*	           fp,
	const char*	   key,
	unsigned int*	   val,
	hu_config_option_t flag
	);


extern int
hu_config_get_double(
	FILE*	           fp,
	const char*	   key,
	double*		   val,
	hu_config_option_t flag
	);


extern int
hu_config_get_float(
	FILE*	           fp,
	const char*	   key,
	float*		   val,
	hu_config_option_t flag
	);


static inline int
hu_config_get_int_req( FILE* fp, const char* key, int* value )
{
    return hu_config_get_int( fp, key, value, HU_REQUIRED );
}


static inline int
hu_config_get_uint_req( FILE* fp, const char* key, unsigned int* value )
{
    return hu_config_get_uint( fp, key, value, HU_REQUIRED );
}


static inline int
hu_config_get_double_req( FILE* fp, const char* key, double* value )
{
    return hu_config_get_double( fp, key, value, HU_REQUIRED );
}


static inline int
hu_config_get_float_req( FILE* fp, const char* key, float* value )
{
    return hu_config_get_float( fp, key, value, HU_REQUIRED );
}


static inline int
hu_config_get_string_req( FILE* fp, const char* key, char** val, size_t* len )
{
    return hu_config_get_string( fp, key, HU_REQUIRED, val, len );
}

static inline int
hu_config_get_int_opt( FILE* fp, const char* key, int* value )
{
    return hu_config_get_int( fp, key, value, HU_OPTIONAL );
}


static inline int
hu_config_get_uint_opt( FILE* fp, const char* key, unsigned int* value )
{
    return hu_config_get_uint( fp, key, value, HU_OPTIONAL );
}


static inline int
hu_config_get_double_opt( FILE* fp, const char* key, double* value )
{
    return hu_config_get_double( fp, key, value, HU_OPTIONAL );
}


static inline int
hu_config_get_float_opt( FILE* fp, const char* key, float* value )
{
    return hu_config_get_float( fp, key, value, HU_OPTIONAL );
}


static inline int
hu_config_get_string_opt( FILE* fp, const char* key, char** val, size_t* len )
{
    return hu_config_get_string( fp, key, HU_OPTIONAL, val, len );
}

static inline int
read_config_string2( FILE* fp, const char* key, char* val, size_t len )
{
    return hu_config_get_string_req( fp, key, &val, &len );
}

int
hu_config_get_string_def( FILE* fp, const char* key, char** val, size_t* len,
			  const char* default_val );

/**
 * Free the memory pointed by *ptr and then set *ptr to NULL.
 */
static inline void
xfree( void** ptr )
{
    if (ptr != NULL) {
	if (NULL != *ptr) {
	    free( *ptr );
	}
	*ptr = NULL;
    }
}

/*
//static inline void
//xfree_char( char** ptr )
//{
//    xfree( (void**)ptr );
//}
*/

#define DEF_XFREE_TYPE(type)						\
    static inline void xfree_##type(type ** p) { xfree( (void**)p ); } 

DEF_XFREE_TYPE(char);
DEF_XFREE_TYPE(int32_t);

#undef DEF_XFREE_TYPE

extern int
hu_darray_has_nan_nd( const double* v, int dim, const size_t len[],
		      size_t idx[] );

extern int
hu_farray_has_nan_nd( const float* v, int dim, const size_t len[],
		      size_t idx[] );

extern int
hu_darray_has_nan( const double* v, const size_t len, size_t* idx );

extern int
hu_farray_has_nan( const float* v, const size_t len, size_t* idx );

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* HERC_UTIL_H */
