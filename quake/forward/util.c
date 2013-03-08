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
#define  _GNU_SOURCE
#include <mpi.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <math.h>

#include "util.h"
#include "htimer.h"

/* Undo redefinitions in util.h */
#ifdef hu_fread
#undef hu_fread
#endif


struct hu_str_t {
    char*   str;
    ssize_t len;
};

typedef struct hu_str_t hu_str_t;


/**
 * Check for garlic NAN values in a floating point array (float version)
 *
 * \param v Float array of the given length.
 * \param len Array length.
 * \param[out] idx Index of the first element that was found to be a NAN.
 *
 * return 0 if no NAN is found, != 0 otherwise (i.e., there's a NAN).
 */
int
hu_farray_has_nan( const float* v, const size_t len, size_t* idx )
{
    const float* end = v + len;

    /* \todo add a conditional assertion for v != NULL && idx != NULL */

    while (v < end && !isnan(*v)) {
	v++;
    }

    *idx = len - (end - v);

    /* if we got to (past) the end then there were no NAN values so (!(v
     * == end)) is 0, otherwise if there was a nan then v != end, thus the
     * expression is 1  */
    return !(v == end);
}


/**
 * Check for garlic NAN values in a floating point array (float version)
 *
 * \param v Float array of the given length.
 * \param len Array length.
 * \param[out] idx Index of the first element that was found to be a NAN.
 *
 * return 0 if no NAN is found, != 0 otherwise (i.e., there's a NAN).
 */
int
hu_darray_has_nan( const double* v, const size_t len, size_t* idx )
{
    const double* end = v + len;

    /* \todo add a conditional assertion for v != NULL && idx != NULL */

    while (v < end && !isnan(*v)) {
	v++;
    }

    *idx = len - (end - v);

    /* if we got to (past) the end then there were no NAN values so (!(v
     * == end)) is 0, otherwise if there was a nan then v != end, thus the
     * expression is 1  */
    return !(v == end);
}


/**
 * Compute the product of the elements of an array v.  The type of the array
 * entries is size_t
 */
size_t
hu_st_product( const size_t v[], int len )
{
    const size_t* v_end = v + len;
    size_t p = 1;

    /* iterate over elements of the array and multiply the running product */
    while (v < v_end) {
	p *= *v;
	v++;
    }

    return p;
}


void
hu_offset_indices( const size_t len[], int dim, size_t index, size_t idx[] )
{
    dim--;

    for (; dim >= 0; dim--) {
	idx[dim] = index % len[dim];
	index /= len[dim];
    }
}


/**
 * Check for garlic NAN values in a floating point array (float version)
 *
 * \param v Float array / matrix of the given lengths.
 * \param dim Dimensionality of the array / matrix.
 * \param len Length along each dimension for the array / matrix.  This
 *	is an array containing 'dim' lengths.
 * \param[out] idx Indices of the first element that was found to be a NAN.
 *	This is an array of length 'dim'.  On return in contains the
 *	indices.
 *
 * return 0 if no NAN is found, != 0 otherwise (i.e., there's a NAN).
 */
int
hu_farray_has_nan_nd( const float* v, int dim, const size_t len[],
		      size_t idx[] )
{
    /* \todo add assertions for v, len and idx not null */
    size_t index = -1;
    size_t total_len = hu_st_product( len, dim );
    int    ret       = hu_farray_has_nan( v, total_len, &index );

    if (ret != 0) {
	/* get indices */
	hu_offset_indices( len, dim, index, idx );
    }

    return ret;
}


/**
 * Check for garlic NAN values in a floating point array (float version)
 *
 * \param v Float array / matrix of the given lengths.
 * \param dim Dimensionality of the array / matrix.
 * \param len Length along each dimension for the array / matrix.  This
 *	is an array containing 'dim' lengths.
 * \param[out] idx Indices of the first element that was found to be a NAN.
 *	This is an array of length 'dim'.  On return in contains the
 *	indices.
 *
 * return 0 if no NAN is found, != 0 otherwise (i.e., there's a NAN).
 */
int
hu_darray_has_nan_nd( const double* v, int dim, const size_t len[],
		      size_t idx[] )
{
    size_t index = -1;
    size_t total_len = hu_st_product( len, dim );
    int    ret       = hu_darray_has_nan( v, total_len, &index );

    /* \todo add assertions for v, len and idx not null */
    if (ret != 0) {
	/* get indices */
	hu_offset_indices( len, dim, index, idx );
    }

    return ret;
}


/**
 * Error handling routine.  It prints on \c stderr a hopefully
 * informational message about the error condition, and then aborts the
 * execution of the parallel program by calling \c MPI_Abort() and \c exit().
 * The first line of the status message contains the processor id for the
 * process calling hu_solver_abort.
 *
 * \pre myID should be initialized.
 *
 * \param fname function name of the caller, it can be NULL, in that case
 *	is not printed.
 * \param perror_msg Message for the perror() call.
 * \param fmt format string for additional message, followed by the
 *	 optional arguments.
 *
 * \return This function should not return, instead it aborts execution.
 *
 * \author jclopez
 */
int
hu_solver_abort(
	const char* fname,
	const char* perror_msg,
	const char* fmt,
	...
	)
{
    int	mpi_rank;
    int my_errno = errno;

    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

    fprintf( stderr, "\nPE %d called solver_abort() ", mpi_rank );

    if (NULL != fname) {
	fputs( "from ", stderr );
	fputs( fname, stderr );
	fputs( ": ", stderr );
    }

    if (NULL != fmt) {
	va_list ap;

	va_start( ap, fmt );
	vfprintf( stderr, fmt, ap );
	va_end( ap );
	fputc( '\n', stderr );
    }

    if (NULL != perror_msg) {
	fprintf( stderr, "%s: %s\n", perror_msg, strerror( my_errno ) );
    }

    fputc( '\n', stderr );

    MPI_Abort( MPI_COMM_WORLD, HERC_ERROR );
    exit( 1 );

    return -1;
}


void
hu_assert_abort(
	const char*     file,
	const char*     function_name,
	int             line,
	const char*     assertion_text
	)
{
    hu_solver_abort( function_name, NULL,
		     "In %s:%d assertion failed \"(%s)\"\n",
		     file, line, assertion_text );
}


void*
hu_xmalloc( size_t size, const char* varname )
{
    void* ptr = malloc( size );

    if (NULL == ptr) {
	const char* my_var_name = (varname == NULL) ? "Unknow" : varname;

	solver_abort( __FUNCTION_NAME, "malloc",
		      "Memory allocation failed for variable \'%s\'",
		      my_var_name );
    }

    return ptr;
}


/**
 * fopen equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
FILE*
hu_fopen( const char* filename, const char* mode )
{
    FILE* fp;

    HU_ASSERT_PTR_ALWAYS( filename );
    HU_ASSERT_PTR_ALWAYS( mode );

    fp = fopen( filename, mode );

    if (NULL == fp) {
	hu_solver_abort( __FUNCTION_NAME, filename,
			 "fopen( filename = \'%s\', mode = %s ) failed!",
			 filename, mode );
    }

    return fp;
}


/**
 * fclose equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 *
 */
int
hu_fclose( FILE* fp )
{
    int ret;

    HU_ASSERT_PTR_ALWAYS( fp );

    ret = fclose( fp );

    if (0 != ret) {
	// TODO replace with a warning instead of an abort
	hu_solver_abort(__FUNCTION_NAME, "fclose() failed", "fp = 0x%p\n", fp);
    }

    return ret;
}


/**
 * Close a file descriptor (FILE*) and set it to NULL.
 *
 * \return the return value from calling fclose(fp)
 */
int
hu_fclosep( FILE** fp )
{
    int ret;

    HU_ASSERT_PTR_ALWAYS( fp );
    ret = hu_fclose( *fp );
    *fp = NULL; /* nullify this handle so we don't use it again */

    return ret;
}


/**
 * fseeko equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
int
hu_fseeko( FILE* fp, off_t offset, int whence )
{
    int ret;

    HU_ASSERT_PTR_ALWAYS( fp );

    ret = fseeko( fp, offset, whence );

    if (0 != ret) {
	hu_solver_abort( __FUNCTION_NAME, "fseek() failed", "fp = 0x%p"
			 "offset="OFFT_FMT", whence=%d\n", fp, offset, whence );
    }

    return ret;
}


/**
 * ftello equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
off_t
hu_ftello( FILE* fp )
{
    off_t ret;

    HU_ASSERT_PTR_ALWAYS( fp );

    ret = ftello( fp );

    if (-1 == ret) {
	hu_solver_abort(__FUNCTION_NAME, "ftello() failed", "fp = 0x%p\n", fp);
    }

    return ret;
}



/**
 * fwrite equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
size_t
hu_fwrite( void* ptr, size_t size, size_t n, FILE* fp )
{
    size_t ret;

    HU_ASSERT_PTR_ALWAYS( fp );
    /* make sure ptr != NULL unless size or n are 0 */
    HU_ASSERT( ( (ptr != NULL) || (size == 0 || n == 0) ) );

    ret = fwrite( ptr, size, n, fp );

    if (n != ret) {
	hu_solver_abort( __FUNCTION_NAME, "fwrite() failed",
			 "fp=0x%p ptr=0x%p size=%zd n=%zd\n", fp, ptr, size, n );
    }

    return ret;
}

/**
 * fread equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
size_t
hu_fread_trace( void* ptr, size_t size, size_t n, FILE* fp, const char* fname )
{
    fputs( "hu_read(...) called from ", stderr );
    fputs( fname, stderr );
    fputc( '\n', stderr );

    return hu_fread( ptr, size, n, fp );
}


/**
 * fread equivalent routine with error checking.
 * On error it calls hu_solver_abort()
 */
size_t
hu_fread( void* ptr, size_t size, size_t n, FILE* fp )
{
    size_t ret;

    HU_ASSERT_PTR_ALWAYS( fp );
    /* make sure ptr != NULL unless size or n are 0 */
    HU_ASSERT( ( (ptr != NULL) || (size == 0 || n == 0) ) );

    ret = fread( ptr, size, n, fp );

    if (n != ret) {
	hu_solver_abort( __FUNCTION_NAME, "fread() failed",
			 "fp=0x%p ptr=0x%p size=%d n=%d ret=%d\n", fp, ptr,
			 size, n, ret );
    }

    return ret;
}


/**
 * print / split output to two files (stderr and the provided file pointer).
 *
 * \param fp     Pointer to the output file stream.
 * \param format Format string as in fprintf.
 * \param args   List of arguments in a properly initialized va_list.
 */
int
hu_print_tee_va( FILE *fp, const char* format, va_list args )
{
    int ret = 0;

    if (format != NULL) {
	va_list my_args1, my_args2;

	va_copy( my_args1, args );
	ret = vfprintf( fp, format, my_args1 );
	va_end( my_args1 );

	/* restart printing and send it to standard error */
	va_copy( my_args2, args );
	vfprintf( stderr, format, my_args2 );
	va_end( my_args2 );
    }

    return ret;
}


/**
 * print / split output to two files (stderr and the provided file pointer).
 *
 * \param fp     Pointer to the output file stream.
 * \param format Format string as in fprintf.
 */
int
print_tee( FILE* fp, const char* format, ... )
{
    int ret = 0;

    if (format != NULL) {
	va_list args;

	va_start( args, format );
	hu_print_tee_va( fp, format, args );
	va_end( args );
    }

    return ret;
}


int
hu_parse_int( const char* str, int base, int* value )
{
    int   my_value;
    char* endptr;

    HU_ASSERT_PTR_ALWAYS( str );
    HU_ASSERT_PTR_ALWAYS( value );

    my_value = strtol( str, &endptr, base );

    if (*endptr == 0 || isspace(*endptr)) {
	*value = my_value;
	return 0;
    }

    return -1;
}


/**
 * Parse a float from a string (char*).  This function returns an
 * error if the string cannot be parsed.
 *
 * \param arg string containing the float to parse
 * \param value float ptr. where the (output) result is stored.
 *
 * \return 0 on success, -1 on error.
 */
int
hu_parse_float( const char* str, float* value )
{
    float my_value;
    char* endptr;

    HU_ASSERT_PTR_ALWAYS( str );
    HU_ASSERT_PTR_ALWAYS( value );

    my_value = strtof( str, &endptr );

    if (*endptr != '\0' && !isspace(*endptr)) {
	return -1;
    }

    *value = my_value;

    return 0;
}


/**
 * Parse a float from a string (char*).  This function returns an
 * error if the string cannot be parsed.
 *
 * \param arg string containing the float to parse
 * \param value float ptr. where the (output) result is stored.
 *
 * \return 0 on success, -1 on error.
 */
int
hu_parse_double( const char* str, double* value )
{
    double my_value;
    char* endptr;

    HU_ASSERT_PTR_ALWAYS( str );
    HU_ASSERT_PTR_ALWAYS( value );

    my_value = strtod( str, &endptr );

    if (*endptr != '\0' && !isspace(*endptr)) {
	return -1;
    }

    *value = my_value;

    return 0;
}


static int
hu_config_parse_double( const char* str, void* value )
{
    return hu_parse_double( str, (double*)value );
}


static int
hu_config_parse_float( const char* str, void* value )
{
    return hu_parse_float( str, (float*)value );
}


static int
hu_config_parse_int( const char* str, void* value )
{
    return hu_parse_int( str, 0, (int*) value );
}


static int
hu_config_parse_uint( const char* str, void* value )
{
    unsigned int my_val;

    if (sscanf( str, "%u", &my_val ) == 1) {
	*((unsigned int*)value) = my_val;
	return 0;
    }

    return -1;
}


static int
hu_config_parse_str( const char* str, void* value )
{
    hu_str_t* valstr = (hu_str_t*)value;

    HU_ASSERT_PTR_ALWAYS( str );
    HU_ASSERT_PTR_ALWAYS( valstr );

    if (valstr->str == NULL) {  /* allocate new string */
	valstr->str = strdup( str );
	HU_ASSERT_PTR_ALWAYS( valstr->str );
	valstr->len = strlen( valstr->str ) + 1;
	return 0;
    }

    /* else: copy the string to the given buffer if there's enough space */
    if (strlen( str ) < valstr->len) {
	strcpy( valstr->str, str );
	return 0;
    }

    return -1;
}


static int
hu_config_parse_str_unsafe( const char* str, void* value )
{
    hu_str_t* valstr = (hu_str_t*)value;

    HU_ASSERT_PTR_ALWAYS( str );
    HU_ASSERT_PTR_ALWAYS( valstr );
    HU_ASSERT_PTR_ALWAYS( valstr->str );

    strcpy( valstr->str, str );

    return 0;
}


#ifndef HAVE_GETLINE
/*
 * Quick workaround for systems without getline().  This implementation
 * uses fgets() and does not have exactly the same semantics as getline.
 * I.e., the output string parameter is not automatically resized
 */
ssize_t
getline( char** lineptr, size_t* n, FILE* stream)
{
    ssize_t len;
    char* str;

    if (NULL == *lineptr) {
	return -1;
    }

    if (0 == *n) {
	return -1;
    }

    str = fgets( *lineptr, *n, stream );

    if (NULL == str) { /* check for errors */
	return -1;
    }

    len = strlen( str );

    /* check whether or not a full line was read */
    if ( (len + 1 == *n) && ((*lineptr)[len - 1] != '\n') ) {
	return -1;
    }

    return len;
}
#endif /* HAVE_GETLINE */

/**
 * Parse a text file and return the value of a match string.
 *
 * \return 1 if found, 0 if not found, -1 on error.
 */
int
hu_config_read(
	FILE*              fp,
	const char*        key,
	hu_config_parse_fn parse_val,
	hu_config_option_t flag,
	void*		   val
	)
{
    static size_t CONFIG_LINE_SIZE_DEFAULT = 256;
    static const char* config_err_msg_fmt
	= "Did not find variable in configuration file\n"
	  "variable name: %s\n";

    const static char delimiters[] = " =\n\t";

    char*   line;
    size_t  keylen;
    ssize_t ln  = 0;
    size_t  n   = CONFIG_LINE_SIZE_DEFAULT;
    int     ret = 0, found = 0;

    HU_ASSERT_PTR_ALWAYS( fp );
    HU_ASSERT_PTR_ALWAYS( key );
    HU_ASSERT_PTR_ALWAYS( parse_val );
    HU_ASSERT_PTR_ALWAYS( val );

    rewind( fp );               /* start from the beginning of the file */
    line = (char*)malloc( n * sizeof(char) );

    keylen = strlen( key );

    /* sequentially look for variable name */
    while (!found && !feof( fp ) && ln >= 0 ) {
	ssize_t ln = getline( &line, &n, fp );  /* read in one line */

	if (ln < 0 && !feof( fp )) { /* error and not EOF, then I/O error */
	    ret = -1;
	    break;
	}

	if (ln > 0 && strncmp( line, key, keylen ) == 0) {
	    found =1;
	    break;
	}
    } /* while */

    if (found) {		/* process entry if found */
	char* strtok_state;
	char* name = strtok_r( line, delimiters, &strtok_state );

	ret = -1;
	if ((name != NULL) && (strcmp( name, key ) == 0)) {
	    char* value = strtok_r( NULL, delimiters, &strtok_state );

	    if (value != NULL && parse_val( value, val ) == 0 ) {
		ret = 1;
	    }
	}
    }

    free( line );

    /* error handling */
    if (ret == -1) {		/* an error occurred, abort */
	hu_solver_abort( __FUNCTION_NAME, NULL,
			 "Error while reading configuration variable from"
			 "configuration file\n"
			 "variable name: %s\n",
			 key );
    }

    else if (ret == 0) {	/* not found, then take action accordingly */
	switch (flag) {
	case HU_REQUIRED:
	    hu_solver_abort( __FUNCTION_NAME, NULL, config_err_msg_fmt, key );
	    break;
	case HU_OPTIONAL_WARN:
	case HU_REQ_WARN:
	    fprintf( stderr, config_err_msg_fmt, key );
	    break;
	case HU_OPTIONAL:
	    break;
	}
    }

    return ret;
}




int
hu_config_get_string( FILE* fp, const char* key, hu_config_option_t flag,
		      char** val, size_t* len )
{
    int ret;
    hu_str_t valstr;

    HU_ASSERT_PTR_ALWAYS( val );
    HU_ASSERT_PTR_ALWAYS( len );

    valstr.str = *val;
    valstr.len = *len;

    ret = hu_config_read( fp, key, hu_config_parse_str, flag, &valstr );

    if (ret == 1) {
	*val = valstr.str;
	*len = valstr.len;
    }

    return ret;
}


int
hu_config_get_string_unsafe(
	FILE* fp,
	const char* key,
	hu_config_option_t flag,
	char* val
	)
{
    int	     ret;
    hu_str_t valstr;

    HU_ASSERT_PTR_ALWAYS( val );

    valstr.str = val;
    valstr.len = -1;

    ret = hu_config_read( fp, key, hu_config_parse_str_unsafe, flag, &valstr );

    return ret;
}


int
hu_config_get_string_def( FILE* fp, const char* key, char** val, size_t* len,
			  const char* default_value )
{
    int ret = hu_config_get_string_opt( fp, key, val, len );

    if (ret == 0) { /* not found */
	*len = strlen( default_value );
	*val = strdup( default_value );
    }

    return ret;
}



int
hu_config_get_int(FILE* fp, const char* key, int* val, hu_config_option_t flag)
{
    return hu_config_read( fp, key, hu_config_parse_int, flag, val );
}


int
hu_config_get_uint(
	FILE*	           fp,
	const char*	   key,
	unsigned int*	   val,
	hu_config_option_t flag
	)
{
    return hu_config_read( fp, key, hu_config_parse_uint, flag, val );
}


int
hu_config_get_double(
	FILE*	           fp,
	const char*	   key,
	double*		   val,
	hu_config_option_t flag
	)
{
    return hu_config_read( fp, key, hu_config_parse_double, flag, val );
}


int
hu_config_get_float(
	FILE*	           fp,
	const char*	   key,
	float*		   val,
	hu_config_option_t flag
	)
{
    return hu_config_read( fp, key, hu_config_parse_float, flag, val );
}


int
hu_parse_text( FILE* fp, const char* key, const char type, void* result )
{
    switch (type) {
    case 'i': return hu_config_get_int_req( fp, key, result );
    case 'u': return hu_config_get_uint_req( fp, key, result );
    case 'f': return hu_config_get_float_req( fp, key, result );
    case 'd': return hu_config_get_double_req( fp, key, result );
    case 's': return hu_config_get_string_unsafe(fp, key, HU_REQUIRED, result);
    default:
	fputs( "parsetext: unknown type ", stderr );
	fputc( type, stderr );
	fputc( '\n', stderr );
	return -1;
    }
}

int
hu_parsetext_v(
	FILE* fp,
	const char* key,
	const char type, 
	void* result
	)
{
    int ret = hu_parse_text( fp, key, type, result );

    fputs( "parsetext: ", stderr );
    fputs( key, stderr );
    fprintf( stderr, " %c %d\n", type, ret );

    return ret;
}
