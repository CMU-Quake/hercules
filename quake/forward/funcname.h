/* -*- C -*- */
#ifndef FUNCNAME_H
#define FUNCNAME_H

#ifndef __GNUC_PREREQ
#  define __GNUC_PREREQ(foo,bar)	0
#endif


/* Version 2.4 and later of GCC define a magical variable
 * `__PRETTY_FUNCTION__' which contains the name of the function currently
 * being defined.  This is broken in G++ before version 2.6.  C9x has a
 * similar variable called __func__, but prefer the GCC one since it
 * demangles C++ function names.
 */
#if defined(__cplusplus) ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
# define __FUNCTION_NAME                __PRETTY_FUNCTION__
#else
# if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#  define __FUNCTION_NAME       __func__
# else
#  define __FUNCTION_NAME       ((__const char *) 0)
# endif
#endif

#endif /* FUNCNAME_H */
