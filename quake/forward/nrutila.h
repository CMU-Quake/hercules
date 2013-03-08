/* -*- C -*- */

/*
 * CAUTION: This is the ANSI C (only) version of the Numerical Recipes
 * utility file nrutil.c.  Do not confuse this file with the same-named
 * file nrutil.c that is supplied in the same subdirectory or archive as
 * the header file nrutil.h.  *That* file contains both ANSI and
 * traditional K&R versions, along with #ifdef macros to select the
 * correct version.  *This* file contains only ANSI C.
 */

void nrerror(char error_text[]);

float *vector(long nl, long nh);

int *ivector(long nl, long nh);

unsigned char *cvector(long nl, long nh);

unsigned long *lvector(long nl, long nh);

double *dvector(long nl, long nh);

float **matrix(long nrl, long nrh, long ncl, long nch);

double **dmatrix(long nrl, long nrh, long ncl, long nch);

int **imatrix(long nrl, long nrh, long ncl, long nch);

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
		  long newrl, long newcl);

float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch);

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

void free_vector(float *v, long nl, long nh);

void free_ivector(int *v, long nl, long nh);

void free_cvector(unsigned char *v, long nl, long nh);

void free_lvector(unsigned long *v, long nl, long nh);

void free_dvector(double *v, long nl, long nh);

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch);

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch);

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
		   long ndl, long ndh);

void read_double_matrix_from_file(FILE *fp,
                                  double **matrix, int rows, int columns);
