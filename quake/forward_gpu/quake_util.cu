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

#include <math.h>
#include <string.h>

#include "psolve.h"
#include "quake_util.h"

#include <cuda_runtime.h>


/**
 * For regular simulations use 1e-20.  For performance and scalability,
 * uncomment the immediate return in vector_is_zero() and vector_is_all_zero().
 * Alternatives with each platform underflow precision limit for 'double'
 * may also work, though this have not been thoroughly tested.
 *
 * Known underflow limits:
 * - NICS' Kraken: Limit is ~2.2e-308 --> use 1e-200
 *
 */
#define UNDERFLOW_CAP 1e-20


/**
 * Return minimum of two int32_t integers.
 */
int32_t imin(int32_t x, int32_t y)
{
  if (x < y) {
    return(x);
  } else {
    return(y);
  }
}


/**
 * For effective stiffness method:
 *
 * Check whether all components of a 3D vector are close to zero,
 * i.e., less than a small threshold.
 *
 * \return 1 when there is at least one "non-zero" component;
 *         0 when all the components are "zero".
 */
//int vector_is_zero( const fvector_t* v )
//{
//    /*
//     * For scalability studies, uncomment the immediate return.
//     */
//
//    /* return 1; */
//
//    int i,j;
//
//    for (i = 0; i < 8; i++) {
//        for(j = 0; j < 3; j++){
//            if (fabs( v[i].f[j] ) > UNDERFLOW_CAP) {
//                return 1;
//            }
//        }
//    }
//
//    return 0;
//}

/**
 * For conventional stiffness method:
 *
 * Check whether all components of a 3D vector are close to zero,
 * i.e., less than a small threshold.
 *
 * \return 1 when there is at least one "non-zero" component;
 *         0 when all the components are "zero".
 */
int vector_is_all_zero( const fvector_t* v )
{
    /*
     * For scalability studies, uncomment the immediate return.
     */

    /* return 1; */

    int i;

    for (i = 0; i < 3; i++) {
        if (fabs( v->f[i] ) > UNDERFLOW_CAP) {
            return 1;
        }
    }

    return 0;
}


/**
 * MultAddMatVec: Multiply a 3 x 3 Matrix (M) with a 3 x 1 vector (V1)
 *                and a constant (c). Then add the result to the same
 *                target vector (V2)
 *
 *  V2 = V2 + M * V1 * c
 *
 */
void MultAddMatVec( fmatrix_t* M, fvector_t* V1, double c, fvector_t* V2 )
{
    int row, col;
    fvector_t tmpV;

    tmpV.f[0] = tmpV.f[1] = tmpV.f[2] = 0;

    for (row = 0; row < 3; row++)
        for (col = 0; col < 3; col++)
            tmpV.f[row] += M->f[row][col] * V1->f[col];

    for (row = 0; row < 3; row++)
        V2->f[row] += c * tmpV.f[row];

    return;
}

/**
 * Search Quality Table to find the corresponding coefficients for the implementation of BKT model
 */

int Search_Quality_Table(double Q, double *theQTABLE, int QTable_Size)
{

	if(Q <= 500)
	{
		int i, range;
		double diff, min;

		range = Q / 5.;
		min = 1000.;

		for(i = 0; i < QTable_Size; i++)
		{
			diff = Q - theQTABLE[ i * 6 ];

			diff = (diff < 0) ? -diff : diff;

			if(diff < min){
				min = diff;
			}
			else
			{
				return (i-1);
			}
		}

		return (-2);
	}
	else
	{
		return (-1);
	}

	// statement is unreachable
	// return (-2);

}

/**
 * Parse an array of doubles.
 *
 * \return 0 on success, -1 on error
 */
int
parsedarray( FILE* fp, const char* querystring, int size, double* array )
{
    static const char delimiters[] = " =\n\t";

    int iSize;
    int found = 0;

    rewind( fp );

    while (!found) {
        char* name;
        char  line[LINESIZE];

        /* Read in one line */
        if (fgets(line, LINESIZE, fp) == NULL) {
            break;
        }

        name = strtok(line, delimiters);

        if ((name != NULL) && (strcmp( name, querystring ) == 0)) {
            found = 1;

            for (iSize = 0; iSize < size; iSize++) {
                if (fscanf( fp," %lf ", &(array[iSize]) )  == 0) {
                    fprintf( fp,"\nUnable to read %s", querystring );
                    return -1;
                }
            }

            return 0;
        }
    }

    return -1;
}



/**
 * Checks whether a leaf octant needs to be expanded or not according to Vs Rule
 *
 *\Return 1 if true, 0 otherwise.
 */
int vsrule( edata_t *edata, double theFactor )
{

    if (edata->edgesize <= edata->Vs / theFactor) {

        return 0;

    } else {

        return 1;
    }
}




