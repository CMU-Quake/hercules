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

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "geometrics.h"
#include "nonlinear.h"
#include "octor.h"
#include "psolve.h"
#include "quake_util.h"
#include "util.h"

#define  QC  qc = 0.577350269189 /* sqrt(3.0)/3.0; */

#define MAX(a, b) ((a)>(b)?(a):(b))

#define  XI  xi[3][8] = { {-1,  1, -1,  1, -1,  1, -1, 1} , \
                          {-1, -1,  1,  1, -1, -1,  1, 1} , \
                          {-1, -1, -1, -1,  1,  1,  1, 1} }

static int superflag = 0;

/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static nlsolver_t           *myNonlinSolver;
static int32_t               thePropertiesCount;
static materialmodel_t       theMaterialModel;
static plasticitytype_t      thePlasticityModel;
static noyesflag_t           theApproxGeoState  = NO;
static double               *theVsLimits;
static double               *theAlphaCohes;
static double               *theKayPhis;
static double               *theStrainRates;
static double               *theSensitivities;
static double               *theHardeningModulus;
static double               *theBetaDilatancy;
static double                theGeostaticLoadingT = 0;
static double                theGeostaticCushionT = 0;
static int                   theGeostaticFinalStep;
static int32_t              *myStationsElementIndices;
//static nlstation_t          *myNonlinStations;
static int32_t              *myNonlinStationsMapping;
static int32_t               myNumberOfNonlinStations;
static int32_t               myNonlinElementsCount = 0;
static int32_t              *myNonlinElementsMapping;
static int32_t               myBottomElementsCount = 0;
static bottomelement_t      *myBottomElements;
static int32_t               theNonlinearFlag = 0;

static double totalWeight = 0;

/* -------------------------------------------------------------------------- */
/*                                 Utilities                                  */
/* -------------------------------------------------------------------------- */

double get_geostatic_total_time() {
    return theGeostaticLoadingT + theGeostaticCushionT;
}

/*
 * Return YES if an element is to be considered nonlinear, NO otherwise.
 */
int isThisElementNonLinear(mesh_t *myMesh, int32_t eindex) {

	elem_t  *elemp;
	edata_t *edata;

	if ( theNonlinearFlag == 0 )
		return NO;

	elemp = &myMesh->elemTable[eindex];
	edata = (edata_t *)elemp->data;

	if ( ( edata->Vs <=  theVsLimits[thePropertiesCount-1] ) && ( edata->Vs >=  theVsLimits[0] ) )
		return YES;

	return NO;
}

/*
 * Performs linear interpolation between to given vectors for a given value
 * and gives back the corresponding pair for the requested Vs.
 *
 */
double interpolate_property_value(double vsRequest, double *propVector)
{
    int i;
    double vs0, vs1;
    double prop0, prop1;

    /* below floor case */
    if ( vsRequest <= theVsLimits[0] ) {
        return propVector[0];
    }

    /* above ceiling case */
    if ( vsRequest > theVsLimits[thePropertiesCount-1] ) {
        return propVector[thePropertiesCount-1];
    }

    /* in between cases */

    for ( i = 0; i < thePropertiesCount-1; i++ ) {

        if ( ( vsRequest >  theVsLimits[i]   ) &&
             ( vsRequest <= theVsLimits[i+1] ) ) {

            vs0 = theVsLimits[i];
            vs1 = theVsLimits[i+1];

            prop0 = propVector[i];
            prop1 = propVector[i+1];
        }
    }

    double result = prop0 + (vsRequest - vs0)*( (prop1 - prop0)/(vs1 - vs0) );

    return result;
}

/*
 * Returns the value of the constant alpha for Drucker-Prager's material
 * model
 */
double get_alpha(double vs, double phi) {

    double alpha;
    alpha = 2. * sin(phi) / ( sqrt(3.0) * ( 3. - sin(phi) ) );
    if ( alpha > 0.40 ) {
    	fprintf(stderr,"Illegal alpha= %f "
    			"Friction angle larger that 50deg\n", alpha);
    	MPI_Abort(MPI_COMM_WORLD, ERROR);
    	exit(1);
    }

    return alpha;
}

/*
 * Returns the value of the constant kay in Drucker-Prager's material
 * model
 */
//double get_kay(double vs, double phi) {
//
//    double k, c;
//
//    c     = interpolate_property_value(vs, theAlphaCohes);
//    k     = 6. * c * cos(phi) / ( sqrt(3.0) * ( 3. - sin(phi) ) );
//
//    return k;
//}

/*
 * Returns the value of the constant gamma in Drucker-Prager's material
 * model
 */
double get_gamma(double vs, double phi) {

    double gamma;

    gamma     = 6. * cos(phi) / ( sqrt(3.0) * ( 3. - sin(phi) ) );

    return gamma;
}

/*
 * Returns the value of the constant phi  in Drucker-Prager's material
 * model
 */
double get_phi(double vs) {

	double phi;

	phi   = interpolate_property_value(vs, theKayPhis) * PI / 180.0;

	return phi;
}

/*
 * Returns the value of the constant phi  in Drucker-Prager's material
 * model
 */
double get_dilatancy(double vs) {

	double dilt;

	dilt   = interpolate_property_value(vs, theBetaDilatancy) * PI / 180.0;

	return dilt;
}

/*
 * Returns the value of the constant beta (dilatancy angle)  in Drucker-Prager's material
 * model.
 */
double get_beta(double vs) {

	double  beta, dil;

	dil   = interpolate_property_value(vs, theBetaDilatancy) * PI / 180.0;
	beta  = 2. * sin(dil) / ( sqrt(3.0) * ( 3. - sin(dil) ) );

	return beta;
}


double get_cohesion(double vs) {

	double  coh;
	coh   = interpolate_property_value(vs, theAlphaCohes);
	return coh;
}

double get_hardmod(double vs) {

	double  hrd;
	hrd   = interpolate_property_value(vs, theHardeningModulus);

	return hrd;
}


double get_gamma_eff (double vs30, double zo)  {

	double gamma_eff=0;

	if ( ( vs30 >= 760 ) && ( vs30 <= 1500 ) )  { /* Site class B "Rock"  */
		if ( ( zo >= 0 ) && ( zo <= 6.096 ) ) {
			gamma_eff = pow(10.0,-1.73) / 100.0;
		} else if ( ( zo > 6.096 ) && ( zo <= 15.24 ) )   {
			gamma_eff = pow(10.0,-1.66) / 100.0;
		} else if ( ( zo > 15.24 ) && ( zo <= 36.576 ) )  {
			gamma_eff = pow(10.0,-1.58) / 100.0;
		} else if ( ( zo > 36.576 ) && ( zo <= 76.20 ) )  {
			gamma_eff = pow(10.0,-1.45) / 100.0;
		} else if ( ( zo > 76.20 ) && ( zo <= 152.40 ) )  {
			gamma_eff = pow(10.0,-1.39) / 100.0;
		} else if ( ( zo > 152.40 ) && ( zo <= 304.80 ) ) {
			gamma_eff = pow(10.0,-1.317) / 100.0;
		} else { /* This should not happen */
	    	fprintf(stderr, "nonlinear error for site class B. Attempting to get gamma_eff for vs30= %f at z= %f \n", vs30, zo);
	    	MPI_Abort(MPI_COMM_WORLD, ERROR);
	    	exit(1);
	    }
	} else if ( ( vs30 >= 200 ) && ( vs30 < 760 ) ) { /* Site class C & D "Sand"  */

		if ( ( zo >= 0 ) && ( zo <= 6.096 ) ) {
			gamma_eff = pow(10.0,-1.49) / 100.0;
		} else if ( ( zo > 6.096 ) && ( zo <= 15.24 ) )   {
			gamma_eff = pow(10.0,-1.29) / 100.0;
		} else if ( ( zo > 15.24 ) && ( zo <= 36.576 ) )  {
			gamma_eff = pow(10.0,-1.14) / 100.0;
		} else if ( ( zo > 36.576 ) && ( zo <= 76.20 ) )  {
			gamma_eff = pow(10.0,-1.00) / 100.0;
		} else if ( ( zo > 76.20 ) && ( zo <= 152.40 ) )  {
			gamma_eff = pow(10.0,-0.87) / 100.0;
		} else if ( ( zo > 152.40 ) && ( zo <= 304.80 ) ) {
			gamma_eff = pow(10.0,-0.7) / 100.0;
		} else { /* This should not happen */
	    	fprintf(stderr, "nonlinear error for site class C and D.  Attempting to get gamma_eff for vs30= %f at z= %f \n", vs30, zo);
	    	MPI_Abort(MPI_COMM_WORLD, ERROR);
	    	exit(1);
	    }
	} else {

			fprintf(stderr, "nonlinear error attempting to get gamma_eff - vs30: %f or z= %f are out of range \n", vs30, zo);
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
	}

return gamma_eff;
}

/* -------------------------------------------------------------------------- */
/*       Initialization of parameters, structures and memory allocations      */
/* -------------------------------------------------------------------------- */



void nonlinear_init( int32_t     myID,
                     const char *parametersin,
                     double      theDeltaT,
                     double      theEndT )
{
    double  double_message[2];
    int     int_message[6];

    /* Capturing data from file --- only done by PE0 */
    if (myID == 0) {
        if (nonlinear_initparameters(parametersin, theDeltaT, theEndT) != 0) {
            fprintf(stderr,"Thread 0: nonlinear_local_init: "
                    "nonlinear_initparameters error\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }
    }

    /* Broadcasting data */
    double_message[0] = theGeostaticLoadingT;
    double_message[1] = theGeostaticCushionT;

    int_message[0] = (int)theMaterialModel;
    int_message[1] = thePropertiesCount;
    int_message[2] = theGeostaticFinalStep;
    int_message[3] = (int)thePlasticityModel;
    int_message[4] = (int)theApproxGeoState;
    int_message[5] = (int)theNonlinearFlag;

    MPI_Bcast(double_message, 2, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(int_message,    6, MPI_INT,    0, comm_solver);

    theGeostaticLoadingT  = double_message[0];
    theGeostaticCushionT  = double_message[1];

    theMaterialModel      = int_message[0];
    thePropertiesCount    = int_message[1];
    theGeostaticFinalStep = int_message[2];
    thePlasticityModel    = int_message[3];
    theApproxGeoState     = int_message[4];
    theNonlinearFlag      = int_message[5];

    /* allocate table of properties for all other PEs */

    if (myID != 0) {
        theVsLimits         = (double*)malloc(sizeof(double) * thePropertiesCount);
        theAlphaCohes       = (double*)malloc(sizeof(double) * thePropertiesCount);
        theKayPhis          = (double*)malloc(sizeof(double) * thePropertiesCount);
        theStrainRates      = (double*)malloc(sizeof(double) * thePropertiesCount);
        theSensitivities    = (double*)malloc(sizeof(double) * thePropertiesCount);
        theHardeningModulus = (double*)malloc(sizeof(double) * thePropertiesCount);
        theBetaDilatancy    = (double*)malloc(sizeof(double) * thePropertiesCount);
    }

    /* Broadcast table of properties */
    MPI_Bcast(theVsLimits,         thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theAlphaCohes,       thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theKayPhis,          thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theStrainRates,      thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theSensitivities,    thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theHardeningModulus, thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theBetaDilatancy,    thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
}

/*
 * Reads from parameters.in and stores in PE0 globals.
 */
int32_t nonlinear_initparameters ( const char *parametersin,
                                   double      theDeltaT,
                                   double      theEndT )
{
    FILE    *fp;
    int32_t  properties_count;
    int      row;
    double   geostatic_loading_t, geostatic_cushion_t,
            *auxiliar;
    char     material_model[64],
             plasticity_type[64], approx_geostatic_state[64];

    materialmodel_t      materialmodel;
    plasticitytype_t     plasticitytype;
    noyesflag_t          approxgeostatic = -1;

    /* Opens numericalin file */
    if ((fp = fopen(parametersin, "r")) == NULL) {
        fprintf(stderr, "Error opening %s\n at nl_init_parameters", parametersin);
        return -1;
    }

    /* Parses parameters.in to capture nonlinear single-value parameters */

    if ( (parsetext(fp, "geostatic_loading_time_sec",   'd', &geostatic_loading_t    ) != 0) ||
         (parsetext(fp, "geostatic_cushion_time_sec",   'd', &geostatic_cushion_t    ) != 0) ||
         (parsetext(fp, "material_model",               's', &material_model         ) != 0) ||
         (parsetext(fp, "approximate_geostatic_state",  's', &approx_geostatic_state ) != 0) ||
         (parsetext(fp, "material_plasticity_type",     's', &plasticity_type        ) != 0) ||
         (parsetext(fp, "material_properties_count",    'i', &properties_count       ) != 0) )
    {
        fprintf(stderr, "Error parsing nonlinear parameters from %s\n", parametersin);
        return -1;
    }

    /* Performs sanity checks */
    if ( (geostatic_loading_t < 0) || (geostatic_cushion_t < 0) ||
         (geostatic_loading_t + geostatic_cushion_t > theEndT) ) {
        fprintf(stderr, "Illegal geostatic loading/cushion time %f/%f\n",
                geostatic_loading_t, geostatic_cushion_t);
        return -1;
    }

    if ( strcasecmp(material_model, "linear") == 0 ) {
        materialmodel = LINEAR;
    } else if ( strcasecmp(material_model, "vonMises") == 0 ) {
        materialmodel = VONMISES;
    }  else if ( strcasecmp(material_model, "MohrCoulomb") == 0 ) {
        materialmodel = MOHR_COULOMB;
    } else if ( strcasecmp(material_model, "DruckerPrager") == 0 ) {
        materialmodel = DRUCKERPRAGER;
    }
    else {
        fprintf(stderr,
                "Illegal material model for nonlinear analysis"
                "(linear, vonMises, DruckerPrager): %s\n", material_model);
        return -1;
    }



    if (properties_count < 1) {
        fprintf(stderr, "Illegal material properties count %d\n", properties_count);
        return -1;
    }

    if ( strcasecmp(plasticity_type, "rate_dependant") == 0 ) {
        plasticitytype = RATE_DEPENDANT;
    } else if ( strcasecmp(plasticity_type, "rate_independant") == 0 ) {
        plasticitytype = RATE_INDEPENDANT;
    } else {
        fprintf(stderr,
                "Illegal material plasticity type for nonlinear "
                "analysis (rate_dependant, rate_independant): %s\n",
                plasticity_type);
        return -1;
    }

    if ( strcasecmp(approx_geostatic_state, "yes") == 0 ) {
        approxgeostatic = YES;
    } else if ( strcasecmp(approx_geostatic_state, "no") == 0 ) {
    	approxgeostatic = NO;
    } else {
        fprintf(stderr,
        		":Unknown response for considering an "
                "approximate geostatic state (yes or no): %s\n",
                approx_geostatic_state );
        return -1;
    }

    if ( ( (geostatic_loading_t > 0) || (geostatic_cushion_t > 0) ) &&
         ( approxgeostatic == YES ) ) {
        fprintf(stderr, "Approximate geostatic-state must be set to (no) when geostatic loading/cushion time > 0. %s\n",
        		approx_geostatic_state);
        return -1;
    }


    /* Initialize the static global variables */
    theGeostaticLoadingT  = geostatic_loading_t;
    theGeostaticCushionT  = geostatic_cushion_t;
    theGeostaticFinalStep = (int)( (geostatic_loading_t + geostatic_cushion_t) / theDeltaT );
    theMaterialModel      = materialmodel;
    thePropertiesCount    = properties_count;
    thePlasticityModel    = plasticitytype;
    theApproxGeoState     = approxgeostatic;

    auxiliar             = (double*)malloc( sizeof(double) * thePropertiesCount * 7 );
    theVsLimits          = (double*)malloc( sizeof(double) * thePropertiesCount );
    theAlphaCohes        = (double*)malloc( sizeof(double) * thePropertiesCount );
    theKayPhis           = (double*)malloc( sizeof(double) * thePropertiesCount );
    theStrainRates       = (double*)malloc( sizeof(double) * thePropertiesCount );
    theSensitivities     = (double*)malloc( sizeof(double) * thePropertiesCount );
    theHardeningModulus  = (double*)malloc( sizeof(double) * thePropertiesCount );
    theBetaDilatancy     = (double*)malloc( sizeof(double) * thePropertiesCount );

    if ( parsedarray( fp, "material_properties_list", thePropertiesCount * 7, auxiliar ) != 0) {
        fprintf(stderr, "Error parsing nonlinear material properties list from %s\n", parametersin);
        return -1;
    }

    for ( row = 0; row < thePropertiesCount; row++) {
        theVsLimits[row]          = auxiliar[ row * 7     ];
        theAlphaCohes[row]        = auxiliar[ row * 7 + 1 ];
        theKayPhis[row]           = auxiliar[ row * 7 + 2 ];
        theStrainRates[row]       = auxiliar[ row * 7 + 3 ];
        theSensitivities[row]     = auxiliar[ row * 7 + 4 ];
        theHardeningModulus[row]  = auxiliar[ row * 7 + 5 ];
        theBetaDilatancy[row]     = auxiliar[ row * 7 + 6 ];
    }

    theNonlinearFlag = 1;

    return 0;
}

/*
 * Counts the number of nonlinear elements in my local mesh
 */
void nonlinear_elements_count(int32_t myID, mesh_t *myMesh) {

    int32_t eindex;
    int32_t count = 0;

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        if ( isThisElementNonLinear(myMesh, eindex) == YES ) {
            count++;
        }
    }

    if ( count > myMesh-> lenum ) {
        fprintf(stderr,"Thread %d: nl_elements_count: "
                "more elements than expected\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    myNonlinElementsCount = count;

    return;
}

/*
 * Re-counts and stores the nonlinear element indices to a static local array
 * that will serve as mapping tool to the local mesh elements table.
 */
void nonlinear_elements_mapping(int32_t myID, mesh_t *myMesh) {

    int32_t eindex;
    int32_t count = 0;

    XMALLOC_VAR_N(myNonlinElementsMapping, int32_t, myNonlinElementsCount);

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        if ( isThisElementNonLinear(myMesh, eindex) == YES ) {
            myNonlinElementsMapping[count] = eindex;
            count++;
        }
    }

    if ( count != myNonlinElementsCount ) {
        fprintf(stderr,"Thread %d: nl_elements_mapping: "
                "more elements than the count\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    return;
}

noyesflag_t isThisElementsAtTheBottom( mesh_t  *myMesh,
                                       int32_t  eindex,
                                       double   depth )
{
    elem_t  *elemp;
    int32_t  nindex;
    double   z_m;

    /* Capture the element's last node at the bottom */
    elemp  = &myMesh->elemTable[eindex];
    nindex = elemp->lnid[7];

    z_m = (myMesh->ticksize)*(double)myMesh->nodeTable[nindex].z;

    if ( z_m == depth ) {
        return YES;
    }

    return NO;
}

void bottom_elements_count(int32_t myID, mesh_t *myMesh, double depth ) {

    int32_t eindex;
    int32_t count = 0;

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        if ( isThisElementsAtTheBottom(myMesh, eindex, depth) == YES ) {
            count++;
        }
    }

    if ( count > myMesh-> lenum ) {
        fprintf(stderr,"Thread %d: bottom_elements_count: "
                "more elements than expected\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    myBottomElementsCount = count;

    return;
}

void bottom_elements_mapping(int32_t myID, mesh_t *myMesh, double depth) {

    int32_t eindex;
    int32_t count = 0;

    XMALLOC_VAR_N(myBottomElements, bottomelement_t, myBottomElementsCount);

    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        if ( isThisElementsAtTheBottom(myMesh, eindex, depth) == YES ) {
            myBottomElements[count].element_id = eindex;
            count++;
        }
    }

    if ( count != myBottomElementsCount ) {
        fprintf(stderr,"Thread %d: bottom_elements_mapping: "
                "more elements than expected\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    return;
}

/*
 * Prints statistics about number of nonlinear elements and stations in nonlinear
 * elements.
 */
void nonlinear_print_stats(int32_t *nonlinElementsCount,
                           int32_t *nonlinStationsCount,
                           int32_t *bottomElementsCount,
                           int32_t  theGroupSize)
{

    int pid;
    global_id_t totalElements = 0;
    global_id_t totalStations = 0;
    global_id_t totalBottom   = 0;

    FILE *fp = hu_fopen( "stat-nonlin.txt", "w" );

    fputs( "\n"
           "# ---------------------------------------- \n"
           "# Nonlinear elements and stations count:   \n"
           "# ---------------------------------------- \n"
           "# Rank    Elements    Stations      Bottom \n"
           "# ---------------------------------------- \n", fp );

    for ( pid = 0; pid < theGroupSize; pid++ ) {

        fprintf( fp, "%06d %11d %11d %11d\n", pid,
                 nonlinElementsCount[pid],
                 nonlinStationsCount[pid],
                 bottomElementsCount[pid] );

        totalElements += nonlinElementsCount[pid];
        totalStations += nonlinStationsCount[pid];
        totalBottom   += bottomElementsCount[pid];
    }

    fprintf( fp,
             "# ---------------------------------------- \n"
             "# Total%11"INT64_FMT" %11"INT64_FMT" %11"INT64_FMT" \n"
             "# ---------------------------------------- \n\n",
             totalElements, totalStations, totalBottom );

    hu_fclosep( &fp );

    /* output aggregate information to the monitor file / stdout */
    fprintf( stdout,
             "\nNonlinear mesh information\n"
             "Total number of nonlinear elements: %11"INT64_FMT"\n"
             "Total number of nonlinear stations: %11"INT64_FMT"\n"
             "Total number of bottom elements:    %11"INT64_FMT"\n\n",
             totalElements, totalStations, totalBottom );

    fflush( stdout );

}

/*
 * Collects statistics about number of nonlinear elements and stations in
 * nonlinear elements.
 */
void nonlinear_stats(int32_t myID, int32_t theGroupSize) {

    int32_t *nonlinElementsCount = NULL;
    int32_t *nonlinStationsCount = NULL;
    int32_t *bottomElementsCount = NULL;

    if ( myID == 0 ) {
        XMALLOC_VAR_N( nonlinElementsCount, int32_t, theGroupSize);
        XMALLOC_VAR_N( nonlinStationsCount, int32_t, theGroupSize);
        XMALLOC_VAR_N( bottomElementsCount, int32_t, theGroupSize);
    }

    MPI_Gather( &myNonlinElementsCount,    1, MPI_INT,
                nonlinElementsCount,       1, MPI_INT, 0, comm_solver );
    MPI_Gather( &myNumberOfNonlinStations, 1, MPI_INT,
                nonlinStationsCount,       1, MPI_INT, 0, comm_solver );
    MPI_Gather( &myBottomElementsCount,    1, MPI_INT,
                bottomElementsCount,       1, MPI_INT, 0, comm_solver );

    if ( myID == 0 ) {

        nonlinear_print_stats( nonlinElementsCount, nonlinStationsCount,
                               bottomElementsCount, theGroupSize);

        xfree_int32_t( &nonlinElementsCount );
    }

    return;
}

/*
 * nonlinear_solver_init: Initialize all the structures needed for nonlinear
 *                        analysis and the material/element constants.
 */
void nonlinear_solver_init(int32_t myID, mesh_t *myMesh, double depth) {

    int32_t eindex, nl_eindex;

    nonlinear_elements_count(myID, myMesh);
    nonlinear_elements_mapping(myID, myMesh);

    if ( theGeostaticLoadingT > 0 ) {
        bottom_elements_count(myID, myMesh, depth);
        bottom_elements_mapping(myID, myMesh, depth);
    }

    /* Memory allocation for mother structure */

    myNonlinSolver = (nlsolver_t *)malloc(sizeof(nlsolver_t));

    if (myNonlinSolver == NULL) {
        fprintf(stderr, "Thread %d: nonlinear_init: out of memory\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    /* Memory allocation for internal structures */

    myNonlinSolver->constants =
        (nlconstants_t *)calloc(myNonlinElementsCount, sizeof(nlconstants_t));
    myNonlinSolver->stresses =
        (qptensors_t *)calloc(myNonlinElementsCount, sizeof(qptensors_t));
    myNonlinSolver->strains =
        (qptensors_t *)calloc(myNonlinElementsCount, sizeof(qptensors_t));
    myNonlinSolver->pstrains1 =
        (qptensors_t *)calloc(myNonlinElementsCount, sizeof(qptensors_t));
    myNonlinSolver->pstrains2 =
        (qptensors_t *)calloc(myNonlinElementsCount, sizeof(qptensors_t));
    myNonlinSolver->ep1 =
        (qpvectors_t *)calloc(myNonlinElementsCount, sizeof(qpvectors_t));
    myNonlinSolver->ep2 =
        (qpvectors_t *)calloc(myNonlinElementsCount, sizeof(qpvectors_t));

    if ( (myNonlinSolver->constants  == NULL) ||
         (myNonlinSolver->stresses   == NULL) ||
         (myNonlinSolver->strains    == NULL) ||
         (myNonlinSolver->ep1        == NULL) ||
         (myNonlinSolver->ep2        == NULL) ||
         (myNonlinSolver->pstrains1  == NULL) ||
         (myNonlinSolver->pstrains2  == NULL) ) {

        fprintf(stderr, "Thread %d: nonlinear_init: out of memory\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    /* Initialization of element constants
     * Tensors have been initialized to 0 by calloc
     */

    for (nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++) {

        elem_t     *elemp;
        edata_t    *edata;
        nlconstants_t *ecp;
        double      mu, lambda;
        double      elementVs, elementVp;

        eindex = myNonlinElementsMapping[nl_eindex];

        elemp = &myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;
        ecp   = myNonlinSolver->constants + nl_eindex;

        int32_t lnid0 = elemp->lnid[0];
        double  zo    = myMesh->ticksize * myMesh->nodeTable[lnid0].z;

        /* get element Vs */

        elementVs   = (double)edata->Vs;
        elementVp   = (double)edata->Vp;

        /* Calculate the lame constants and store in element */

        mu_and_lambda(&mu, &lambda, edata, eindex);
        ecp->lambda = lambda;
        ecp->mu     = mu;

        /* Calculate the vertical stress as a homogeneous half-space */
        if ( theApproxGeoState == YES )
        	ecp->sigmaZ_st = edata->rho * 9.80 * ( zo + edata->edgesize / 2.0 );

        /* Calculate the yield function constants */
        switch (theMaterialModel) {

            case LINEAR:
                ecp->alpha = 0.;
                ecp->k     = 0.;
                ecp->phi   = 0.;
                ecp->beta  = 0.;
                ecp->h     = 0.;
                break;

            case VONMISES:
                ecp->c     = get_cohesion(elementVs);
                ecp->phi   = get_phi(elementVs);
                ecp->dil_angle = 0.0;

                ecp->alpha = 0.;
                ecp->beta  = 0.;
                ecp->gamma = get_gamma(elementVs,ecp->phi);

                ecp->k           = ecp->gamma * ecp->c;
                ecp->h           = get_hardmod(elementVs);
                break;
            case DRUCKERPRAGER:
                ecp->c         = get_cohesion(elementVs);
                ecp->phi       = get_phi(elementVs);
                ecp->dil_angle = get_dilatancy(elementVs);

                ecp->alpha = get_alpha(elementVs, ecp->phi);
                ecp->beta  = get_beta(elementVs);
                ecp->gamma = get_gamma(elementVs,ecp->phi);

                ecp->k     = ecp->gamma * ecp->c;
                ecp->h     = get_hardmod(elementVs);

            	break;
            case MOHR_COULOMB:
                ecp->c     = get_cohesion(elementVs);
                ecp->phi   = get_phi(elementVs);
                ecp->dil_angle = get_dilatancy(elementVs);

                ecp->alpha = 0.;
                ecp->beta  = 0.;
                ecp->gamma = 0.;

                ecp->h = get_hardmod(elementVs);
            	break;

            default:
                fprintf(stderr, "Thread %d: nonlinear_solver_init:\n"
                        "\tUnexpected error with the material model\n", myID);
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
                break;
        }


        ecp->strainrate  =
        		interpolate_property_value(elementVs, theStrainRates  );
        ecp->sensitivity =
        		interpolate_property_value(elementVs, theSensitivities );
//        ecp->hardmodulus =
//        		interpolate_property_value(elementVs, theHardeningModulus );

//        if ( theApproxGeoState == NO ) {
//            ecp->I1_st        =  -1.;
//            ecp->J2square_st  =   0.;
//        } else {
//            ecp->I1_st        = -S_zz * ( 3.0 - 2.0 * sin ( ecp->phi ) );
//            ecp->J2square_st  =  S_zz * S_zz * sin ( ecp->phi ) * sin ( ecp->phi ) / 3.0;
//        }


    } /* for all elements */

}

/* -------------------------------------------------------------------------- */
/*                   Auxiliary tensor manipulation methods                    */
/* -------------------------------------------------------------------------- */

/*
 * tensor_I1: Returns the invariant I1 of a tensor.
 */
double tensor_I1(tensor_t tensor) {

    return tensor.xx + tensor.yy + tensor.zz;
}

/*
 * Returns the octahedral of a tensor given its first invariant I1
 */
double tensor_octahedral(double I1) {

    return I1/3.0;
}

/*
 * Returns the deviator of a tensor given the tensor and its octahedral.
 */
tensor_t tensor_deviator(tensor_t tensor, double oct) {

    tensor_t deviator;

    deviator.xx = tensor.xx - oct;
    deviator.yy = tensor.yy - oct;
    deviator.zz = tensor.zz - oct;
    deviator.xy = tensor.xy;
    deviator.yz = tensor.yz;
    deviator.xz = tensor.xz;

    return deviator;
}

/*
 * tensos_J2: Returns the second invariant of a tensor given its deviator.
 */
double tensor_J2(tensor_t dev) {

    return ( (dev.xx * dev.xx) + (dev.yy * dev.yy) + (dev.zz * dev.zz) ) * 0.5
           + (dev.xy * dev.xy) + (dev.yz * dev.yz) + (dev.xz * dev.xz);
}

/*
 * tensos_Det: Returns the determinan of a tensor.
 */
double tensor_J3(tensor_t dev) {

    return ( dev.xx * ( dev.yy * dev.zz - dev.yz * dev.yz ) - dev.xy * ( dev.xy * dev.zz - dev.xz * dev.yz ) + dev.xz * ( dev.xy * dev.yz - dev.xz * dev.yy ) );
}

/*
 * Computes the contribution of the i-th node to the three derivatives
 * dx, dy, dz of the shape functions evaluated at local coordinates
 * lx, ly, lz.
 */
void point_dxi ( double *dx, double *dy, double *dz,
                 double  lx, double  ly, double  lz, double h, int i )
{
    double XI;
    double Jij = 0.25 / h; /* Jacobian 1/(4h) */

    *dx = Jij * (       xi[0][i]      )
              * ( 1.0 + xi[1][i] * ly )
              * ( 1.0 + xi[2][i] * lz );

    *dy = Jij * ( 1.0 + xi[0][i] * lx )
              * (       xi[1][i]      )
              * ( 1.0 + xi[2][i] * lz );

    *dz = Jij * ( 1.0 + xi[0][i] * lx )
              * ( 1.0 + xi[1][i] * ly )
              * (       xi[2][i]      );

    return;
}

/*
 * Computes the three derivatives of the shape functions evaluated at
 * coordinates j-th quadrature point.
 */
void compute_qp_dxi (double *dx, double *dy, double *dz, int i, int j, double h)
{
    double XI, QC;

    /* quadrature point local coordinates */
    double lx = xi[0][j] * qc ;
    double ly = xi[1][j] * qc ;
    double lz = xi[2][j] * qc ;

    point_dxi(dx, dy, dz, lx, ly, lz, h, i);

    return;
}

/*
 * Resets a tensor to zero in all its components.
 */
tensor_t init_tensor() {

    tensor_t tensor;

    tensor.xx = 0.0;
    tensor.yy = 0.0;
    tensor.zz = 0.0;
    tensor.xy = 0.0;
    tensor.yz = 0.0;
    tensor.xz = 0.0;

    return tensor;
}

void init_tensorptr(tensor_t *tensor) {

    tensor->xx = 0.0;
    tensor->yy = 0.0;
    tensor->zz = 0.0;
    tensor->xy = 0.0;
    tensor->yz = 0.0;
    tensor->xz = 0.0;

    return;
}

/*
 * Compute strain tensor of a given point in the element.
 */
tensor_t point_strain (fvector_t *u, double lx, double ly, double lz, double h) {

    int i;

    tensor_t strain = init_tensor();

    /* Contribution of each node */
    for (i = 0; i < 8; i++) {

        double dx, dy, dz;

        point_dxi(&dx, &dy, &dz, lx, ly, lz, h, i);

        strain.xx += dx * u[i].f[0];
        strain.yy += dy * u[i].f[1];
        strain.zz += dz * u[i].f[2];

        strain.xy += 0.5 * ( dy * u[i].f[0] + dx * u[i].f[1] );
        strain.yz += 0.5 * ( dz * u[i].f[1] + dy * u[i].f[2] );
        strain.xz += 0.5 * ( dz * u[i].f[0] + dx * u[i].f[2] );

    } /* nodes contribution */

    return strain;
}

/*
 * Computes the stress tensor in a given point in the element from the point's
 * strain tensor and the element properties according to the linear elastic
 * stress-strain relationship.
 */
tensor_t point_stress (tensor_t strain, double mu, double lambda) {

    double mu2, lkk;
    double strain_kk;
    tensor_t stress;

    /* calculate strtain_kk */
    strain_kk = tensor_I1(strain);

    mu2 = 2.0 * mu;
    lkk = lambda * strain_kk;

    stress.xx = mu2 * strain.xx + lkk;
    stress.yy = mu2 * strain.yy + lkk;
    stress.zz = mu2 * strain.zz + lkk;

    stress.xy = mu2 * strain.xy;
    stress.yz = mu2 * strain.yz;
    stress.xz = mu2 * strain.xz;

    return stress;
}

/*
 * Computes the strain tensor given the stress tensor and the material properties.
 */
tensor_t elastic_strains (tensor_t stress, double mu, double kappa) {

    double Skk, e_kk;
    tensor_t strain, strain_dev;

    Skk = tensor_I1(stress);
    e_kk = Skk / ( 9.0 * kappa);

    strain_dev =  tensor_deviator( stress, Skk / 3.0 );

    strain.xx = 1./(2. * mu) * strain_dev.xx + e_kk;
    strain.yy = 1./(2. * mu) * strain_dev.yy + e_kk;
    strain.zz = 1./(2. * mu) * strain_dev.zz + e_kk;
    strain.xy = 1./(2. * mu) * strain_dev.xy;
    strain.xz = 1./(2. * mu) * strain_dev.xz;
    strain.yz = 1./(2. * mu) * strain_dev.yz;

    return strain;
}

/*
 * Computes the stress tensors for the quadrature points of an element given
 * the strain tensors.
 */
qptensors_t compute_qp_stresses (qptensors_t strains, double mu, double lambda)
{
    int i;
    qptensors_t stresses;

    /* Loop over the quadrature points */
    for ( i = 0; i < 8; i++ ) {
        stresses.qp[i] = point_stress(strains.qp[i], mu, lambda);
    }

    return stresses;
}

/*
 * Returns the subtraction between two tensors.
 */
tensor_t subtrac_tensors(tensor_t A, tensor_t B) {

    tensor_t C;

    C.xx = A.xx - B.xx;
    C.yy = A.yy - B.yy;
    C.zz = A.zz - B.zz;
    C.xy = A.xy - B.xy;
    C.yz = A.yz - B.yz;
    C.xz = A.xz - B.xz;

    return C;
}

/*
 * Returns the summation of two tensors.
 */
tensor_t add_tensors(tensor_t A, tensor_t B) {

    tensor_t C;

    C.xx = A.xx + B.xx;
    C.yy = A.yy + B.yy;
    C.zz = A.zz + B.zz;
    C.xy = A.xy + B.xy;
    C.yz = A.yz + B.yz;
    C.xz = A.xz + B.xz;

    return C;
}

/*
 * Returns the ZERO tensor.
 */
tensor_t zero_tensor() {

    tensor_t C;

    C.xx = 0.0;
    C.yy = 0.0;
    C.zz = 0.0;
    C.xy = 0.0;
    C.yz = 0.0;
    C.xz = 0.0;

    return C;
}

/*
 * Returns the approximated self-weight tensor.
 */
tensor_t ApproxGravity_tensor(double Szz, double phi, double h, double lz, double rho) {

	double Ko = 1 - sin(phi);
	double Sigma = -( Szz + 9.8 * rho * h * lz * 0.5);

    tensor_t C;

    C.xx = Ko * Sigma;
    C.yy = Ko * Sigma;
    C.zz = Sigma;
    C.xy = 0.0;
    C.yz = 0.0;
    C.xz = 0.0;

    return C;
}

/*
 * Returns the subtraction between two tensors for all quadrature points.
 */
qptensors_t subtrac_qptensors(qptensors_t A, qptensors_t B) {

    int i;
    qptensors_t C;

    /* Loop over quadrature points */
    for (i = 0; i < 8; i++) {
        C.qp[i] = subtrac_tensors(A.qp[i], B.qp[i]);
    }

    return C;
}

tensor_t copy_tensor (tensor_t original) {

    tensor_t copy;

    copy.xx = original.xx;
    copy.yy = original.yy;
    copy.zz = original.zz;
    copy.xy = original.xy;
    copy.yz = original.yz;
    copy.xz = original.xz;

    return copy;
}

double compute_yield_surface_stateII ( double J3, double J2, double I1, double alpha, double phi, tensor_t Sigma ) {

	double Yf=0., p, q, r, teta, Rmc, s1, s3;

	if ( ( theMaterialModel == VONMISES ) || ( theMaterialModel == DRUCKERPRAGER ) ) {
		Yf = alpha * I1 + sqrt( J2 );
	} else {

		p = (1./3.) * I1;
		q = 2.0 * pow( J2, 1.5 );
		r = -3.0 * sqrt(3.0) * (J3);

		if ( ( r/q <= 1.00000001 ) && ( r/q >= 0.99999999 ) )
			teta = PI / 6.0;
		else if ( ( r/q >= -1.00000001 ) && ( r/q <= -0.99999999 ) )
			teta = -PI / 6.0;
		else
			teta = 1./3. * asin(r/q);

		if ( ( teta > PI / 6.0 ) || ( teta < -PI / 6.0 ) ) {

			vect1_t n1, n2, n3, eig_vals;

		    specDecomp(Sigma, &n1, &n2, &n3, &eig_vals);

		    s1 = eig_vals.x;
		    s3 = eig_vals.z;

		    Yf = s1 - s3 + ( s1 + s3 )*sin(phi);

		} else {
			Rmc = -1./sqrt(3.0) * ( sin(phi)*sin(teta) ) + cos(teta);
			Yf = 2. * ( Rmc * sqrt(J2) + p * sin(phi) );
		}

	}

	return Yf;

}

double compute_hardening ( double gamma, double c, double h, double ep_bar, double phi ) {
	double H=0.;

	if ( ( theMaterialModel == VONMISES ) || ( theMaterialModel == DRUCKERPRAGER ) ) {
		H = gamma * ( c + h * ep_bar);
	} else {
		H = 2.0 * ( c + h * ep_bar) * cos(phi);
	}

	return H;

}


/*
double compute_dLambdaII ( nlconstants_t constants, double fs, double eff_ps, double J2, double I1, double J2_st, double I1_st, double *po) {

	double phi_pt, s, c, kappa, mu, beta, alpha, tanpsi_min, tanpsi, FsT, delta=0;   variables needed for the plastic strain update

	if ( thePlasticityModel == RATE_DEPENDANT ) {
		double factor      = fs / constants.k;
		double strainRate  = constants.strainrate;
		double sensitivity = constants.sensitivity;
		double oneOverM    = 1.0 / sensitivity;

		return strainRate * pow(factor, oneOverM);
	}

	 Dorian: Rate independent plastic multiplier computation stage.
	 This expression is exact for the Drucker-Prager material model with Linear hardening Rule

	s      = constants.h;
	c      = constants.k;
	mu     = constants.mu;
	beta   = constants.beta;
	alpha  = constants.alpha;

	kappa  = ( constants.lambda + 2. * mu / 3. );
	phi_pt = sqrt ( 1/2. + 3.0 * beta * beta );

	tanpsi_min = mu / ( 9.0 * kappa * beta );
	tanpsi     = sqrt(J2 + J2_st) / ( I1 + I1_st - c / alpha );

	FsT = fs - c - s * eff_ps;

	if ( FsT < 0 )
		return 0.;

	if ( theMaterialModel == VONMISES ) {

		delta = FsT / ( constants.mu + s * phi_pt );

	} else if ( ( tanpsi < 0 ) || ( tanpsi > tanpsi_min ) ) {

		delta = FsT / ( constants.mu + 9.0 * kappa * constants.alpha * constants.beta + s * phi_pt );

	} else {
		FsT = alpha * ( I1 + I1_st ) - c - s * eff_ps;
		delta = FsT / ( 9.0 * kappa * constants.alpha * constants.beta + s * phi_pt );
		*po   = ( ( I1 + I1_st ) - 9.0 * kappa * beta * delta ) / 3.0;


		Sanity Check
		if ( ( *po < 0.0 ) )  {
		 this should not happen
            fprintf(stderr, "Thread po = %8e in: nonlinear_compute_dLambda:\n"
                    "\t Negative pressure capacity at the apex of the Drucker-Prager cone \n", *po);
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);

		}
	}
	return delta;
}
*/


void material_update ( nlconstants_t constants, tensor_t e_n, tensor_t ep, double ep_barn, tensor_t sigma0, double dt,
		tensor_t *epl, tensor_t *sigma, double *ep_bar, double *fs) {
	/* INPUTS:
	 * constants: Material constants
	 * e_n      : Total strain tensor
	 * ep       : Plastic strain tensor at t-1. Used to compute the predictor state
	 * ep_barn  : equivalent plastic strain at t-1. Used to compute the predictor hardening function
	 * sigma0   : Approximated self-weight tensor.
	 * dt       : Time step
	 *
	 * OUTPUTS:
	 * fs       : Updated yield function value
	 * epl      : Updated plastic strain
	 * sigma    : Updated stress tensor
	 * ep_bar   : Updated equivalent hardening variable
	 */

	double phi_pt, c, h, kappa, mu, beta, alpha, gamma, phi, dil, Fs_pr, Lambda, dLambda=0.0,
			Tol_sigma = 5e-10, cond1, cond2, dep_bar; /*  variables needed for the plastic strain update */

	h      = constants.h;
	c      = constants.c;

	phi = constants.phi;
	dil = constants.dil_angle;

	mu     = constants.mu;
	Lambda = constants.lambda;
	kappa  = constants.lambda + 2.0 * mu / 3.0;

	beta   = constants.beta;
	alpha  = constants.alpha;
	gamma  = constants.gamma;

	phi_pt = gamma / (3.0*beta);

	/* ---------      get the stress predictor tensor      ---------*/
	tensor_t estrain     = subtrac_tensors ( e_n, ep );             /* strain predictor   */
	tensor_t stresses    = point_stress ( estrain, mu, Lambda );    /* stress predictor   */
	tensor_t sigma_trial = add_tensors(stresses,sigma0);            /* stress predictor TOTAL  */

	/* compute the invariants of the stress predictor*/
	double   I1_pr   = tensor_I1 ( sigma_trial );
	double   oct_pr  = tensor_octahedral ( I1_pr );
	tensor_t dev_pr  = tensor_deviator ( sigma_trial, oct_pr );
	double   J2_pr   = tensor_J2 ( dev_pr );
	double   J3_pr   = tensor_J3 (dev_pr);
	tensor_t dfds_pr = compute_dfds ( dev_pr, J2_pr, beta );

	if ( ( theMaterialModel == VONMISES ) || ( theMaterialModel == DRUCKERPRAGER ) ){

		Fs_pr = compute_yield_surface_stateII ( J3_pr, J2_pr, I1_pr, alpha, phi, sigma_trial) - compute_hardening(gamma,c,h,ep_barn,phi); /* Fs predictor */

		if ( Fs_pr < 0.0 ) {
			*epl    = copy_tensor(ep);
			*sigma  = copy_tensor(stresses); /* return stresses without self-weight  */
			*ep_bar = ep_barn;
			*fs     = Fs_pr;
			return;
		}

		dLambda = Fs_pr / ( mu + 9.0 * kappa * alpha * beta + h * gamma * gamma );

		/* Updated plastic strains */
		epl->xx = ep.xx +  dLambda * dfds_pr.xx;
		epl->yy = ep.yy +  dLambda * dfds_pr.yy;
		epl->zz = ep.zz +  dLambda * dfds_pr.zz;
		epl->xy = ep.xy +  dLambda * dfds_pr.xy;
		epl->yz = ep.yz +  dLambda * dfds_pr.yz;
		epl->xz = ep.xz +  dLambda * dfds_pr.xz;

		/* Updated stresses */
		estrain     = subtrac_tensors ( e_n, *epl );
		stresses    = point_stress ( estrain, mu, Lambda );
		*sigma      = stresses;
		stresses    = add_tensors ( stresses, sigma0 );

		/* updated invariants*/
		double I1     = tensor_I1 ( stresses );
		double oct    = tensor_octahedral ( I1 );
		tensor_t dev  = tensor_deviator ( stresses, oct );
		double J2     = tensor_J2 ( dev );
		double J3     = tensor_J3 ( dev );

		/* Updated equivalent plastic strain */
		*ep_bar = ep_barn + dLambda * gamma;

		/* Updated yield function value  */
		*fs = compute_yield_surface_stateII ( J3, J2, I1, alpha, phi, stresses) - compute_hardening(gamma,c,h,*ep_bar,phi);

		/* check for apex zone in DP model */
		if (  theMaterialModel == DRUCKERPRAGER  ){

			double Imin = I1_pr - 9.0 * kappa * beta / mu * sqrt(J2_pr);
			double Imax = I1_pr + sqrt(J2_pr)/alpha;

			if ( (I1 < Imin) || (I1 > Imax) || (sqrt(J2_pr) - mu * dLambda < 0.0) ) { /*return to the apex  */

				dep_bar = (  compute_yield_surface_stateII ( 0.0, 0.0, I1_pr, alpha, phi, sigma_trial ) - compute_hardening(gamma,c,h,ep_barn,phi) ) / ( 9.0 * kappa * alpha * beta / gamma + h * gamma );

				/* Updated equivalent plastic strain */
				*ep_bar = ep_barn + dep_bar;

				double Skk = I1_pr - 9.0 * kappa * beta / gamma * dep_bar;

				/* Updated stresses:
				 * It must be isotropic at the apex*/
				stresses.xx    = Skk/3.0;
				stresses.yy    = Skk/3.0;
				stresses.zz    = Skk/3.0;
				stresses.xy    = 0.0;
				stresses.xz    = 0.0;
				stresses.yz    = 0.0;
				*sigma      = subtrac_tensors ( stresses, sigma0 ); /* Sigma is still isotropic since sigma0 is isotropic */

				double Skk_rlt = tensor_I1 ( *sigma );  /* Relative trace. */

				/* Updated strains */
				estrain.xx = Skk_rlt / (3.0 * kappa);
				estrain.yy = Skk_rlt / (3.0 * kappa);
				estrain.zz = Skk_rlt / (3.0 * kappa);
				estrain.xy = 0.0;
				estrain.xz = 0.0;
				estrain.yz = 0.0;

				*epl  = subtrac_tensors ( e_n, estrain );

				*fs = alpha * Skk - compute_hardening(gamma,c,h,*ep_bar,phi);
			}
		}

	} else { /* Must be MohrCoulomb soil */

		/* Spectral decomposition of the sigma_trial tensor*/
		vect1_t n1, n2, n3, sigma_ppal_trial;
		int edge;

		Fs_pr = compute_yield_surface_stateII ( J3_pr, J2_pr, I1_pr, alpha, phi, sigma_trial) - compute_hardening(gamma,c,h,ep_barn,phi);

		if ( Fs_pr < 0.0 ) {
			*epl    = copy_tensor(ep);
			*sigma  = copy_tensor(stresses); /* return stresses without self-weight  */
			*ep_bar = ep_barn;
			*fs     = Fs_pr;
			return;
		}

		/* check for return to the main plane */
		specDecomp(sigma_trial, &n1, &n2, &n3, &sigma_ppal_trial); /* eig_values.x > eig_values.y > eig_values.z   */

		vect1_t sigma_ppal;
		BOX85_l(ep_barn, sigma_ppal_trial, phi, dil, h, c, kappa, mu, &sigma_ppal, ep_bar); /* Return to the main plan */

		/* Check assumption of returning to the main plan */
		if ( ( sigma_ppal.x <= sigma_ppal.y ) || ( sigma_ppal.y <= sigma_ppal.z ) ) {

			if ( ( 1. - sin(dil) ) * sigma_ppal_trial.x - 2. * sigma_ppal_trial.y + ( 1. + sin(dil) ) * sigma_ppal_trial.z > 0 )
				edge = 1; /*return to the right edge*/
			else
				edge = -1; /*return to the left edge*/

			BOX86_l    (ep_barn, sigma_ppal_trial, phi, dil, h, c, kappa, mu, edge, &sigma_ppal, ep_bar);

			cond1 = sigma_ppal.x - sigma_ppal.y;
			cond2 = sigma_ppal.y - sigma_ppal.z;
			double p_trial = ( sigma_ppal_trial.x + sigma_ppal_trial.y + sigma_ppal_trial.z )/3.0;

			if ( (cond1 <= 0.0 ) && ( abs(cond1) >= Tol_sigma)  ) { /* return to the apex */
				BOX87_l(ep_barn, p_trial, phi, dil, h, c, kappa, &sigma_ppal, ep_bar);
			} else if ( (cond2 <= 0.0) && (abs(cond2) >= Tol_sigma) ){
				BOX87_l(ep_barn, p_trial, phi, dil, h, c, kappa, &sigma_ppal, ep_bar);
			}
		}

		/* get updated stress tensor "sigma" */
		tensor_t stressRecomp = specRecomp(sigma_ppal, n1, n2, n3);
		*sigma  = subtrac_tensors(stressRecomp,sigma0);

		estrain = elastic_strains (*sigma, mu, kappa);
		*epl  = subtrac_tensors ( e_n, estrain );

		/* updated invariants*/
		double I1     = tensor_I1 ( stressRecomp );
		double oct    = tensor_octahedral ( I1 );
		tensor_t dev  = tensor_deviator ( stressRecomp, oct );
		double J2     = tensor_J2 ( dev );
		double J3     = tensor_J3 ( dev );

		*fs = compute_yield_surface_stateII ( J3, J2, I1, alpha, phi, stressRecomp) - compute_hardening(gamma,c,h,*ep_bar,phi);

	}



	//	if ( thePlasticityModel == RATE_DEPENDANT ) { /*TODO: Add implementation for rate dependant material model  */
	//
	//		/* Rate dependant material is considered as a Drucker-Prager material   */
	//
	//		double factor      = fs / constants.k;
	//		double strainRate  = constants.strainrate;
	//		double sensitivity = constants.sensitivity;
	//		double oneOverM    = 1.0 / sensitivity;
	//
	//		dLambda =	strainRate * pow(factor, oneOverM);
	//
	//		epl->xx = ep.xx + dt * dLambda * dfds.xx;
	//		epl->yy = ep.yy + dt * dLambda * dfds.yy;
	//		epl->zz = ep.zz + dt * dLambda * dfds.zz;
	//		epl->xy = ep.xy + dt * dLambda * dfds.xy;
	//		epl->yz = ep.yz + dt * dLambda * dfds.yz;
	//		epl->xz = ep.xz + dt * dLambda * dfds.xz;
	//
	//		estrain   = subtrac_tensors ( e_n, *epl );
	//		*sigma    = point_stress ( estrain, constants.mu, constants.lambda );
	//
	//		*ep_bar   = ep_barn + gamma * dLambda;
	//
	//
	//	}


}

/* computes the derivatives of the flow potential for a Drucker-Prager material */
tensor_t compute_dfds (tensor_t dev, double J2, double beta) {

    tensor_t dfds;

    dfds.xx = dev.xx / ( 2.0 * sqrt(J2) ) + beta;
    dfds.yy = dev.yy / ( 2.0 * sqrt(J2) ) + beta;
    dfds.zz = dev.zz / ( 2.0 * sqrt(J2) ) + beta;
    dfds.xy = dev.xy / ( 2.0 * sqrt(J2) );
    dfds.yz = dev.yz / ( 2.0 * sqrt(J2) );
    dfds.xz = dev.xz / ( 2.0 * sqrt(J2) );

    return dfds;

}

/*tensor_t compute_pstrain2 ( nlconstants_t constants, tensor_t pstrain1, tensor_t tstrain,
							tensor_t dfds, double dLambda, double dt, double J2, double I1,
							double J2_st, double I1_st, double po ) {

    tensor_t pstrain2;
    double kappa;

	kappa  = ( constants.lambda + 2.0 * constants.mu / 3.0 );

	if ( dLambda == 0 )
		return pstrain1;

    if ( thePlasticityModel == RATE_DEPENDANT ) {
    	pstrain2.xx = pstrain1.xx + dt * dLambda * dfds.xx;
    	pstrain2.yy = pstrain1.yy + dt * dLambda * dfds.yy;
    	pstrain2.zz = pstrain1.zz + dt * dLambda * dfds.zz;
    	pstrain2.xy = pstrain1.xy + dt * dLambda * dfds.xy;
    	pstrain2.yz = pstrain1.yz + dt * dLambda * dfds.yz;
    	pstrain2.xz = pstrain1.xz + dt * dLambda * dfds.xz;

    } else if ( po >  0.0  ) {

        pstrain2.xx = tstrain.xx -  ( po - I1_st / 3. ) / ( 3.0 * kappa );
        pstrain2.yy = tstrain.yy -  ( po - I1_st / 3. ) / ( 3.0 * kappa );
        pstrain2.zz = tstrain.zz -  ( po - I1_st / 3. ) / ( 3.0 * kappa );
        pstrain2.xy = tstrain.xy;
        pstrain2.yz = tstrain.yz;
        pstrain2.xz = tstrain.xz;

    } else {
    	pstrain2.xx = pstrain1.xx +  dLambda * dfds.xx;
    	pstrain2.yy = pstrain1.yy +  dLambda * dfds.yy;
    	pstrain2.zz = pstrain1.zz +  dLambda * dfds.zz;
        pstrain2.xy = pstrain1.xy +  dLambda * dfds.xy;
        pstrain2.yz = pstrain1.yz +  dLambda * dfds.yz;
        pstrain2.xz = pstrain1.xz +  dLambda * dfds.xz;
    }

    return pstrain2;
}*/

int get_displacements(mysolver_t *solver, elem_t *elemp, fvector_t *u) {

    int i;
    int res = 0;

    /* Capture displacements for each node */
    for (i = 0; i < 8; i++) {

        int32_t    lnid;
        fvector_t *dis;

        lnid = elemp->lnid[i];
        dis  = solver->tm1 + lnid;

        res += vector_is_all_zero( dis );

        u[i].f[0] = dis->f[0];
        u[i].f[1] = dis->f[1];
        u[i].f[2] = dis->f[2];

    }

    return res;
}

/* -------------------------------------------------------------------------- */
/*                              Stability methods                             */
/* -------------------------------------------------------------------------- */

void check_yield_limit(mesh_t *myMesh, int32_t eindex, double vs, double fs,
                       double k, int qp)
{
    if ( fs > 1.5 * k ) {

        if ( superflag > 0 ) {

            double  north_m, east_m, depth_m;
            int32_t lnid0;

            lnid0 = myMesh->elemTable[eindex].lnid[0];

            north_m = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].x;
            east_m  = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].y;
            depth_m = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].z;

            fprintf(stderr, "\n\n\tcompute_nonlinear_entities:"
                    "\n\tAn element exceeded the yield surface."
                    "\n\tThe element origin is at:\n"
                    "\n\tnorth (m) = %f"
                    "\n\teast  (m) = %f"
                    "\n\tdepth (m) = %f"
                    "\n\tVs  (m/s) = %f"
                    "\n\tFs        = %f"
                    "\n\tk         = %f"
                    "\n\tqp        = %d\n"
                    "\n\tA smaller dt or coarser mesh is required.\n",
                    north_m, east_m, depth_m, vs, fs, k, qp);

            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);

        } else {

            superflag++;
        }
    }

    return;
}

void check_strain_stability ( double dLambda, double dt, mesh_t *myMesh,
                              int32_t eindex, double vs, double fs,
                              double k, int qp)
{
    double STRAINRATELIMIT = 0.002;

    if ( dLambda > STRAINRATELIMIT / dt ) {

        if ( superflag > 0 ) {

            double  north_m, east_m, depth_m;
            int32_t lnid0;

            lnid0 = myMesh->elemTable[eindex].lnid[0];

            north_m = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].x;
            east_m  = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].y;
            depth_m = (myMesh->ticksize)*(double)myMesh->nodeTable[lnid0].z;

            fprintf(stderr, "\n\n\tcompute_nonlinear_entities:"
                    "\n\tAn element violated dlambda condition"
                    "\n\tThe element origin is at:\n"
                    "\n\tnorth (m) = %f"
                    "\n\teast  (m) = %f"
                    "\n\tdepth (m) = %f"
                    "\n\tVs  (m/s) = %f"
                    "\n\tFs        = %f"
                    "\n\tk         = %f"
                    "\n\tqp        = %d\n"
                    "\n\tdLambda   = % 8e\n",
                    north_m, east_m, depth_m, vs, fs, k, qp, dLambda);

            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);

        } else {

            superflag++;
        }
    }

    return;
}


/* -------------------------------------------------------------------------- */
/*                   Nonlinear core computational methods                     */
/* -------------------------------------------------------------------------- */
void BOX85_l(double ep_bar_n,vect1_t sigma_ppal_trial,double Phi, double Psi, double H, double c0, double K, double G, vect1_t *sigma_ppal, double *ep_bar_n1) {

/* Return mapping algorithm copied from:
   EA de Souza Neto, D Peric, D.O (2008). Computational methods for plasticity. Wiley */

	double a, dGamma;

	sigma_ppal->x = 0.0;
	sigma_ppal->y = 0.0;
	sigma_ppal->z = 0.0;

	a = ( 4. * G * ( 1. + 1./3. * sin(Psi) * sin(Phi) ) + 4. * K * sin(Phi) * sin(Psi) );

	dGamma = ( sigma_ppal_trial.x - sigma_ppal_trial.z + ( sigma_ppal_trial.x + sigma_ppal_trial.z ) * sin(Phi) - 2. * ( c0 + H * ep_bar_n ) * cos(Phi) ) /
			 ( 4. * H * cos(Phi) * cos(Phi) +  a );

	sigma_ppal->x = sigma_ppal_trial.x - ( 2. * G * ( 1. + 1./3. * sin(Psi) ) + 2. * K * sin(Psi) ) * dGamma;

	sigma_ppal->y = sigma_ppal_trial.y + ( 4./3. * G - 2. * K ) * sin(Psi) * dGamma;

	sigma_ppal->z = sigma_ppal_trial.z + ( 2. * G * ( 1. - 1./3. * sin(Psi) ) - 2. * K * sin(Psi) ) * dGamma;

	*ep_bar_n1 = ep_bar_n + 2.0 * cos(Phi) * dGamma;

}

void BOX86_l(double ep_bar_n,vect1_t sigma_ppal_trial,double Phi, double Psi, double H, double c0, double K, double G, double id, vect1_t *sigma_ppal, double *ep_bar_n1) {

	/* Return mapping algorithm copied from:
	   EA de Souza Neto, D Peric, D.O (2008). Computational methods for plasticity. Wiley */

	double aux, phi_a_bar, phi_b_bar, a, b, a11, b11, dGamma[2]={0}, sum_dGamma;

	sigma_ppal->x = 0.0;
	sigma_ppal->y = 0.0;
	sigma_ppal->z = 0.0;

    aux = 2. * cos(Phi) * ( c0 + H * ep_bar_n );

    phi_a_bar = sigma_ppal_trial.x - sigma_ppal_trial.z + ( sigma_ppal_trial.x + sigma_ppal_trial.z ) * sin(Phi) - aux;

    a = 4. * G * ( 1. + 1. / 3. * sin(Phi) * sin(Psi) ) + 4. * K * sin(Phi) * sin(Psi);

    if ( id == 1 ) {
        b = 2. * G * ( 1. + sin(Phi) + sin(Psi) - (1./3.) * sin(Phi) * sin(Psi) ) + 4. * K * sin(Phi) * sin(Psi);
        phi_b_bar = sigma_ppal_trial.x - sigma_ppal_trial.y + ( sigma_ppal_trial.x + sigma_ppal_trial.y ) * sin(Phi) - aux;
    } else {
        b = 2. * G * ( 1. - sin(Phi) - sin(Psi) - (1./3.) * sin(Phi) * sin(Psi) ) + 4. * K * sin(Phi) * sin(Psi);
        phi_b_bar = sigma_ppal_trial.y - sigma_ppal_trial.z + ( sigma_ppal_trial.y + sigma_ppal_trial.z ) * sin(Phi) - aux;
    }

    a11 = ( a + 4. * H * ( cos(Phi) * cos(Phi) ) );
    b11 = ( b + 4. * H * ( cos(Phi) * cos(Phi) ) );

    dGamma[0] = 1. / ( a11 * a11 - b11 * b11 ) * ( a11 * phi_a_bar - b11 * phi_b_bar );
    dGamma[1] = 1. / ( a11 * a11 - b11 * b11 ) * ( a11 * phi_b_bar - b11 * phi_a_bar );
    sum_dGamma    = dGamma[0] + dGamma[1];

   *ep_bar_n1 = ep_bar_n + 2. * cos(Phi) * sum_dGamma;

   if ( id == 1 ) {
    sigma_ppal->x = sigma_ppal_trial.x - ( 2. * G * ( 1. + (1./3.) * sin(Psi) ) + 2. * K * sin(Psi) ) * (sum_dGamma);
    sigma_ppal->y = sigma_ppal_trial.y + ( 4. * G / 3. - 2. * K ) * sin(Psi) * dGamma[0] + ( 2. * G * ( 1. - (1./3.) * sin(Psi) ) - 2.* K * sin(Psi) ) * dGamma[1];
    sigma_ppal->z = sigma_ppal_trial.z + ( 2. * G * ( 1. - (1./3.) * sin(Psi) ) - 2. * K * sin(Psi) ) * dGamma[0] + ( ( 4. * G / 3. ) - 2. * K ) * sin(Psi) * dGamma[1];
   } else {
    sigma_ppal->x = sigma_ppal_trial.x - ( 2. * G * ( 1. + (1./3.) * sin(Psi) ) + 2. * K * sin(Psi) ) * dGamma[0] + ( ( 4. * G / 3.) - 2. * K ) * sin(Psi) * dGamma[1];
    sigma_ppal->y = sigma_ppal_trial.y + ( 4. * G / 3. - 2. * K ) * sin(Psi) * dGamma[0] - ( 2. * G * ( 1. + (1./3.) * sin(Psi) ) + 2. * K * sin(Psi) ) * dGamma[1];
    sigma_ppal->z = sigma_ppal_trial.z + ( 2. * G * ( 1.- (1./3.) * sin(Psi) ) - 2. * K * sin(Psi) ) * sum_dGamma;
   }

}

void BOX87_l(double ep_bar_n,double p_trial,double Phi, double Psi, double H, double c0, double K, vect1_t *sigma_ppal, double *ep_bar_n1) {

	/* Return mapping algorithm copied from:
	   EA de Souza Neto, D Peric, D.O (2008). Computational methods for plasticity. Wiley */
	double omega, dep_bar, cot_phi, p;

	sigma_ppal->x = 0.0;
	sigma_ppal->y = 0.0;
	sigma_ppal->z = 0.0;

	omega = sin(Psi)/cos(Phi);
	cot_phi = cos(Phi) / sin (Phi);

	dep_bar = ( p_trial - ( c0 + H * ep_bar_n ) * cot_phi ) / ( H * cot_phi + omega * K); /*Here we deviate from the original formulation in
	                                                                                        order be able to use zero dilatancy  */
	*ep_bar_n1 = ep_bar_n + dep_bar;
	p = p_trial - K * omega * dep_bar;

	sigma_ppal->x = p;
	sigma_ppal->y = p;
	sigma_ppal->z = p;

}

tensor_t specRecomp(vect1_t eig_val, vect1_t n1, vect1_t n2, vect1_t n3) {
	tensor_t stress = zero_tensor();

	/* from first eigen_vector */
	stress.xx += eig_val.x * ( n1.x * n1.x);
	stress.yy += eig_val.x * ( n1.y * n1.y);
	stress.zz += eig_val.x * ( n1.z * n1.z);
	stress.xy += eig_val.x * ( n1.x * n1.y);
	stress.xz += eig_val.x * ( n1.x * n1.z);
	stress.yz += eig_val.x * ( n1.y * n1.z);

	/* from 2nd eigen_vector */
	stress.xx += eig_val.y * ( n2.x * n2.x);
	stress.yy += eig_val.y * ( n2.y * n2.y);
	stress.zz += eig_val.y * ( n2.z * n2.z);
	stress.xy += eig_val.y * ( n2.x * n2.y);
	stress.xz += eig_val.y * ( n2.x * n2.z);
	stress.yz += eig_val.y * ( n2.y * n2.z);

	/* from 3rd eigen_vector */
	stress.xx += eig_val.z * ( n3.x * n3.x);
	stress.yy += eig_val.z * ( n3.y * n3.y);
	stress.zz += eig_val.z * ( n3.z * n3.z);
	stress.xy += eig_val.z * ( n3.x * n3.y);
	stress.xz += eig_val.z * ( n3.x * n3.z);
	stress.yz += eig_val.z * ( n3.y * n3.z);

	return stress;

}



/*      Computes the spectral decomposition of a 3x3 symmetric matrix         */
/* -------------------------------------------------------------------------- */
/* Eigen decomposition code for symmetric 3x3 matrices, copied from the public
 domain Java Matrix library JAMA. */
// #define MAX(a, b) ((a)>(b)?(a):(b))

double hypot2(double x, double y) {
    return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

void tred2(double V[3][3], double *d, double *e) {

    int n=3, j, i, k;
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.

    for (j = 0; j < n; j++) {
        d[j] = V[n-1][j];
    }

    // Householder reduction to tridiagonal form.

    for (i = n-1; i > 0; i--) {

        // Scale to avoid under/overflow.

        double scale = 0.0;
        double h = 0.0;
        for (k = 0; k < i; k++) {
            scale = scale + fabs(d[k]);
        }
        if (scale == 0.0) {
            e[i] = d[i-1];
            for ( j = 0; j < i; j++) {
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
                V[j][i] = 0.0;
            }
        } else {

            // Generate Householder vector.

            for (k = 0; k < i; k++) {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            double f = d[i-1];
            double g = sqrt(h);
            if (f > 0) {
                g = -g;
            }
            e[i] = scale * g;
            h = h - f * g;
            d[i-1] = f - g;
            for ( j = 0; j < i; j++) {
                e[j] = 0.0;
            }

            // Apply similarity transformation to remaining columns.

            for ( j = 0; j < i; j++) {
                f = d[j];
                V[j][i] = f;
                g = e[j] + V[j][j] * f;
                for ( k = j+1; k <= i-1; k++) {
                    g += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for ( j = 0; j < i; j++) {
                e[j] /= h;
                f += e[j] * d[j];
            }
            double hh = f / (h + h);
            for ( j = 0; j < i; j++) {
                e[j] -= hh * d[j];
            }
            for ( j = 0; j < i; j++) {
                f = d[j];
                g = e[j];
                for ( k = j; k <= i-1; k++) {
                    V[k][j] -= (f * e[k] + g * d[k]);
                }
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
            }
        }
        d[i] = h;
    }

    // Accumulate transformations.

    for ( i = 0; i < n-1; i++) {
        V[n-1][i] = V[i][i];
        V[i][i] = 1.0;
        double h = d[i+1];
        if (h != 0.0) {
            for ( k = 0; k <= i; k++) {
                d[k] = V[k][i+1] / h;
            }
            for ( j = 0; j <= i; j++) {
                double g = 0.0;
                for ( k = 0; k <= i; k++) {
                    g += V[k][i+1] * V[k][j];
                }
                for ( k = 0; k <= i; k++) {
                    V[k][j] -= g * d[k];
                }
            }
        }
        for ( k = 0; k <= i; k++) {
            V[k][i+1] = 0.0;
        }
    }
    for ( j = 0; j < n; j++) {
        d[j] = V[n-1][j];
        V[n-1][j] = 0.0;
    }
    V[n-1][n-1] = 1.0;
    e[0] = 0.0;

}



void tql2(double V[3][3], double* d, double* e) {

    //  This is derived from the Algol procedures tql2, by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.
    int i, l, k, j,  n=3;


    for ( i = 1; i < n; i++) {
        e[i-1] = e[i];
    }
    e[n-1] = 0.0;

    double f = 0.0;
    double tst1 = 0.0;
    double eps = pow(2.0,-52.0);
    for ( l = 0; l < n; l++) {

        // Find small subdiagonal element

        tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
        int m = l;
        while (m < n) {
            if (fabs(e[m]) <= eps*tst1) {
                break;
            }
            m++;
        }

        // If m == l, d[l] is an eigenvalue,
        // otherwise, iterate.

        if (m > l) {
            int iter = 0;
            do {
                iter = iter + 1;  // (Could check iteration count here.)

                // Compute implicit shift

                double g = d[l];
                double p = (d[l+1] - g) / (2.0 * e[l]);
                double r = hypot2(p,1.0);
                if (p < 0) {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l+1] = e[l] * (p + r);
                double dl1 = d[l+1];
                double h = g - d[l];
                for ( i = l+2; i < n; i++) {
                    d[i] -= h;
                }
                f = f + h;

                // Implicit QL transformation.

                p = d[m];
                double c = 1.0;
                double c2 = c;
                double c3 = c;
                double el1 = e[l+1];
                double s = 0.0;
                double s2 = 0.0;
                for ( i = m-1; i >= l; i--) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot2(p,e[i]);
                    e[i+1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i+1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation.

                    for ( k = 0; k < n; k++) {
                        h = V[k][i+1];
                        V[k][i+1] = s * V[k][i] + c * h;
                        V[k][i] = c * V[k][i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence.

            } while (fabs(e[l]) > eps*tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0.0;
    }

    // Sort eigenvalues and corresponding vectors.

    for ( i = 0; i < n-1; i++) {
         k = i;
        double p = d[i];
        for ( j = i+1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for ( j = 0; j < n; j++) {
                p = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = p;
            }
        }
    }
}


void  specDecomp(tensor_t sigma, vect1_t *n1, vect1_t *n2, vect1_t *n3, vect1_t *eig_values){
  double V[3][3], d[3], e[3];

    V[0][0] = sigma.xx;
    V[0][1] = sigma.xy;
    V[0][2] = sigma.xz;
    V[1][0] = sigma.xy;
    V[1][1] = sigma.yy;
    V[1][2] = sigma.yz;
    V[2][0] = sigma.xz;
    V[2][1] = sigma.yz;
    V[2][2] = sigma.zz;

    tred2(V, d, e);
    tql2(V, d, e);

    eig_values->x = d[2];
    eig_values->y = d[1];
    eig_values->z = d[0];

    n3->x = V[0][0];
    n3->y = V[1][0];
    n3->z = V[2][0];

    n2->x = V[0][1];
    n2->y = V[1][1];
    n2->z = V[2][1];

    n1->x = V[0][2];
    n1->y = V[1][2];
    n1->z = V[2][2];

}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */


double smooth_rise_factor(int32_t step, double dt) {

    /* TODO: Consider to eliminate t1, i.e. t1 = 0 */

    static noyesflag_t preloaded = NO;
    static double n1, n2, n3, n22, n31;
    static double C1, C2, B1, B2;
    static int    N;

    /* Pre-load constants, executed only once */
    if ( preloaded == NO ) {

        N = (int)(theGeostaticLoadingT / dt);

        n1 = (int)( 0.1 * N );
        n2 = (int)( 0.5 * N );
        n3 = (int)( 0.9 * N );

        n31 = n3 - n1;

        C1 = 2.0 / ( n31 * (n2 - n1) ) ;
        C2 = 2.0 / ( n31 * (n2 - n3) ) ;

        B1 = 0.5 * n1 * n1;
        B2 = 0.5 * ( n31*(n2-n3) + n3*n3 );

        preloaded = YES;
    }

    /* Started dynamic simulation (the most common case) */
    if ( step > n3 ) {
        return 1.0;
    }

    /* Has not even really started (a zeros-buffer zone) */
    if ( step <= n1 ) {
        return 0.0;
    }

    /* The actual geostatic loading cases */

    n22 = 0.5 * step * step;

    if ( (step > n1) && (step <= n2) ) {

        return C1 * (n22 - step*n1 + B1);

    } else if ( (step > n2) && (step <= n3) ) {

        return C2 * (n22 - step*n3 + B2);
    }

    /* The code should never get here */
    fprintf(stderr, "Smooth Rise Error %d %d\n", N, step);
    MPI_Abort(MPI_COMM_WORLD, ERROR);
    exit(1);
}

void add_force_reactions ( mesh_t     *myMesh,
                           mysolver_t *mySolver )
{

    int       i;
    int32_t   beindex, eindex;

    /* Loop on the number of elements */
    for (beindex = 0; beindex < myBottomElementsCount; beindex++) {

        elem_t    *elemp;

        eindex = myBottomElements[beindex].element_id;
        elemp  = &myMesh->elemTable[eindex];

        for (i = 4; i < 8; i++) {

            int32_t    lnid;
            fvector_t *nodalForce;

            lnid = elemp->lnid[i];
            nodalForce = mySolver->force + lnid;

            nodalForce->f[2] += myBottomElements[beindex].nodal_force[i-4];

        } /* element nodes */
    }

    return;
}

void check_balance( int32_t myID ) {

    double totalReaction = 0;
    int i;
    int32_t   beindex;

    for (beindex = 0; beindex < myBottomElementsCount; beindex++) {
        for ( i = 0; i < 4; i++ ) {
            totalReaction += myBottomElements[beindex].nodal_force[i];
        }
    }

    double theTotalWeight;
    double theTotalReaction;

    MPI_Reduce( &totalWeight, &theTotalWeight, 1,
                MPI_DOUBLE, MPI_SUM, 0, comm_solver );
    MPI_Reduce( &totalReaction, &theTotalReaction, 1,
                MPI_DOUBLE, MPI_SUM, 0, comm_solver );

    if ( myID == 0 ) {
        fprintf( stdout,
                 "\tTotal Weight   = %20.6f\n"
                 "\tTotal Reaction = %20.6f\n"
                 "\tDifference     = %20.6f\n\n",
                 theTotalWeight, theTotalReaction,
                 theTotalWeight+theTotalReaction );
    }

    return;
}

void compute_addforce_gravity( mesh_t     *myMesh,
                               mysolver_t *mySolver,
                               int         step,
                               double      dt )
{
    /*
     * Some gravity values:
     *
     * Los Angeles    9.796 m/s2      Athens    9.800 m/s2
     * Mexico City    9.779 m/s2      Auckland  9.799 m/s2
     * San Francisco  9.800 m/s2      Rome      9.803 m/s2
     * Istanbul       9.808 m/s2      Tokyo     9.798 m/s2
     * Vancouver      9.809 m/s2      Ottawa    9.806 m/s2
     *
     */

    #define G 9.8

    int32_t   eindex;

    /* Loop on the number of elements */
    for (eindex = 0; eindex < myMesh->lenum; eindex++) {

        int      i;
        elem_t  *elemp;
        edata_t *edata;
        double   h, h3;
        double   rho, W;

        /* Capture element data structure */
        elemp = &myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;

        /* capture element data */
        h   = (double)edata->edgesize;
        rho = (double)edata->rho;

        /* Compute nodal total weight contribution */
        h3 = h * h * h;
        W  = h3 * rho * G * 0.125; /* volume x density x gravity / 8 */

        if ( step == theGeostaticFinalStep+10 ) {
            totalWeight += W*dt*dt*8;
        }

        /* Loop over the 8 element nodes:
         * Add the gravitational force contribution calculated with respect
         * to the current time-step to the nodal force vector.
         */
        for (i = 0; i < 8; i++) {

            int32_t    lnid;
            fvector_t *nodalForce;

            lnid       = elemp->lnid[i];
            nodalForce = mySolver->force + lnid;

            /* Force due to gravity is positive in Z-axis */
            nodalForce->f[2] += W * smooth_rise_factor(step, dt) * dt * dt;

        } /* element nodes */

    }

    if ( step > theGeostaticFinalStep ) {
        add_force_reactions(myMesh, mySolver);
    }

    return;
}

void compute_bottom_reactions ( mesh_t     *myMesh,
                                mysolver_t *mySolver,
                                fmatrix_t (*theK1)[8],
                                fmatrix_t (*theK2)[8],
                                int         step,
                                double      dt )
{
    if ( step != theGeostaticFinalStep ) {
        return;
    }

    int32_t   beindex;
    int32_t   eindex;
    fvector_t localForce[8];

    for ( beindex = 0; beindex < myBottomElementsCount; beindex++ ) {

        int        i, j;
        elem_t    *elemp;
        e_t*       ep;

        eindex = myBottomElements[beindex].element_id;
        elemp  = &myMesh->elemTable[eindex];
        ep     = &mySolver->eTable[eindex];

        /* -------------------------------
         * Ku DONE IN THE CONVENTIONAL WAY
         * ------------------------------- */

        /* step 1: calculate the force due to the element stiffness */
        memset( localForce, 0, 8 * sizeof(fvector_t) );

        /* contribution by node j to node i (only for nodes at the bottom) */
        for (i = 4; i < 8; i++) {

            fvector_t* toForce = &localForce[i];

            for (j = 0; j < 8; j++) {

                int32_t    nodeJ  = elemp->lnid[j];
                fvector_t *myDisp = mySolver->tm1 + nodeJ;

                MultAddMatVec( &theK1[i][j], myDisp, ep->c1, toForce );
                MultAddMatVec( &theK2[i][j], myDisp, ep->c2, toForce );
            }
        }

        /* step 2: store the forces */
        for (i = 4; i < 8; i++) {
            myBottomElements[beindex].nodal_force[i-4] = localForce[i].f[2];
        }

        edata_t *edata;
        double   h, h3;
        double   rho, W;

        edata = (edata_t *)elemp->data;
        h     = (double)edata->edgesize;
        rho   = (double)edata->rho;
        h3    = h * h * h;
        W     = h3 * rho * G * 0.125;

        for (i = 0; i < 4; i++) {
            myBottomElements[beindex].nodal_force[i] -= W * dt * dt;
        }
    }

    return;
}

void geostatic_displacements_fix( mesh_t     *myMesh,
                                  mysolver_t *mySolver,
                                  double      totalDomainDepth,
                                  double      dt,
                                  int         step )
{

    if ( step > theGeostaticFinalStep ) {
        return;
    }

    int32_t nindex;

    for ( nindex = 0; nindex < myMesh->nharbored; nindex++ ) {

        double z_m = (myMesh->ticksize)*(double)myMesh->nodeTable[nindex].z;

        if ( z_m == totalDomainDepth ) {
            fvector_t *tm2Disp;
            tm2Disp = mySolver->tm2 + nindex;
            tm2Disp->f[2] = 0;
        }
    }

    return;
}

/*
 * compute_addforce_nl: Adds a fictitious force that accounts for
 *                      nonlinearity in the soil --- a correction
 *                      to the 'trial' stress within K-elastic
 *
 * The computation done here corresponds to calculate the last expression
 * in the manuscript prepared by Ricardo (part I) for the implementation of
 * nonlinear soil into the program.
 *
 *      Integral(grad(Phi) * Cijkl * PlasticStrain) * delta_t^2
 */
void compute_addforce_nl (mesh_t     *myMesh,
                          mysolver_t *mySolver,
                          double      theDeltaTSquared)
{

    int       i, j;
    int32_t   eindex;
    int32_t   nl_eindex;
    fvector_t localForce[8];

    /* Loop on the number of elements */
    for (nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++) {

        elem_t  *elemp;
        edata_t *edata;
        double   h, h3, WiJi;
        double   mu, lambda;

        eindex = myNonlinElementsMapping[nl_eindex];

        qptensors_t stresses;

        /* Capture the table of elements from the mesh and the size
         * This is what gives me the connectivity to nodes */
        elemp = &myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;

        h    = (double)edata->edgesize;
        h3   = h * h * h;
        WiJi = h3 * 0.125; /* (h^3)/8 */

        nlconstants_t ec = myNonlinSolver->constants[nl_eindex];

        mu     = ec.mu;
        lambda = ec.lambda;

//        int kk=0;
//        if ( (nl_eindex==462) )
//        	kk=89;

//        if ( theMaterialModel == LINEAR ) {

            stresses = myNonlinSolver->stresses[nl_eindex];

//        } else {
//
//            qptensors_t tstrains, pstrains, estrains;
//
//            tstrains = myNonlinSolver->strains   [nl_eindex];
//            pstrains = myNonlinSolver->pstrains1[nl_eindex];
//
//            /* compute total strain - plastic strain */
//            estrains = subtrac_qptensors(tstrains, pstrains);
//
//            /* compute the corresponding stress */
//            stresses = compute_qp_stresses(estrains, mu, lambda);
//        }

        /* Clean memory for the local force vector */
        memset(localForce, 0, 8 * sizeof(fvector_t));

        /* Loop over the 8 element nodes:
         * Calculates the forces on each vertex */
        for (i = 0; i < 8; i++) {

            fvector_t *toForce;

            /* Points the loop force to the element force */
            toForce = &localForce[i];

            /* Loop over the 8 quadrature points */
            for (j = 0; j < 8; j++) {

                double dx, dy, dz;

                compute_qp_dxi (&dx, &dy, &dz, i, j, h);

                /* Gauss integration: Sum(DeltaPsi * Sigma * wi * Ji) */

                toForce->f[0] += ( ( dx * stresses.qp[j].xx )
                                 + ( dy * stresses.qp[j].xy )
                                 + ( dz * stresses.qp[j].xz ) ) * WiJi;

                toForce->f[1] += ( ( dy * stresses.qp[j].yy )
                                 + ( dx * stresses.qp[j].xy )
                                 + ( dz * stresses.qp[j].yz ) ) * WiJi;

                toForce->f[2] += ( ( dz * stresses.qp[j].zz )
                                 + ( dy * stresses.qp[j].yz )
                                 + ( dx * stresses.qp[j].xz ) ) * WiJi;

            } /* quadrature points */

        } /* element nodes */

        /* Loop over the 8 element nodes:
         * Add the contribution calculated above to the node
         * forces carried from the source and stiffness.
         */
        for (i = 0; i < 8; i++) {

            int32_t    lnid;
            fvector_t *nodalForce;

            lnid = elemp->lnid[i];

            nodalForce = mySolver->force + lnid;

            nodalForce->f[0] -= localForce[i].f[0] * theDeltaTSquared;
            nodalForce->f[1] -= localForce[i].f[1] * theDeltaTSquared;
            nodalForce->f[2] -= localForce[i].f[2] * theDeltaTSquared;

        	if (  ( ( nodalForce->f[0] >=0 ) || ( nodalForce->f[0] < 0 ) ) &&
        		  ( ( nodalForce->f[1] >=0 ) || ( nodalForce->f[1] < 0 ) ) &&
        		  ( ( nodalForce->f[2] >=0 ) || ( nodalForce->f[2] < 0 ) ) ) {
        	} else {

        	}


        } /* element nodes */

    } /* all elements */

    return;
}

/*
 * compute_nonlinear_state: Compute the necessary quantities to determine
 *                             if an element has undergone plastic deformation.
 *
 * The steps described in this method are in reference to the manuscript
 * prepared by Ricardo about the implementation of nonlinear soil in the
 * program (Part III).
 *
 * The convention for the strain and stress tensors is:
 *
 *      T[6] = {Txx, Tyy, Tzz, Txy, Tyz, Txz}
 */
void compute_nonlinear_state ( mesh_t     *myMesh,
                               mysolver_t *mySolver,
                               int32_t     theNumberOfStations,
                               int32_t     myNumberOfStations,
                               station_t  *myStations,
                               double      theDeltaT,
                               int         step )
{
	/* In general, j-index refers to the quadrature point in a loop (0 to 7 for
	 * eight points), and i-index refers to the tensor component (0 to 5), with
	 * the following order xx[0], yy[1], zz[2], xy[3], yz[4], xz[5]. i-index is
	 * also some times used for the number of nodes (8, 0 to 7).
	 */

	int     i;
	int32_t eindex, nl_eindex;


	/* Loop over the number of local elements */
	for (nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++) {

		elem_t        *elemp;
		edata_t       *edata;
		nlconstants_t *enlcons;

		double         h;          /* Element edge-size in meters   */
		double         alpha, k;   /* Drucker-Prager constants      */
		double         mu, lambda; /* Elasticity material constants */
		double		   hrd;        /* Hardening Modulus  */
		double         beta;       /* Plastic flow rule constant */
		double         XI, QC;
		fvector_t      u[8];
		qptensors_t   *stresses, *tstrains, *pstrains1, *pstrains2;
		qpvectors_t   *epstr1, *epstr2;

		/* Capture data from the element and mesh */

		eindex = myNonlinElementsMapping[nl_eindex];

		elemp = &myMesh->elemTable[eindex];
		edata = (edata_t *)elemp->data;
		h     = edata->edgesize;

		/* Capture data from the nonlinear element structure */

		enlcons = myNonlinSolver->constants + nl_eindex;

		mu     = enlcons->mu;
		lambda = enlcons->lambda;
		alpha  = enlcons->alpha;
		beta   = enlcons->beta;
		k      = enlcons->k;
		hrd    = enlcons->h;


		/* Capture the current state in the element */
		tstrains  = myNonlinSolver->strains   + nl_eindex;
		stresses  = myNonlinSolver->stresses  + nl_eindex;
		pstrains1 = myNonlinSolver->pstrains1 + nl_eindex;
		pstrains2 = myNonlinSolver->pstrains2 + nl_eindex;
		epstr1    = myNonlinSolver->ep1       + nl_eindex;
		epstr2    = myNonlinSolver->ep2       + nl_eindex;

		/* Capture displacements */
		if ( get_displacements(mySolver, elemp, u) == 0 ) {
			/* If all displacements are zero go for next element */
			continue;
		}

		/* Loop over the quadrature points */
		for (i = 0; i < 8; i++) {

			tensor_t  sigma0;

			/* Quadrature point local coordinates */
			double lx = xi[0][i] * qc ;
			double ly = xi[1][i] * qc ;
			double lz = xi[2][i] * qc ;

			/* Calculate total strains */
			tstrains->qp[i] = point_strain(u, lx, ly, lz, h);

			/* Calculate stresses */
			if ( ( theMaterialModel == LINEAR ) || ( step <= theGeostaticFinalStep ) ){
				stresses->qp[i]  = point_stress ( tstrains->qp[i], mu, lambda );
				continue;
			} else {

				if ( theApproxGeoState == YES )
					sigma0 = ApproxGravity_tensor(enlcons->sigmaZ_st, enlcons->phi, h, lz, edata->rho);
				else
					sigma0 = zero_tensor();

				material_update ( *enlcons,  tstrains->qp[i], pstrains1->qp[i], epstr1->qv[i], sigma0, theDeltaT,
						&pstrains2->qp[i], &stresses->qp[i], &epstr2->qv[i], &enlcons->fs[i]);
			}
		} /* for all quadrature points */
	} /* for all nonlinear elements */
}

/* -------------------------------------------------------------------------- */
/*                        Nonlinear Finalize and Stats                        */
/* -------------------------------------------------------------------------- */

void nonlinear_yield_stats(mesh_t *myMesh, int32_t myID, int32_t theTotalSteps, int32_t theGroupSize) {

    static double VSMIN = 0;
    static double VSMAX = 10000;

    int32_t  nl_eindex, eindex;
    int      r;
    int      ranges = thePropertiesCount+1;
    double  *myFsMaxs;
    double  *myFsAvgs;
    double   vs, vs0, vs1;
    int32_t *myFsAvgCount;

    myFsMaxs     =  (double *)calloc(ranges, sizeof(double));
    myFsAvgs     =  (double *)calloc(ranges, sizeof(double));
    myFsAvgCount = (int32_t *)calloc(ranges, sizeof(int32_t));

    for ( nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++ ) {

        elem_t        *elemp;
        edata_t       *edata;
        nlconstants_t  ec;

        eindex = myNonlinElementsMapping[nl_eindex];
        elemp  = &myMesh->elemTable[eindex];
        edata  = (edata_t *)elemp->data;
        ec     = myNonlinSolver->constants[nl_eindex];
        vs     = edata->Vs;

        for ( r = 0; r < ranges; r++ ) {

            /* set bottom vs limit */
            if ( r == 0 ) {
                vs0 = VSMIN;
            } else {
                vs0 = theVsLimits[r-1];
            }

            /* set top vs limit */
            if ( r == ranges-1 ) {
                vs1 = VSMAX;
            } else {
                vs1 = theVsLimits[r];
            }

            if ( (vs > vs0) && (vs <= vs1) ) {
                myFsAvgs[r] += ec.avgFs;
                myFsAvgCount[r]++;
                if ( ec.maxFs > myFsMaxs[r] ) {
                    myFsMaxs[r] = ec.maxFs;
                }
            }
        }
    }

    if ( myNonlinElementsCount > 0 ) {
        for ( r = 0; r < ranges; r++ ) {
            myFsAvgs[r] /= theTotalSteps;
        }
    }


    double  *theFsMaxs     = NULL;
    double  *theFsAvgs     = NULL;
    int32_t *theFsAvgCount = NULL;

    theFsMaxs     =  (double *)calloc(ranges, sizeof(double));
    theFsAvgs     =  (double *)calloc(ranges, sizeof(double));
    theFsAvgCount = (int32_t *)calloc(ranges, sizeof(int32_t));

    MPI_Reduce(myFsMaxs, theFsMaxs, ranges, MPI_DOUBLE, MPI_MAX, 0, comm_solver );
    MPI_Reduce(myFsAvgs, theFsAvgs, ranges, MPI_DOUBLE, MPI_SUM, 0, comm_solver );
    MPI_Reduce(myFsAvgCount, theFsAvgCount, ranges, MPI_INT, MPI_SUM, 0, comm_solver );

    if ( myID == 0 ) {

        for ( r = 0; r < ranges; r++ ) {
            if ( theFsAvgCount[r] > 0 ) {
                theFsAvgs[r] /= theFsAvgCount[r];
            }
        }

        FILE *fp = hu_fopen( "stat-fs-yield.txt", "w" );

        fputs( "\n"
               "# ------------------------------------------- \n"
               "# Nonlinear Fs maximum and average values:    \n"
               "# ------------------------------------------- \n"
               "#   Vs >    Vs <=           Avg           Max \n"
               "# ------------------------------------------- \n", fp );

        for ( r = 0; r < ranges; r++ ) {

            /* set bottom vs limit */
            if ( r == 0 ) {
                vs0 = VSMIN;
            } else {
                vs0 = theVsLimits[r-1];
            }

            /* set top vs limit */
            if ( r == ranges-1 ) {
                vs1 = VSMAX;
            } else {
                vs1 = theVsLimits[r];
            }

            fprintf( fp, "%8.0f %8.0f % 10e % 10e\n",
                     vs0, vs1, theFsAvgs[r], theFsMaxs[r]);

        }

        fprintf( fp, "# ------------------------------------------- \n\n");

        hu_fclosep( &fp );

    }
}

/* -------------------------------------------------------------------------- */
/*                        Nonlinear Output to Stations                        */
/* -------------------------------------------------------------------------- */

void nonlinear_stations_init(mesh_t    *myMesh,
                             station_t *myStations,
                             int32_t    myNumberOfStations)
{

    if ( myNumberOfStations == 0 ) {
        return;
    }

    int32_t     eindex, nl_eindex;
    int32_t     iStation=0;
    vector3D_t  point;
    octant_t   *octant;
    int32_t     lnid0;

    myNumberOfNonlinStations = 0;
    for (iStation = 0; iStation < myNumberOfStations; iStation++) {

        for ( nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++ ) {

            /* capture the stations coordinates */
            point = myStations[iStation].coords;

            /* search the octant */
            if ( search_point(point, &octant) != 1 ) {
                fprintf(stderr,
                        "nonlinear_stations_init: "
                        "No octant with station coords\n");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }

            eindex = myNonlinElementsMapping[nl_eindex];

            lnid0 = myMesh->elemTable[eindex].lnid[0];

            if ( (myMesh->nodeTable[lnid0].x == octant->lx) &&
                 (myMesh->nodeTable[lnid0].y == octant->ly) &&
                 (myMesh->nodeTable[lnid0].z == octant->lz) ) {

                /* I have a match for the element's origin */

                /* Now, perform level sanity check */
                if (myMesh->elemTable[eindex].level != octant->level) {
                    fprintf(stderr,
                            "nonlinear_stations_init: First pass: "
                            "Wrong level of octant\n");
                    MPI_Abort(MPI_COMM_WORLD, ERROR);
                    exit(1);
                }

                myNumberOfNonlinStations++;

                break;
            }
        }
    }

    XMALLOC_VAR_N( myStationsElementIndices, int32_t, myNumberOfNonlinStations);
    XMALLOC_VAR_N( myNonlinStationsMapping, int32_t, myNumberOfNonlinStations);
 //   XMALLOC_VAR_N( myNonlinStations, nlstation_t, myNumberOfNonlinStations);

    int32_t nlStationsCount = 0;
    for (iStation = 0; iStation < myNumberOfStations; iStation++) {

        for ( nl_eindex = 0; nl_eindex < myNonlinElementsCount; nl_eindex++ ) {

            /* capture the stations coordinates */
            point = myStations[iStation].coords;

            /* search the octant */
            if ( search_point(point, &octant) != 1 ) {
                fprintf(stderr,
                        "nonlinear_stations_init: "
                        "No octant with station coords\n");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }

            eindex = myNonlinElementsMapping[nl_eindex];

            lnid0 = myMesh->elemTable[eindex].lnid[0];

            if ( (myMesh->nodeTable[lnid0].x == octant->lx) &&
                 (myMesh->nodeTable[lnid0].y == octant->ly) &&
                 (myMesh->nodeTable[lnid0].z == octant->lz) ) {

                /* I have a match for the element's origin */

                /* Now, perform level sanity check */
                if (myMesh->elemTable[eindex].level != octant->level) {
                    fprintf(stderr,
                            "nonlinear_stations_init: Second pass: "
                            "Wrong level of octant\n");
                    MPI_Abort(MPI_COMM_WORLD, ERROR);
                    exit(1);
                }

                if ( nlStationsCount >= myNumberOfNonlinStations ) {
                    fprintf(stderr,
                            "nonlinear_stations_init: Second pass: "
                            "More stations than initially counted\n");
                    MPI_Abort(MPI_COMM_WORLD, ERROR);
                    exit(1);
                }

                /* Store the element index and mapping to stations */
                myStationsElementIndices[nlStationsCount] = nl_eindex;
                myNonlinStationsMapping[nlStationsCount] = iStation;

                nlStationsCount++;

                break;
            }

        } /* for all my elements */

    } /* for all my stations */

/*    for ( iStation = 0; iStation < myNumberOfNonlinStations; iStation++ ) {

        tensor_t *stress, *strain, *pstrain1, *pstrain2;
        double   *ep1;

        strain   = &(myNonlinStations[iStation].strain);
        stress   = &(myNonlinStations[iStation].stress);
        pstrain1 = &(myNonlinStations[iStation].pstrain1);
        pstrain2 = &(myNonlinStations[iStation].pstrain2);
        ep1      = &(myNonlinStations[iStation].ep );
        *ep1     = 0.;

        init_tensorptr(strain);
        init_tensorptr(stress);
        init_tensorptr(pstrain1);
        init_tensorptr(pstrain2);

    }*/

}

void print_nonlinear_stations(mesh_t     *myMesh,
                              mysolver_t *mySolver,
                              station_t  *myStations,
                              int32_t     myNumberOfStations,
                              double      dt,
                              int         step,
                              int         rate)
{

    int32_t eindex;
    int32_t nl_eindex;
    int32_t iStation;
    int32_t mappingIndex;

    for ( iStation = 0; iStation < myNumberOfNonlinStations; iStation++ ) {
    	tensor_t       *stress, *tstrain, tstress;
    	qptensors_t    *stressF, *tstrainF;
    	double         bStrain = 0., bStress = 0., Fy, h;
    	tensor_t       sigma0;

    	elem_t         *elemp;
		edata_t        *edata;
    	nlconstants_t  *enlcons;

    	nl_eindex    = myStationsElementIndices[iStation];
    	eindex       = myNonlinElementsMapping[nl_eindex];
    	mappingIndex = myNonlinStationsMapping[iStation];
    	enlcons      = myNonlinSolver->constants + nl_eindex;

		elemp = &myMesh->elemTable[eindex];
		edata = (edata_t *)elemp->data;
		h     = edata->edgesize;

		/* compute the self-weight stresses at the first Gauss point*/
		double lz = -0.577350269189;
		if ( theApproxGeoState == YES )
			sigma0 = ApproxGravity_tensor(enlcons->sigmaZ_st, enlcons->phi, h, lz, edata->rho);
		else
			sigma0 = zero_tensor();

    	/* Capture data from the nonlinear element structure
    	 * corresponding to the first Gauss point*/
    	tstrainF   = myNonlinSolver->strains   + nl_eindex;
    	stressF    = myNonlinSolver->stresses  + nl_eindex;

    	stress      = &(stressF->qp[0]);            /* relative stresses of the first Gauss point */
    	tstress     = add_tensors(*stress,sigma0); /* compute the total stress tensor */

    	tstrain    = &(tstrainF->qp[0]);

    	Fy         = (myNonlinSolver->constants   + nl_eindex)->fs[0];

    	bStrain = tstrain->xx + tstrain->yy + tstrain->zz;
    	bStress = tstress.xx + tstress.yy + tstress.zz;

    	if (step % rate == 0) {
    		fprintf( myStations[mappingIndex].fpoutputfile,

    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e % 8e"
    				" % 8e",

    				tstrain->xx, tstress.xx, // 11 12
    				tstrain->yy, tstress.yy, // 13 14
    				tstrain->zz, tstress.zz, // 15 16
    				bStrain,     bStress,    // 17 18
    				tstrain->xy, tstress.xy, // 19 20
    				tstrain->yz, tstress.yz, // 21 22
    				tstrain->xz, tstress.xz,
    				Fy); // 23 24
    	}
    } /* for all my stations */

}
