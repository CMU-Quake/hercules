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

#define  XI  xi[3][8] = { {-1,  1, -1,  1, -1,  1, -1, 1} , \
                          {-1, -1,  1,  1, -1, -1,  1, 1} , \
                          {-1, -1, -1, -1,  1,  1,  1, 1} }

static int superflag = 0;

/* -------------------------------------------------------------------------- */
/*                             Global Variables                               */
/* -------------------------------------------------------------------------- */

static nlsolver_t           *myNonlinSolver;
static double                theNonLinVsCut = 0;
static double                theNonLinVsMin = 0;
static int32_t               thePropertiesCount;
static materialmodel_t       theMaterialModel;
static plasticitytype_t      thePlasticityModel;
static materialproperties_t  thePropertiesType;
static double               *theVsLimits;
static double               *theAlphaCohes;
static double               *theKayPhis;
static double               *theStrainRates;
static double               *theSensitivities;
static double               *theHardeningModulus;
static double                theGeostaticLoadingT = 0;
static double                theGeostaticCushionT = 0;
static int                   theGeostaticFinalStep;
static int32_t              *myStationsElementIndices;
static nlstation_t          *myNonlinStations;
static int32_t              *myNonlinStationsMapping;
static int32_t               myNumberOfNonlinStations;
static int32_t               myNonlinElementsCount;
static int32_t              *myNonlinElementsMapping;
static int32_t               myBottomElementsCount = 0;
static bottomelement_t      *myBottomElements;

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

    elemp = &myMesh->elemTable[eindex];
    edata = (edata_t *)elemp->data;

    if ( ( edata->Vs <= theNonLinVsCut ) && ( edata->Vs >= theNonLinVsMin ) )  {
        return YES;
    }

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
double get_alpha(double vs) {

    double alpha;
    double phi;

    switch ( thePropertiesType ) {

        case ALPHAKAY:
            alpha = interpolate_property_value(vs, theAlphaCohes);
            break;

        case COHEFRICTION:
            phi   = interpolate_property_value(vs, theKayPhis) * PI / 180.0;
            alpha = 2 * sin(phi) / ( sqrt(3.0) * ( 3 - sin(phi) ) );
            break;

        default:
            alpha = 0;
    }

    return alpha;
}

/*
 * Returns the value of the constant kay for in Drucker-Prager's material
 * model
 */
double get_kay(double vs) {

    double k;
    double c, phi;

    switch ( thePropertiesType ) {

        case ALPHAKAY:
            k     = interpolate_property_value(vs, theKayPhis);
            break;

        case COHEFRICTION:
            c     = interpolate_property_value(vs, theAlphaCohes);
            phi   = interpolate_property_value(vs, theKayPhis) * PI / 180.0;
            k     = 6 * c * cos(phi) / ( sqrt(3.0) * ( 3 - sin(phi) ) );
            break;

        default:
            k     = 0;
    }

    return k;
}

/* -------------------------------------------------------------------------- */
/*       Initialization of parameters, structures and memory allocations      */
/* -------------------------------------------------------------------------- */

void nonlinear_init( int32_t     myID,
                     const char *parametersin,
                     double      theDeltaT,
                     double      theEndT )
{
    double  double_message[4];
    int     int_message[5];

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

    double_message[0] = theNonLinVsCut;
    double_message[1] = theGeostaticLoadingT;
    double_message[2] = theGeostaticCushionT;
    double_message[3] = theNonLinVsMin;

    int_message[0] = (int)theMaterialModel;
    int_message[1] = (int)thePropertiesType;
    int_message[2] = thePropertiesCount;
    int_message[3] = theGeostaticFinalStep;
    int_message[4] = thePlasticityModel;


    MPI_Bcast(double_message, 4, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(int_message,    5, MPI_INT,    0, comm_solver);

    theNonLinVsCut        = double_message[0];
    theGeostaticLoadingT  = double_message[1];
    theGeostaticCushionT  = double_message[2];
    theNonLinVsMin        = double_message[3];

    theMaterialModel      = int_message[0];
    thePropertiesType     = int_message[1];
    thePropertiesCount    = int_message[2];
    theGeostaticFinalStep = int_message[3];
    thePlasticityModel    = int_message[4];

    /* allocate table of properties for all other PEs */

    if (myID != 0) {
        theVsLimits         = (double*)malloc(sizeof(double) * thePropertiesCount);
        theAlphaCohes       = (double*)malloc(sizeof(double) * thePropertiesCount);
        theKayPhis          = (double*)malloc(sizeof(double) * thePropertiesCount);
        theStrainRates      = (double*)malloc(sizeof(double) * thePropertiesCount);
        theSensitivities    = (double*)malloc(sizeof(double) * thePropertiesCount);
        theHardeningModulus = (double*)malloc(sizeof(double) * thePropertiesCount);
    }

    /* Broadcast table of properties */
    MPI_Bcast(theVsLimits,         thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theAlphaCohes,       thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theKayPhis,          thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theStrainRates,      thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theSensitivities,    thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(theHardeningModulus, thePropertiesCount, MPI_DOUBLE, 0, comm_solver);
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
    double   nonlin_vscut, nonlin_vsbott, geostatic_loading_t, geostatic_cushion_t,
            *auxiliar;
    char     material_model[64],
             material_properties[64],
             plasticity_type[64];

    materialmodel_t      materialmodel;
    materialproperties_t materialproperties;
    plasticitytype_t     plasticitytype;


    /* Opens numericalin file */
    if ((fp = fopen(parametersin, "r")) == NULL) {
        fprintf(stderr, "Error opening %s\n at nl_init_parameters", parametersin);
        return -1;
    }

    /* Parses parameters.in to capture nonlinear single-value parameters */

    if ( (parsetext(fp, "nonlinear_shear_velocity_cut", 'd', &nonlin_vscut        ) != 0) ||
         (parsetext(fp, "nonlinear_shear_velocity_min", 'd', &nonlin_vsbott        ) != 0) ||
         (parsetext(fp, "geostatic_loading_time_sec",   'd', &geostatic_loading_t ) != 0) ||
         (parsetext(fp, "geostatic_cushion_time_sec",   'd', &geostatic_cushion_t ) != 0) ||
         (parsetext(fp, "material_model",               's', &material_model      ) != 0) ||
         (parsetext(fp, "material_properties_type",     's', &material_properties ) != 0) ||
         (parsetext(fp, "material_plasticity_type",     's', &plasticity_type     ) != 0) ||
         (parsetext(fp, "material_properties_count",    'i', &properties_count    ) != 0) )
    {
        fprintf(stderr, "Error parsing nonlinear parameters from %s\n", parametersin);
        return -1;
    }

    /* Performs sanity checks */

    if (nonlin_vscut < 0) {
        fprintf(stderr, "Illegal Vs cut value for nonlinear analysis %f\n", nonlin_vscut);
        return -1;
    }

    if (nonlin_vsbott < 0) {
        fprintf(stderr, "Illegal Vs min value for nonlinear analysis %f\n", nonlin_vsbott);
        return -1;
    }


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
    } else if ( strcasecmp(material_model, "DruckerPrager") == 0 ) {
        materialmodel = DRUCKERPRAGER;
    } else {
        fprintf(stderr,
                "Illegal material model for nonlinear analysis"
                "(linear, vonMises, DruckerPrager): %s\n", material_model);
        return -1;
    }

    if ( strcasecmp(material_properties, "cohefriction") == 0 ) {
        materialproperties = COHEFRICTION;
    } else if ( strcasecmp(material_properties, "alphakay") == 0 ) {
        materialproperties = ALPHAKAY;
    } else {
        fprintf(stderr,
                "Illegal material properties type for nonlinear "
                "analysis (cohefriction, alphakay): %s\n",
                material_properties);
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


    /* Initialize the static global variables */

    theNonLinVsCut        = nonlin_vscut;
    theNonLinVsMin        = nonlin_vsbott;
    theGeostaticLoadingT  = geostatic_loading_t;
    theGeostaticCushionT  = geostatic_cushion_t;
    theGeostaticFinalStep = (int)( (geostatic_loading_t + geostatic_cushion_t) / theDeltaT );
    theMaterialModel      = materialmodel;
    thePropertiesType     = materialproperties;
    thePropertiesCount    = properties_count;
    thePlasticityModel    = plasticitytype;

    auxiliar             = (double*)malloc( sizeof(double) * thePropertiesCount * 6 );
    theVsLimits          = (double*)malloc( sizeof(double) * thePropertiesCount );
    theAlphaCohes        = (double*)malloc( sizeof(double) * thePropertiesCount );
    theKayPhis           = (double*)malloc( sizeof(double) * thePropertiesCount );
    theStrainRates       = (double*)malloc( sizeof(double) * thePropertiesCount );
    theSensitivities     = (double*)malloc( sizeof(double) * thePropertiesCount );
    theHardeningModulus  = (double*)malloc( sizeof(double) * thePropertiesCount );


    if ( parsedarray( fp, "material_properties_list", thePropertiesCount * 6, auxiliar ) != 0) {
        fprintf(stderr, "Error parsing nonlinear material properties list from %s\n", parametersin);
        return -1;
    }

    for ( row = 0; row < thePropertiesCount; row++) {
        theVsLimits[row]          = auxiliar[ row * 6     ];
        theAlphaCohes[row]        = auxiliar[ row * 6 + 1 ];
        theKayPhis[row]           = auxiliar[ row * 6 + 2 ];
        theStrainRates[row]       = auxiliar[ row * 6 + 3 ];
        theSensitivities[row]     = auxiliar[ row * 6 + 4 ];
        theHardeningModulus[row]  = auxiliar[ row * 6 + 5 ];
    }

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

    int     i;
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
        double      elementVs;

        eindex = myNonlinElementsMapping[nl_eindex];

        elemp = &myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;
        ecp   = myNonlinSolver->constants + nl_eindex;

        /* get element Vs */

        elementVs = (double)edata->Vs;

        /* Calculate the lame constants and store in element */

        mu_and_lambda(&mu, &lambda, edata, eindex);
        ecp->lambda = lambda;
        ecp->mu     = mu;

        /* Calculate the yield function constants */

        switch (theMaterialModel) {

            case LINEAR:
                ecp->alpha = 0;
                ecp->k     = 0;
                break;

            case VONMISES:
                ecp->alpha = 0;
                ecp->k     = get_kay(elementVs);
                break;

            case DRUCKERPRAGER:
                ecp->alpha = get_alpha(elementVs);
                ecp->k     = get_kay(elementVs);
                break;

            default:
                fprintf(stderr, "Thread %d: nonlinear_solver_init:\n"
                        "\tUnexpected error with the material model\n", myID);
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
                break;
        }

        for (i = 0; i < 8; i++) {
            ecp->fs[i]     = 0;
            ecp->dLambda[i] = 0;
        }

        ecp->strainrate  =
            interpolate_property_value(elementVs, theStrainRates  );
        ecp->sensitivity =
            interpolate_property_value(elementVs, theSensitivities );
        ecp->hardmodulus =
            interpolate_property_value(elementVs, theHardeningModulus );


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

    tensor.xx = 0;
    tensor.yy = 0;
    tensor.zz = 0;
    tensor.xy = 0;
    tensor.yz = 0;
    tensor.xz = 0;

    return tensor;
}

void init_tensorptr(tensor_t *tensor) {

    tensor->xx = 0;
    tensor->yy = 0;
    tensor->zz = 0;
    tensor->xy = 0;
    tensor->yz = 0;
    tensor->xz = 0;

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

double compute_yield_surface_state ( double J2, double I1, double alpha ) {

//    if ( theMaterialModel == VONMISES ) {
//        return sqrt(J2);
//    }
//
//    if ( theMaterialModel == DRUCKERPRAGER ) {
        return alpha * I1 + sqrt(J2);
//    }
//
//    return 0;
}

double compute_dLambda ( nlconstants_t constants, double fs, double eff_ps, double J2, double I1 ) {

	/* This function is no longer used and was updated by compute_dLambdaII */
	double phi_pt, eta_pt, psi_pt, s, c, kappa, A1, B1, C1; /*  variables needed for the plastic strain update*/

	if ( thePlasticityModel == RATE_DEPENDANT ) {
		double factor      = fs / constants.k;
		double strainRate  = constants.strainrate;
		double sensitivity = constants.sensitivity;
		double oneOverM    = 1.0 / sensitivity;

		return strainRate * pow(factor, oneOverM);
	}

	/* Dorian: Rate independent plastic multiplier computation stage. */
	/* This expression is exact for the Drucker-Prager material model with Linear hardening Rule */

	s      = constants.hardmodulus;
	c      = constants.k;
	kappa  = ( constants.lambda + 2 * constants.mu / 3 );

	phi_pt = sqrt ( 1/2. + 3 * constants.alpha * constants.alpha );
	eta_pt = s * phi_pt + 9 * kappa * constants.alpha *  constants.alpha;
	psi_pt = s * eff_ps + c - constants.alpha * I1;

	A1 = constants.mu * constants.mu - eta_pt * eta_pt;
	B1 = 2 * eta_pt * psi_pt + 2 * sqrt( J2 ) * constants.mu;
	C1 = J2 - psi_pt * psi_pt;


	if ( fs > c + s * eff_ps ) {

		if ( ( B1*B1 - 4 * A1 * C1 ) < 0 ) {
        fprintf(stderr, "Thread compute_dLambda:\n"
                "\t Illegal value computing plastic strain multiplier\n" );
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);

		}

		return ( B1 - sqrt ( B1 * B1 - 4 * A1 * C1 ) ) / ( 2 * A1 );
	}
	else
		return 0;

}


double compute_dLambdaII ( nlconstants_t constants, double fs, double eff_ps, double J2, double I1 ) {

	double phi_pt, s, c, kappa, FsT; /*  variables needed for the plastic strain update*/

	if ( thePlasticityModel == RATE_DEPENDANT ) {
		double factor      = fs / constants.k;
		double strainRate  = constants.strainrate;
		double sensitivity = constants.sensitivity;
		double oneOverM    = 1.0 / sensitivity;

		return strainRate * pow(factor, oneOverM);
	}

	/* Dorian: Rate independent plastic multiplier computation stage. */
	/* This expression is exact for the Drucker-Prager material model with Linear hardening Rule */

	s      = constants.hardmodulus;
	c      = constants.k;
	kappa  = ( constants.lambda + 2 * constants.mu / 3 );
	phi_pt = sqrt ( 1/2. + 3 * constants.alpha * constants.alpha );

	FsT = fs - c - s * eff_ps;

	if (  FsT > 0 )
		return FsT / ( constants.mu + 9 * kappa * constants.alpha * constants.alpha + s * phi_pt );

	return 0;
}





tensor_t compute_dfds (tensor_t dev, double J2, double alpha) {

    tensor_t dfds;

    dfds.xx = dev.xx / ( 2.0 * sqrt(J2) ) + alpha;
    dfds.yy = dev.yy / ( 2.0 * sqrt(J2) ) + alpha;
    dfds.zz = dev.zz / ( 2.0 * sqrt(J2) ) + alpha;
    dfds.xy = dev.xy / ( 2.0 * sqrt(J2) );
    dfds.yz = dev.yz / ( 2.0 * sqrt(J2) );
    dfds.xz = dev.xz / ( 2.0 * sqrt(J2) );

    return dfds;

}

tensor_t compute_pstrain2 (tensor_t pstrain1, tensor_t dfds, double dLambda,
                           double dt) {

    tensor_t pstrain2;

    if ( thePlasticityModel == RATE_DEPENDANT ) {
    	pstrain2.xx = pstrain1.xx + dt * dLambda * dfds.xx;
    	pstrain2.yy = pstrain1.yy + dt * dLambda * dfds.yy;
    	pstrain2.zz = pstrain1.zz + dt * dLambda * dfds.zz;
    	pstrain2.xy = pstrain1.xy + dt * dLambda * dfds.xy;
    	pstrain2.yz = pstrain1.yz + dt * dLambda * dfds.yz;
    	pstrain2.xz = pstrain1.xz + dt * dLambda * dfds.xz;
    }
    else {
    pstrain2.xx = pstrain1.xx +  dLambda * dfds.xx;
    pstrain2.yy = pstrain1.yy +  dLambda * dfds.yy;
    pstrain2.zz = pstrain1.zz +  dLambda * dfds.zz;
    pstrain2.xy = pstrain1.xy +  dLambda * dfds.xy;
    pstrain2.yz = pstrain1.yz +  dLambda * dfds.yz;
    pstrain2.xz = pstrain1.xz +  dLambda * dfds.xz;

    }

    return pstrain2;
}

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

        if ( theMaterialModel == LINEAR ) {

            stresses = myNonlinSolver->stresses[nl_eindex];

        } else {

            qptensors_t tstrains, pstrains, estrains;

            tstrains = myNonlinSolver->strains   [nl_eindex];
            pstrains = myNonlinSolver->pstrains1[nl_eindex];

            /* compute total strain - plastic strain */
            estrains = subtrac_qptensors(tstrains, pstrains);

            /* compute the corresponding stress */
            stresses = compute_qp_stresses(estrains, mu, lambda);
        }

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
                               double      theDeltaT)
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
        double         XI, QC;
        double         Fs, I1, oct, J2, Fs2, ept=0;
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
        k      = enlcons->k;
        hrd    = enlcons->hardmodulus;

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

            tensor_t    estrain, dev;

            /* Quadrature point local coordinates */
            double lx = xi[0][i] * qc ;
            double ly = xi[1][i] * qc ;
            double lz = xi[2][i] * qc ;

            /* Calculate strains */
            tstrains->qp[i] = point_strain(u, lx, ly, lz, h);

            /* Calculate stresses */
            if ( theMaterialModel == LINEAR ) {
                stresses->qp[i]  = point_stress ( tstrains->qp[i], mu, lambda );
            } else {
                pstrains1->qp[i] = copy_tensor ( pstrains2->qp[i] );     /* strain predictor: equal to the previous strain tens   */
                estrain          = subtrac_tensors ( tstrains->qp[i], pstrains1->qp[i] ); /* stress predictor   */
                stresses->qp[i]  = point_stress ( estrain, mu, lambda );
                ept              = epstr2->qv[i];

            }


            /* Stress invariants */
            I1  = tensor_I1 ( stresses->qp[i] );
            oct = tensor_octahedral ( I1 );
            dev = tensor_deviator ( stresses->qp[i], oct );
            J2  = tensor_J2 ( dev );
            Fs  = compute_yield_surface_state ( J2, I1, alpha ); /* Fs predictor */


            enlcons->avgFs += Fs * 0.125; /* 1/8 = 0.125 is qp contribution */
            if ( Fs > enlcons->maxFs ) {
                enlcons->maxFs = Fs;
            }

            /* Do not compute plasticity for the linear case */
//            if ( ( theMaterialModel == LINEAR ) || ( J2 == 0 ) ) {
            if ( ( theMaterialModel == LINEAR ) ) {
            	continue;
            }

            /* Next plastic strain correction */
            enlcons->dLambda[i] = compute_dLambdaII ( *enlcons, Fs, ept , J2, I1 );
            tensor_t dfds       = compute_dfds ( dev, J2, alpha );
            pstrains2->qp[i]    = compute_pstrain2 ( pstrains1->qp[i], dfds, enlcons->dLambda[i], theDeltaT ); /* real plastic strain */
        	epstr2->qv[i]       = ept + enlcons->dLambda[i] * sqrt ( 1/2. + 3 * alpha * alpha);

//            if ( hrd != 0)
//            	epstr2->qv[i]   = ept + enlcons->dLambda[i] * sqrt ( 1/2. + 3 * alpha * alpha);
//            else
//            	epstr2->qv[i]   = 0;

            /* if fs > k the stress and strain tensors must be corrected. Dorian*/
            if ( thePlasticityModel != RATE_DEPENDANT ) {
            	if ( enlcons->dLambda[i] > 0 ){
            		estrain          = subtrac_tensors ( tstrains->qp[i], pstrains2->qp[i] );
            		stresses->qp[i]  = point_stress ( estrain, mu, lambda ); /* real stress tensor */
            		I1               = tensor_I1 ( stresses->qp[i] );
            		oct              = tensor_octahedral ( I1 );
            		dev              = tensor_deviator ( stresses->qp[i], oct );
            		J2               = tensor_J2 ( dev );
            		Fs2              = compute_yield_surface_state ( J2, I1, alpha );  /* final Fs value, must be equal to the value from the hardening function */
            	}
            }

            /* Storing and checking */
            if ( thePlasticityModel == RATE_DEPENDANT ) {
            	enlcons->fs[i] = Fs;
            	check_yield_limit ( myMesh, eindex, edata->Vs, Fs, k, i);
            }
//            check_strain_stability ( enlcons->dLambda[i], theDeltaT, myMesh, eindex, edata->Vs, Fs, k, i );



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
    XMALLOC_VAR_N( myNonlinStations, nlstation_t, myNumberOfNonlinStations);

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

    for ( iStation = 0; iStation < myNumberOfNonlinStations; iStation++ ) {

        tensor_t *stress, *strain, *pstrain1, *pstrain2;
        double   *ep1;

        strain   = &(myNonlinStations[iStation].strain);
        stress   = &(myNonlinStations[iStation].stress);
        pstrain1 = &(myNonlinStations[iStation].pstrain1);
        pstrain2 = &(myNonlinStations[iStation].pstrain2);
        ep1      = &(myNonlinStations[iStation].ep );
        *ep1     = 0;

        init_tensorptr(strain);
        init_tensorptr(stress);
        init_tensorptr(pstrain1);
        init_tensorptr(pstrain2);

    }

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

        vector3D_t     localcoords;
        tensor_t      *stress, *tstrain, *pstrain1, *pstrain2, estrain, dev;
        elem_t        *elemp;
        edata_t       *edata;
        nlconstants_t *constants;
        fvector_t      u[8];
        double         alpha, lambda, dLambda = 0, mu, k, hrd,  Fs = 0, J2, I1, h, oct;
        double         bStrain = 0, bStress = 0, *ept, ep=0;
        double         lx, ly, lz;

        nl_eindex    = myStationsElementIndices[iStation];
        eindex       = myNonlinElementsMapping[nl_eindex];
        mappingIndex = myNonlinStationsMapping[iStation];

        elemp = &myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;
        h     = edata->edgesize;

        /* Capture data from the nonlinear element structure */
        constants = myNonlinSolver->constants + nl_eindex;

        mu     = constants->mu;
        lambda = constants->lambda;
        alpha  = constants->alpha;
        k      = constants->k;
        hrd    = constants->hardmodulus;

        /* Capture the current state in the station */
        tstrain  = &(myNonlinStations[iStation].strain);
        stress   = &(myNonlinStations[iStation].stress);
        pstrain1 = &(myNonlinStations[iStation].pstrain1);
        pstrain2 = &(myNonlinStations[iStation].pstrain2);
        ept      = &(myNonlinStations[iStation].ep);
        ep       = *ept;

        /* Capture displacements */
        if ( get_displacements(mySolver, elemp, u) == 0 ) {
            /* If all displacements are zero go directly to printing */
            goto NLPRINT;
        }

        /* Capture local coordinates */
        localcoords = myStations[mappingIndex].localcoords;

        localcoords.x[0] = -1/sqrt(3.);
        localcoords.x[1] = -1/sqrt(3.);
        localcoords.x[2] = -1/sqrt(3.);

        lx = localcoords.x[0];
        ly = localcoords.x[1];
        lz = localcoords.x[2];

        /* Calculate strains */
        *tstrain = point_strain(u, lx, ly, lz, h);

        /* Calculate stresses */
        if ( theMaterialModel == LINEAR ) {
            *stress = point_stress ( *tstrain, mu, lambda);
        } else {
            *pstrain1 = copy_tensor ( *pstrain2 );
            estrain   = subtrac_tensors ( *tstrain, *pstrain1 );
            *stress   = point_stress ( estrain, mu, lambda );
        }

        /* Stress invariants */
        I1  = tensor_I1 ( *stress );
        oct = tensor_octahedral ( I1 );
        dev = tensor_deviator ( *stress, oct );
        J2  = tensor_J2 ( dev );

        Fs  = compute_yield_surface_state ( J2, I1, alpha );


        /* Do not compute plasticity for the linear case */
        if ( theMaterialModel != LINEAR ) {
//            if ( J2 != 0 ) {
                /* Next plastic strain correction */
                dLambda       = compute_dLambdaII ( *constants, Fs, *ept, J2, I1 );
                tensor_t dfds = compute_dfds ( dev, J2, alpha );
                *pstrain2     = compute_pstrain2 ( *pstrain1, dfds, dLambda, dt );
                *ept          = ep +  dLambda * sqrt (1/2. + 3 * alpha * alpha );


                if ( thePlasticityModel != RATE_DEPENDANT ) {
                	if ( dLambda > 0) {

                		estrain   = subtrac_tensors ( *tstrain, *pstrain2 );
                		*stress   = point_stress ( estrain, mu, lambda );
                		I1  = tensor_I1 ( *stress );
                		oct = tensor_octahedral ( I1 );
                		dev = tensor_deviator ( *stress, oct );
                		J2  = tensor_J2 ( dev );
                		Fs  = compute_yield_surface_state ( J2, I1, alpha );  /* final Fs value, must be equal to the value from the hardening function */
                	}

                }

//            }
        }

        bStrain = tstrain->xx + tstrain->yy + tstrain->zz;
        bStress = stress->xx + stress->yy + stress->zz;

NLPRINT:
        if (step % rate == 0) {
            fprintf( myStations[mappingIndex].fpoutputfile,

                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e"
                    " % 8e % 8e % 8e",

                    tstrain->xx, stress->xx, // 11 12
                    tstrain->yy, stress->yy, // 13 14
                    tstrain->zz, stress->zz, // 15 16
                    bStrain,     bStress,    // 17 18
                    tstrain->xy, stress->xy, // 19 20
                    tstrain->yz, stress->yz, // 21 22
                    tstrain->xz, stress->xz, // 23 24
                    dLambda, Fs, k + hrd * (*ept) ); // 25 26 27
        }
    } /* for all my stations */

}
