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
 * Generate an unstructured mesh and solve the linear system thereof
 * derived.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <float.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <assert.h>
#include <errno.h>
#include <stdarg.h>

#include "util.h"
#include "commutil.h"
#include "timers.h"
#include "etree.h"
#include "octor.h"
#include "psolve.h"
#include "cvm.h"
#include "nrutila.h"
#include "quakesource.h"
#include "output.h"
#include "nonlinear.h"
#include "io_planes.h"
#include "io_checkpoint.h"
#include "stiffness.h"
#include "damping.h"
#include "quake_util.h"
#include "buildings.h"
#include "drm.h"
#include "meshformatlab.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

/* ONLY GLOBAL VARIABLES ALLOWED OUTSIDE OF PARAM. and GLOBAL. IN ALL OF PSOLVE!! */
MPI_Comm comm_solver;
MPI_Comm comm_IO;

#ifndef PROCPERNODE
#define PROCPERNODE     4
#endif

#define PI		3.14159265358979323846264338327

#define GOAHEAD_MSG     100
#define MESH_MSG	101
#define STAT_MSG	102
#define OUT4D_MSG       103
#define DN_MASS_MSG     104
#define AN_MASS_MSG     105
#define DN_FORCE_MSG    106
#define AN_FORCE_MSG    107
#define DN_DISP_MSG     108
#define AN_DISP_MSG     109
#define CVMRECORD_MSG   110

#define BATCH		(1 << 20)
#define LINESIZE	512
#define FILEBUFSIZE     (1 << 25)


#define CONTRIBUTION    901  /**< Harboring processors to owner processors */
#define SHARING		902  /**< Owner processors to harboring processors */

#define DISTRIBUTION    903  /**< Dangling nodes to anchored nodes */
#define ASSIGNMENT      904  /**< Anchored nodes to dangling nodes */


/*---------------Initialization and cleanup routines----------------------*/
static void    read_parameters(int argc, char **argv);
static int32_t parse_parameters(const char *numericalin);
static void    local_finalize(void);

/*---------------- Mesh generation data structures -----------------------*/
#ifdef USECVMDB

#ifndef CVMBUFSIZE
#define CVMBUFSIZE      100
#endif

static void     replicateDB(const char *dbname);
static void     open_cvmdb(void);

#else

static int32_t zsearch(void *base, int32_t count, int32_t recordsize,
		       const point_t *searchpt);
static cvmrecord_t *sliceCVM(const char *cvm_flatfile);

#endif
/** cvmrecord_t: cvm record.  Onlye used if USECVMDB not defined **/
typedef struct cvmrecord_t {
    char key[12];
    float Vp, Vs, density;
} cvmrecord_t;

/**
 * mrecord_t: Complete mesh database record
 *
 */
typedef struct mrecord_t {
    etree_addr_t addr;
    mdata_t mdata;
} mrecord_t;

/* Mesh generation related routines */
static int32_t toexpand(octant_t *leaf, double ticksize, const void *data);
static void    setrec(octant_t *leaf, double ticksize, void *data);
static void    mesh_generate(void);
static int32_t bulkload(etree_t *mep, mrecord_t *partTable, int32_t count);
static void    mesh_output(void);
static void    mesh_correct_properties( etree_t* cvm );

static void    gpu_init(void);
static void    solver_init(void);
static void    solver_printstat(mysolver_t* solver);
static void    solver_delete(void);
static void    solver_run(void);
       void    solver_output_seq(void);
static int     solver_print_schedules(mysolver_t* solver);
static schedule_t* schedule_new(void);
static void    schedule_build(mesh_t *mesh, schedule_t *dnsched,
			      schedule_t *ansched);
static void    schedule_allocMPIctl(schedule_t *sched);
static void    schedule_allocmapping(schedule_t *sched);
static void    schedule_delete(schedule_t *sched);
static void    schedule_prepare(schedule_t *sched, int32_t c_outsize,
				int32_t c_insize, int32_t s_outsize,
				int32_t s_insize);
static void    schedule_senddata(schedule_t *sched, void *valuetable,
				 int32_t itemsperentry, int32_t direction,
				 int32_t msgtag);
static int     schedule_print( schedule_t *sched, char type, FILE* out );
static int     schedule_print_detail(schedule_t* sched, char type, FILE* out);
static int     schedule_print_messenger_list( schedule_t* sched,
					      messenger_t* msg, int count,
					      char type, char cs, FILE* out );

static messenger_t *messenger_new(int32_t procid);
static void    messenger_delete(messenger_t *messenger);
static void    messenger_set(messenger_t *messenger, int32_t outsize,
			     int32_t insize);
static int32_t messenger_countnodes(messenger_t *first);

static void    compute_K(void);
static void constract_Quality_Factor_Table(void);

#ifdef BOUNDARY
static char    compute_setflag(tick_t ldb[3], tick_t ruf[3],
			       tick_t nearendp[3], tick_t farendp[3]);
static void    compute_setboundary(float size, float Vp, float Vs,
				   float rho, int flag, double dashpot[8][3]);
#endif /* BOUNDARY */

static void    compute_setab(double freq, double *aBasePtr, double *bBasePtr);
static void    compute_addforce_s(int32_t timestep);
static void    compute_adjust(void *valuetable, int32_t itemsperentry,
			      int32_t how);

static int     interpolate_station_displacements(int32_t step);


/* ---------- Static global variables ------------------------------------ */

/* These are all of the input parameters - add new ones here */
static struct Param_t {
    char  FourDOutFile[256]; 
    FILE*  FourDOutFp;
    FILE*  theMonitorFileFp;
    char*  theMonitorFileName;
    char  parameters_input_file[256];
    char  cvmdb_input_file[256];
    char  mesh_etree_output_file[256];
    char  planes_input_file[256];
    double  theVsCut;
    double  theFactor;
    double  theFreq;
    double  theFreq_Vel;
    double  theDeltaT;
    double  theDeltaTSquared;
    double  theEndT;
    double  theStartT;
    double  theDomainAzimuth;
    int    monitor_stats_rate;
    double  theSofteningFactor;
    int     theStepMeshingFactor;
    int32_t  theTotalSteps;
    int32_t  theRate;
    damping_type_t  theTypeOfDamping;
    double	theThresholdDamping;
    double	theThresholdVpVs;
    int	   theDampingStatisticsFlag;
    int	   theSchedulePrintErrorCheckFlag;
    int	   theSchedulePrintToStdout;
    int	   theSchedulePrintToFile;
    char   theSchedulePrintFilename[256];
    char*	theScheduleStatFilename;
    char*       theMeshStatFilename;
    noyesflag_t  printStationVelocities;
    noyesflag_t  printK;
    noyesflag_t  printStationAccelerations;
    noyesflag_t  includeBuildings;
    noyesflag_t  includeNonlinearAnalysis;
    noyesflag_t  useInfQk;
    int  theTimingBarriersFlag;
    stiffness_type_t   theStiffness;
    int      theStationsPrintRate;
    double*  theStationX;
    double*  theStationY;
    double*  theStationZ;
    int32_t  theNumberOfStations;
    int32_t  myNumberOfStations;
    int      IO_pool_pe_count;
    int32_t  thePlanePrintRate;
    int      theNumberOfPlanes;
    char     theStationsDirOut[256];
    station_t*  myStations;
    int  theCheckPointingRate;
    int    theUseCheckPoint;
    char   theCheckPointingDirOut[256];
    noyesflag_t  storeMeshCoordinatesForMatlab;
    double  the4DOutSize;
    int    theMeshOutFlag;
    char  theCVMFlatFile[128];
    output_parameters_t  theOutputParameters;
    double  theRegionLong;
    double  theRegionLat; 
    double  theRegionDepth;
    double  region_depth_deep_m;
    double  theSurfaceCornersLong[4];
    double  theSurfaceCornersLat[4];
    double  theDomainX;
    double  theDomainY;
    double  theDomainZ;
    noyesflag_t  drmImplement;
    drm_part_t   theDrmPart;

} Param;


/* These are all of the remaining global variables - this list should not grow */
static struct Global_t {
    int32_t  myID;
    int32_t  theGroupSize;
    octree_t*  myOctree;
    mesh_t*  myMesh;
    int64_t  theETotal;
    int64_t  theNTotal;
    mysolver_t*  mySolver;
    fvector_t*  myVelocityTable;
    fmatrix_t  theK1[8][8];
    fmatrix_t  theK2[8][8];
    fmatrix_t  theK3[8][8];
    double  theQTABLE[26][6];
    double  theABase;
    double  theBBase;
    double  theCriticalT;
    double  fastestTimeSteps;
    double  slowestTimeSteps;
    numerics_info_t  theNumericsInformation;
    mpi_info_t  theMPIInformation;
    int32_t  theNodesLoaded;
    int32_t*  theNodesLoadedList;
    vector3D_t*  myForces;
    FILE*  fpsource;
    etree_t*  theCVMEp;
    int32_t  theCVMQueryStage;
    double  theXForMeshOrigin;
    double  theYForMeshOrigin;
    double  theZForMeshOrigin;
    cvmrecord_t*  theCVMRecord;
    int  theCVMRecordSize;
    int  theCVMRecordCount;
    gpu_spec_t gpu_spec;
} Global;


/* ------------------------------End of declarations------------------------------ */

void getGPUHardware(int device, gpu_spec_t *gpuSpecs, int dispFlag)
{
    int i;
    const int kb = 1024;
    const int mb = kb * kb;

    cudaDeviceProp props;
    memset(&props, 0, sizeof(cudaDeviceProp));
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
      fprintf(stderr, "Failed to get GPU device properties\n");
      MPI_Abort(MPI_COMM_WORLD, ERROR);
      exit(1);
    }

    gpuSpecs->device = device;
    gpuSpecs->max_threads = props.maxThreadsPerBlock;
    for (i = 0; i < 3; i++) {
      gpuSpecs->max_block_dim[i] = props.maxThreadsDim[i];
      gpuSpecs->max_grid_dim[i] = props.maxGridSize[i];
    }

    if (dispFlag) {
      cout << "GPU Device " << device << ": " << props.name 
	   << ": " << props.major << "." << props.minor << endl;
      cout << "  Global memory:   " << props.totalGlobalMem / mb 
	   << "mb" << endl;
      cout << "  Shared memory:   " << props.sharedMemPerBlock / kb 
	   << "kb" << endl;
      cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
      cout << "  Block registers: " << props.regsPerBlock << endl << endl;
      
      cout << "  Warp size:         " << props.warpSize << endl;
      cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
      cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] 
	   << ", " << props.maxThreadsDim[1]  << ", " 
	   << props.maxThreadsDim[2] << " ]" << endl;
      cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " 
	   << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" 
	   << endl;
      cout << endl;
    }
}

static inline int
monitor_print( const char* format, ... )
{
    int ret = 0;

    if (format != NULL) {
	va_list args;

	va_start( args, format );

	if (Param.theMonitorFileFp == NULL) {
	    ret = vfprintf( stderr, format, args );
	} else {
	    ret = hu_print_tee_va( Param.theMonitorFileFp, format, args );
	}

	va_end( args );
    }

    return ret;
}



/*-----------Parameter input routines---------------------------------*/

static void read_parameters( int argc, char** argv ){

#define LOCAL_INIT_DOUBLE_MESSAGE_LENGTH 18  /* Must adjust this if adding double params */
#define LOCAL_INIT_INT_MESSAGE_LENGTH 20     /* Must adjust this if adding int params */

    double  double_message[LOCAL_INIT_DOUBLE_MESSAGE_LENGTH];
    int     int_message[LOCAL_INIT_INT_MESSAGE_LENGTH];

    strcpy(Param.parameters_input_file, argv[1]);

    /* PE 0 reads all params from disk */
    if (Global.myID == 0) {
        if (parse_parameters(Param.parameters_input_file) != 0) {
            fprintf(stderr, "Thread 0: Problem reading parameters!\n");
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }
    }

    /*Broadcast all double params*/
    double_message[0]  = Param.theVsCut;
    double_message[1]  = Param.theFactor;
    double_message[2]  = Param.theFreq;
    double_message[3]  = Param.theDeltaT;
    double_message[4]  = Param.theDeltaTSquared;
    double_message[5]  = Param.theEndT;
    double_message[6]  = Param.theStartT;
    double_message[7]  = Param.theDomainX;
    double_message[8]  = Param.theDomainY;
    double_message[9]  = Param.theDomainZ;
    double_message[10] = Param.theDomainAzimuth;
    double_message[11] = Param.theThresholdDamping;
    double_message[12] = Param.theThresholdVpVs;
    double_message[13] = Param.theSofteningFactor;
    double_message[14] = Param.theFreq_Vel;
    double_message[15] = Param.theRegionLat;
    double_message[16] = Param.theRegionLong;
    double_message[17] = Param.theRegionDepth;


    MPI_Bcast(double_message, LOCAL_INIT_DOUBLE_MESSAGE_LENGTH, MPI_DOUBLE, 0, comm_solver);

    Param.theVsCut            = double_message[0];
    Param.theFactor           = double_message[1];
    Param.theFreq             = double_message[2];
    Param.theDeltaT           = double_message[3];
    Param.theDeltaTSquared    = double_message[4];
    Param.theEndT             = double_message[5];
    Param.theStartT           = double_message[6];
    Param.theDomainX          = double_message[7];
    Param.theDomainY          = double_message[8];
    Param.theDomainZ          = double_message[9];
    Param.theDomainAzimuth	= double_message[10];
    Param.theThresholdDamping = double_message[11];
    Param.theThresholdVpVs    = double_message[12];
    Param.theSofteningFactor  = double_message[13];
    Param.theFreq_Vel		= double_message[14];
    Param.theRegionLat		= double_message[15];
    Param.theRegionLong		= double_message[16];
    Param.theRegionDepth    = double_message[17];

    /*Broadcast all integer params*/
    int_message[0]  = Param.theTotalSteps;
    int_message[1]  = Param.theRate;
    int_message[2]  = Param.theNumberOfPlanes;
    int_message[3]  = Param.theNumberOfStations;
    int_message[4]  = (int)Param.theTypeOfDamping;
    int_message[5]  = Param.theDampingStatisticsFlag;
    int_message[6]  = Param.theMeshOutFlag;
    int_message[7]  = Param.theCheckPointingRate;
    int_message[8]  = Param.theUseCheckPoint;
    int_message[9]  = (int)Param.includeNonlinearAnalysis;
    int_message[10] = (int)Param.theStiffness;
    int_message[11] = (int)Param.printK;
    int_message[12] = (int)Param.printStationVelocities;
    int_message[13] = (int)Param.printStationAccelerations;
    int_message[14] = Param.theTimingBarriersFlag;
    int_message[15] = (int)Param.includeBuildings;
    int_message[16] = (int)Param.storeMeshCoordinatesForMatlab;
    int_message[17] = (int)Param.drmImplement;
    int_message[18] = (int)Param.useInfQk;
    int_message[19] = Param.theStepMeshingFactor;


    MPI_Bcast(int_message, LOCAL_INIT_INT_MESSAGE_LENGTH, MPI_INT, 0, comm_solver);

    Param.theTotalSteps            = int_message[0];
    Param.theRate                  = int_message[1];
    Param.theNumberOfPlanes              = int_message[2];
    Param.theNumberOfStations            = int_message[3];
    Param.theTypeOfDamping         = (damping_type_t)int_message[4];
    Param.theDampingStatisticsFlag = int_message[5];
    Param.theMeshOutFlag                 = int_message[6];
    Param.theCheckPointingRate           = int_message[7];
    Param.theUseCheckPoint               = int_message[8];
    Param.includeNonlinearAnalysis       = (noyesflag_t)int_message[9];
    Param.theStiffness                   = (stiffness_type_t)int_message[10];
    Param.printK                         = (noyesflag_t)int_message[11];
    Param.printStationVelocities         = (noyesflag_t)int_message[12];
    Param.printStationAccelerations      = (noyesflag_t)int_message[13];
    Param.theTimingBarriersFlag          = int_message[14];
    Param.includeBuildings               = (noyesflag_t)int_message[15];
    Param.storeMeshCoordinatesForMatlab  = (noyesflag_t)int_message[16];
    Param.drmImplement                   = (noyesflag_t)int_message[17];
    Param.useInfQk                       = (noyesflag_t)int_message[18];
    Param.theStepMeshingFactor           = int_message[19];

    /*Broadcast all string params*/
    MPI_Bcast (Param.parameters_input_file,  256, MPI_CHAR, 0, comm_solver);
    MPI_Bcast (Param.theCheckPointingDirOut, 256, MPI_CHAR, 0, comm_solver);
    MPI_Bcast (Param.FourDOutFile,           256, MPI_CHAR, 0, comm_solver);
    MPI_Bcast (Param.cvmdb_input_file,       256, MPI_CHAR, 0, comm_solver);
    MPI_Bcast (Param.mesh_etree_output_file, 256, MPI_CHAR, 0, comm_solver);
    MPI_Bcast (Param.planes_input_file,      256, MPI_CHAR, 0, comm_solver);

    return;
}



static void
local_finalize()
{

    /* Free memory associated with the octree and mesh */
    octor_deletetree(Global.myOctree);
    octor_deletemesh(Global.myMesh);

    /* Free the memory used by the source */
    free(Global.myForces);

    if (Global.myVelocityTable != NULL)
	free(Global.myVelocityTable);

    /* Free memory associated with the stiffness module */
    stiffness_delete(Global.myID);

    /* Free memory associated with the solver */
    solver_delete();

    return;
}


/**
 * Parse a text file and return the value of a match string.
 *
 * \return 0 if OK, -1 on error.
 */
int
parsetext (FILE* fp, const char* querystring, const char type, void* result)
{
    const static char delimiters[] = " =\n\t";

    int32_t res = 0, found = 0;

    /* Start from the beginning */
    rewind(fp);


    /* Look for the string until found */
    while (!found) {
	char line[LINESIZE];
	char *name, *value;

	/* Read in one line */
	if (fgets(line, LINESIZE, fp) == NULL)
	    break;

	name = strtok(line, delimiters);
	if ((name != NULL) && (strcmp(name, querystring) == 0)) {
	    found = 1;
	    value = strtok(NULL, delimiters);

	    switch (type) {
	    case 'i':
		res = sscanf(value, "%d", (int *)result);
		break;
	    case 'f':
		res = sscanf(value, "%f", (float *)result);
		break;
	    case 'd':
		res = sscanf(value, "%lf", (double *)result);
		break;
	    case 's':
		res = 1;
		strcpy((char *)result, value);
		break;
	    case 'u':
		res = sscanf(value, "%u", (uint32_t *)result);
		break;
	    default:
		fprintf(stderr, "parsetext: unknown type %c\n", type);
		return -1;
	    }
	}

    }

    return (res == 1) ? 0 : -1;
}




/**
 * This is like \c parsetext with the following differences:
 * - works only for strings;
 * - avoids writting past the end of the string;
 * - return convention is different, it distinguishes between "key not found"
 *   and other type of errors.
 *
 * \return
 *	1 if the key name is found and the value is stored in \c value;
 *	0 if the key name was not found; -1 on error.
 */
static int
read_config_string (FILE* fp, const char* key, char* value_ptr, size_t size)
{
    static const char delimiters[] = " =\n\t";

    int  ret;
    char line[LINESIZE];
    char state[LINESIZE];
    char *name, *value, *state_ptr;


    HU_ASSERT_PTR_ALWAYS( fp );
    HU_ASSERT_PTR_ALWAYS( value_ptr );
    HU_ASSERT_PTR_ALWAYS( key );

    rewind (fp);
    ret   = 0;
    *value_ptr = '\0';

    while (0 == ret && !ferror (fp)) {

	if (fgets (line, LINESIZE, fp) == NULL) {
	    if (!feof (fp)) {
		ret = -1;	/* input error */
	    }
	    break;
	}

	state_ptr = state;
	name      = strtok_r (line, delimiters, &state_ptr);

	if ((name != NULL) && (strcmp (name, key) == 0)) {
	    size_t value_len;

	    value = strtok_r (NULL, delimiters, &state_ptr);

	    if (NULL != value) {
		value_len = strlen (value);

		if (value_len >= size) {
		    ret = -2;	/* return buffer is too short */
		} else {
		    strncpy (value_ptr, value, size);
		    ret = 1;
		}
	    }

	    break;
	}
    }

    return ret;
}


/**
 * Open material database and initialize various static global variables.
 *
 * \return 0 if OK, -1 on error.
 */
static int32_t parse_parameters( const char* numericalin )
{
    FILE* fp;

    /*
     * This used to be a seperate file, now aliased to numericalin.
     * This fakes a seperate file as per old code
     */
    const char* physicsin = numericalin;

    int32_t   samples, rate;
    int       number_output_planes, number_output_stations,
              damping_statistics, use_checkpoint, checkpointing_rate,
              step_meshing;

    double    freq, vscut,
              region_origin_latitude_deg, region_origin_longitude_deg,
              region_azimuth_leftface_deg,
              region_depth_shallow_m, region_length_east_m,
              region_length_north_m, region_depth_deep_m,
              startT, endT, deltaT, softening_factor,
              threshold_damping, threshold_VpVs, freq_vel;
    char      type_of_damping[64],
	      	  checkpoint_path[256],
              include_buildings[64],
              include_nonlinear_analysis[64],
              stiffness_calculation_method[64],
              print_matrix_k[64],
              print_station_velocities[64],
              print_station_accelerations[64],
	      	  mesh_coordinates_for_matlab[64],
    		  implement_drm[64],
    		  use_infinite_qk[64];

    // Casting -1 to enum type is a bad idea
    //damping_type_t   typeOfDamping     = -1;
    //stiffness_type_t stiffness_method  = -1;
    //noyesflag_t      have_buildings    = -1;
    //noyesflag_t      includeNonlinear  = -1;
    //noyesflag_t      printMatrixK      = -1;
    //noyesflag_t      printStationVels  = -1;
    //noyesflag_t      printStationAccs  = -1;
    //noyesflag_t      useInfQk          = -1;

    //noyesflag_t      meshCoordinatesForMatlab  = -1;
    //noyesflag_t      implementdrm  = -1;

    damping_type_t   typeOfDamping;
    stiffness_type_t stiffness_method;
    noyesflag_t      have_buildings;
    noyesflag_t      includeNonlinear;
    noyesflag_t      printMatrixK;
    noyesflag_t      printStationVels;
    noyesflag_t      printStationAccs;
    noyesflag_t      useInfQk;

    noyesflag_t      meshCoordinatesForMatlab;
    noyesflag_t      implementdrm;


    /* Obtain the specification of the simulation */
    if ((fp = fopen(physicsin, "r")) == NULL)
    {
	fprintf(stderr, "Error opening %s\n", physicsin);
	return -1;
    }

    /* Julio, I know this violates the 80 chars printing limit, but we never
     * print this code and is a heck of a lot easier to read it like this
     * --Ricardo
     */
    if ((parsetext(fp, "region_origin_latitude_deg",  'd', &region_origin_latitude_deg  ) != 0) ||
        (parsetext(fp, "region_origin_longitude_deg", 'd', &region_origin_longitude_deg ) != 0) ||
        (parsetext(fp, "region_depth_shallow_m",      'd', &region_depth_shallow_m      ) != 0) ||
        (parsetext(fp, "region_length_east_m",        'd', &region_length_east_m        ) != 0) ||
        (parsetext(fp, "region_length_north_m",       'd', &region_length_north_m       ) != 0) ||
        (parsetext(fp, "region_depth_deep_m",         'd', &region_depth_deep_m         ) != 0) ||
        (parsetext(fp, "region_azimuth_leftface_deg", 'd', &region_azimuth_leftface_deg ) != 0) ||
        (parsetext(fp, "type_of_damping",             's', &type_of_damping             ) != 0) )
    {
        fprintf(stderr, "Error reading region origin from %s\n", physicsin);
        return -1;
    }

    if ( strcasecmp(type_of_damping, "rayleigh") == 0) {
    	typeOfDamping = RAYLEIGH;
    } else if (strcasecmp(type_of_damping, "mass") == 0) {
    	typeOfDamping = MASS;
    } else if (strcasecmp(type_of_damping, "none") == 0) {
    	typeOfDamping = NONE;
    } else if (strcasecmp(type_of_damping, "bkt") == 0) {
    	typeOfDamping = BKT;
    } else {
    	solver_abort( __FUNCTION_NAME, NULL,
    			"Unknown damping type: %s\n",
    			type_of_damping );
    }


    fclose(fp); /* physics.in */

    if ((fp = fopen(numericalin, "r")) == NULL) {
	fprintf(stderr, "Error opening %s\n", numericalin);
	return -1;
    }

    size_t monitor_name_len = 0;
    hu_config_get_string_def( fp, "monitor_file", &Param.theMonitorFileName,
			      &monitor_name_len, "monitor.txt" );

    /* open the monitor file of the simulation in pe 0 */
    Param.theMonitorFileFp = fopen( Param.theMonitorFileName, "w" );
    if (Param.theMonitorFileFp == NULL) {
	fprintf( stderr,"\n Err opening the monitor file" );
    } else {
	setlinebuf ( Param.theMonitorFileFp );
    }

    xfree_char( &Param.theMonitorFileName );

    /* numerical.in parse texts */
    if ((parsetext(fp, "simulation_wave_max_freq_hz",    'd', &freq                        ) != 0) ||
        (parsetext(fp, "simulation_node_per_wavelength", 'i', &samples                     ) != 0) ||
        (parsetext(fp, "simulation_shear_velocity_min",  'd', &vscut                       ) != 0) ||
        (parsetext(fp, "simulation_start_time_sec",      'd', &startT                      ) != 0) ||
        (parsetext(fp, "simulation_end_time_sec",        'd', &endT                        ) != 0) ||
        (parsetext(fp, "simulation_delta_time_sec",      'd', &deltaT                      ) != 0) ||
        (parsetext(fp, "softening_factor",               'd', &softening_factor            ) != 0) ||
        (parsetext(fp, "use_progressive_meshing",        'i', &step_meshing                ) != 0) ||
        (parsetext(fp, "simulation_output_rate",         'i', &rate                        ) != 0) ||
        (parsetext(fp, "number_output_planes",           'i', &number_output_planes        ) != 0) ||
        (parsetext(fp, "number_output_stations",         'i', &number_output_stations      ) != 0) ||
        (parsetext(fp, "the_threshold_damping",          'd', &threshold_damping           ) != 0) ||
        (parsetext(fp, "the_threshold_Vp_over_Vs",       'd', &threshold_VpVs              ) != 0) ||
        (parsetext(fp, "do_damping_statistics",          'i', &damping_statistics          ) != 0) ||
        (parsetext(fp, "use_checkpoint",                 'i', &use_checkpoint              ) != 0) ||
        (parsetext(fp, "checkpointing_rate",             'i', &checkpointing_rate          ) != 0) ||
        (parsetext(fp, "checkpoint_path",                's', &checkpoint_path             ) != 0) ||
        (parsetext(fp, "4D_output_file",                 's', &Param.FourDOutFile          ) != 0) ||
        (parsetext(fp, "cvmdb_input_file",               's', &Param.cvmdb_input_file      ) != 0) ||
        (parsetext(fp, "mesh_etree_output_file",         's', &Param.mesh_etree_output_file) != 0) ||
        (parsetext(fp, "planes_input_file",              's', &Param.planes_input_file     ) != 0) ||
        (parsetext(fp, "include_nonlinear_analysis",     's', &include_nonlinear_analysis  ) != 0) ||
        (parsetext(fp, "stiffness_calculation_method",   's', &stiffness_calculation_method) != 0) ||
        (parsetext(fp, "print_matrix_k",                 's', &print_matrix_k              ) != 0) ||
        (parsetext(fp, "print_station_velocities",       's', &print_station_velocities    ) != 0) ||
        (parsetext(fp, "print_station_accelerations",    's', &print_station_accelerations ) != 0) ||
        (parsetext(fp, "include_buildings",              's', &include_buildings           ) != 0) ||
        (parsetext(fp, "mesh_coordinates_for_matlab",    's', &mesh_coordinates_for_matlab ) != 0) ||
        (parsetext(fp, "implement_drm",    				 's', &implement_drm               ) != 0) ||
        (parsetext(fp, "simulation_velocity_profile_freq_hz",'d', &freq_vel                ) != 0) ||
        (parsetext(fp, "use_infinite_qk",                's', &use_infinite_qk             ) != 0) )
    {
        fprintf( stderr, "Error parsing simulation parameters from %s\n",
                numericalin );
        return -1;
    }

    hu_config_get_int_opt(fp, "output_mesh", &Param.theMeshOutFlag );
    hu_config_get_int_opt(fp, "enable_timing_barriers",&Param.theTimingBarriersFlag);
    hu_config_get_int_opt(fp, "forces_buffer_size", &theForcesBufferSize );
    hu_config_get_int_opt(fp, "schedule_print_file", &Param.theSchedulePrintToFile );

    hu_config_get_int_opt(fp, "schedule_print_error_check",
			  &Param.theSchedulePrintErrorCheckFlag);
    hu_config_get_int_opt(fp, "schedule_print_stdout",
			  &Param.theSchedulePrintToStdout );

    size_t schedule_stat_len = 0;
    hu_config_get_string_def( fp, "stat_schedule_filename",
			      &Param.theScheduleStatFilename,
			      &schedule_stat_len, "stat-sched.txt" );
    size_t mesh_stat_len = 0;
    hu_config_get_string_def( fp, "stat_mesh_filename", &Param.theMeshStatFilename,
			      &mesh_stat_len, "stat-mesh.txt" );

    fclose( fp );

    /* sanity check */
    if (freq <= 0) {
        fprintf(stderr, "Illegal frequency value %f\n", freq);
        return -1;
    }

    if (freq_vel < 0 || freq_vel > freq) {
        fprintf(stderr, "Illegal frequency value, velocity profile frequency can not be smaller than zero or bigger than max freq %f\n", freq_vel);
        return -1;
    }

    if (samples <= 0) {
        fprintf(stderr, "Illegal samples value %d\n", samples);
        return -1;
    }

    if (vscut <= 0) {
        fprintf(stderr, "Illegal vscut value %f\n", vscut);
        return -1;
    }

    if ((startT < 0) || (endT < 0) || (startT > endT)) {
        fprintf(stderr, "Illegal startT %f or endT %f\n", startT, endT);
        return -1;
    }

    if (deltaT <= 0) {
        fprintf(stderr, "Illegal deltaT %f\n", deltaT);
        return -1;
    }

    if ( (softening_factor <= 1) && (softening_factor != 0) ) {
        fprintf(stderr, "Illegal softening factor %f\n", softening_factor);
        return -1;
    }

    if (step_meshing < 0) {
        fprintf(stderr, "Illegal progressive meshing factor %d\n", step_meshing);
        return -1;
    }

    if (rate <= 0) {
        fprintf(stderr, "Illegal output rate %d\n", rate);
        return -1;
    }

    if (number_output_planes < 0) {
        fprintf(stderr, "Illegal number of output planes %d\n",
                number_output_planes);
        return -1;
    }

    if (number_output_stations < 0) {
        fprintf(stderr, "Illegal number of output stations %d\n",
                number_output_planes);
        return -1;
    }

    if (threshold_damping < 0) {
        fprintf(stderr, "Illegal threshold damping %f\n",
                threshold_damping);
        return -1;
    }

    if (threshold_VpVs < 0) {
        fprintf(stderr, "Illegal threshold Vp over Vs %f\n",
                threshold_VpVs);
        return -1;
    }

    if ( (damping_statistics < 0) || (damping_statistics > 1) ) {
        fprintf(stderr, "Illegal do damping statistics flag %d\n",
                damping_statistics);
        return -1;
    }

    if ( (use_checkpoint < 0) || (use_checkpoint > 1) ) {
        fprintf(stderr, "Illegal use checkpoint flag %d\n",
                use_checkpoint);
        return -1;
    }

    if ( checkpointing_rate < 0 ) {
        fprintf(stderr, "Illegal checkpointing rate %d\n",
                use_checkpoint);
        return -1;
    }

    if ( strcasecmp(include_nonlinear_analysis, "yes") == 0 ) {
        includeNonlinear = YES;
    } else if ( strcasecmp(include_nonlinear_analysis, "no") == 0 ) {
        includeNonlinear = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
        	"Unknown response for including"
                "nonlinear analysis (yes or no): %s\n",
                include_nonlinear_analysis );
    }

    if ( strcasecmp(stiffness_calculation_method, "effective") == 0 ) {
        stiffness_method = EFFECTIVE;
    } else if ( strcasecmp(stiffness_calculation_method, "conventional") == 0 ) {
        stiffness_method = CONVENTIONAL;
    } else {
        solver_abort( __FUNCTION_NAME, NULL, "Unknown response for stiffness"
                "calculation method (effective or conventional): %s\n",
                stiffness_calculation_method );
    }

    if ( strcasecmp(print_matrix_k, "yes") == 0 ) {
        printMatrixK = YES;
    } else if ( strcasecmp(print_matrix_k, "no") == 0 ) {
        printMatrixK = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
                "Unknown response for printing K matrix (yes or no): %s\n",
                print_matrix_k );
    }

    if ( strcasecmp(print_station_velocities, "yes") == 0 ) {
        printStationVels = YES;
    } else if ( strcasecmp(print_station_velocities, "no") == 0 ) {
        printStationVels = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
                "Unknown response for printing"
                "station velocities (yes or no): %s\n",
                print_station_velocities );
    }

    if ( strcasecmp(print_station_accelerations, "yes") == 0 ) {
        printStationAccs = YES;
    } else if ( strcasecmp(print_station_accelerations, "no") == 0 ) {
        printStationAccs = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
                "Unknown response for printing"
                "station accelerations (yes or no): %s\n",
                print_station_accelerations );
    }

    if ( strcasecmp(mesh_coordinates_for_matlab, "yes") == 0 ) {
    	meshCoordinatesForMatlab = YES;
    } else if ( strcasecmp(mesh_coordinates_for_matlab, "no") == 0 ) {
    	meshCoordinatesForMatlab = NO;
    } else {
    	solver_abort( __FUNCTION_NAME, NULL,
    			"Unknown response for mesh coordinates"
    			"for matlab (yes or no): %s\n",
    			mesh_coordinates_for_matlab );
    }

    if ( strcasecmp(include_buildings, "yes") == 0 ) {
        have_buildings = YES;
    } else if ( strcasecmp(include_buildings, "no") == 0 ) {
        have_buildings = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
                "Unknown response for including buildings (yes or no): %s\n",
                include_buildings );
    }

    if ( strcasecmp(implement_drm, "yes") == 0 ) {
        implementdrm = YES;
    } else if ( strcasecmp(implement_drm, "no") == 0 ) {
        implementdrm = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
                "Unknown response for impelement_drm (yes or no): %s\n",
                implement_drm );
    }

    if ( strcasecmp(use_infinite_qk, "yes") == 0 ) {
        useInfQk = YES;
    } else if ( strcasecmp(use_infinite_qk, "no") == 0 ) {
        useInfQk = NO;
    } else {
        solver_abort( __FUNCTION_NAME, NULL,
            "Unknown response using infinite Qk (yes or no): %s\n",
                use_infinite_qk);
    }

    /* Init the static global variables */

    Param.theRegionLat      = region_origin_latitude_deg;
    Param.theRegionLong     = region_origin_longitude_deg ;
    Param.theRegionDepth    = region_depth_shallow_m ;

    Param.theVsCut	      = vscut;
    Param.theFactor	      = freq * samples;
    Param.theFreq         = freq;
    Param.theFreq_Vel	  = freq_vel;
    Param.theDeltaT	      = deltaT;
    Param.theDeltaTSquared  = deltaT * deltaT;
    Param.theStartT	      = startT;
    Param.theEndT           = endT;
    Param.theTotalSteps     = (int)(((endT - startT) / deltaT));

    Param.theDomainX	      = region_length_north_m;
    Param.theDomainY	      = region_length_east_m;
    Param.region_depth_deep_m = region_depth_deep_m;
    Param.theDomainZ	      = region_depth_deep_m - region_depth_shallow_m;
    Param.theDomainAzimuth  = region_azimuth_leftface_deg;
    Param.theTypeOfDamping  = typeOfDamping;
    Param.useInfQk          = useInfQk;

    Param.theRate           = rate;

    Param.theNumberOfPlanes	      = number_output_planes;
    Param.theNumberOfStations	      = number_output_stations;

    Param.theSofteningFactor        = softening_factor;
    Param.theStepMeshingFactor     = step_meshing;
    Param.theThresholdDamping	      = threshold_damping;
    Param.theThresholdVpVs	      = threshold_VpVs;
    Param.theDampingStatisticsFlag  = damping_statistics;

    Param.theCheckPointingRate      = checkpointing_rate;
    Param.theUseCheckPoint	      = use_checkpoint;

    Param.includeNonlinearAnalysis  = includeNonlinear;
    Param.theStiffness              = stiffness_method;

    Param.printK                    = printMatrixK;
    Param.printStationVelocities    = printStationVels;
    Param.printStationAccelerations = printStationAccs;

    Param.includeBuildings          = have_buildings;

    Param.storeMeshCoordinatesForMatlab  = meshCoordinatesForMatlab;

    Param.drmImplement              = implementdrm;

    strcpy( Param.theCheckPointingDirOut, checkpoint_path );

    monitor_print("\n\n---------------- Some Input Data ----------------\n\n");
    monitor_print("Vs cut:                             %f\n", Param.theVsCut);
    monitor_print("Softening factor:                   %f\n", Param.theSofteningFactor);
    monitor_print("Number of stations:                 %d\n", Param.theNumberOfStations);
    monitor_print("Number of planes:                   %d\n", Param.theNumberOfPlanes);
    monitor_print("Stiffness calculation method:       %s\n", stiffness_calculation_method);
    monitor_print("Include buildings:                  %s\n", include_buildings);
    monitor_print("Include nonlinear analysis:         %s\n", include_nonlinear_analysis);
    monitor_print("Printing velocities on stations:    %s\n", print_station_velocities);
    monitor_print("Printing accelerations on stations: %s\n", print_station_accelerations);
    monitor_print("Mesh Coordinates For Matlab:        %s\n", mesh_coordinates_for_matlab);
    monitor_print("cvmdb_input_file:                   %s\n", Param.cvmdb_input_file);
    monitor_print("Implement drm:      	               %s\n", implement_drm);
    monitor_print("\n-------------------------------------------------\n\n");

    fflush(Param.theMonitorFileFp);

    return 0;
}



/*-----------Mesh generation related routines------------------------------*/

static void  open_cvmdb(void){

#ifdef USECVMDB

    MPI_Barrier(comm_solver);
    replicateDB(Param.cvmdb_input_file);

    MPI_Barrier(comm_solver);
    Global.theCVMEp = etree_open(Param.cvmdb_input_file, O_RDONLY, CVMBUFSIZE, 0, 0 );

    if (Global.theCVMEp == NULL) {
	fprintf( stderr, "Thread %d: open_cvmdb: error opening CVM etree %s\n",
		 Global.myID, Param.cvmdb_input_file );
	MPI_Abort(MPI_COMM_WORLD, ERROR );
	exit( 1 );
    }

    dbctl_t  *myctl;
    /* Obtain the material database application control/meta data */
    if ((myctl = cvm_getdbctl(Global.theCVMEp)) == NULL) {
    	fprintf(stderr, "Error reading CVM etree control data\n");
    	MPI_Abort(MPI_COMM_WORLD, ERROR );
    	exit( 1 );
    }

    /* Check the ranges of the mesh and the scope of the CVM etree */
    if ((Param.theRegionLat < myctl->region_origin_latitude_deg) ||
        (Param.theRegionLong < myctl->region_origin_longitude_deg) ||
        (Param.theRegionDepth < myctl->region_depth_shallow_m) ||
        (Param.region_depth_deep_m > myctl->region_depth_deep_m) ||
        (Param.theRegionLat + Param.theDomainX / DIST1LAT
         > myctl->region_origin_latitude_deg
         + myctl->region_length_north_m / DIST1LAT) ||
        (Param.theRegionLong + Param.theDomainY / DIST1LON
         > myctl->region_origin_longitude_deg +
         myctl->region_length_east_m / DIST1LON)) {
        fprintf(stderr, "Mesh area out of the CVM etree\n");
        MPI_Abort(MPI_COMM_WORLD, ERROR );
    	exit( 1 );
    }

    /* Compute the coordinates of the origin of the mesh coordinate
       system in the CVM etree domain coordinate system */
    Global.theXForMeshOrigin = (Param.theRegionLat
				- myctl->region_origin_latitude_deg) * DIST1LAT;
    Global.theYForMeshOrigin = (Param.theRegionLong
				- myctl->region_origin_longitude_deg) * DIST1LON;
    Global.theZForMeshOrigin = Param.theRegionDepth - myctl->region_depth_shallow_m;

    /* Free memory used by myctl */
    cvm_freedbctl(myctl);

    double  double_message_extra[3];

    double_message_extra[0] = Global.theXForMeshOrigin;
    double_message_extra[1] = Global.theYForMeshOrigin;
    double_message_extra[2] = Global.theZForMeshOrigin;

    MPI_Bcast(double_message_extra, 3, MPI_DOUBLE, 0, comm_solver);

    Global.theXForMeshOrigin = double_message_extra[0];
    Global.theYForMeshOrigin = double_message_extra[1];
    Global.theZForMeshOrigin = double_message_extra[2];

#else
    strcpy(Param.theCVMFlatFile, cvmdb_input_file);
#endif

}



#ifdef USECVMDB

/**
 * replicateDB: Copy the material database to local disks.
 *
 */
static void
replicateDB(const char *dbname)
{
    char* destdir;

#ifndef SCEC
    char* srcpath;
    MPI_Comm replica_comm;
#endif /* SCEC */


    /* Change the working directory to $LOCAL */
#ifndef CVM_DESTDIR
    char  curdir[256];
    destdir = getenv( "CVM_DESTDIR" );
    if (destdir == NULL) { /* then use current directory */
	destdir = getcwd( curdir, 256 );
    }
#else
    destdir = CVM_DESTDIR;
#endif

    /* Clean the disks:
     * NOTE: Guys! cleanup the disk in your job script before launching
     * psolve, using rm -rf.
     * E.g., on Lemieux add the following line to your PBS job script,
     * before launching psolve.
     *   prun -m cyclic -n ${RMS_NODES} -N ${RMS_NODES} rm -rf <dirname>/
     * where dirname is the directory you want to wipe out.
     * This will take care of this issue.
     *
     * On BigBen it is even easier, it can be done from the front end
     * since everything is a shared parallel file system.
     *
     * Unfortunately the 'system' function is not supported on all platforms,
     * e.g., Cray's XT3 catamount platform (BigBen) does not have it.
     */
#ifndef SCEC
    if (chdir(destdir) != 0) {
	fprintf(stderr, "Thread %d: replicateDB: cannot chdir to %s\n",
		Global.myID, destdir);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    MPI_Barrier(comm_solver);

    /* Replicate the material database among the processors */
    if (Global.myID % PROCPERNODE != 0) {
	MPI_Comm_split(comm_solver, MPI_UNDEFINED, Global.myID, &replica_comm);

    } else {
	int replica_id;
	off_t filesize, remains, batchsize;
	void *filebuf;
	int src_fd = -1, dest_fd;

	MPI_Comm_split(comm_solver, 0, Global.myID, &replica_comm);
	MPI_Comm_rank(replica_comm, &replica_id);

	if (replica_id == 0) {
	    struct stat statbuf;

#ifndef CVM_SRRCPATH
	    srcpath = getenv("CVM_SRCPATH");
#else
	    srcpath = CVM_SRCPATH;
#endif

	    if (stat(srcpath, &statbuf) != 0) {
		fprintf(stderr,
			"Thread 0: replicateDB: Cannot get stat of %s\n",
			srcpath);
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }

	    filesize = statbuf.st_size;
	    src_fd = open(srcpath, O_RDONLY);
	    if (src_fd == -1) {
		fprintf(stderr,
			"Thread 0: replicateDB: Cannot open cvm source db\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
	}


	MPI_Bcast(&filesize, sizeof(off_t), MPI_CHAR, 0, replica_comm);
	//theDBSize = filesize;

	if ((filebuf = malloc(FILEBUFSIZE)) == NULL) {
	    fprintf(stderr, "Thread %d: replicateDB: ", Global.myID);
	    fprintf(stderr, "run out of memory while ");
	    fprintf(stderr, "preparing to receive material database\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Everyone opens a replicate db */
	dest_fd = open(dbname, O_CREAT|O_TRUNC|O_WRONLY, S_IRUSR|S_IWUSR);
	if (dest_fd == -1) {
	    fprintf(stderr, "Thread %d: replicateDB: ", Global.myID);
	    fprintf(stderr, "cannot create replica database\n");
	    perror("open");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	remains = filesize;
	while (remains > 0) {
	    batchsize = (remains > FILEBUFSIZE) ? FILEBUFSIZE : remains;

	    if (replica_id == 0) {
		if (read(src_fd, filebuf, batchsize) !=	 batchsize) {
		    fprintf(stderr, "Thread 0: replicateDB: ");
		    fprintf(stderr, "Cannot read database\n");
		    perror("read");
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}
	    }

	    MPI_Bcast(filebuf, batchsize, MPI_CHAR, 0, replica_comm);

	    if (write(dest_fd, filebuf, batchsize) != batchsize) {
		fprintf(stderr, "Thread %d: replicateDB: ", Global.myID);
		fprintf(stderr, "Cannot write replica database\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }

	    remains -= batchsize;
	}

	free(filebuf);

	if (close(dest_fd) != 0) {
	    fprintf(stderr, "Thread %d: replicateDB: ", Global.myID);
	    fprintf(stderr, "cannot close replica database\n");
	    perror("close");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	if (replica_id == 0) {
	    close(src_fd);
	}

	MPI_Comm_free(&replica_comm);

    } /* processors participating in the replication */

#endif /* SCEC */

    return ;
}



/**
 * Assign values (material properties) to a leaf octant specified by
 * octleaf.  In order to refine the mesh, select the minimum Vs of 27
 * sample points: 8 near the corners and 19 midpoints.
 */
static void
setrec( octant_t* leaf, double ticksize, void* data )
{
    double x_m, y_m, z_m;	/* x:south-north, y:east-west, z:depth */
    tick_t halfticks;
    cvmpayload_t g_props;	/* cvm record with ground properties */
    cvmpayload_t g_props_min;	/* cvm record with the min Vs found */

    int i_x, i_y, i_z, n_points = 3;
    double points[3];

    int res = 0;
    edata_t* edata = (edata_t*)data;

    points[0] = 0.01;
    points[1] = 1;
    points[2] = 1.99;

    halfticks = (tick_t)1 << (PIXELLEVEL - leaf->level - 1);
    edata->edgesize = ticksize * halfticks * 2;

    /* Check for buildings and proceed according to the buildings setrec */
    if ( Param.includeBuildings == YES ) {
		if ( bldgs_setrec( leaf, ticksize, edata, Global.theCVMEp,Global.theXForMeshOrigin,Global.theYForMeshOrigin,Global.theZForMeshOrigin ) ) {
            return;
        }
    }

    g_props_min.Vs  = FLT_MAX;
    g_props_min.Vp  = NAN;
    g_props_min.rho = NAN;

    for ( i_x = 0; i_x < n_points; i_x++ ) {

	x_m = (Global.theXForMeshOrigin
	       + (leaf->lx + points[i_x] * halfticks) * ticksize);

	for ( i_y = 0; i_y < n_points; i_y++ ) {

	    y_m  = Global.theYForMeshOrigin
		+ (leaf->ly + points[i_y] * halfticks) * ticksize;

	    for ( i_z = 0; i_z < n_points; i_z++) {

		z_m = Global.theZForMeshOrigin
		    + (leaf->lz +  points[i_z] * halfticks) * ticksize;

		/* Shift the domain if buildings are considered */
		if ( Param.includeBuildings == YES ) {
                    z_m -= get_surface_shift();
		}

		res = cvm_query( Global.theCVMEp, y_m, x_m, z_m, &g_props );

		if (res != 0) {
		    continue;
		}

		if ( g_props.Vs < g_props_min.Vs ) {
		    /* assign minimum value of vs to produce elements
		     * that are small enough to rightly represent the model */
		    g_props_min = g_props;
		}

		if (g_props.Vs <= Param.theVsCut) {
		    /* stop early if needed, completely break out of all
		     * the loops, the label is just outside the loop */
		    goto outer_loop_label;
		}
	    }
	}
    }
 outer_loop_label: /* in order to completely break out from the inner loop */

    edata->Vp  = g_props_min.Vp;
    edata->Vs  = g_props_min.Vs;
    edata->rho = g_props_min.rho;

    if (res != 0 && g_props_min.Vs == DBL_MAX) {
	/* all the queries failed, then center out of bound point. Set Vs
	 * to force split */
	edata->Vs = Param.theFactor * edata->edgesize / 2;
    } else if (edata->Vs <= Param.theVsCut) {	/* adjust Vs and Vp */
	double VpVsRatio = edata->Vp / edata->Vs;

	edata->Vs = Param.theVsCut;
	edata->Vp = Param.theVsCut * VpVsRatio;
    }

    return;
}

#else /* USECVMDB */

static int32_t
zsearch(void *base, int32_t count, int32_t recordsize, const point_t *searchpt)
{
    int32_t start, end, offset, found;

    start = 0;
    end = count - 1;
    offset = (start + end ) / 2;

    found = 0;
    do {
	if (end < start) {
	    /* the two pointer crossed each other */
	    offset = end;
	    found = 1;
	} else {
	    const void *pivot = (char *)base + offset * recordsize;

	    switch (octor_zcompare(searchpt, pivot)) {
	    case (0): /* equal */
		found = 1;
		break;
	    case (1): /* searchpoint larger than the pivot */
		start = offset + 1;
		offset = (start + end) / 2;
		break;
	    case (-1): /* searchpoint smaller than the pivot */
		end = offset - 1;
		offset = (start + end) / 2;
		break;
	    }
	}
    } while (!found);

    return offset;
}


static cvmrecord_t *sliceCVM(const char *cvm_flatfile)
{
    cvmrecord_t *cvmrecord;
    int32_t bufferedbytes, bytecount, recordcount;
    if (Global.myID == Global.theGroupSize - 1) {
	/* the last processor reads data and
	   distribute to other processors*/

	struct timeval starttime, endtime;
	float iotime = 0, memmovetime = 0;
	MPI_Request *isendreqs;
	MPI_Status *isendstats;
	FILE *fp;
	int fd, procid;
	struct stat statbuf;
	void *maxbuf;
	const point_t *intervaltable;
	off_t bytesent;
	int32_t offset;
	const int maxcount =  (1 << 29) / sizeof(cvmrecord_t);
	const int maxbufsize = maxcount * sizeof(cvmrecord_t);

	fp = fopen(cvm_flatfile, "r");
	if (fp == NULL) {
	    fprintf(stderr, "Thread %d: Cannot open flat CVM file\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	fd = fileno(fp);
	if (fstat(fd, &statbuf) != 0) {
	    fprintf(stderr, "Thread %d: Cannot get the status of CVM file\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	intervaltable = octor_getintervaltable(Global.myOctree);

	/*
	for (procid = 0; procid <= Global.myID; procid++) {
	    fprintf(stderr, "interval[%d] = {%d, %d, %d}\n", procid,
		    intervaltable[procid].x << 1, intervaltable[procid].y << 1,
		    intervaltable[procid].z << 1);
	}
	*/

	bytesent = 0;
	maxbuf = malloc(maxbufsize) ;
	if (maxbuf == NULL) {
	    fprintf(stderr, "Thread %d: Cannot allocate send buffer\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	isendreqs = (MPI_Request *)malloc(sizeof(MPI_Request) * Global.theGroupSize);
	isendstats = (MPI_Status *)malloc(sizeof(MPI_Status) * Global.theGroupSize);
	if ((isendreqs == NULL) || (isendstats == NULL)) {
	    fprintf(stderr, "Thread %d: Cannot allocate isend controls\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Try to read max number of CVM records as allowed */
	//	gettimeofday(&starttime, NULL);
	recordcount = fread(maxbuf, sizeof(cvmrecord_t),
			    maxbufsize / sizeof(cvmrecord_t), fp);
	//	gettimeofday(&endtime, NULL);

	iotime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
	    + (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	if (recordcount != maxbufsize / sizeof(cvmrecord_t)) {
	    fprintf(stderr, "Thread %d: Cannot read-init buffer\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* start with proc 0 */
	procid = 0;

	while (procid < Global.myID) { /* repeatedly fill the buffer */
	    point_t searchpoint, *point;
	    int newreads;
	    int isendcount = 0;

	    /* we have recordcount to work with */
	    cvmrecord = (cvmrecord_t *)maxbuf;

	    while (procid < Global.myID) { /* repeatedly send out data */

		searchpoint.x = intervaltable[procid + 1].x << 1;
		searchpoint.y = intervaltable[procid + 1].y << 1;
		searchpoint.z = intervaltable[procid + 1].z << 1;

		offset = zsearch(cvmrecord, recordcount, Global.theCVMRecordSize,
				 &searchpoint);

		point = (point_t *)(cvmrecord + offset);

		if ((point->x != searchpoint.x) ||
		    (point->y != searchpoint.y) ||
		    (point->z != searchpoint.z)) {
		    break;
		} else {
		    bytecount = offset * sizeof(cvmrecord_t);
		    MPI_Isend(cvmrecord, bytecount, MPI_CHAR, procid,
			      CVMRECORD_MSG, comm_solver,
			      &isendreqs[isendcount]);
		    isendcount++;

		    /*
		      fprintf(stderr,
		      "Procid = %d offset = %qd bytecount = %d\n",
		      procid, (int64_t)bytesent, bytecount);
		    */

		    bytesent += bytecount;

		    /* prepare for the next processor */
		    recordcount -= offset;
		    cvmrecord = (cvmrecord_t *)point;
		    procid++;
		}
	    }

	    /* Wait till the data in the buffer has been sent */
	    MPI_Waitall(isendcount, isendreqs, isendstats);

	    /* Move residual data to the beginning of the buffer
	       and try to fill the newly free space */
	    bufferedbytes = sizeof(cvmrecord_t) * recordcount;

	    // gettimeofday(&starttime, NULL);
	    memmove(maxbuf, cvmrecord, bufferedbytes);
	    // gettimeofday(&endtime, NULL);
	    memmovetime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
		+ (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	    // gettimeofday(&starttime, NULL);
	    newreads = fread((char *)maxbuf + bufferedbytes,
			     sizeof(cvmrecord_t), maxcount - recordcount, fp);
	    // gettimeofday(&endtime, NULL);
	    iotime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
		+ (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	    recordcount += newreads;

	    if (newreads == 0)
		break;
	}

	free(maxbuf);
	free(isendreqs);
	free(isendstats);

	/* I am supposed to accomodate the remaining octants */
	bytecount = statbuf.st_size - bytesent;

	cvmrecord = (cvmrecord_t *)malloc(bytecount);
	if (cvmrecord == NULL) {
	    fprintf(stderr, "Thread %d: out of memory\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* fseek exiting the for loop has file cursor propertly */
	if (fseeko(fp, bytesent, SEEK_SET) != 0) {
	    fprintf(stderr, "Thread %d: fseeko failed\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	//	gettimeofday(&starttime, NULL);
	if (fread(cvmrecord, 1, bytecount, fp) != (size_t)bytecount) {
	    fprintf(stderr, "Thread %d: fail to read the last chunk\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
	//	gettimeofday(&endtime, NULL);
	iotime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
	    + (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	/*
	fprintf(stderr, "Procid = %d offset = %qd bytecount = %d\n",
		Global.myID, (int64_t)bytesent, bytecount);
	*/

	fclose(fp);

	fprintf(stdout, "Read %s (%.2fMB) in %.2f seconds (%.2fMB/sec)\n",
		cvm_flatfile, (float)statbuf.st_size / (1 << 20),
		iotime / 1000,
		(float)statbuf.st_size / (1 << 20) / (iotime / 1000));

	fprintf(stdout, "Memmove takes %.2f seconds\n",
		(float)memmovetime / 1000);

    } else {
	/* wait for my turn till PE(n - 1) tells me to go ahead */

	MPI_Status status;

	MPI_Probe(Global.theGroupSize - 1, CVMRECORD_MSG, comm_solver, &status);
	MPI_Get_count(&status, MPI_CHAR, &bytecount);

	cvmrecord = (cvmrecord_t *)malloc(bytecount);
	if (cvmrecord == NULL) {
	    fprintf(stderr, "Thread %d: out of memory\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	MPI_Recv(cvmrecord, bytecount, MPI_CHAR, Global.theGroupSize - 1,
		 CVMRECORD_MSG, comm_solver,	 &status);

    }

    /* Every processor should set these parameters correctly */
    Global.theCVMRecordCount = bytecount / sizeof(cvmrecord_t);
    if (Global.theCVMRecordCount * sizeof(cvmrecord_t) != (size_t)bytecount) {
	fprintf(stderr, "Thread %d: received corrupted CVM data\n",
		Global.myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    return cvmrecord;
}


static cvmrecord_t *sliceCVM_old(const char *cvm_flatfile)
{
    cvmrecord_t *cvmrecord;
    int32_t bufferedbytes, bytecount, recordcount;

    if (Global.myID == Global.theGroupSize - 1) {
	/* the last processor reads data and
	   distribute to other processors*/

	FILE *fp;
	int fd, procid;
	struct stat statbuf;
	void *maxbuf;
	const point_t *intervaltable;
	off_t bytesent;
	int32_t offset;
	const int maxcount =  (1 << 29) / sizeof(cvmrecord_t);
	const int maxbufsize = maxcount * sizeof(cvmrecord_t);

	fp = fopen(cvm_flatfile, "r");
	if (fp == NULL) {
	    fprintf(stderr, "Thread %d: Cannot open flat CVM file\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	fd = fileno(fp);
	if (fstat(fd, &statbuf) != 0) {
	    fprintf(stderr, "Thread %d: Cannot get the status of CVM file\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	intervaltable = octor_getintervaltable(Global.myOctree);
	/*
	for (procid = 0; procid <= Global.myID; procid++) {
	    fprintf(stderr, "interval[%d] = {%d, %d, %d}\n", procid,
		    intervaltable[procid].x << 1, intervaltable[procid].y << 1,
		    intervaltable[procid].z << 1);
	}
	*/

	bytesent = 0;
	maxbuf = malloc(maxbufsize) ;
	if (maxbuf == NULL) {
	    fprintf(stderr, "Thread %d: Cannot allocate send buffer\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Try to read max number of CVM records as allowed */
	recordcount = fread(maxbuf, sizeof(cvmrecord_t),
			    maxbufsize / sizeof(cvmrecord_t), fp);

	if (recordcount != maxbufsize / sizeof(cvmrecord_t)) {
	    fprintf(stderr, "Thread %d: Cannot read-init buffer\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* start with proc 0 */
	procid = 0;

	while (procid < Global.myID) { /* repeatedly fill the buffer */
	    point_t searchpoint, *point;
	    int newreads;

	    /* we have recordcount to work with */
	    cvmrecord = (cvmrecord_t *)maxbuf;

	    while (procid < Global.myID) { /* repeatedly send out data */
		searchpoint.x = intervaltable[procid + 1].x << 1;
		searchpoint.y = intervaltable[procid + 1].y << 1;
		searchpoint.z = intervaltable[procid + 1].z << 1;

		offset = zsearch(cvmrecord, recordcount, Global.theCVMRecordSize,
				 &searchpoint);

		point = (point_t *)(cvmrecord + offset);

		if ((point->x != searchpoint.x) ||
		    (point->y != searchpoint.y) ||
		    (point->z != searchpoint.z)) {
		    break;
		} else {
		    bytecount = offset * sizeof(cvmrecord_t);
		    MPI_Send(cvmrecord, bytecount, MPI_CHAR, procid,
			     CVMRECORD_MSG, comm_solver);
		    /*
		    fprintf(stderr,
			    "Procid = %d offset = %qd bytecount = %d\n",
			    procid, (int64_t)bytesent, bytecount);
		    */

		    bytesent += bytecount;

		    /* prepare for the next processor */
		    recordcount -= offset;
		    cvmrecord = (cvmrecord_t *)point;
		    procid++;
		}
	    }

	    /* Move residual data to the beginning of the buffer
	       and try to fill the newly free space */
	    bufferedbytes = sizeof(cvmrecord_t) * recordcount;
	    memmove(maxbuf, cvmrecord, bufferedbytes);
	    newreads = fread((char *)maxbuf + bufferedbytes,
			     sizeof(cvmrecord_t), maxcount - recordcount, fp);
	    recordcount += newreads;

	    if (newreads == 0)
		break;
	}

	free(maxbuf);

	/* I am supposed to accomodate the remaining octants */
	bytecount = statbuf.st_size - bytesent;

	cvmrecord = (cvmrecord_t *)malloc(bytecount);
	if (cvmrecord == NULL) {
	    fprintf(stderr, "Thread %d: out of memory for %d bytes\n",
		    Global.myID, bytecount);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* fseek exiting the for loop has file cursor propertly */
	if (fseeko(fp, bytesent, SEEK_SET) != 0) {
	    fprintf(stderr, "Thread %d: fseeko failed\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	if (fread(cvmrecord, 1, bytecount, fp) != (size_t)bytecount) {
	    fprintf(stderr, "Thread %d: fail to read the last chunk\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/*
	  fprintf(stderr, "Procid = %d offset = %qd bytecount = %d\n",
	  Global.myID, (int64_t)bytesent, bytecount);
	*/

	fclose(fp);

    } else {
	/* wait for my turn till PE(n - 1) tells me to go ahead */

	MPI_Status status;

	MPI_Probe(Global.theGroupSize - 1, CVMRECORD_MSG, comm_solver, &status);
	MPI_Get_count(&status, MPI_CHAR, &bytecount);

	cvmrecord = (cvmrecord_t *)malloc(bytecount);
	if (cvmrecord == NULL) {
	    fprintf(stderr, "Thread %d: out of memory\n", Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	MPI_Recv(cvmrecord, bytecount, MPI_CHAR, Global.theGroupSize - 1,
		 CVMRECORD_MSG, comm_solver,	 &status);

    }

    /* Every processor should set these parameters correctly */
    Global.theCVMRecordCount = bytecount / sizeof(cvmrecord_t);
    if (Global.theCVMRecordCount * sizeof(cvmrecord_t) != (size_t)bytecount) {
	fprintf(stderr, "Thread %d: received corrupted CVM data\n",
		Global.myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    return cvmrecord;
}



/**
 * setrec: Search the CVM record array to obtain the material property of
 *	   a leaf octant.
 *
 */
void setrec(octant_t *leaf, double ticksize, void *data)
{
    cvmrecord_t *agghit;
    edata_t *edata;
    etree_tick_t x, y, z;
    etree_tick_t halfticks;
    point_t searchpoint;

    edata = (edata_t *)data;

    halfticks = (tick_t)1 << (PIXELLEVEL - leaf->level - 1);

    edata->edgesize = ticksize * halfticks * 2;

    searchpoint.x = x = leaf->lx + halfticks;
    searchpoint.y = y = leaf->ly + halfticks;
    searchpoint.z = z = leaf->lz + halfticks;

    if ((x * ticksize >= Param.theDomainX) ||
	(y * ticksize >= Param.theDomainY) ||
	(z * ticksize >= Param.theDomainZ)) {
	/* Center point out the bound. Set Vs to force split */
	edata->Vs = Param.theFactor * edata->edgesize / 2;
    } else {
	int offset;

	/* map the coordinate from the octor address space to the
	   etree address space */
	searchpoint.x = x << 1;
	searchpoint.y = y << 1;
	searchpoint.z = z << 1;

	/* Inbound */
	offset = zsearch(Global.theCVMRecord, Global.theCVMRecordCount, Global.theCVMRecordSize,
			 &searchpoint);
	if (offset < 0) {
	    fprintf(stderr, "setrec: fatal error\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	agghit = Global.theCVMRecord + offset;
	edata->Vs = agghit->Vs;
	edata->Vp = agghit->Vp;
	edata->rho = agghit->density;

	/* Adjust the Vs */
	edata->Vs = (edata->Vs < Param.theVsCut) ? Param.theVsCut : edata->Vs;
    }

    return;
}
#endif	/* USECVMDB */


/**
 * mesh_generate: Generate and partition an unstructured octree mesh.
 *
 */
static void
mesh_generate()
{

    int mstep, step = 1;
    double originalFactor = Param.theFactor;
    double ppwl = Param.theFactor / Param.theFreq;
    double prevtref = 0, prevtbal = 0, prevtpar = 0;
    int64_t tote, mine, maxe;

    if (Global.myID == 0) {
        fprintf(stdout, "Meshing: ");
        if (Param.theStepMeshingFactor == 0) {
            fprintf(stdout, "Conventional\n\n");
        } else {
            fprintf(stdout, "Progressive\n\n");
        }
        fprintf(stdout, "Stage %14s Min %7s Max %5s Total    Time(s)","","","");
        if (Param.theStepMeshingFactor == 0) {
            fprintf(stdout, "\n\n");
        } else {
            fprintf(stdout, "   Step  f(Hz)\n\n");
        }
    }

    /*----  Generate and partition an unstructured octree mesh ----*/
    MPI_Barrier(comm_solver);
    Timer_Start("Octor Newtree");
    if (Global.myID == 0) {
        fprintf(stdout, "New tree %41s","");
    }
    Global.myOctree = octor_newtree( Param.theDomainX, Param.theDomainY, Param.theDomainZ,
            sizeof(edata_t), Global.myID, Global.theGroupSize,
            comm_solver, get_surface_shift());

    /* NOTE:
     * If you want to see the carving process, replace by:
     *     Global.myOctree = octor_newtree(
     *             Param.theDomainX, Param.theDomainY, Param.theDomainZ,
     *             sizeof(edata_t), Global.myID, Global.theGroupSize,
     *             comm_solver, 0);
     */

    if (Global.myOctree == NULL) {
        fprintf(stderr, "Thread %d: mesh_generate: fail to create octree\n",
                Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    MPI_Barrier(comm_solver);
    Timer_Stop("Octor Newtree");
    if (Global.myID == 0) {
        fprintf(stdout, "%9.2f\n\n", Timer_Value("Octor Newtree", (TimerKind)0) );
    }

    /* Essential for DRM implementation */
    if (Param.drmImplement == YES) {
        drm_fix_coordinates(Global.myOctree->ticksize);
    }

#ifdef USECVMDB
    Global.theCVMQueryStage = 0; /* Query CVM database to refine the mesh */
#else
     /* Use flat data record file and distibute the data in memories */
    if (Global.myID == 0) {
	fprintf(stdout, "slicing CVMDB ...");
    }
    Timer_Start("Slice CVM");
    Global.theCVMRecord = sliceCVM(Param.theCVMFlatFile);
    MPI_Barrier(comm_solver);
    Timer_Stop("Slice CVM");
    if (Global.theCVMRecord == NULL) {
	fprintf(stderr, "Thread %d: Error obtaining the CVM records from %s\n",
		Global.myID, Param.theCVMFlatFile);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    };
    if (Global.myID == 0) {
	fprintf(stdout, "done : %9.2f seconds\n", Timer_Value("Slice CVM", (TimerKind)0));
    }
#endif

    for ( mstep = Param.theStepMeshingFactor; mstep >= 0; mstep-- ) {

        double myFactor = (double)(1 << mstep); // 2^mstep
        Param.theFactor = originalFactor / myFactor;

        /* Refinement */
        Timer_Start("Octor Refinetree");
        if (Global.myID == 0) {
            fprintf(stdout, "Refining     ");
            fflush(stdout);
        }
        if (octor_refinetree(Global.myOctree, toexpand, setrec) != 0) {
            fprintf(stderr, "Thread %d: mesh_generate: fail to refine octree\n",Global.myID);
            MPI_Abort(MPI_COMM_WORLD, ERROR); exit(1);
        }
        MPI_Barrier(comm_solver);
        tote = octor_getleavescount(Global.myOctree, GLOBAL);
        mine = octor_getminleavescount(Global.myOctree, GLOBAL);
        maxe = octor_getmaxleavescount(Global.myOctree, GLOBAL);
        if (Global.myID == 0) {
            fprintf(stdout, "%11"INT64_FMT" %11"INT64_FMT" %11"INT64_FMT, mine, maxe, tote);
            fflush(stdout);
        }
        Timer_Stop("Octor Refinetree");
        if (Global.myID == 0) {
            fprintf(stdout, "%11.2f", Timer_Value("Octor Refinetree", (TimerKind)0) - prevtref);
            if (Param.theStepMeshingFactor == 0 ) {
                fprintf(stdout, "\n");
            } else {
                fprintf(stdout, "   %4d %6.2f\n", step, Param.theFactor/ppwl);
            }
            prevtref = Timer_Value("Octor Refinetree", (TimerKind)0);
            fflush(stdout);
        }

        /* Balancing */
        Timer_Start("Octor Balancetree");
        if (Global.myID == 0) {
            fprintf(stdout, "Balancing    ");
            fflush(stdout);
        }
        if (octor_balancetree(Global.myOctree, setrec, Param.theStepMeshingFactor) != 0) {
            fprintf(stderr, "Thread %d: mesh_generate: fail to balance octree\n",Global.myID);
            MPI_Abort(MPI_COMM_WORLD, ERROR); exit(1);
        }
        MPI_Barrier(comm_solver);
        tote = octor_getleavescount(Global.myOctree, GLOBAL);
        mine = octor_getminleavescount(Global.myOctree, GLOBAL);
        maxe = octor_getmaxleavescount(Global.myOctree, GLOBAL);
        if (Global.myID == 0) {
            fprintf(stdout, "%11"INT64_FMT" %11"INT64_FMT" %11"INT64_FMT, mine, maxe, tote);
            fflush(stdout);
        }
        Timer_Stop("Octor Balancetree");
        if (Global.myID == 0) {
            fprintf(stdout, "%11.2f\n", Timer_Value("Octor Balancetree", (TimerKind)0) - prevtbal);
            prevtbal = Timer_Value("Octor Balancetree", (TimerKind)0);
            fflush(stdout);
        }

        /* Partitioning */
        Timer_Start("Octor Partitiontree");
        if (Global.myID == 0) {
            fprintf(stdout, "Partitioning ");
            fflush(stdout);
        }
        if (octor_partitiontree(Global.myOctree, bldgs_nodesearch) != 0) {
            fprintf(stderr, "Thread %d: mesh_generate: fail to balance load\n",Global.myID);
            MPI_Abort(MPI_COMM_WORLD, ERROR); exit(1);
        }
        MPI_Barrier(comm_solver);
        tote = octor_getleavescount(Global.myOctree, GLOBAL);
        mine = octor_getminleavescount(Global.myOctree, GLOBAL);
        maxe = octor_getmaxleavescount(Global.myOctree, GLOBAL);
        if (Global.myID == 0) {
            fprintf(stdout, "%11"INT64_FMT" %11"INT64_FMT" %11"INT64_FMT, mine, maxe, tote);
            fflush(stdout);
        }
        Timer_Stop("Octor Partitiontree");
        if (Global.myID == 0) {
            fprintf(stdout, "%11.2f\n\n", Timer_Value("Octor Partitiontree", (TimerKind)0) - prevtpar);
            prevtpar = Timer_Value("Octor Partitiontree", (TimerKind)0);
            fflush(stdout);
        }

        step++;
        fflush(stdout);
        MPI_Barrier(comm_solver);
    }

    /* Buildings Carving */
    if ( Param.includeBuildings == YES ) {

        Timer_Start("Carve Buildings");
        if (Global.myID == 0) {
            fprintf(stdout, "Carving buildings");
            fflush(stdout);
        }

        /* NOTE: If you want to see the carving process, comment next line */
        octor_carvebuildings(Global.myOctree, 1, bldgs_nodesearch);
        MPI_Barrier(comm_solver);
        Timer_Stop("Carve Buildings");
        if (Global.myID == 0) {
	  fprintf(stdout, "%9.2f\n", Timer_Value("Carve Buildings", (TimerKind)0) );
            fflush(stdout);
        }

        Timer_Start("Octor Partitiontree");
        if (Global.myID == 0) {
            fprintf(stdout, "Repartitioning");
            fflush(stdout);
        }
        if (octor_partitiontree(Global.myOctree, bldgs_nodesearch) != 0) {
            fprintf(stderr, "Thread %d: mesh_generate: fail to balance load\n",
                    Global.myID);
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }
        MPI_Barrier(comm_solver);
        Timer_Stop("Octor Partitiontree");
        if (Global.myID == 0) {
            fprintf(stdout, "%9.2f\n", Timer_Value("Octor Partitiontree", (TimerKind)0));
            fflush(stdout);
        }
    }

    if ( Global.myID == 0 && Param.theStepMeshingFactor !=0 ) {
        fprintf(stdout, "Total refine    %33s %9.2f\n", "", Timer_Value("Octor Refinetree", (TimerKind)0));
        fprintf(stdout, "Total balance   %33s %9.2f\n", "", Timer_Value("Octor Balancetree", (TimerKind)0));
        fprintf(stdout, "Total partition %33s %9.2f\n\n", "", Timer_Value("Octor Partitiontree", (TimerKind)0));
        fflush(stdout);
    }

    Timer_Start("Octor Extractmesh");
    if (Global.myID == 0) {
        fprintf(stdout, "Extracting the mesh %30s","");
        fflush(stdout);
    }
    Global.myMesh = octor_extractmesh(Global.myOctree, bldgs_nodesearch);
    if (Global.myMesh == NULL) {
        fprintf(stderr, "Thread %d: mesh_generate: fail to extract mesh\n",
                Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    MPI_Barrier(comm_solver);
    Timer_Stop("Octor Extractmesh");
    if (Global.myID == 0) {
        fprintf(stdout, "%9.2f\n", Timer_Value("Octor Partitiontree", (TimerKind)0));
    }

    Timer_Start( "Mesh correct properties" );
    /* Re-populates the mesh with actual values from the CVM-etree */
    if (Global.myID == 0) {
        fprintf(stdout,"Correcting mesh properties %23s","");
        fflush(stdout);
    }

    mesh_correct_properties( Global.theCVMEp );

    MPI_Barrier( comm_solver );
    Timer_Stop( "Mesh correct properties" );
    if (Global.myID == 0) {
        fprintf(stdout, "%9.2f\n\n",Timer_Value( "Mesh correct properties", (TimerKind)0 ) );
        fflush(stdout);
    }

#ifdef USECVMDB
    /* Close the material database */
    etree_close(Global.theCVMEp);
#else
    free(Global.theCVMRecord);
#endif /* USECVMDB */
}


/**
 * toexpand: Instruct the Octor library whether a leaf octant needs to
 *	     be expanded or not. Return 1 if true, 0 otherwise.
 *
 */
static int32_t
toexpand(octant_t *leaf, double ticksize, const void *data) {

	if ( data == NULL ) {
		return 1;
	}

	int      res;
	edata_t *edata = (edata_t *)data;

	if ( Param.includeBuildings == YES ) {
		res = bldgs_toexpand( leaf, ticksize, edata, Param.theFactor );
		if ( res != -1 ) {
			return res;
		}
	}

	if ( Param.drmImplement == YES) {
		//if( Param.drmImplement == YES && Param.theDrmPart != PART1 ) {
		res = drm_toexpand( leaf, ticksize, edata );
		if ( res != -1 ) {
			return res;
		}
	}

	return vsrule( edata, Param.theFactor );
}

/**
 * bulkload: Append the data to the end of the mesh database. Return 0 if OK,
 *	     -1 on error.
 *
 */
static int32_t
bulkload(etree_t *mep, mrecord_t *partTable, int32_t count)
{
    int index;

    for (index = 0; index < count; index++) {
	void *payload = &partTable[index].mdata;

	if (etree_append(mep, partTable[index].addr, payload) != 0) {
	    /* Append error */
	    return -1;
	}
    }

    return 0;
}


/** Enumeration of the counters used in the mesh statistics */
enum mesh_stat_count_t {
    ELEMENT_COUNT, NODE_COUNT, DANGLING_COUNT, HARBORED_COUNT, MESH_COUNT_LENGTH
};



static void
mesh_print_stat_imp( int32_t* st, int group_size, FILE* out )
{
    int pid;
    global_id_t total_count[MESH_COUNT_LENGTH] = { 0, 0, 0, 0 };

    fputs( "\n"
	   "# ------------------------------------------------------------\n"
	   "# Mesh statistics:\n"
	   "# ------------------------------------------------------------\n"
	   "# Rank    Elements       Nodes     D-nodes     H-nodes\n", out );

    for (pid = 0; pid < group_size; pid++) {
	fprintf( out, "%06d %11d %11d %11d %11d\n", pid, st[ELEMENT_COUNT],
		 st[NODE_COUNT], st[DANGLING_COUNT], st[HARBORED_COUNT] );

	/* add to total count */
	total_count[ELEMENT_COUNT]  += st[ELEMENT_COUNT];
	total_count[NODE_COUNT]	    += st[NODE_COUNT];
	total_count[DANGLING_COUNT] += st[DANGLING_COUNT];
	total_count[HARBORED_COUNT] += st[HARBORED_COUNT];

	st += MESH_COUNT_LENGTH;	/* move to next row */
    }

    fputs( "\n\n"
	    "# ------------------------------------------------------------\n"
	   "# Total\n"
	   "# ------------------------------------------------------------\n",
	   out );

    fprintf( out, "       %11"INT64_FMT" %11"INT64_FMT" %11"INT64_FMT
	     " %11"INT64_FMT"\n\n",
	     total_count[ELEMENT_COUNT], total_count[NODE_COUNT],
	     total_count[DANGLING_COUNT], total_count[HARBORED_COUNT] );

    /* copy totals to static globals */
    /* TODO this should be computed through different means */
    Global.theETotal  = total_count[ELEMENT_COUNT];
    Global.theNTotal  = total_count[NODE_COUNT];


    /* output aggregate information to the monitor file / stdout */
    monitor_print(
		   "Total elements:                      %11"INT64_FMT"\n"
		   "Total nodes:                         %11"INT64_FMT"\n"
		   "Total dangling nodes:                %11"INT64_FMT"\n\n",
		   total_count[ELEMENT_COUNT], total_count[NODE_COUNT],
		   total_count[DANGLING_COUNT] );
}


static int
mesh_collect_print_stats( local_id_t mesh_stat[MESH_COUNT_LENGTH], int my_id,
			  int group_size, const char* fname )
{
    local_id_t* st = NULL;

    if (0 == my_id) { /* only the root node allocates memory */
	XMALLOC_VAR_N( st, local_id_t, (group_size * MESH_COUNT_LENGTH) );
    }

    MPI_Gather( mesh_stat, MESH_COUNT_LENGTH, MPI_INT,
		st,        MESH_COUNT_LENGTH, MPI_INT, 0, comm_solver );

    if (0 == my_id) { /* the root node prints the stats */
        const size_t bufsize = 1048576;  // 1MB
	FILE* out = hu_fopen( Param.theMeshStatFilename, "w" );

	setvbuf( out, NULL, _IOFBF, bufsize );
	mesh_print_stat_imp( st, group_size, out );
	hu_fclosep( &out );
	xfree_int32_t( &st );
    }

    return 0;
}


static void
mesh_printstat_imp( const mesh_t* mesh, int my_id, int group_size,
		    const char* fname )
{
    local_id_t mesh_stat[MESH_COUNT_LENGTH];

    mesh_stat[ ELEMENT_COUNT  ] = mesh->lenum;
    mesh_stat[ NODE_COUNT     ] = mesh->lnnum;
    mesh_stat[ DANGLING_COUNT ] = mesh->ldnnum;
    mesh_stat[ HARBORED_COUNT ] = mesh->nharbored;

    mesh_collect_print_stats( mesh_stat, my_id, group_size, fname );
}


/**
 * Gather and print mesh statistics to a file with the given name.
 *
 * \param fname Name of the file where the statistics should be stored.
 */
static void
mesh_print_stat( const octree_t* oct, const mesh_t* mesh, int my_id,
		 int group_size, const char* fname )
{
    /* collective function calls */
    int32_t gmin = octor_getminleaflevel( oct, GLOBAL );
    int32_t gmax = octor_getmaxleaflevel( oct, GLOBAL );

    mesh_printstat_imp( mesh, my_id, group_size, fname );

    if (Global.myID == 0) {
	monitor_print( "Maximum leaf level = %d\n", gmax );
	monitor_print( "Minimum leaf level = %d\n", gmin );
    }
}


/**
 * Join elements and nodes, and send to Thread 0 for output.
 */
static void
mesh_output()
{
    int32_t eindex;
    int32_t remains, batch, batchlimit, idx;
    mrecord_t *partTable;

    Timer_Start("Mesh Out");

    batchlimit = BATCH;

    /* Allocate a fixed size buffer space to store the join results */
    partTable = (mrecord_t *)calloc(batchlimit, sizeof(mrecord_t));
    if (partTable == NULL) {
	fprintf(stderr,	 "Thread %d: mesh_output: out of memory\n", Global.myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    if (Global.myID == 0) {
	etree_t *mep;
	int32_t procid;

	printf("mesh_output ... ");

	mep = etree_open(Param.mesh_etree_output_file, O_CREAT|O_RDWR|O_TRUNC, 0,
			 sizeof(mdata_t),3);
	if (mep == NULL) {
	    fprintf(stderr, "Thread 0: mesh_output: ");
	    fprintf(stderr, "cannot create mesh etree\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Begin an appending operation */
	if (etree_beginappend(mep, 1) != 0) {
	    fprintf(stderr, "Thread 0: mesh_output: \n");
	    fprintf(stderr, "cannot begin an append operation\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	eindex = 0;
	while (eindex < Global.myMesh->lenum) {
	    remains = Global.myMesh->lenum - eindex;
	    batch = (remains < batchlimit) ? remains : batchlimit;

	    for (idx = 0; idx < batch; idx++) {
		mrecord_t *mrecord;
		int32_t whichnode;
		int32_t localnid0;

		mrecord = &partTable[idx];

		/* Fill the address field */
		localnid0 = Global.myMesh->elemTable[eindex].lnid[0];

		mrecord->addr.x = Global.myMesh->nodeTable[localnid0].x;
		mrecord->addr.y = Global.myMesh->nodeTable[localnid0].y;
		mrecord->addr.z = Global.myMesh->nodeTable[localnid0].z;
		mrecord->addr.level = Global.myMesh->elemTable[eindex].level;
		mrecord->addr.type = ETREE_LEAF;

		/* Find the global node ids for the vertices */
		for (whichnode = 0; whichnode < 8; whichnode++) {
		    int32_t localnid;
		    int64_t globalnid;

		    localnid = Global.myMesh->elemTable[eindex].lnid[whichnode];
		    globalnid = Global.myMesh->nodeTable[localnid].gnid;

		    mrecord->mdata.nid[whichnode] = globalnid;
		}

		/* data points to mdata_t type */
		memcpy(&mrecord->mdata.edgesize,
		       Global.myMesh->elemTable[eindex].data,
		       sizeof(edata_t));

		eindex++;
	    } /* for a batch */

	    if (bulkload(mep, partTable, batch) != 0) {
		fprintf(stderr, "Thread 0: mesh_output: ");
		fprintf(stderr, "error bulk-loading data\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
	} /* for all the elements Thread 0 has */

	/* Receive data from other processors */
	for (procid = 1; procid < Global.theGroupSize; procid++) {
	    MPI_Status status;
	    int32_t rcvbytecount;

	    /* Signal the next processor to go ahead */
	    MPI_Send(NULL, 0, MPI_CHAR, procid, GOAHEAD_MSG, comm_solver);

	    while (1) {
		MPI_Probe(procid, MESH_MSG, comm_solver, &status);
		MPI_Get_count(&status, MPI_CHAR, &rcvbytecount);

		batch = rcvbytecount / sizeof(mrecord_t);

		MPI_Recv(partTable, rcvbytecount, MPI_CHAR, procid,
			 MESH_MSG, comm_solver, &status);

		if (batch == 0) {
		    /* Done */
		    break;
		}

		if (bulkload(mep, partTable, batch) != 0) {
		    fprintf(stderr, "Thread 0: mesh_output: ");
		    fprintf(stderr, "cannot bulkloading data from ");
		    fprintf(stderr, "Thread %d\n", procid);
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}
	    } /* while there is more data to be received from procid */
	} /* for all the processors */

	/* End the appending operation */
	etree_endappend(mep);

	/* Close the mep to ensure the data is on disk */
	if (etree_close(mep) != 0) {
	    fprintf(stderr, "Thread 0: mesh_output ");
	    fprintf(stderr, "error closing the etree database\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

    } else {
	/* Processors other than 0 needs to send data to 0 */
	int32_t sndbytecount;
	MPI_Status status;

	/* Wait for me turn */
	MPI_Recv(NULL, 0, MPI_CHAR, 0, GOAHEAD_MSG, comm_solver, &status);

	eindex = 0;
	while (eindex < Global.myMesh->lenum) {
	    remains = Global.myMesh->lenum - eindex;
	    batch = (remains < batchlimit) ? remains : batchlimit;

	    for (idx = 0; idx < batch; idx++) {
		mrecord_t *mrecord;
		int32_t whichnode;
		int32_t localnid0;

		mrecord = &partTable[idx];

		/* Fill the address field */
		localnid0 = Global.myMesh->elemTable[eindex].lnid[0];

		mrecord->addr.x = Global.myMesh->nodeTable[localnid0].x;
		mrecord->addr.y = Global.myMesh->nodeTable[localnid0].y;
		mrecord->addr.z = Global.myMesh->nodeTable[localnid0].z;
		mrecord->addr.level = Global.myMesh->elemTable[eindex].level;
		mrecord->addr.type = ETREE_LEAF;

		/* Find the global node ids for the vertices */
		for (whichnode = 0; whichnode < 8; whichnode++) {
		    int32_t localnid;
		    int64_t globalnid;

		    localnid = Global.myMesh->elemTable[eindex].lnid[whichnode];
		    globalnid = Global.myMesh->nodeTable[localnid].gnid;

		    mrecord->mdata.nid[whichnode] = globalnid;
		}

		memcpy(&mrecord->mdata.edgesize,
		       Global.myMesh->elemTable[eindex].data,
		       sizeof(edata_t));

		eindex++;
	    } /* for a batch */


	    /* Send data to proc 0 */
	    sndbytecount = batch * sizeof(mrecord_t);
	    MPI_Send(partTable, sndbytecount, MPI_CHAR, 0, MESH_MSG,
		     comm_solver);
	} /* While there is data left to be sent */

	/* Send an empty message to indicate the end of my transfer */
	MPI_Send(NULL, 0, MPI_CHAR, 0, MESH_MSG, comm_solver);
    }

    /* Free the memory for the partial join results */
    free(partTable);

    Timer_Stop("Mesh Out");

    if (Global.myID == 0) {
	printf("done : %9.2f seconds\n", Timer_Value("Mesh Out", (TimerKind)0) );
    }

    return;
}


/*-----------Computation routines ------------------------------*/

/*
   Macros to facitilate computation

   INTEGRAL_1: Integral_Delfixl_Delfjxl()
   INTEGRAL_2: Integral_Delfixl_Delfjxm()
 */

#define INTEGRAL_1(xki, xkj, xli, xlj, xmi, xmj) \
(4.5 * xki * xkj * (1 + xli * xlj / 3) * (1 + xmi * xmj / 3) / 8)

#define INTEGRAL_2(xki, xlj, xmi, xmj) \
(4.5 * xki * xlj * (1 + xmi * xmj / 3) / 8)

#define DS_TOTAL_INTERVALS	    40
#define DS_TOTAL_PARAMETERS	     6

/**
 * Compute histograms for xi, zeta and associated values to understand
 * what is happenning with those parameters involved the damping and
 * delta T.
 */
static void
damping_statistics (
	double min_xi,
	double max_xi,
	double min_zeta,
	double max_zeta,
	double min_VsVp,
	double max_VsVp,
	double min_VpVsZ,
	double max_VpVsZ,
	double min_Vs,
	double max_Vs
	)
{
    static const int totalintervals  = DS_TOTAL_INTERVALS;
    static const int totalparameters = DS_TOTAL_PARAMETERS;

    int interval, parameter, row, col, matrixelements;

    double  min_VpVs, max_VpVs;

    double  themins[DS_TOTAL_PARAMETERS],
	    themaxs[DS_TOTAL_PARAMETERS],
	    spacing[DS_TOTAL_PARAMETERS];

    int32_t counters[DS_TOTAL_PARAMETERS][DS_TOTAL_INTERVALS],
	    global_counter[DS_TOTAL_PARAMETERS][DS_TOTAL_INTERVALS],
	    global_total[DS_TOTAL_PARAMETERS],
	    eindex;

    /* Initializing clue values and variables */

    min_VpVs = 1 / max_VsVp;
    max_VpVs = 1 / min_VsVp;

    themins[0] = min_zeta;
    themins[1] = min_xi;
    themins[2] = min_VsVp;
    themins[3] = min_VpVs;
    themins[4] = min_VpVsZ;
    themins[5] = min_Vs;

    themaxs[0] = max_zeta;
    themaxs[1] = max_xi;
    themaxs[2] = max_VsVp;
    themaxs[3] = max_VpVs;
    themaxs[4] = max_VpVsZ;
    themaxs[5] = max_Vs;

    for ( row = 0; row < totalparameters; row++ )
    {
	for ( col = 0; col < totalintervals; col++ )
	{
	    counters[row][col] = 0;
	}
	global_total[row] = 0;
    }

    for ( row = 0; row < totalparameters; row++ )
    {
	spacing[row] = ( themaxs[row] - themins[row] ) / totalintervals;
    }

    /* loop over the elements */
    for ( eindex = 0; eindex < Global.myMesh->lenum; eindex++)
    {
	/* loop variables */
	elem_t	*elemp;
	edata_t *edata;
	double	 a, b,
		 omega,
		 elementvalues[6];

	/* capturing the elements */
	elemp = &Global.myMesh->elemTable[eindex];
	edata = (edata_t *)elemp->data;

	/* the parameteres */
	elementvalues[0] = 10 / edata->Vs;
	  /* (edata->Vs < 1500) ? 25 / edata->Vs : 5 / edata->Vs; */
	  /* zeta */
	omega = 3.46410161514 / ( edata->edgesize / edata->Vp );     /* freq in rad */
	a = elementvalues[0] * Global.theABase;			    /* a     */
	b = elementvalues[0] * Global.theBBase;			    /* b     */
	elementvalues[1] = ( a / (2 * omega)) + ( b * omega / 2 );  /* xi    */
	elementvalues[2] = ( edata->Vs / edata->Vp);		    /* Vs/Vp */
	elementvalues[3] = edata->Vp / edata->Vs;		    /* Vp/Vs */
	elementvalues[4] = elementvalues[0] * ( edata->Vp / edata->Vs );
	/* Vp/Vs*zeta  */
	elementvalues[5] = edata->Vs;				    /* Vs    */

	/* loop over the parameters */
	for ( parameter = 0; parameter < totalparameters; parameter++ )
	{
	    /* loop over each interval */
	    for ( interval = 0; interval < totalintervals; interval++)
	    {
		/* loop variables */
		double liminf, limsup;

		/* histogram limits */
		liminf = themins[parameter] + (interval * spacing[parameter]);
		limsup = liminf + spacing[parameter];

		/* for the last interval adjust to the max value */
		if ( interval == totalintervals-1 )
		{
		    limsup = themaxs[parameter];
		}

		/* counting elements within the interval */
		if ( ( elementvalues[parameter] >  liminf ) &&
		     ( elementvalues[parameter] <= limsup ) )
		{
		    counters[parameter][interval]++;
		}
	    } /* ends loop on intervals */
	} /* ends loop on parameters */
    } /*ends loop on elements */

    /* add all counting results from each processor */
    matrixelements = totalparameters * totalintervals;
    MPI_Reduce (&counters[0][0], &global_counter[0][0], matrixelements,
		MPI_INT, MPI_SUM, 0, comm_solver);
    MPI_Bcast (&global_counter[0][0], matrixelements, MPI_INT,0,comm_solver);

    /* sums the total of elements for each histogram */
    if (Global.myID == 0)
    {
	for ( parameter = 0; parameter < totalparameters; parameter++)
	{
	    global_counter[parameter][0]++;
	    for ( interval = 0; interval < totalintervals; interval++)
	    {
		global_total[parameter] = global_total[parameter]
		    + global_counter[parameter][interval];
	    }
	}
    }

    /* MPI Barrier */
    MPI_Barrier( comm_solver );

    /* prints to the terminal the histograms */
    if (Global.myID == 0)
    {
	/* header to identify each column */
	printf("\n\n\tThe histograms of the following parameters: \n\n");
	printf("\t 1. Zeta\n");
	printf("\t 2. Xi\n");
	printf("\t 3. Vs/Vp\n");
	printf("\t 4. Vp/Vs\n");
	printf("\t 5. Vp/Vs*zeta\n");
	printf("\t 6. Vs\n\n");
	printf("\tAre given in the following table\n");
	printf("\t(each column is one of the parameters)\n\n\t");

	/* printing the histograms */
	for ( interval = 0; interval < totalintervals; interval++)
	{
	    for ( parameter = 0; parameter < totalparameters; parameter++)
	    {
		printf("%12d",global_counter[parameter][interval]);
	    }
	    printf("\n\t");
	}

	/* prints the total of elements for each column */
	printf("\n\tTotals:\n\n\t");
	for ( parameter = 0; parameter < totalparameters; parameter++)
	{
	    printf("%12d",global_total[parameter]);
	}

	/* prints the interval witdth */
	printf("\n\n\tAnd the intervals width is:");
	for ( parameter = 0; parameter < totalparameters; parameter++)
	{
	    printf("\n\t %2d. %.6f ",parameter+1,spacing[parameter]);
	}
	printf ("\n\n");
	fflush (stdout);
    }

    return;
} /* end damping_statistics */


/**
 * Determine the limit values associated with the damping and delta_t problem.
 */
static void solver_set_critical_T()
{
    int32_t eindex;			/* element index	 */

    double  min_h_over_Vp = 1e32;	/* the min h/Vp group	 */
    double  min_h_over_Vp_global;
    int32_t min_h_over_Vp_elem_index = -1;

    double  min_dt_factor_X = 1e32;	/* the min delta_t group */
    double  min_dt_factor_Z = 1e32,
	    min_dt_factor_X_global,
	    min_dt_factor_Z_global;
    int32_t min_dt_factor_X_elem_index = -1,
	    min_dt_factor_Z_elem_index = -1;

    double  min_zeta = 1e32;		/* the zeta group	 */
    double  max_zeta = 0,
	    min_zeta_global,
	    max_zeta_global;
    int32_t min_zeta_elem_index = -1,
	    max_zeta_elem_index = -1;

    double  min_xi = 1e32;		/* the xi group		 */
    double  max_xi = 0,
	    min_xi_global,
	    max_xi_global;
    int32_t min_xi_elem_index = -1,
	    max_xi_elem_index = -1;

    double  min_VsVp = 1e32;		/* the Vs/Vp group	 */
    double  min_VsVp_global,
	    max_VsVp = 0,
	    max_VsVp_global;
    int32_t min_VsVp_elem_index = -1,
	    max_VsVp_elem_index = -1;

    double  min_VpVsZ = 1e32;		/* the Vp/Vs group	 */
    double  min_VpVsZ_global,
	    max_VpVsZ = 0,
	    max_VpVsZ_global;
    int32_t min_VpVsZ_elem_index = -1,
	    max_VpVsZ_elem_index = -1;

    double  min_Vs = 1e32;		/* the Vs group		 */
    double  min_Vs_global,
	    max_Vs = 0,
	    max_Vs_global;
    int32_t min_Vs_elem_index = -1,
	    max_Vs_elem_index = -1;

    /* Find the minima and maxima for all needed coefficients */
    /* Start loop over the mesh elements */
    for (eindex = 0; eindex < Global.myMesh->lenum; eindex++)
    {
	/* Loop local variables */

	elem_t	*elemp;	      /* pointer to the mesh database		      */
	edata_t *edata;	      /* pointer to the element data		      */

	double	 ratio;	      /* the h/Vp ratio				      */
	double	 zeta;	      /* the time domain zeta-damping		      */
	double	 xi;	      /* the freq domain xi-damping		      */
	double	 omega;	      /* the element associated freq from w=3.46*Vp/h */
	double	 a, b;	      /* the same constants we use for C = aM + bK    */
	double	 dt_factor_X; /* the factor of 0.577(1-xi)*h/Vp		      */
	double	 dt_factor_Z; /* the factor of 0.577(1-zeta)*h/Vp	      */
	double	 VsVp;	      /* the quotient Vs/Vp			      */
	double	 VpVsZ;	      /* the result of Vp / Vs * zeta		      */
	double	 Vs;	      /* the Vs					      */

	/* Captures the element */

	elemp = &Global.myMesh->elemTable[eindex];
	edata = (edata_t *)elemp->data;

	/* Calculate the clue quantities */

	ratio	    = edata->edgesize / edata->Vp;

	/* Old formula for damping */
	/* zeta	       = (edata->Vs < 1500) ? 25 / edata->Vs : 5 / edata->Vs; */
	/* New formula acording to Graves */

	zeta	    = 10 / edata->Vs;

	omega	    = 3.46410161514 / ratio;
	a	    = zeta * Global.theABase;
	b	    = zeta * Global.theBBase;
	xi	    = ( a / (2 * omega)) + ( b * omega / 2 );
	dt_factor_X = 0.57735026919 * ( 1 - xi ) * ratio;
	dt_factor_Z = 0.57735026919 * ( 1 - zeta ) * ratio;
	VsVp	    = edata->Vs / edata->Vp;
	VpVsZ	    = zeta * ( edata->Vp / edata->Vs );
	Vs	    = edata->Vs;

	/* Updating for extreme values */

	/* ratio */
	if ( ratio < min_h_over_Vp )
	{
	    min_h_over_Vp = ratio;
	    min_h_over_Vp_elem_index = eindex;
	}

	/* dt_factors */
	if ( dt_factor_X < min_dt_factor_X )
	{
	    min_dt_factor_X = dt_factor_X;
	    min_dt_factor_X_elem_index = eindex;
	}
	if ( dt_factor_Z < min_dt_factor_Z )
	{
	    min_dt_factor_Z = dt_factor_Z;
	    min_dt_factor_Z_elem_index = eindex;
	}

	/* min_zeta and max_zeta */
	if ( zeta < min_zeta )
	{
	    min_zeta = zeta;
	    min_zeta_elem_index = eindex;
	}
	if ( zeta > max_zeta )
	{
	    max_zeta = zeta;
	    max_zeta_elem_index = eindex;
	}

	/* min_xi and max_xi */
	if ( xi < min_xi )
	{
	    min_xi = xi;
	    min_xi_elem_index = eindex;
	}
	if ( xi > max_xi )
	{
	    max_xi = xi;
	    max_xi_elem_index = eindex;
	}

	/* min Vs/Vp */
	if ( VsVp < min_VsVp )
	{
	    min_VsVp = VsVp;
	    min_VsVp_elem_index = eindex;
	}
	if ( VsVp > max_VsVp )
	{
	    max_VsVp = VsVp;
	    max_VsVp_elem_index = eindex;
	}

	/* min and max VpVsZ */
	if ( VpVsZ < min_VpVsZ )
	{
	    min_VpVsZ = VpVsZ;
	    min_VpVsZ_elem_index = eindex;
	}
	if ( VpVsZ > max_VpVsZ )
	{
	    max_VpVsZ = VpVsZ;
	    max_VpVsZ_elem_index = eindex;
	}

	/* min Vs */
	if ( Vs < min_Vs )
	{
	    min_Vs = Vs;
	    min_Vs_elem_index = eindex;
	}
	if ( Vs > max_Vs )
	{
	    max_Vs = Vs;
	    max_Vs_elem_index = eindex;
	}

    } /* End of the loop over the mesh elements */

    /* Reducing to global values */
    MPI_Reduce(&min_h_over_Vp,	 &min_h_over_Vp_global,	  1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
    if ( Param.theDampingStatisticsFlag == 1 )
    {
	MPI_Reduce(&min_dt_factor_X, &min_dt_factor_X_global, 1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&min_dt_factor_Z, &min_dt_factor_Z_global, 1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&min_zeta,	     &min_zeta_global,	      1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&max_zeta,	     &max_zeta_global,	      1, MPI_DOUBLE, MPI_MAX, 0, comm_solver);
	MPI_Reduce(&min_xi,	     &min_xi_global,	      1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&max_xi,	     &max_xi_global,	      1, MPI_DOUBLE, MPI_MAX, 0, comm_solver);
	MPI_Reduce(&min_VsVp,	     &min_VsVp_global,	      1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&max_VsVp,	     &max_VsVp_global,	      1, MPI_DOUBLE, MPI_MAX, 0, comm_solver);
	MPI_Reduce(&min_VpVsZ,	     &min_VpVsZ_global,	      1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&max_VpVsZ,	     &max_VpVsZ_global,	      1, MPI_DOUBLE, MPI_MAX, 0, comm_solver);
	MPI_Reduce(&min_Vs,	     &min_Vs_global,	      1, MPI_DOUBLE, MPI_MIN, 0, comm_solver);
	MPI_Reduce(&max_Vs,	     &max_Vs_global,	      1, MPI_DOUBLE, MPI_MAX, 0, comm_solver);
    }

    /* Inform everyone about the global values */
    MPI_Bcast(&min_h_over_Vp_global,   1, MPI_DOUBLE, 0, comm_solver);
    if ( Param.theDampingStatisticsFlag == 1 )
    {
	MPI_Bcast(&min_dt_factor_X_global, 1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_dt_factor_Z_global, 1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_zeta_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&max_zeta_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_xi_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&max_xi_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_VsVp_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&max_VsVp_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_VpVsZ_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&max_VpVsZ_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&min_Vs_global,	   1, MPI_DOUBLE, 0, comm_solver);
	MPI_Bcast(&max_Vs_global,	   1, MPI_DOUBLE, 0, comm_solver);
    }

    /* go for damping statistics */
    if ( Param.theDampingStatisticsFlag == 1 )
    {
	damping_statistics(min_xi_global,   max_xi_global,   min_zeta_global,  max_zeta_global,
			   min_VsVp_global, max_VsVp_global, min_VpVsZ_global, max_VpVsZ_global,
			   min_Vs_global,   max_Vs_global);
    }

    /* Static global variable for the critical delta t */
    Global.theCriticalT = min_h_over_Vp_global;

    /* Printing of information */
    MPI_Barrier( comm_solver );
    if (Global.myID == 0)
    {
	if ( Param.theDampingStatisticsFlag == 1 )
	{
	    printf("\n\n Critical delta t related information: \n\n");
	    printf("\t 1. The minimum h/Vp	   = %.6f \n", min_h_over_Vp_global);
	    printf("\t 2. The minimum dt X	   = %.6f \n", min_dt_factor_X_global);
	    printf("\t 3. The minimum dt Z	   = %.6f \n", min_dt_factor_Z_global);
	    printf("\t 4. The minimum zeta	   = %.6f \n", min_zeta_global);
	    printf("\t 5. The maximum zeta	   = %.6f \n", max_zeta_global);
	    printf("\t 6. The minimum xi	   = %.6f \n", min_xi_global);
	    printf("\t 7. The maximum xi	   = %.6f \n", max_xi_global);
	    printf("\t 8. The minimum Vs/Vp	   = %.6f \n", min_VsVp_global);
	    printf("\t 9. The maximum Vs/Vp	   = %.6f \n", max_VsVp_global);
	    printf("\t10. The minimum (Vp/Vs)*zeta = %.6f \n", min_VpVsZ_global);
	    printf("\t11. The maximum (Vp/Vs)*zeta = %.6f \n", max_VpVsZ_global);
	    printf("\t12. The minimum Vs	   = %.6f \n", min_Vs_global);
	    printf("\t13. The maximum Vs	   = %.6f \n", max_Vs_global);
	}
	else
	{
	    printf("\n\n Critical delta t related information: \n\n");
	    printf("\t The minimum h/Vp = %.6f \n\n", min_h_over_Vp_global);
	}
    }

#ifdef AUTO_DELTA_T
    /* Set the critical delta T */
    Param.theDeltaT	     = Global.theCriticalT;
    Param.theDeltaTSquared = Param.theDeltaT * Param.theDeltaT;

    /* Set the total steps */
    Param.theTotalSteps    = (int)(((Param.theEndT - Param.theStartT) / Param.theDeltaT));
#endif /* AUTO_DELTA_T */

    /* Printing location and element properties of the maximum values */
    if ( Param.theDampingStatisticsFlag == 1 )
    {
	/* Local variables */

	double	local_extremes[13],
		global_extremes[13];
	int32_t element_indices[13];
	int32_t extreme_index;

	local_extremes[0]  = min_h_over_Vp;
	local_extremes[1]  = min_dt_factor_X;
	local_extremes[2]  = min_dt_factor_Z;
	local_extremes[3]  = min_zeta;
	local_extremes[4]  = max_zeta;
	local_extremes[5]  = min_xi;
	local_extremes[6]  = max_xi;
	local_extremes[7]  = min_VsVp;
	local_extremes[8]  = max_VsVp;
	local_extremes[9]  = min_VpVsZ;
	local_extremes[10] = max_VpVsZ;
	local_extremes[11] = min_Vs;
	local_extremes[12] = max_Vs;

	global_extremes[0]  = min_h_over_Vp_global;
	global_extremes[1]  = min_dt_factor_X_global;
	global_extremes[2]  = min_dt_factor_Z_global;
	global_extremes[3]  = min_zeta_global;
	global_extremes[4]  = max_zeta_global;
	global_extremes[5]  = min_xi_global;
	global_extremes[6]  = max_xi_global;
	global_extremes[7]  = min_VsVp_global;
	global_extremes[8]  = max_VsVp_global;
	global_extremes[9]  = min_VpVsZ_global;
	global_extremes[10] = max_VpVsZ_global;
	global_extremes[11] = min_Vs_global;
	global_extremes[12] = max_Vs_global;

	element_indices[0]  = min_h_over_Vp_elem_index;
	element_indices[1]  = min_dt_factor_X_elem_index;
	element_indices[2]  = min_dt_factor_Z_elem_index;
	element_indices[3]  = min_zeta_elem_index;
	element_indices[4]  = max_zeta_elem_index;
	element_indices[5]  = min_xi_elem_index;
	element_indices[6]  = max_xi_elem_index;
	element_indices[7]  = min_VsVp_elem_index;
	element_indices[8]  = max_VsVp_elem_index;
	element_indices[9]  = min_VpVsZ_elem_index;
	element_indices[10] = max_VpVsZ_elem_index;
	element_indices[11] = min_Vs_elem_index;
	element_indices[12] = max_Vs_elem_index;

	/* Printing section title */
	MPI_Barrier( comm_solver );
	if (Global.myID == 0)
	{
	    printf("\n\t Their corresponding element properties and coordinates are: \n\n");
	}

	/* Loop over the six extreme values */
	MPI_Barrier( comm_solver );
	for ( extreme_index = 0; extreme_index < 13; extreme_index++ )
	{
	    MPI_Barrier( comm_solver );
	    if ( local_extremes[extreme_index] == global_extremes[extreme_index] )
	    {
		tick_t	 ldb[3];
		elem_t	*elemp;
		edata_t *edata;
		int lnid0 = Global.myMesh->elemTable[element_indices[extreme_index]].lnid[0];

		ldb[0] = Global.myMesh->nodeTable[lnid0].x;
		ldb[1] = Global.myMesh->nodeTable[lnid0].y;
		ldb[2] = Global.myMesh->nodeTable[lnid0].z;

		elemp  = &Global.myMesh->elemTable[element_indices[extreme_index]];
		edata  = (edata_t *)elemp->data;

		printf("\t For extreme value No. %d:", extreme_index + 1);
		printf("\n\t\t h = %.6f, Vp = %.6f Vs = %.6f rho = %.6f",
		       edata->edgesize, edata->Vp , edata->Vs, edata->rho);
		printf("\n\t\t by thread %d, element_coord = (%.6f, %.6f, %.6f)\n\n",
		       Global.myID, ldb[0] * Global.myMesh->ticksize, ldb[1] * Global.myMesh->ticksize,
		       ldb[2] * Global.myMesh->ticksize);
	    }
	    MPI_Barrier( comm_solver );
	} /* End of loop over the extreme values */

	if (Global.myID == 0) {
	    fflush (stdout);
	}
    } /* end if damping statistics */

    return;
} /* End solver_set_critical_T */


/**
 * Iterates through all processor to obtain the minimum * edgesize of the
 * mesh.
 */
static void get_minimum_edge()
{
    int32_t eindex;
    double min_h = 1e32, min_h_global;
    int32_t min_h_elem_index;

    /* Find the minimal h/Vp in the domain */
    for (eindex = 0; eindex < Global.myMesh->lenum; eindex++) {
        elem_t *elemp;	 /* pointer to the mesh database */
        edata_t *edata;
        double h;

        elemp = &Global.myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;

        /* Update the min_h value. (h = edgesize)  */
        h = edata->edgesize;
        if (h < min_h) {
            min_h = h;
            min_h_elem_index = eindex;
        }
    }

    MPI_Reduce(&min_h, &min_h_global, 1, MPI_DOUBLE,
            MPI_MIN, 0, comm_solver);
    /* Inform everyone of this value */
    MPI_Bcast(&min_h_global, 1, MPI_DOUBLE, 0, comm_solver);

    if (Global.myID == 0) {
        printf("\nThe minimal h	 = %.6f\n\n\n", min_h_global);
    }

    Global.theNumericsInformation.minimumh=min_h_global;

    return;
}


/**
 * Print the stiffness matrix K1, K2 and K3 to the given output stream.
 */
static void
print_K_stdoutput()
{
    int i, j, iloc, jloc;

    fprintf(stdout, "\n\nStiffness Matrix K1 \n\n");
    for ( i = 0 ; i < 8 ; i++){
        for ( iloc = 0; iloc < 3; iloc++){
            for ( j = 0; j < 8; j++) {
                for (jloc = 0; jloc < 3; jloc++) {
                    fprintf(stdout, "%10.2e", Global.theK1[i][j].f[iloc][jloc]);
                }
            }
            fprintf(stdout, "\n");
        }
    }

    fprintf(stdout, "\n\nStiffness Matrix K2 \n\n");
    for ( i = 0 ; i < 8 ; i++){
        for ( iloc = 0; iloc < 3; iloc++){
            for ( j = 0; j < 8; j++) {
                for (jloc = 0; jloc < 3; jloc++) {
                    fprintf(stdout, "%10.2e", Global.theK2[i][j].f[iloc][jloc]);
                }
            }
            fprintf(stdout, "\n");
        }
    }

    fprintf(stdout, "\n\nStiffness Matrix K3 \n\n");
    for ( i = 0 ; i < 8 ; i++){
        for ( iloc = 0; iloc < 3; iloc++){
            for ( j = 0; j < 8; j++) {
                for (jloc = 0; jloc < 3; jloc++) {
                    fprintf(stdout, "%10.2e", Global.theK3[i][j].f[iloc][jloc]);
                }
            }
            fprintf(stdout, "\n");
        }
    }

    fprintf(stdout, "\n\n");
}



/**
 * mu_and_lambda: Calculates mu and lambda according to the element values
 *                of Vp, Vs, and Rho and verifies/applies some rules defined
 *                by Jacobo, Leonardo and Ricardo.  It was originally within
 *                solver_init but was moved out because it needs to be used
 *                in other places as well (nonlinear)
 */
void mu_and_lambda(double *theMu, double *theLambda,
                   edata_t *edata, int32_t eindex)
{

    double mu, lambda;

    mu = edata->rho * edata->Vs * edata->Vs;

    if ( edata->Vp > (edata->Vs * Param.theThresholdVpVs) ) {
        lambda = edata->rho * edata->Vs * edata->Vs * Param.theThresholdVpVs
               * Param.theThresholdVpVs - 2 * mu;
    } else {
        lambda = edata->rho * edata->Vp * edata->Vp - 2 * mu;
    }

    /* Adjust Vs, Vp to fix Poisson ratio problem, formula provided by Jacobo */
    if ( lambda < 0 ) {
        if ( edata->Vs < 500 )
            edata->Vp = 2.45 * edata->Vs;
        else if ( edata->Vs < 1200 )
            edata->Vp = 2 * edata->Vs;
        else
            edata->Vp = 1.87 * edata->Vs;

        lambda = edata->rho * edata->Vp * edata->Vp;
    }

    if ( lambda < 0) {
        fprintf(stderr, "\nThread %d: %d element produces negative lambda = %.6f; Vp = %f; Vs = %f; Rho = %f",
                Global.myID, eindex, lambda, edata->Vp, edata->Vs, edata->rho);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
    }

    /* assign results to returning values */
    *theMu = mu;
    *theLambda = lambda;
}


/**
 * gpu_init: Select one of the attached GPU devices and retrieve its
 *           specifications.
 *
 */
static void gpu_init()
{
    int myDevID, devCount;

    /* Select a GPU device for this process */
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) {
      fprintf(stderr, "Failed to get GPU device count\n");
      MPI_Abort(MPI_COMM_WORLD, ERROR);
      exit(1);
    }

    myDevID = (Global.myID % devCount);
    
    /* Retrieve device hardware details */
    if (Global.myID == 0) {
      /* Only Rank 0 dumps GPU information */
      getGPUHardware(myDevID, &(Global.gpu_spec), 1);
    } else {
      getGPUHardware(myDevID, &(Global.gpu_spec), 0);
    }
}


/**
 * Init matrices and constants, build comm schedule, allocate/init space
 * for the solver.
 */
static void solver_init()
{
    /* local variables */
    int32_t eindex;
    int32_t c_outsize, c_insize, s_outsize, s_insize;

    /* compute the damping parameters a/zeta and b/zeta */
    compute_setab(Param.theFreq, &Global.theABase, &Global.theBBase);

    /* find out the critical delta T of the current simulation */
    /* and goes for the damping statistics if falg is == 1     */
    MPI_Barrier( comm_solver );
    solver_set_critical_T();

    /* find the minimum edge size */

    get_minimum_edge();

    /* Init stiffness matrices and other constants */
    compute_K();

    /* For debugging */
    if ( ( Param.printK == YES ) && ( Global.myID == 0 ) ) {
        print_K_stdoutput();
    }

    compute_setab(Param.theFreq, &Global.theABase, &Global.theBBase);

    /* allocation of memory */
    Global.mySolver = (mysolver_t *)malloc(sizeof(mysolver_t));
    if (Global.mySolver == NULL) {
        fprintf(stderr, "Thread %d: solver_init: out of memory\n", Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    /* Allocate memory */
    Global.mySolver->eTable = (e_t *)calloc(Global.myMesh->lenum, sizeof(e_t));
    Global.mySolver->nTable = (n_t *)calloc(Global.myMesh->nharbored, sizeof(n_t));
    Global.mySolver->tm1    = (fvector_t *)calloc(Global.myMesh->nharbored, sizeof(fvector_t));
    Global.mySolver->tm2    = (fvector_t *)calloc(Global.myMesh->nharbored, sizeof(fvector_t));
    Global.mySolver->force  = (fvector_t *)calloc(Global.myMesh->nharbored, sizeof(fvector_t));
    Global.mySolver->conv_shear_1 = (fvector_t *)calloc(8 * Global.myMesh->lenum, sizeof(fvector_t));
    Global.mySolver->conv_shear_2 = (fvector_t *)calloc(8 * Global.myMesh->lenum, sizeof(fvector_t));
    Global.mySolver->conv_kappa_1 = (fvector_t *)calloc(8 * Global.myMesh->lenum, sizeof(fvector_t));
    Global.mySolver->conv_kappa_2 = (fvector_t *)calloc(8 * Global.myMesh->lenum, sizeof(fvector_t));

    Global.mySolver->dn_sched = schedule_new();
    Global.mySolver->an_sched = schedule_new();

    if ( (Global.mySolver->eTable == NULL) ||
         (Global.mySolver->nTable == NULL) ||
         (Global.mySolver->tm1    == NULL) ||
         (Global.mySolver->tm2    == NULL) ||
         (Global.mySolver->force  == NULL) ||
         (Global.mySolver->conv_shear_1 == NULL) ||
         (Global.mySolver->conv_shear_2 == NULL) ||
         (Global.mySolver->conv_kappa_1 == NULL) ||
         (Global.mySolver->conv_kappa_2 == NULL) ) {

        fprintf(stderr, "Thread %d: solver_init: out of memory\n", Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    if ( Param.printStationAccelerations == YES ) {

        Global.mySolver->tm3 = (fvector_t *)calloc(Global.myMesh->nharbored, sizeof(fvector_t));

        if ( Global.mySolver->tm3 == NULL ) {

            fprintf(stderr, "Thread %d: solver_init: out of memory for accs\n", Global.myID);
            MPI_Abort(MPI_COMM_WORLD, ERROR);
            exit(1);
        }
    }

    /* For each element:
     * Initialize the data structures. tm1, tm2 and force have
     * already been initialized to 0 by calloc(). */
    for (eindex = 0; eindex < Global.myMesh->lenum; eindex++)
    {
        elem_t  *elemp; /* pointer to the mesh database */
        edata_t *edata;
        e_t     *ep;    /* pointer to the element constant table */
        double   mass, M, mu, lambda;
        int j;

#ifdef BOUNDARY
        tick_t  edgeticks;
        tick_t  ldb[3], ruf[3];
        int32_t lnid0;
        char    flag;
        double  dashpot[8][3];
#endif

        double zeta, a, b;

        /* Note the difference between the two tables */
        elemp = &Global.myMesh->elemTable[eindex];
        edata = (edata_t *)elemp->data;
        ep    = &Global.mySolver->eTable[eindex];

        /* Calculate the Lame constants */
        mu_and_lambda(&mu, &lambda, edata, eindex);

        /* coefficients for term (deltaT_squared * Ke * Ut) */
        ep->c1 = Param.theDeltaTSquared * edata->edgesize * mu / 9;
        ep->c2 = Param.theDeltaTSquared * edata->edgesize * lambda / 9;

        /* coefficients for term (b * deltaT * Ke_off * (Ut-1 - Ut)) */
        /* Anelastic attenuation (material damping) */

        /* Old formula for damping */
        /* zeta = (edata->Vs < 1500) ? 25 / edata->Vs : 5 / edata->Vs; */

        /* New formula for damping according to Graves */
        	zeta = 10 / edata->Vs;

        if ( zeta > Param.theThresholdDamping ) {
        	zeta = Param.theThresholdDamping;
        }

        /* the a,b coefficients */
        a = zeta * Global.theABase;
        b = zeta * Global.theBBase;

        /* coefficients for term (b * deltaT * Ke_off * (Ut-1 - Ut)) */
        ep->c3 = b * Param.theDeltaT * edata->edgesize * mu / 9;
        ep->c4 = b * Param.theDeltaT * edata->edgesize * lambda / 9;

#ifdef BOUNDARY

        /* Set the flag for the element */
        lnid0 = elemp->lnid[0];

        ldb[0] = Global.myMesh->nodeTable[lnid0].x;
        ldb[1] = Global.myMesh->nodeTable[lnid0].y;
        ldb[2] = Global.myMesh->nodeTable[lnid0].z;

        edgeticks = (tick_t)1 << (PIXELLEVEL - elemp->level);
        ruf[0] = ldb[0] + edgeticks;
        ruf[1] = ldb[1] + edgeticks;
        ruf[2] = ldb[2] + edgeticks;

        flag = compute_setflag(ldb, ruf, Global.myOctree->nearendp,
                Global.myOctree->farendp);
        if (flag != 13) {
            compute_setboundary(edata->edgesize, edata->Vp, edata->Vs,
                                edata->rho, flag, dashpot);
        }
#endif /* BOUNDARY */

        /* Assign the element mass to its vertices */
        /* mass is the total mass of the element   */
        /* and M is the mass assigned to each node */
        mass = edata->rho * edata->edgesize * edata->edgesize *edata->edgesize;
        M    = mass / 8;

        /* For each node */
        for (j = 0; j < 8; j++)
        {
            int32_t lnid;
            int axis;
            n_t *np;

            lnid = elemp->lnid[j];
            np   = &Global.mySolver->nTable[lnid];

            np->mass_simple += M;

            /* loop for each axis */
            for (axis = 0; axis < 3; axis++ )
            {
                np->mass_minusaM[axis]	-= (Param.theDeltaT * a * M);
                np->mass2_minusaM[axis] -= (Param.theDeltaT * a * M);

#ifdef BOUNDARY
                if (flag != 13)
                {
                    /* boundary impact */
                    np->mass_minusaM[axis]  -= (Param.theDeltaT * dashpot[j][axis]);
                    np->mass2_minusaM[axis] -= (Param.theDeltaT * dashpot[j][axis]);
                }
#endif /* BOUNDARY */

                np->mass_minusaM[axis]	+= M;
                np->mass2_minusaM[axis] += (M * 2);

            } /* end loop for each axis */

        } /* end loop for each node */

    } /* eindex for elements */

    /* Build the communication schedules */
    schedule_build(Global.myMesh, Global.mySolver->dn_sched, Global.mySolver->an_sched);

#ifdef DEBUG
    /* For debug purpose, add gnid into the data field. */
    c_outsize = sizeof(n_t) + sizeof(int64_t);
    c_insize  = sizeof(n_t) + sizeof(int64_t);
    s_outsize = sizeof(n_t) + sizeof(int64_t);
    s_insize  = sizeof(n_t) + sizeof(int64_t);
#else
    c_outsize = sizeof(n_t);
    c_insize  = sizeof(n_t);
    s_outsize = sizeof(n_t);
    s_insize  = sizeof(n_t);
#endif /* DEBUG */

    schedule_prepare(Global.mySolver->dn_sched, c_outsize, c_insize,
		     s_outsize, s_insize);

    schedule_prepare(Global.mySolver->an_sched, c_outsize, c_insize,
		     s_outsize, s_insize);

    /* Send mass information of dangling nodes to their owners */
    schedule_senddata(Global.mySolver->dn_sched, Global.mySolver->nTable,
	      sizeof(n_t) / sizeof(solver_float), CONTRIBUTION, DN_MASS_MSG);

    /* Distribute the mass from dangling nodes to anchored nodes. (local) */
    compute_adjust(Global.mySolver->nTable, sizeof(n_t) / sizeof(solver_float),
		   DISTRIBUTION);

    /* Send mass information of anchored nodes to their owner processors*/
    schedule_senddata(Global.mySolver->an_sched, Global.mySolver->nTable,
	      sizeof(n_t) / sizeof(solver_float), CONTRIBUTION, AN_MASS_MSG);

    /* Save device configuration */
    Global.mySolver->gpu_spec = &(Global.gpu_spec);

    /* Allocate device memory */
    if (cudaMalloc((void**)&(Global.mySolver->elemTableDevice), 
		   Global.myMesh->lenum * sizeof(elem_t)) != cudaSuccess) {
        fprintf(stderr, "Thread %d: Failed to allocate elemTable memory\n", 
		Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    if (cudaMalloc((void**)&(Global.mySolver->eTableDevice), 
		   Global.myMesh->lenum * sizeof(e_t)) != cudaSuccess) {
        fprintf(stderr, "Thread %d: Failed to allocate etable memory\n", 
		Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    if (cudaMalloc((void**)&(Global.mySolver->tm1Device), 
		   Global.myMesh->nharbored * sizeof(fvector_t)) != cudaSuccess) {
        fprintf(stderr, "Thread %d: Failed to allocate tm1 memory\n", 
		Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    if (cudaMalloc((void**)&(Global.mySolver->forceDevice), 
		   Global.myMesh->nharbored * sizeof(fvector_t)) != cudaSuccess) {
        fprintf(stderr, "Thread %d: Failed to allocate force memory\n", 
		Global.myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    /* Copy mesh and solver elements to device */
    cudaMemcpy(Global.mySolver->elemTableDevice, Global.myMesh->elemTable, 
	       Global.myMesh->lenum * sizeof(elem_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Global.mySolver->eTableDevice, Global.mySolver->eTable, 
	       Global.myMesh->lenum * sizeof(e_t), cudaMemcpyHostToDevice);

    return;
}


/**
 * Print the communication schedule of each processor to the given output
 * stream.
 */
static void
solver_printstat_to_stream( mysolver_t* solver, FILE* out )
{
    /**
     * Enumeration for the indices in the counts array
     * the first 4 values are for the dangling nodes:
     * the last 4 values are for the anchored nodes.
     */
    enum solver_msg_idx_t {
	I_DN_C_P,	/**< 0: Dangling contribution processor count */
	I_DN_C_N,	/**< 1: Dangling contribution node count */
	I_DN_S_P,	/**< 2: Dangling shared processor count */
	I_DN_S_N,	/**< 3: Dangling shared node count */
	I_AN_C_P,	/**< 4: Shared contribution processor count */
	I_AN_C_N,	/**< 5: Shared contribution node count */
	I_AN_S_P,	/**< 6: Shared shared processor count */
	I_AN_S_N,	/**< 7: Shared shared node count */
	I_SCHED_LAST	/**< 8: Number of counters */
    };

    int32_t* recv_counts = NULL;
    int32_t  send_counts[I_SCHED_LAST];

    /* number of processors this PE communicates with */
    send_counts[I_DN_C_P] = solver->dn_sched->c_count;
    send_counts[I_DN_S_P] = solver->dn_sched->s_count;
    send_counts[I_AN_C_P] = solver->an_sched->c_count;
    send_counts[I_AN_S_P] = solver->an_sched->s_count;

    /* number of nodes this PE communicates */
    send_counts[I_DN_C_N] = messenger_countnodes( solver->dn_sched->first_c );
    send_counts[I_DN_S_N] = messenger_countnodes( solver->dn_sched->first_s );
    send_counts[I_AN_C_N] = messenger_countnodes( solver->an_sched->first_c );
    send_counts[I_AN_S_N] = messenger_countnodes( solver->an_sched->first_s );


    if (Global.myID == 0) {
	/* allocate a buffer in PE 0 to receive stats from all other PEs */
	XMALLOC_VAR_N( recv_counts, int32_t, I_SCHED_LAST * Global.theGroupSize );
    }

    MPI_Gather( send_counts, I_SCHED_LAST, MPI_INT,
		recv_counts, I_SCHED_LAST, MPI_INT, 0, comm_solver );

    if (Global.myID == 0) {
	int      pe_id;
	int32_t* pe_counts = recv_counts;

	fprintf( out, "# Solver communication schedule summary:\n"
		 "#   PE dc_p     dc_n ds_p     ds_n ac_p     ac_n as_p"
		 "     as_n total_ncnt\n" );

	for (pe_id = 0; pe_id < Global.theGroupSize; pe_id++) {
	    /* print a row for a PE, each row has I_SCHED_LAST (8) items
	     * besides the pe_id and the total node sum at the end */
	    int pe_comm_count = pe_counts[I_DN_C_N] +
				pe_counts[I_DN_S_N] +
				pe_counts[I_AN_C_N] +
				pe_counts[I_AN_S_N];

	    fprintf( out, "%6d %4d %8d %4d %8d %4d %8d %4d %8d %10d\n",
		     pe_id,
		     pe_counts[I_DN_C_P],
		     pe_counts[I_DN_C_N],
		     pe_counts[I_DN_S_P],
		     pe_counts[I_DN_S_N],
		     pe_counts[I_AN_C_P],
		     pe_counts[I_AN_C_N],
		     pe_counts[I_AN_S_P],
		     pe_counts[I_AN_S_N],
		     pe_comm_count );

	    pe_counts += I_SCHED_LAST; /* advance a row of size I_SCHED_LAST */
	}

	fprintf( out, "\n\n" );
	fflush( out );

	free( recv_counts );
    }

    return;
}

/**
 * Print the communication schedule of each processor to the given output
 * stream.
 */
static void
solver_printstat( mysolver_t* solver )
{
    FILE* stat_out = NULL;

    if (Global.myID == 0) {
	stat_out = hu_fopen( Param.theScheduleStatFilename, "w" );
    }

    solver_printstat_to_stream( solver, stat_out );

    if (Global.myID == 0) {
	hu_fclose( stat_out );
	xfree_char( & Param.theScheduleStatFilename );
    }
}


/**
 * solver_delete: Release all the memory associate with Global.mySolver.
 *
 */
static void solver_delete()
{
    if (Global.mySolver == NULL) {
	return;
    }

    /* Free device memory */
    cudaFree(Global.mySolver->elemTableDevice);
    cudaFree(Global.mySolver->eTableDevice);
    cudaFree(Global.mySolver->tm1Device);
    cudaFree(Global.mySolver->forceDevice);

    free(Global.mySolver->eTable);
    free(Global.mySolver->nTable);

    free(Global.mySolver->tm1);
    free(Global.mySolver->tm2);
    free(Global.mySolver->force);

    free(Global.mySolver->conv_shear_1);
    free(Global.mySolver->conv_shear_2);
    free(Global.mySolver->conv_kappa_1);
    free(Global.mySolver->conv_kappa_2);

    schedule_delete(Global.mySolver->dn_sched);
    schedule_delete(Global.mySolver->an_sched);

    free(Global.mySolver);
}

static int
read_myForces( int32_t timestep )
{
    off_t   whereToRead;
    size_t  to_read, read_count;

    whereToRead = ((off_t)sizeof(int32_t))
		+ Global.theNodesLoaded * sizeof(int32_t)
		+ Global.theNodesLoaded * timestep * sizeof(double) * 3;

    hu_fseeko( Global.fpsource, whereToRead, SEEK_SET );

    to_read    = Global.theNodesLoaded * 3;
    read_count = hu_fread( Global.myForces, sizeof(double), to_read, Global.fpsource );

    return 0;	/* if we got here everything went OK */
}


/**
 * check the max and min value of displacement.
 */
static void
solver_debug_overflow( mysolver_t* solver, mesh_t* mesh, int step )
{
    int nindex;
    double max_disp, min_disp, global_max_disp, global_min_disp;

    max_disp = DBL_MIN;
    min_disp = DBL_MAX;

    /* find the min and max X displacement components */
    for (nindex = 0; nindex < mesh->nharbored; nindex++) {
	fvector_t* tm2Disp;
	n_t* np;

	np      = &solver->nTable[nindex];
	tm2Disp = solver->tm2 + nindex;

	max_disp = (max_disp > tm2Disp->f[0]) ? max_disp : tm2Disp->f[0];
	min_disp = (min_disp < tm2Disp->f[0]) ? min_disp : tm2Disp->f[0];
    }

    /* get global min and max values */
    MPI_Reduce( &min_disp, &global_min_disp, 1, MPI_DOUBLE,
		MPI_MIN, 0, comm_solver );

    MPI_Reduce( &max_disp, &global_max_disp, 1, MPI_DOUBLE,
		MPI_MAX, 0, comm_solver );

    if (Global.myID == 0) {
	printf("Timestep %d: max_dx = %.6f min_dx = %.6f\n",
	       step, global_max_disp, global_min_disp);
    }
}


int
darray_has_nan_nd( const double* v, int dim, const size_t len[] )
{
    HU_ASSERT_PTR_ALWAYS( v );
    HU_ASSERT_PTR_ALWAYS( len );
    HU_ASSERT_ALWAYS( dim > 0 );

    size_t* idx = XMALLOC_N( size_t, dim );
    int     ret = hu_darray_has_nan_nd( v, dim, len, idx );

    if (ret != 0) {
	int i;

	fputs( "WARNING!: Found NAN value at index", stderr );

	for (i = 0; i < dim; i++) {
	    fprintf( stderr, " %zu", idx[i] );
	}

	fputc( '\n', stderr );
    }

    free( idx );
    idx = NULL;

    return ret;
}



/**
 * Check a fvector_t array for NAN values.
 */
static int
fvector_array_has_nan( const fvector_t* f, size_t len, const char* varname )
{
    size_t lengths[2];
    const double* v = (double*)f;

    lengths[0] = len;
    lengths[1] = 3;

    if (darray_has_nan_nd( v, 2, lengths ) != 0) {
        if (NULL == varname) {
	    varname = "(unknown)";
	}
	fputs( "fvector_t varname=", stderr );
	fputs( varname, stderr );
	fputc( '\n', stderr );

	return -1;
    }

    return 0;
}

/**
 * Debug function to check the solver structures for NAN.
 * The checked structures are the displacement arrays (tm1 & tm2) and
 * the forces array.
 */
void
solver_check_nan( mysolver_t* solver, int node_count, int time_step )
{
    HU_ASSERT_PTR_ALWAYS( solver );

    int ret1 = fvector_array_has_nan( solver->tm1, node_count, "tm1" );
    int ret2 = fvector_array_has_nan( solver->tm2, node_count, "tm2" );
    int ret3 = fvector_array_has_nan( solver->force, node_count, "force" );

    if ((ret1 | ret2 | ret3) != 0) {
	hu_solver_abort( __FUNCTION_NAME, NULL,
			 "Found a NAN value at timestep %d", time_step );
    }
}


static void solver_run_init_comm( mysolver_t* solver )
{
    /* The int64_t (global node id) is for debug purpose */
    static const int debug_size = (DO_DEBUG) ? sizeof(int64_t) : 0;

    /* properly setup the communication schedule */
    int c_outsize = sizeof(fvector_t) + debug_size;     /* force */
    int c_insize  = sizeof(fvector_t) + debug_size;     /* displacement */
    int s_outsize = sizeof(fvector_t) + debug_size;     /* displacement */
    int s_insize  = sizeof(fvector_t) + debug_size;     /* force */

    schedule_prepare(solver->dn_sched,c_outsize,c_insize,s_outsize,s_insize);
    schedule_prepare(solver->an_sched,c_outsize,c_insize,s_outsize,s_insize);

    solver_print_schedules( solver );
}


/**
 * \note Globals used:
 * - Global.myID (read)
 * - Param.theDeltaT (read)
 * - startingStep (read)
 */
/* show a progress bar to make the process less anxious! */
static void solver_update_status( int step, const int start_step ){

    static double lastCheckedTime = 0;
    double interval = 0;
    double CurrTime;

    if (Global.myID == 0) {

	CurrTime = Timer_Value( "Total Wall Clock", (TimerKind)0 );

	if (lastCheckedTime==0) {
	    lastCheckedTime = CurrTime;
	    return;
	}

	monitor_print( "*" );

        if (step % Param.monitor_stats_rate == 0) {
	    interval = CurrTime - lastCheckedTime;
	    if (interval > Global.slowestTimeSteps) Global.slowestTimeSteps = interval;
	    if (interval < Global.fastestTimeSteps) Global.fastestTimeSteps = interval;
            monitor_print( "     Sim=% 12.6f     ETA=% 6.1f    WC=% 6.1f\n",
                           step * Param.theDeltaT,
			   ((Param.theTotalSteps - step) / Param.monitor_stats_rate) * interval,
                           CurrTime);
	    lastCheckedTime = CurrTime;
        }

    }
}


static void solver_write_checkpoint( int step, int start_step ){

    if ((Param.theCheckPointingRate != 0) && (step != start_step) &&
        ((step % Param.theCheckPointingRate) == 0)) {

        checkpoint_write( step, Global.myID, Global.myMesh, Param.theCheckPointingDirOut, Global.theGroupSize,
			     Global.mySolver, comm_solver );
    }

}


/**
 * \note Globals used:
 * - Param.theRate (read)
 */
static void solver_output_wavefield( int step )
{
    if (DO_OUTPUT && (step % Param.theRate == 0)) {
        /* output the current timestep */
        do_solver_output();
    }
}


/**
 * \note Globals used:
 * - thePlanePrintRate (read)
 */
static void solver_output_planes( mysolver_t* solver, int my_id, int step )
{
    if (Param.theNumberOfPlanes != 0) {
        if (step % Param.thePlanePrintRate == 0) {
            Timer_Start( "Print Planes" );
            planes_print( my_id, Param.IO_pool_pe_count, Param.theNumberOfPlanes, solver );
            Timer_Stop( "Print Planes" );
        }
    }
}


static void solver_output_stations( int step )
{
    if (Param.theNumberOfStations !=0) {
        if (step % Param.theStationsPrintRate == 0) {
            Timer_Start( "Print Stations" );
            interpolate_station_displacements( step );
            Timer_Stop( "Print Stations" );
        }
    }
}


/**
 * Calculate the nonlinear entities necessary for the next step computation
 * of force correction.
 *
 * \note Globals used:
 * - Param.theNumberOfStations
 * - Param.myNumberOfStations
 * - Param.myStations
 * - Param.theDeltaT
 */
static void solver_nonlinear_state( mysolver_t *solver,
                                    mesh_t     *mesh,
                                    fmatrix_t   k1[8][8],
                                    fmatrix_t   k2[8][8],
                                    int step )
{
    if ( Param.includeNonlinearAnalysis == YES ) {
        Timer_Start( "Compute Non-linear Entities" );
        compute_nonlinear_state ( mesh, solver, Param.theNumberOfStations,
                                  Param.myNumberOfStations, Param.myStations, Param.theDeltaT );
        if ( get_geostatic_total_time() > 0 ) {
            compute_bottom_reactions( mesh, solver, k1, k2, step, Param.theDeltaT );
        }
        Timer_Stop( "Compute Non-linear Entities" );
        if (Param.theNumberOfStations != 0) {
            Timer_Start( "Print Stations" );
            print_nonlinear_stations( mesh, solver, Param.myStations,
                                      Param.myNumberOfStations, Param.theDeltaT,
                                      step, Param.theStationsPrintRate);
            Timer_Stop( "Print Stations" );
        }
    }
}


static void solver_read_source_forces( int step )
{
    Timer_Start( "Read My Forces" );
    if (Global.theNodesLoaded > 0) {
        read_myForces( step );
    }
    Timer_Stop( "Read My Forces" );
}

/**
 * TODO: This method uses global variable Param.theDeltaT
 */
static void solver_load_fixedbase_displacements( mysolver_t* solver, int step )
{
    Timer_Start( "Load Fixedbase Disps" );
    if ( get_fixedbase_flag() == YES ) {
        bldgs_load_fixedbase_disps ( solver, Param.theDeltaT, step);
    }
    Timer_Stop( "Load Fixedbase Disps" );
}


/** Compute the force due to the earthquake source. */
static void solver_compute_force_source( int step )
{
    Timer_Start( "Compute addforces s" );
    compute_addforce_s( step );
    Timer_Stop( "Compute addforces s" );
}


/** Compute the force due to element stiffness matrices. */
static void
solver_compute_force_stiffness( mysolver_t *solver,
                                mesh_t     *mesh,
                                fmatrix_t   k1[8][8],
                                fmatrix_t   k2[8][8] )
{
	Timer_Start( "Compute addforces e" );
	if(Param.theTypeOfDamping != BKT)
	{
		if (Param.theStiffness == EFFECTIVE) {
			compute_addforce_effective_gpu( mesh, solver );
		}
		else if (Param.theStiffness == CONVENTIONAL) {
			compute_addforce_conventional( mesh, solver, k1, k2 );
		}
	}
	Timer_Stop( "Compute addforces e" );
}


/** Compute contribution of damping to the force vector */
static void
solver_compute_force_damping( mysolver_t *solver,
                              mesh_t     *mesh,
                              fmatrix_t   k1[8][8],
                              fmatrix_t   k2[8][8] )
{
	Timer_Start( "Damping addforce" );

	if(Param.theTypeOfDamping == RAYLEIGH  || Param.theTypeOfDamping == MASS)
	{
		damping_addforce(Global.myMesh, Global.mySolver, Global.theK1, Global.theK2);
	}
	else if(Param.theTypeOfDamping == BKT)
	{
		calc_conv(Global.myMesh, Global.mySolver, Param.theFreq, Param.theDeltaT, Param.theDeltaTSquared);
		//addforce_conv(myMesh, mySolver, theFreq, theDeltaT, theDeltaTSquared);
		constant_Q_addforce(Global.myMesh, Global.mySolver, Param.theFreq, Param.theDeltaT, Param.theDeltaTSquared);
	}
	else
	{}

	Timer_Stop( "Damping addforce" );
}

/**
 * Compute the nonlinear contribution to the force.
 * \param deltaT2 Delta t^2 (i.e., squared).
 */
static void
solver_compute_force_nonlinear( mysolver_t *solver,
                                mesh_t     *mesh,
                                double      deltaT2 )
{
    if ( Param.includeNonlinearAnalysis == YES ) {
        Timer_Start( "Compute addforces Non-linear" );
        compute_addforce_nl( mesh, solver, deltaT2 );
        Timer_Stop( "Compute addforces Non-linear" );
    }
}

static void
solver_compute_force_gravity( mysolver_t *solver, mesh_t *mesh, int step )
{
    if ( Param.includeNonlinearAnalysis == YES ) {
        Timer_Start( "Compute addforces gravity" );
        if ( get_geostatic_total_time() > 0 ) {
            compute_addforce_gravity( mesh, solver, step, Param.theDeltaT );
        }
        Timer_Stop( "Compute addforces gravity" );
    }
}

/** Send the forces on dangling nodes to their owner processors */
static void solver_send_force_dangling( mysolver_t* solver )
{
//    HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );

    Timer_Start( "1st schedule send data (contribution)" );
    schedule_senddata( solver->dn_sched, solver->force,
                       sizeof(fvector_t) / sizeof(solver_float), CONTRIBUTION,
                       DN_FORCE_MSG);
    Timer_Stop( "1st schedule send data (contribution)" );
}


static void solver_adjust_forces(  mysolver_t* solver )
{
    Timer_Start( "1st compute adjust (distribution)" );
    /* Distribute the forces to LOCAL anchored nodes */
    compute_adjust( solver->force, sizeof(fvector_t) / sizeof(solver_float),
                    DISTRIBUTION);
    Timer_Stop( "1st compute adjust (distribution)" );
}


static void solver_send_force_anchored( mysolver_t* solver )
{
//    HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );

    Timer_Start( "2nd schedule send data (contribution)" );
    /* Send the forces on anchored nodes to their owner processors */
    schedule_senddata( solver->an_sched, solver->force,
                       sizeof(fvector_t) / sizeof(solver_float), CONTRIBUTION,
                       AN_FORCE_MSG );
    Timer_Stop( "2nd schedule send data (contribution)" );
}


/** Compute new displacements of my harbored nodes */
static void
solver_compute_displacement( mysolver_t* solver, mesh_t* mesh )
{
    lnid_t nindex;

    Timer_Start( "Compute new displacement" );
    for (nindex = 0; nindex < mesh->nharbored; nindex++) {

        const n_t*       np         = &solver->nTable[nindex];
        fvector_t        nodalForce = solver->force[nindex];
        const fvector_t* tm1Disp    = solver->tm1 + nindex;
        fvector_t*       tm2Disp    = solver->tm2 + nindex;

        /* total nodal forces */
        nodalForce.f[0] += np->mass2_minusaM[0] * tm1Disp->f[0]
                         - np->mass_minusaM[0]  * tm2Disp->f[0];
        nodalForce.f[1] += np->mass2_minusaM[1] * tm1Disp->f[1]
                         - np->mass_minusaM[1]  * tm2Disp->f[1];
        nodalForce.f[2] += np->mass2_minusaM[2] * tm1Disp->f[2]
                         - np->mass_minusaM[2]  * tm2Disp->f[2];

        /* Save tm3 for accelerations */
        if ( Param.printStationAccelerations == YES ) {

            fvector_t* tm3Disp = solver->tm3 + nindex;

            tm3Disp->f[0] = tm2Disp->f[0];
            tm3Disp->f[1] = tm2Disp->f[1];
            tm3Disp->f[2] = tm2Disp->f[2];
        }

        /* overwrite tm2 */
        tm2Disp->f[0] = nodalForce.f[0] / np->mass_simple;
        tm2Disp->f[1] = nodalForce.f[1] / np->mass_simple;
        tm2Disp->f[2] = nodalForce.f[2] / np->mass_simple;

    } /* for (nindex ...): all my harbored nodes */

    /* zero out the force vector for all nodes */
    memset( solver->force, 0, sizeof(fvector_t) * mesh->nharbored );

    Timer_Stop( "Compute new displacement" );
}

static void
solver_geostatic_fix(int step)
{
    if ( Param.includeNonlinearAnalysis == YES ) {
        Timer_Start( "Compute addforces gravity" );
        if ( get_geostatic_total_time() > 0 ) {
            geostatic_displacements_fix( Global.myMesh, Global.mySolver, Param.theDomainZ,
                                         Param.theDeltaT, step );
        }
        Timer_Stop( "Compute addforces gravity" );
    }
}

/** Share the displacement of anchored nodes with other processors. */
static void solver_send_displacement_anchored( mysolver_t* solver )
{
//    HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );

    Timer_Start("3rd schedule send data (sharing)");
    schedule_senddata(Global.mySolver->an_sched, Global.mySolver->tm2,
                      sizeof(fvector_t) / sizeof(solver_float), SHARING,
                      AN_DISP_MSG);
    Timer_Stop("3rd schedule send data (sharing)");

}


/** Adjust the displacements of my LOCAL dangling nodes. */
static void solver_adjust_displacement(  mysolver_t* solver )
{
    Timer_Start( "2nd compute adjust (assignment)" );
    compute_adjust( Global.mySolver->tm2, sizeof(fvector_t) / sizeof(solver_float),
                    ASSIGNMENT );
    Timer_Stop( "2nd compute adjust (assignment)" );
}


/** Share the displacement of dangling nodes with other processors. */
static void solver_send_displacement_dangling( mysolver_t* solver )
{
//    HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );

    Timer_Start( "4th schadule send data (sharing)" );
    schedule_senddata( Global.mySolver->dn_sched, Global.mySolver->tm2,
                       sizeof(fvector_t) / sizeof(solver_float), SHARING,
                       DN_DISP_MSG );
    Timer_Stop( "4th schadule send data (sharing)" );
}


/**
 * Hook for instrumentation or other functionality in the solver_run loop.
 */
static void
solver_loop_hook_bottom( mysolver_t* solver, mesh_t* mesh, int step )
{
    if (0) {
        solver_debug_overflow( solver, mesh, step );
    }
}


static void solver_output_wavefield_close( void )
{
    if (DO_OUTPUT && (Param.FourDOutFp != NULL)) {
        fclose( Param.FourDOutFp );     /* close the output file */
    }
}


static void solver_run_collect_timers( void )
{
    /* Get min and max of individual cumulative times.
     * this is NOT best and worst single step times
     */
    if( Timer_Exists("Print Planes") ) {
        Timer_Reduce("Print Planes",   (TimerKind)(MAX | MIN), comm_solver);
    }

    if( Timer_Exists("Print Stations") ) {
        Timer_Reduce("Print Stations", (TimerKind)(MAX | MIN), comm_solver);
    }

    if ( Timer_Exists("Compute Non-linear Entities") ) {
        Timer_Reduce("Compute Non-linear Entities", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    }

    if ( Timer_Exists("Compute addforces Non-linear") ) {
        Timer_Reduce("Compute addforces Non-linear", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
        Timer_Reduce("Compute addforces gravity",    (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    }

    if ( Timer_Exists("Solver drm output") ) {
        Timer_Reduce("Solver drm output", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    }

    if ( Timer_Exists("Solver drm read displacements") ) {
        Timer_Reduce("Solver drm read displacements", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    }

    if ( Timer_Exists("Solver drm force compute") ) {
        Timer_Reduce("Solver drm force compute", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    }

    Timer_Reduce("Read My Forces",                        (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Compute addforces s",                   (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Compute addforces e",                   (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Damping addforce",                      (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("1st schedule send data (contribution)", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("1st compute adjust (distribution)",     (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("2nd schedule send data (contribution)", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Compute new displacement",              (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("3rd schedule send data (sharing)",      (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("2nd compute adjust (assignment)",       (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("4th schadule send data (sharing)",      (TimerKind)(MAX | MIN | AVERAGE), comm_solver);

    Timer_Reduce("Solver I/O",      (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Compute Physics", (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
    Timer_Reduce("Communication",   (TimerKind)(MAX | MIN | AVERAGE), comm_solver);
}


/**
 * March forward in time and output the result whenever necessary.
 */
static void solver_run()
{
    int32_t step, startingStep;

    solver_run_init_comm( Global.mySolver );

    /* sets new starting step if loading checkpoint */
    if (Param.theUseCheckPoint == 1) {
        startingStep = checkpoint_read(Global.myID, Global.myMesh, Param.theCheckPointingDirOut,
				       Global.theGroupSize, Global.mySolver,comm_solver);
    } else {
        startingStep = 0;
    }

    if (Global.myID == 0) {
        /* print header for monitor file */
        monitor_print( "solver_run() start\nStarting time step = %d\n\n",
                       startingStep );
        monitor_print( "Sim = Simulation time (s), Sol = Solver time (s), WC = Wall Clock Time (s)\n");
    }

    MPI_Barrier( comm_solver );

    /* march forward in time */
    for (step = startingStep; step < Param.theTotalSteps; step++) {

        fvector_t* tmpvector;

        /* prepare for a new iteration
         * swap displacement vectors for t(n) and t(n-1) */
        tmpvector     = Global.mySolver->tm2;
        Global.mySolver->tm2 = Global.mySolver->tm1;
        Global.mySolver->tm1 = tmpvector;

        Timer_Start( "Solver I/O" );
        solver_write_checkpoint( step, startingStep );
        solver_update_status( step, startingStep );
        solver_output_wavefield( step );
        solver_output_planes( Global.mySolver, Global.myID, step );
        solver_output_stations( step );
        solver_output_drm_nodes( Global.mySolver, step, Param.theTotalSteps );
        solver_read_source_forces( step );
        solver_read_drm_displacements( step , Param.theDeltaT ,Param.theTotalSteps );
        Timer_Stop( "Solver I/O" );

        Timer_Start( "Compute Physics" );
        solver_nonlinear_state( Global.mySolver, Global.myMesh, Global.theK1, Global.theK2, step );
        solver_compute_force_source( step );
        solver_compute_effective_drm_force( Global.mySolver, Global.myMesh,Global.theK1, Global.theK2, step, Param.theDeltaT );
        solver_compute_force_stiffness( Global.mySolver, Global.myMesh, Global.theK1, Global.theK2 );
        solver_compute_force_damping( Global.mySolver, Global.myMesh, Global.theK1, Global.theK2 );
        solver_compute_force_gravity( Global.mySolver, Global.myMesh, step );
        solver_compute_force_nonlinear( Global.mySolver, Global.myMesh, Param.theDeltaTSquared );
        Timer_Stop( "Compute Physics" );

        Timer_Start( "Communication" );
        HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );
        solver_send_force_dangling( Global.mySolver );
        solver_adjust_forces( Global.mySolver );
        HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );
        solver_send_force_anchored( Global.mySolver );
        Timer_Stop( "Communication" );

        Timer_Start( "Compute Physics" );
        solver_compute_displacement( Global.mySolver, Global.myMesh );
        solver_geostatic_fix( step );
        solver_load_fixedbase_displacements( Global.mySolver, step );
        Timer_Stop( "Compute Physics" );

        Timer_Start( "Communication" );
        HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );
        solver_send_displacement_anchored( Global.mySolver );
        solver_adjust_displacement( Global.mySolver );
        HU_COND_GLOBAL_BARRIER( Param.theTimingBarriersFlag );
        solver_send_displacement_dangling( Global.mySolver );
        Timer_Stop( "Communication" );

        solver_loop_hook_bottom( Global.mySolver, Global.myMesh, step );
    } /* for (step = ....): all steps */

    solver_drm_close();
    solver_output_wavefield_close();
    solver_run_collect_timers();
}


/**
 * Output the velocity of the mesh nodes for the current timestep. Send
 * the data to Thread 0 who is responsible for dumping the data to disk.
 *
 * \note This is only useful for very small distributed setups where:
 * - There is no parallel file system.
 * - The distributed file system does not have POSIX semantics.
 * - The distributed file system performs poorly with just a few clients
 *   and the effective overall performance is better when only one client
 *   (i.e., PE 0) is writing to the FS.
 */
void
solver_output_seq()
{
    int32_t nindex;
    int32_t batchlimit, idx;

#ifdef DEBUG
    int64_t gnid_prev, gnid_current;
    int32_t first_counted;
#endif /* DEBUG */

    batchlimit = BATCH * 10;

    /* Allocate a fixed size buffer space if not initiazlied */
    if (Global.myVelocityTable == NULL) {
	Global.myVelocityTable = (fvector_t *)calloc(batchlimit, sizeof(fvector_t));
	if (Global.myVelocityTable == NULL) {
	    fprintf(stderr,  "Thread %d: solver_output_seq: out of memory\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
    }

    if (Global.myID == 0) {
	int32_t procid;
#ifdef DEBUG
	first_counted = 0;
#endif

	if (Param.FourDOutFp == NULL) {
	    out_hdr_t out_hdr;

	    /* First output, create the output file */
	    Param.FourDOutFp = fopen(Param.FourDOutFile, "w+");
	    if (Param.FourDOutFp == NULL) {
		fprintf(stderr, "Thread 0: solver_output_seq: ");
		fprintf(stderr, "cannot create %s\n", Param.FourDOutFile);
		perror("fopen");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }

	    /* Write the header that contains the metadata */
	    out_hdr.domain_x = Param.theDomainX;
	    out_hdr.domain_y = Param.theDomainY;
	    out_hdr.domain_z = Param.theDomainZ;
	    out_hdr.total_nodes = Global.theNTotal;
	    out_hdr.total_elements = Global.theETotal;
	    out_hdr.mesh_ticksize = Global.myMesh->ticksize;
	    out_hdr.output_steps = (Param.theTotalSteps - 1) / Param.theRate + 1;

	    if (fwrite(&out_hdr, sizeof(out_hdr_t), 1, Param.FourDOutFp) != 1){
		fprintf(stderr, "Thread 0: solver_output_seq: ");
		fprintf(stderr, "fail to write 4D-out header info\n");
		perror("fwrite");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
	}

	/* Output my owned nodes' velocities */
	nindex = 0;
	while (nindex < Global.myMesh->nharbored) {
	    fvector_t vel;

	    if (Global.myMesh->nodeTable[nindex].ismine) {
		vel.f[0] =
		    (Global.mySolver->tm1[nindex].f[0]
		     - Global.mySolver->tm2[nindex].f[0])  / Param.theDeltaT;
		vel.f[1] =
		    (Global.mySolver->tm1[nindex].f[1]
		     - Global.mySolver->tm2[nindex].f[1])  / Param.theDeltaT;
		vel.f[2] =
		    (Global.mySolver->tm1[nindex].f[2]
		     - Global.mySolver->tm2[nindex].f[2])  / Param.theDeltaT;


		if (fwrite(&vel, sizeof(fvector_t), 1, Param.FourDOutFp) != 1) {
		    fprintf(stderr, "Thread 0: solver_output_seq: error\n");
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}

		Param.the4DOutSize += sizeof(fvector_t);

#ifdef DEBUG
		gnid_current = Global.myMesh->nodeTable[nindex].gnid;

		if (first_counted) {
		    if (gnid_prev != (gnid_current - 1)) {
			fprintf( stderr, "PE 0: uncontinuous gnid\n"
				 "   gnid_prev = %" INT64_FMT
				 ", gnid_current = %" INT64_FMT "\n",
				 gnid_prev, gnid_current);
		    }
		} else {
		    first_counted = 1;
		}

		gnid_prev = gnid_current;

		if ((vel.f[0] != 0) ||
		    (vel.f[1] != 0) ||
		    (vel.f[2] != 0)) {
		    /*
		    fprintf(stderr, "Thread 0: Node %ld	 non-zero values\n",
			    gnid_current);
		    */
		}
#endif /* DEBUG */

	    }

	    nindex++;
	}

	/* Receive data from other processors */
	for (procid = 1; procid < Global.theGroupSize; procid++) {
	    MPI_Status status;
	    int32_t rcvbytecount;

	    /* Signal the next processor to go ahead */
	    MPI_Send(NULL, 0, MPI_CHAR, procid, GOAHEAD_MSG, comm_solver);

	    while (1) {
		MPI_Probe(procid, OUT4D_MSG, comm_solver, &status);
		MPI_Get_count(&status, MPI_CHAR, &rcvbytecount);

		/* Receive the data even if rcvbytecount == 0. Otherwise
		   the 0-byte message would get stuck in the message queue */
		MPI_Recv(Global.myVelocityTable, rcvbytecount, MPI_CHAR, procid,
			 OUT4D_MSG, comm_solver, &status);

		if (rcvbytecount == 0) {
		    /* Done */
		    break;
		}

		if (fwrite(Global.myVelocityTable, rcvbytecount, 1, Param.FourDOutFp) != 1) {
		    fprintf(stderr, "Thread 0: solver_output_seq: error\n");
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}

		Param.the4DOutSize += rcvbytecount;

	    } /* while there is more data to be received from procid */
	} /* for all the processors */

    } else {
	/* Processors other than 0 needs to send data to 0 */
	int32_t sndbytecount;
	MPI_Status status;

	/* Wait for me turn */
	MPI_Recv(NULL, 0, MPI_CHAR, 0, GOAHEAD_MSG, comm_solver, &status);

#ifdef DEBUG
	first_counted = 0;
#endif


	nindex = 0;
	while (nindex < Global.myMesh->nharbored) {
	    fvector_t *velp;

	    idx = 0;
	    while ((idx < batchlimit) &&
		   (nindex < Global.myMesh->nharbored)) {

		if (Global.myMesh->nodeTable[nindex].ismine) {

		    velp = &Global.myVelocityTable[idx];

		    velp->f[0] =
			(Global.mySolver->tm1[nindex].f[0]
			 - Global.mySolver->tm2[nindex].f[0])	/ Param.theDeltaT;
		    velp->f[1] =
			(Global.mySolver->tm1[nindex].f[1]
			 - Global.mySolver->tm2[nindex].f[1])	/ Param.theDeltaT;
		    velp->f[2] =
			(Global.mySolver->tm1[nindex].f[2]
			 - Global.mySolver->tm2[nindex].f[2])	/ Param.theDeltaT;


		    idx++;

#ifdef DEBUG
		    gnid_current = Global.myMesh->nodeTable[nindex].gnid;

		    if (first_counted) {
			if (gnid_prev != (gnid_current - 1)) {
			    fprintf( stderr, "PE %d uncontinuous gnid\n"
				     "	gnid_prev = %" INT64_FMT
				     ", gnid_current = %" INT64_FMT "\n",
				     Global.myID, gnid_prev, gnid_current );
			}
		    } else {
			first_counted = 1;
		    }

		    gnid_prev = gnid_current;

		    /* debug */
		    /*
		    if ((velp->f[0] != 0) ||
			(velp->f[1] != 0) ||
			(velp->f[2] != 0)) {
			fprintf(stderr,
				"Thread %d: there are non-zero values\n",
				Global.myID);
				}
		    */
#endif /* DEBUG */

		}

		nindex++;
	    }

	    /* Send data to proc 0 */

	    if (idx > 0) {
		/* I have some real data to send */
		sndbytecount = idx * sizeof(fvector_t);
		MPI_Send(Global.myVelocityTable, sndbytecount, MPI_CHAR, 0, OUT4D_MSG,
			 comm_solver);
	    }
	} /* While there is data left to be sent */

	/* Send an empty message to indicate the end of my transfer */
	MPI_Send(NULL, 0, MPI_CHAR, 0, OUT4D_MSG, comm_solver);
    }

    return;
}


/**
 * Allocate and initialize a scheduler.
 */
static schedule_t*
schedule_new()
{
    schedule_t *sched;

    sched = (schedule_t *)malloc(sizeof(schedule_t));
    if (sched == NULL)
	return NULL;

    sched->c_count = 0;
    sched->first_c = NULL;
    sched->messenger_c = (messenger_t **)
	calloc(Global.theGroupSize, sizeof(messenger_t *));
    if (sched->messenger_c == NULL)
	return NULL;

    sched->s_count = 0;
    sched->first_s = NULL;
    sched->messenger_s = (messenger_t **)
	calloc(Global.theGroupSize, sizeof(messenger_t *));
    if (sched->messenger_s == NULL)
	return NULL;

    return sched;
}



/**
 * Allocate the mapping table for each of my messenger.
 */
static void
schedule_allocmapping( schedule_t *sched )
{
    messenger_t *messenger;
    int32_t nodecount;

    messenger = sched->first_c;
    while (messenger != NULL) {
	nodecount = messenger->nodecount;

	messenger->mapping =
	    (int32_t *)calloc(nodecount, sizeof(int32_t));

	if (messenger->mapping == NULL) {
	    fprintf(stderr, "Thread %d: schedule_allocamapping: ", Global.myID);
	    fprintf(stderr, " out of memory\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	messenger = messenger->next;
    }

    messenger = sched->first_s;
    while (messenger != NULL) {
	nodecount = messenger->nodecount;

	messenger->mapping =
	    (int32_t *)calloc(nodecount, sizeof(int32_t));

	if (messenger->mapping == NULL) {
	    fprintf(stderr, "Thread %d: schedule_allocamapping: ", Global.myID);
	    fprintf(stderr, " out of memory\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	messenger = messenger->next;
    }

    return;
}


/**
 * Allocate MPI controls for non-blocing receives.
 */
static void
schedule_allocMPIctl( schedule_t* sched )
{
    if (sched->c_count != 0) {
	sched->irecvreqs_c =
	    (MPI_Request *)malloc(sizeof(MPI_Request) * sched->c_count);
	sched->irecvstats_c =
	    (MPI_Status *)malloc(sizeof(MPI_Status) * sched->c_count);

	if ((sched->irecvreqs_c == NULL) ||
	    (sched->irecvstats_c == NULL)) {
	    fprintf(stderr, "Thread %d: schedule_allocMPIctl: ", Global.myID);
	    fprintf(stderr, "out of memory\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
    } else {
	sched->irecvreqs_c = NULL;
	sched->irecvstats_c = NULL;
    }

    if (sched->s_count != 0) {
	sched->irecvreqs_s =
	    (MPI_Request *)malloc(sizeof(MPI_Request) * sched->s_count);
	sched->irecvstats_s =
	    (MPI_Status *)malloc(sizeof(MPI_Status) * sched->s_count);

	if ((sched->irecvreqs_s == NULL) ||
	    (sched->irecvstats_s == NULL)) {
	    fprintf(stderr, "Thread %d: schedule_allocMPIctl: ", Global.myID);
	    fprintf(stderr, "out of memory\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
    } else {
	sched->irecvreqs_s = NULL;
	sched->irecvstats_s = NULL;
    }

    return;
}


/**
 * Build a communication schedule using local information.
 */
static void
schedule_build( mesh_t* mesh, schedule_t* dnsched, schedule_t* ansched )
{
    int32_t nindex;
    node_t* nodep;
    messenger_t* messenger;

    for (nindex = 0; nindex < mesh->nharbored; nindex++) {
	nodep = &mesh->nodeTable[nindex];
	int32_t owner, sharer;

	if (!nodep->ismine) {
	    /* I do not own this node. Add its owner processor to my c-list */

	    owner = nodep->proc.ownerid;

	    if (nodep->isanchored) {
		messenger = ansched->messenger_c[owner];
	    } else {
		messenger = dnsched->messenger_c[owner];
	    }

	    if (messenger == NULL) {
		messenger = messenger_new(owner);
		if (messenger == NULL) {
		    fprintf(stderr, "Thread %d: schedule_build: ", Global.myID);
		    fprintf(stderr, "out of memory.\n");
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}

		if (nodep->isanchored) {
		    ansched->c_count++;
		    ansched->messenger_c[owner] = messenger;
		    messenger->next = ansched->first_c;
		    ansched->first_c = messenger;
		} else {
		    dnsched->c_count++;
		    dnsched->messenger_c[owner] = messenger;
		    messenger->next = dnsched->first_c;
		    dnsched->first_c = messenger;
		}
	    }

	    /* Update the number of nodecount for the messenger */
	    messenger->nodecount++;

	} else {
	    /* I own this node. Add any sharing processor to my s-list */

	    int32link_t *int32link;

	    int32link = nodep->proc.share;
	    while (int32link != NULL) {
		sharer = int32link->id;

		if (nodep->isanchored) {
		    messenger = ansched->messenger_s[sharer];
		} else {
		    messenger = dnsched->messenger_s[sharer];
		}

		if (messenger == NULL) {
		    messenger = messenger_new(sharer);
		    if (messenger == NULL) {
			fprintf(stderr, "Thread %d: schedule_build: ", Global.myID);
			fprintf(stderr, "out of memory.\n");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		    }

		    if (nodep->isanchored) {
			ansched->s_count++;
			ansched->messenger_s[sharer] = messenger;
			messenger->next = ansched->first_s;
			ansched->first_s = messenger;
		    } else {
			dnsched->s_count++;
			dnsched->messenger_s[sharer] = messenger;
			messenger->next = dnsched->first_s;
			dnsched->first_s = messenger;
		    }
		}

		/* Update the nodecount */
		messenger->nodecount++;

		/* Move to the next sharing processor */
		int32link = int32link->next;
	    }
	}
    }

    /* Allocate MPI controls */
    schedule_allocMPIctl(ansched);
    schedule_allocMPIctl(dnsched);

    /* Allocate localnode table for each of the messegners I have */
    schedule_allocmapping(ansched);
    schedule_allocmapping(dnsched);

    /* Go through the nodes again and fill out the mapping table */
    for (nindex = 0; nindex < mesh->nharbored; nindex++) {
	nodep = &mesh->nodeTable[nindex];
	int32_t owner, sharer;

	if (!nodep->ismine) {
	    /* I do not own this node. Add its owner processor to my c-list */
	    owner = nodep->proc.ownerid;

	    if (nodep->isanchored) {
		messenger = ansched->messenger_c[owner];
	    } else {
		messenger = dnsched->messenger_c[owner];
	    }

	    if (messenger == NULL) {
		fprintf(stderr, "Thread %d: schedule_build: ", Global.myID);
		fprintf(stderr, "encounter NULL messenger.\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }

	    /* Fill in the mapping */
	    messenger->mapping[messenger->nidx] = nindex;
	    messenger->nidx++;

	} else {
	    /* I own this node. Add any sharing processor to my s-list */

	    int32link_t *int32link;

	    int32link = nodep->proc.share;
	    while (int32link != NULL) {
		sharer = int32link->id;

		if (nodep->isanchored) {
		    messenger = ansched->messenger_s[sharer];
		} else {
		    messenger = dnsched->messenger_s[sharer];
		}

		if (messenger == NULL) {
		    fprintf(stderr, "Thread %d: schedule_build: ", Global.myID);
		    fprintf(stderr, "encounter NULL messenger.\n");
		    MPI_Abort(MPI_COMM_WORLD, ERROR);
		    exit(1);
		}

		messenger->mapping[messenger->nidx] = nindex;
		messenger->nidx++;

		/* Move to the next sharing processor */
		int32link = int32link->next;
	    }
	}
    }

    return ;
}



/**
 * Release memory used by a scheduler.
 */
static void
schedule_delete( schedule_t* sched )
{
    messenger_t *current, *next;

    /* Release messengers overseeing my contributions */
    current = sched->first_c;
    while (current != NULL) {
	next = current->next;

	messenger_delete(current);
	current = next;
    }

    /* Release messengers overseeing shareing with others */
    current = sched->first_s;
    while (current != NULL) {
	next = current->next;

	messenger_delete(current);
	current = next;
    }

    if (sched->irecvreqs_c != NULL)
	free(sched->irecvreqs_c);
    if (sched->irecvstats_c != NULL)
	free(sched->irecvstats_c);
    free(sched->messenger_c);

    if (sched->irecvreqs_s != NULL)
	free(sched->irecvreqs_s);
    if (sched->irecvstats_s != NULL)
	free(sched->irecvstats_s);
    free(sched->messenger_s);

    free(sched);

    return;
}



/**
 * Allocate the memory for data exchange.
 */
static void
schedule_prepare( schedule_t* sched, int32_t c_outsize, int32_t c_insize,
		  int32_t s_outsize, int32_t s_insize )
{
    messenger_t* messenger;

    messenger = sched->first_c;
    while (messenger != NULL) {
	messenger_set(messenger, c_outsize, c_insize);
	messenger = messenger->next;
    }

    messenger = sched->first_s;
    while (messenger != NULL) {
	messenger_set(messenger, s_outsize, s_insize);
	messenger = messenger->next;
    }

    return;

}



/**
 * Assemble the proper information for the group of messengers managed by
 * a scheduler and send the data.
 *
 * \param direction: CONTRIBUTION or SHARING.
 */
static void
schedule_senddata(schedule_t *sched, void *valuetable, int32_t itemsperentry,
		  int32_t direction, int32_t msgtag)
{
    messenger_t *send_messenger, *recv_messenger;
    int32_t irecvcount, irecvnum, bytesize;
    MPI_Request *irecvreqs;
    MPI_Status *irecvstats;

#ifdef DEBUG
    int64_t *gnidp;
#endif /* DEBUG */

    if (direction == CONTRIBUTION) {
	send_messenger = sched->first_c;
	recv_messenger = sched->first_s;
	irecvcount = sched->s_count;
	irecvreqs = sched->irecvreqs_s;
	irecvstats = sched->irecvstats_s;
    } else {
	send_messenger = sched->first_s;
	recv_messenger = sched->first_c;
	irecvcount = sched->c_count;
	irecvreqs = sched->irecvreqs_c;
	irecvstats = sched->irecvstats_c;
    }

    /* Post receives */
    irecvnum = 0;
    while (recv_messenger != NULL) {
	bytesize = recv_messenger->nodecount * recv_messenger->insize;
	MPI_Irecv(recv_messenger->indata, bytesize, MPI_CHAR,
		  recv_messenger->procid, msgtag, comm_solver,
		  &irecvreqs[irecvnum]);

	irecvnum++;
	recv_messenger = recv_messenger->next;
    }

    /* Asssemble outgoing messages */
    while (send_messenger != NULL) {
	int32_t lnid, idx, entry;
	solver_float *dvalue;
	solver_float *out;

	for (idx = 0; idx < send_messenger->nidx; idx++) {

	    lnid = send_messenger->mapping[idx];

	    out = (solver_float *)((char *)send_messenger->outdata +
			     send_messenger->outsize * idx);

	    dvalue = (solver_float *)valuetable + itemsperentry * lnid;

	    for (entry = 0; entry < itemsperentry; entry++)
		*(out + entry) = *(dvalue + entry);

#ifdef DEBUG
	    /* For debug, carry the global node id */
	    gnidp = (int64_t *)
		((char *)out + itemsperentry * sizeof(solver_float));
	    *gnidp = Global.myMesh->nodeTable[lnid].gnid;
#endif /* DEBUG */
	}

	send_messenger = send_messenger->next;
    }

    /* Revisit messengers */
    if (direction == CONTRIBUTION) {
	send_messenger = sched->first_c;
	recv_messenger = sched->first_s;
    } else {
	send_messenger = sched->first_s;
	recv_messenger = sched->first_c;
    }

    /* Send the data */
    while (send_messenger != NULL) {
	bytesize = send_messenger->nodecount * send_messenger->outsize;
	MPI_Send(send_messenger->outdata, bytesize, MPI_CHAR,
		 send_messenger->procid, msgtag, comm_solver);
	send_messenger = send_messenger->next;
    }

    /* Wait till I receive all the data I want */
    if (irecvcount != 0) {
	MPI_Waitall(irecvcount, irecvreqs, irecvstats);
    }

    while (recv_messenger != NULL) {
	int32_t lnid, idx, entry;
	solver_float *dvalue;
	solver_float *in;

	for (idx = 0; idx < recv_messenger->nidx; idx++) {

	    lnid = recv_messenger->mapping[idx];

	    in = (solver_float *)((char *)recv_messenger->indata +
			    recv_messenger->insize * idx);

	    dvalue = (solver_float *)valuetable + itemsperentry * lnid;

	    for (entry = 0; entry < itemsperentry; entry++) {
		if (direction == CONTRIBUTION) {
		    *(dvalue + entry) += *(in + entry);
		} else {
		    /* SHARING, overwrite my local value */
		    *(dvalue + entry) = *(in + entry);
		}
	    }

#ifdef DEBUG
	    /* For debug, check the global node id */
	    gnidp = (int64_t *)
		((char *)in + itemsperentry * sizeof(solver_float));

	    if (*gnidp != Global.myMesh->nodeTable[lnid].gnid) {
		fprintf(stderr, "Thread %d: solver_init: gnids do not match\n",
			Global.myID);
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
#endif /* DEBUG */
	} /* for all the incoming data */

	recv_messenger = recv_messenger->next;
    }


    /* MPI_Barrier(comm_solver);  */

    return;
}


/**
 * Print messenger entry data to the given output stream.
 * The output line contains the following information:
 * "pe_id type c/s rpe nodecount outsize insize"
 *
 * \note Printing is not synchronized accross PEs, so lines can be out of
 * order / interleaved
 */
static int
messenger_print( messenger_t* msg, char type, char cs, FILE* out )
{
    int ret;

    assert( NULL != msg );

    ret = fprintf( out, "msg_info: %d %c %c %d %d %d %d %d\n", Global.myID, type, cs,
		   msg->procid, msg->nodecount, msg->outsize, msg->insize,
		   msg->nodecount * msg->outsize );

    if (ret > 0 || !Param.theSchedulePrintErrorCheckFlag) {
	ret = 0;
    } else {  /* ret <= 0 && Param.theSchedulePrintErrorCheckFlag */
	perror( "Warning! fprintf(...) failed" );
	fprintf(stderr, "PE id = %d, ret = %d, out = 0x%p\n", Global.myID, ret, out);
	fflush( stderr );
    }

    return ret;
}


/**
 * Print the messenger list for a scheduler
 *
 * \note Printing is not synchronized accross PEs, so lines can be out of
 * order / interleaved
 */
static int
schedule_print_messenger_list( schedule_t* sched, messenger_t* msg, int count,
			       char type, char cs, FILE* out )
{
    messenger_t* m_p;
    int my_count;
    int ret = 0;

    assert( NULL != sched );

    m_p = msg;


    for (my_count = 0; m_p != NULL && ret >= 0; my_count++, m_p = m_p->next) {
	ret += messenger_print( msg, type, cs, out );
    }


    if (ret < 0) {
	return -1;
    }

    if (my_count != count) {
	fprintf( stderr, "Warning! schedule list count does not match: "
		 "%u %c %c %d %d\n", Global.myID, type, cs, count, my_count );
    }

    return ret;
}


/**
 * Print scheduler communication to a given output stream.
 *
 * Print a line per entry in the schedule_t structure, i.e., an line
 * per messenger_t entry in the list.
 *
 * Each line has the following format:
 * "pe_id type c/s rpe nodecount outsize insize"
 */
static int
schedule_print_detail( schedule_t* sched, char type, FILE* out )
{
    int ret = 0;

    ret += schedule_print_messenger_list( sched, sched->first_c,
					  sched->c_count, type, 'c', out );
    ret += schedule_print_messenger_list( sched, sched->first_s,
					  sched->s_count, type, 's', out );

    return ret;
}


/**
 * Print scheduler communication to a given output stream.
 *
 * Print a line per entry in the schedule_t structure, i.e., an line
 * per messenger_t entry in the list.
 *
 * Each line has the following format:
 * "pe_id type s_count c_count"
 *
 * \note Printing is not synchronized accross PEs, so lines can be out of
 * order / interleaved
 */
static int
schedule_print( schedule_t* sched, char type, FILE* out )
{
    int ret;

    assert( NULL != sched );
    assert( NULL != out );

    ret = fprintf( out, "sch_info: %u %c %d %d\n", Global.myID, type, sched->c_count,
		   sched->s_count );


    if (ret > 0 || !Param.theSchedulePrintErrorCheckFlag) {
	ret = 0;
    }

    return ret;
}


/**
 * \note Printing is not synchronized accross PEs, so lines can be out of
 * order / interleaved
 */
static int
solver_print_schedules_imp( mysolver_t* solver, FILE* out )
{
    int ret = 0;

    assert( solver != NULL );
    assert( out != NULL );

    MPI_Barrier(comm_solver);

    /* print the high-level schedule per PE */
    if (Global.myID == 0) { /* print some header information */
	fputs( "# ----------------------------------------------------------\n"
	       "# Content: Solver schedule information\n"
	       "# pe_id d/a s_count c_count\n", out );
    }
    fflush( out );
    fdatasync( fileno( out ) );

    MPI_Barrier( comm_solver );

    /* this is not synchronized, so it might be out of order */
    ret += schedule_print( solver->an_sched, 'a', out );
    ret += schedule_print( solver->dn_sched, 'd', out );

    fflush( out );
    fdatasync( fileno( out ) );
    MPI_Barrier( comm_solver );

    /* print the schedule detail */
    if (Global.myID == 0) { /* print some header information */
	fputs( "\n\n"
	       "# ----------------------------------------------------------\n"
	       "# Content: Solver schedule detail\n"
	       "# pe_id d/a c/s rpe nodecount outsize insize msgsize\n", out );
	fflush( out );
	fdatasync( fileno( out ) );
    }

    MPI_Barrier( comm_solver );

    /* this is not synchronized, so it might be out of order */
    ret += schedule_print_detail( solver->an_sched, 'a', out );
    ret += schedule_print_detail( solver->dn_sched, 'd', out );

    fflush( out );
    fdatasync( fileno( out ) );
    MPI_Barrier( comm_solver );

    if (Global.myID == 0) {
	fputs( "# ----------------------------------------------------------\n"
	       "\n", out );
	fflush( out );
	fdatasync( fileno( out ) );
    }

    MPI_Barrier( comm_solver );

    return ret;
}


/**
 * Wrapper to print the solver schedules to stdout and a file with
 * the given name.
 *
 * \note Printing is not synchronized accross PEs, so lines can be out of
 * order / interleaved
 */
static int
solver_print_schedules( mysolver_t* solver )
{
    FILE* out;
    int ret = 0;

    if (Param.theSchedulePrintToStdout) {
	/* print schedules to standard output */
	ret += solver_print_schedules_imp( solver, stdout );

	if (ret < 0) {
	    fprintf( stderr, "Warning! printing schedules to standard output "
		     "failed for PE %d\n", Global.myID );
	}
    }

    if (Param.theSchedulePrintToFile && (Param.theSchedulePrintFilename != NULL)) {
	/* print schedules to the given file */
	out = fopen( Param.theSchedulePrintFilename, "a" );

	if (NULL == out) { /* this is not fatal */
	    fprintf( stderr, "Warning!, PE# %d failed to open output file for "
		     "printing the communication schedule\n", Global.myID );
	    return -1;
	}

	ret = solver_print_schedules_imp( solver, out );

	if (ret < 0) {
	    fprintf( stderr, "Warning! PE %d could not print schedules "
		     "to file\n", Global.myID );
	}

	fclose( out );
    }

    return ret;
}


/**
 * messenger_new: Allocate and initialize a messenger.
 *
 */
static messenger_t *messenger_new(int32_t procid)
{
    messenger_t *messenger;

    messenger = (messenger_t *)calloc(1, sizeof(messenger_t));

    if (messenger == NULL)
	return NULL;

    messenger->procid = procid;

    return messenger;
}



/**
 * messenger_delete: Release memory used by a messenger.
 *
 */
static void messenger_delete(messenger_t *messenger)
{
    if (messenger == NULL)
	return;

    if (messenger->outdata != NULL)
	free(messenger->outdata);

    if (messenger->indata != NULL)
	free(messenger->indata);

    free(messenger->mapping);

    free(messenger);

    return;
}



/**
 * messenger_set: Free any data memory used by the messenger for
 *                previous communication. And allocate new memory
 *                for the new round of communication.
 *
 */
static void
messenger_set(messenger_t *messenger, int32_t outsize, int32_t insize)
{
    if (messenger->outdata != NULL) {
	free(messenger->outdata);
	messenger->outdata = NULL;
    }

    if (messenger->indata != NULL) {
	free(messenger->indata);
	messenger->indata = NULL;
    }

    messenger->outsize = outsize;
    messenger->insize = insize;

    if (outsize != 0) {
	messenger->outdata = calloc(messenger->nodecount, outsize);
	if (messenger->outdata == NULL) {
	    fprintf(stderr, "Thread %d: messenger_set: out of memory\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
    }

    if (insize != 0) {
	messenger->indata = calloc(messenger->nodecount, insize);
	if (messenger->indata == NULL) {
	    fprintf(stderr, "Thread %d: messenger_set: out of memory\n",
		    Global.myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
    }

    return;
}



/**
 * messenger_countnodes: Count the total number of nodes (that need to
 *                       be communicated) on the messenger link list.
 *
 */
static int32_t
messenger_countnodes(messenger_t *first)
{
    messenger_t *messenger;
    int32_t total;

    total = 0;
    messenger = first;
    while (messenger != NULL) {
	total += messenger->nodecount;
	messenger = messenger->next;
    }

    return total;
}


/**
 * Compute K1, K2 and K3 for a  linear  elastic,  homogeneous
 * and isotropic cube; and their off-diagonal matrices.
 *
 *            K1 = Integral(   Del(Phi(j))  * T(Del(Phi(i)))     )
 *            K2 = Integral(   Del(Phi(i))  * T(Del(Phi(j)))     )
 *            K3 = Integral( T(Del(Phi(i))) *   Del(Phi(j))  * I )
 *
 *            where:
 *
 *            T(W) = transpose of W.
 *
 * \note K offdiagonal has been commented out because is no longer needed
 * after the changes made on the solution algorith due to the new
 * form to handle the damping involved terms.
 */
static void compute_K()
{
    int i, j; /* indices of the block matrices, i for rows, j for columns */
    int k, l; /* indices of 3 x 3 matrices, k for rows, l for columns     */

    double x[3][8]={ {-1,  1, -1,  1, -1,  1, -1, 1} ,
		     {-1, -1,  1,  1, -1, -1,  1, 1} ,
		     {-1, -1, -1, -1,  1,  1,  1, 1} };

    /* compute K3 first */
    memset(Global.theK3, 0, 8 * 8 * sizeof(fmatrix_t));
    for (i = 0 ;  i < 8; i++) {
	for (j = 0; j <8; j++){
	    /* for  each 3 x 3 matrix representing (node i, node j),
               each of these matrices is diagonal, off-diagonal elements
               have been set to 0 in memset() */

	    fmatrix_t *matPtr;
	    matPtr = &Global.theK3[i][j];

	    for (k = 0; k < 3; k++) {
		/* set the diagonal values for the 3 x 3 matrix.
		   Note the rotation of the index */
		double I1, I2, I3;

		I1 = INTEGRAL_1
		    (x[(k + 0) % 3][i], x[(k + 0) % 3][j],
		     x[(k + 1) % 3][i], x[(k + 1) % 3][j],
		     x[(k + 2) % 3][i], x[(k + 2) % 3][j]);

		I2 = INTEGRAL_1
		    (x[(k + 1) % 3][i], x[(k + 1) % 3][j],
		     x[(k + 2) % 3][i], x[(k + 2) % 3][j],
		     x[(k + 0) % 3][i], x[(k + 0) % 3][j]);


		I3 = INTEGRAL_1
		    (x[(k + 2) % 3][i], x[(k + 2) % 3][j],
		     x[(k + 0) % 3][i], x[(k + 0) % 3][j],
		     x[(k + 1) % 3][i], x[(k + 1) % 3][j]);

		matPtr->f[k][k] = I1 + I2 + I3;
	    } /* k*/
	} /* j*/
    } /* i */

    /* compute K1 and K2. They are not diagonal either, but they are
       indeed symmetric globablly */
    for (i = 0; i < 8; i++) {
	for (j = 0; j < 8; j++) {
	    /* for  each 3 x 3 matrix representing (node i, node j)*/

	    fmatrix_t *matPtr0, *matPtr1;
	    matPtr0 = &Global.theK1[i][j];
	    matPtr1 = &Global.theK2[i][j];

	    for (k = 0; k <  3; k++)
		for (l = 0; l < 3; l++) {
		    if (k == l) {
			/* Initialize the diagnoal elements */
			matPtr0->f[k][k] =
			    INTEGRAL_1
			    (x[(k + 0) % 3][i], x[(k + 0) % 3][j],
			     x[(k + 1) % 3][i], x[(k + 1) % 3][j],
			     x[(k + 2) % 3][i], x[(k + 2) % 3][j]);


			matPtr1->f[k][k] =
			    INTEGRAL_1
			    (x[(k + 0) % 3][j], x[(k + 0) % 3][i],
			     x[(k + 1) % 3][j], x[(k + 1) % 3][i],
			     x[(k + 2) % 3][j], x[(k + 2) % 3][i]);

		    } else {
			/* Initialize off-diagonal elements */

			matPtr0->f[k][l] =
			    INTEGRAL_2
			    (x[k][j], x[l][i],
			     x[3 - (k + l)][j], x[3 - (k + l)][i]);

			matPtr1->f[k][l] =
			    INTEGRAL_2
			    (x[k][i], x[l][j],
			     x[3 - (k + l)][i], x[3 - (k + l)][j]);

		    } /* else */
		} /* for l */
	} /* for j */
    } /* i */

    /* Old code for damping                                    */
    /* Create the off-diagonal matrices                        */
    /* memcpy(theK1_offdiag, Global.theK1, sizeof(double) * 24 * 24); */
    /* memcpy(theK2_offdiag, Global.theK2, sizeof(double) * 24 * 24); */
    /* memcpy(theK3_offdiag, Global.theK3, sizeof(double) * 24 * 24); */
    /* for (i = 0; i <8; i++) {                                */
    /*     theK1_offdiag[i][i].f[0][0] = 0;                    */
    /*     theK1_offdiag[i][i].f[1][1] = 0;                    */
    /*     theK1_offdiag[i][i].f[2][2] = 0;                    */
    /*     theK2_offdiag[i][i].f[0][0] = 0;                    */
    /*     theK2_offdiag[i][i].f[1][1] = 0;                    */
    /*     theK2_offdiag[i][i].f[2][2] = 0;                    */
    /*     theK3_offdiag[i][i].f[0][0] = 0;                    */
    /*     theK3_offdiag[i][i].f[1][1] = 0;                    */
    /*     theK3_offdiag[i][i].f[2][2] = 0;                    */
    /* }                                                       */

    /* New code:
     * First option to solve double K1 K3 computation in the
     * time steps is to merge both of them in K1 only
     */
    for ( i = 0; i < 8; i++ )
    {
	for ( j = 0; j < 8; j++ )
	{
	    for ( k = 0; k < 3; k++ )
	    {
		for ( l = 0; l < 3; l++ )
		{
		    Global.theK1[i][j].f[k][l] += Global.theK3[i][j].f[k][l];
		}
	    }
	}
    }

    return;
}

static void constract_Quality_Factor_Table()
{
int i,j;
double local_QTABLE[26][6] = {{ 5.,	0.211111102, 0.236842104, 0.032142857, 0.271428571,	0.14},
		{6.25,	0.188888889,	0.184210526,	0.039893617,	0.336879433,	0.10152},
		{8.33,	0.157777778,	0.139473684,	0.045,	0.38,	0.07},
		{10., 0.137777765, 0.12105263, 0.032942899, 0.27818448, 0.0683},
		{15., 0.097777765,	0.08105263,	0.032942899,	0.27818448,	0.045},
		{20., 0.078139527, 0.060526314,	0.031409788, 0.277574872, 0.034225},
		{25., 0.064285708, 0.049999999,	0.031578947, 0.285714286, 0.0266},
		{30.,	0.053658537,	0.044736842,	0.026640676,	0.24691358,	0.023085},
		{35.,	0.046341463,	0.038157895,	0.02709848,	0.251156642,	0.019669},
		{40.,	0.040487805,	0.034210526,	0.025949367,	0.240506329,	0.01738},
		{45.,	0.036585366,	0.028947368,	0.031393568,	0.290964778,	0.014366},
		{50.,	0.032926829,	0.026315789,	0.032488114,	0.30110935,	0.01262},
		{60.,     0.0279,    0.0223,    0.0275,    0.2545,    0.0114},
		{70.,   0.024,			0.019,			0.032488114,    0.30110935, 0.0083},
		{80.,  0.0207,    0.0174,    0.0251,    0.2326,    0.0088},
		{90.,    0.0187,    0.0154,    0.0244,    0.2256,    0.0079},
		{100.,	0.017,	0.014,	0.028021016,	0.288966725,	0.006281},
		{120.,     0.0142,    0.0115,    0.0280,   0.2700,    0.0052},
		{150.,  0.0114,    0.0094,    0.0240,    0.2316,    0.0047},
		{200.,	0.0085,	0.00705,	0.022603978,	0.226039783,	0.0035392},
		{250., 0.0069,    0.0055,    0.0269,    0.2596,    0.0027},
		{300.,	0.0057,	0.0047,	0.027072758,	0.279187817,	0.0021276},
		{350,  0.0048,    0.0040,    0.0242,    0.2339,    0.0020},
		{400.,	0.0043,	0.0036,	0.021425572,	0.214255718,	0.0017935},
		{450., 0.0039,    0.0030,   0.0280,    0.2710,    0.0015},
		{500.,	0.0035,	0.00285,	0.023408925,	0.241404535,	0.001367}
};

for(i = 0; i < 18; i++)
{
	for(j = 0; j < 6; j++)
	{
		Global.theQTABLE[i][j] = local_QTABLE[i][j];
//		printf("%f ",theQTABLE[i][j]);
	}
//	printf("\n");
}
return;
}


/**
 * compute_setflag:
 *
 * - results from the discussion with Leo
 * - set flags as if in the full space.
 * - the main() routine will set the flag properly in case half-space
 *   is desired.
 *
 */
#ifdef BOUNDARY
static char
compute_setflag(tick_t ldb[3], tick_t ruf[3], tick_t p1[3], tick_t p2[3])
{
    char flag;

    flag = 13; /* indicate internal element */

    if (ldb[0] == p1[0])
	flag = 12;
    if (ldb[1] == p1[1])
	flag = 10;
    if (ldb[2] == p1[2])
	flag = 4;

    if (ruf[0] == p2[0])
	flag = 14;
    if (ruf[1] == p2[1])
	flag = 16;
    if (ruf[2] == p2[2])
	flag = 22;


    if(ldb[0] == p1[0] && ldb[1] == p1[1])
	flag = 9;

    if(ruf[0] == p2[0] && ldb[1] == p1[1])
	flag = 11;

    if(ldb[0] == p1[0] && ruf[1] == p2[1])
	flag = 15;

    if(ruf[0] == p2[0] &&   ruf[1] == p2[1])
	flag = 17;


    if (ldb[0] == p1[0] && ldb[2] == p1[2])
	flag = 3;

    if (ruf[0] == p2[0] && ldb[2] == p1[2])
	flag = 5;

    if (ldb[0] == p1[0] && ruf[2] == p2[2])
	flag = 21;

    if (ruf[0] == p2[0] && ruf[2] == p2[2])
	flag = 23;


    if (ldb[1] == p1[1] && ldb[2] == p1[2])
	flag = 1;

    if (ruf[1] == p2[1] && ldb[2] == p1[2])
	flag = 7;

    if (ldb[1] == p1[1] && ruf[2] == p2[2])
	flag = 19;

    if (ruf[1] == p2[1] && ruf[2] == p2[2])
	flag = 25;

    if (ldb[0] == p1[0] && ldb[1] == p1[1] && ldb[2] == p1[2])
	flag = 0;

    if (ruf[0] == p2[0] && (ldb[1] == p1[1]) && ldb[2] == p1[2])
	flag = 2;

    if (ldb[0] == p1[0] && ruf[1] == p2[1] && ldb[2] == p1[2])
	flag = 6;

    if (ruf[0] == p2[0] && ruf[1] == p2[1] && ldb[2] == p1[2])
	flag = 8;

    if (ldb[0] == p1[0] && ldb[1] == p1[1] && ruf[2] == p2[2])
	flag = 18;

    if (ruf[0] == p2[0] && ldb[1] == p1[1] && ruf[2] == p2[2])
	flag = 20;

    if (ldb[0] == p1[0] && ruf[1] == p2[1] && ruf[2] == p2[2])
	flag = 24;

    if (ruf[0] == p2[0] && ruf[1] == p2[1] && ruf[2] == p2[2])
	flag = 26;

    return flag;
}



static const int theIDBoundaryMatrix[27][8] = {
    { 7, 6, 5, 4, 3, 2, 1, 0},
    { 6, 6, 4, 4, 2, 2, 0, 0},
    { 6, 7, 4, 5, 2, 3, 0, 1},
    { 5, 4, 5, 4, 1, 0, 1, 0},
    { 4, 4, 4, 4, 0, 0, 0, 0},
    { 4, 5, 4, 5, 0, 1, 0, 1},
    { 5, 4, 7, 6, 1, 0, 3, 2},
    { 4, 4, 6, 6, 0, 0, 2, 2},
    { 4, 5, 6, 7, 0, 1, 2, 3},
    { 3, 2, 1, 0, 3, 2, 1, 0},
    { 2, 2, 0, 0, 2, 2, 0, 0},
    { 2, 3, 0, 1, 2, 3, 0, 1},
    { 1, 0, 1, 0, 1, 0, 1, 0},
    { 0, 0, 0, 0, 0, 0, 0, 0}, /* 13: internal elements */
    { 0, 1, 0, 1, 0, 1, 0, 1},
    { 1, 0, 3, 2, 1, 0, 3, 2},
    { 0, 0, 2, 2, 0, 0, 2, 2},
    { 0, 1, 2, 3, 0, 1, 2, 3},
    { 3, 2, 1, 0, 7, 6, 5, 4},
    { 2, 2, 0, 0, 6, 6, 4, 4},
    { 2, 3, 0, 1, 6, 7, 4, 5},
    { 1, 0, 1, 0, 5, 4, 5, 4},
    { 0, 0, 0, 0, 4, 4, 4, 4},
    { 0, 1, 0, 1, 4, 5, 4, 5},
    { 1, 0, 3, 2, 5, 4, 7, 6},
    { 0, 0, 2, 2, 4, 4, 6, 6},
    { 0, 1, 2, 3, 4, 5, 6, 7},
};





static void
compute_setboundary(float size, float Vp, float Vs, float rho, int flag,
		    double dashpot[8][3])
{
    int whichNode;
    double scale;

    /* init the damping vector to all zeroes */
    memset(dashpot, 0, sizeof(double) * 8 * 3);

#ifdef HALFSPACE
    flag = (flag < 9) ? flag + 9 : flag;
#endif /* HALFSPACE */

    scale = rho * (size / 2) * (size / 2);

    for (whichNode = 0 ; whichNode < 8; whichNode++) {
	int bitmark, component;

	bitmark = theIDBoundaryMatrix[flag][whichNode];

	switch (bitmark) {
	case 0:
	    break;
	case 7:
	    /* Three contributing faces */
	    dashpot[whichNode][0] = dashpot[whichNode][1]
		= dashpot[whichNode][2] = (Vp + 2 * Vs) * scale;
	    break;
	case 3:
	case 5:
	case 6:
	    /* Two contributing faces */
	    for (component = 0; component < 3; component++)
		dashpot[whichNode][component] =
		    (Vs + ((bitmark & (1<< component)) ? Vp : Vs)) * scale;
	    break;
	case 1:
	case 2:
	case 4:
	    /* One contributing face */
	    for (component = 0; component < 3; component++)
		dashpot[whichNode][component] =
		    ((bitmark & (1<<component)) ? Vp : Vs) * scale;
	    break;
	default:
	    fprintf(stderr, "SetBoundary: Unknown bitmark. Panic!\n");
	    exit(1);
	}
    }

    return;
}
#endif /* BOUNDARY */



/**
 * compute_setab: the base a and b values will be scaled by zeta
 *                specific to each element.
 */
static void compute_setab(double freq, double *aBasePtr, double *bBasePtr)
{
    /* old version which caused overflow because of the aproximation in
     * the derivative */

    double w1, w2, lw1, lw2, sw1, sw2, cw1, cw2;
    double numer, denom;

    if (Param.theTypeOfDamping == RAYLEIGH)
    {
	/* the factors 0.2 and 1 were calibrated heuristically by LEO */
	w1 = 2 * PI * freq *.2;
	w2 = 2 * PI * freq * 1;

	/* logs */
	lw1 = log(w1);
	lw2 = log(w2);

	/* squares */
	sw1 = w1 * w1;
	sw2 = w2 * w2;

	/* cubes */
	cw1 = w1 * w1 * w1;
	cw2 = w2 * w2 * w2;

	/* numerator */
	numer = w1 * w2 *
	    ( -2 * sw1 * lw2 + 2 * sw1 * lw1 - 2 * w1 * w2 * lw2
	      + 2 * w1 * w2 * lw1 + 3 * sw2 - 3 * sw1
		  - 2 * sw2 * lw2 + 2 * sw2 * lw1);

	/* denominator */
	denom = (cw1 - cw2 + 3 * sw2 * w1 - 3 * sw1 * w2);

	/* the a over zeta target is... */
	*aBasePtr = numer / denom;

	/* new numerator */
	numer = 3 * (2 * w1 * w2 * lw2 - 2 * w1 * w2 * lw1 + sw1 - sw2);

	/* the b over zeta target is... */
	*bBasePtr = numer / denom;

    }
    else if ( Param.theTypeOfDamping == MASS )
    {
	w1 = 2 * PI * freq * .1;  /* these .1 and 8 heuristics */
	w2 = 2 * PI * freq * 8;

	numer = 2 * w2 * w1 * log(w2 / w1);
	denom = w2 - w1;

	*aBasePtr = 1.3*numer / denom;  /* this 1.3 comes out from heuristics */
	*bBasePtr = 0;
    }
    else if ( Param.theTypeOfDamping == NONE || Param.theTypeOfDamping == BKT )
    {
	*aBasePtr = 0;
	*bBasePtr = 0;
    }

    return;
}



int
is_nodeloaded( int32_t iNode, char* onoff )
{
    /* \todo use a general bitmap data structure for this */
    /* \todo move the routine declaration to the source generation file */

    int32_t whichByte, whichBit;

    char mask,test;

    whichByte = iNode/8;
    whichBit = 7 - iNode % 8;

    mask = ( char )pow(2,whichBit);

    test = onoff[whichByte] & mask;

    return (test == mask);  /* 1 if equal, 0 otherwise */
}


/**
 * Add the force due to earthquake source.
 *
 * Globals accessed:
 * - Global.mySolver->force (W)
 * - Param.theDeltaTSquared (R)
 * - Global.myForces (R)
 *
 * Iterates over the nodes that are loaded by the source and set the
 * respective forces for those nodes.
 */
static void
compute_addforce_s( int32_t timestep )
{
    int i;	/* index for loaded nodes (from the source) */

    for (i = 0; i <  Global.theNodesLoaded; i++) {
	int lnid = Global.theNodesLoadedList[i];	/* local node id */

	/* node's force vector */
	fvector_t* nodalForce =	Global.mySolver->force + lnid;

	/* vector-scalar multiply */
	nodalForce->f[0] = ( Global.myForces [ i ].x [0] ) * Param.theDeltaTSquared;
	nodalForce->f[1] = ( Global.myForces [ i ].x [1] ) * Param.theDeltaTSquared;
	nodalForce->f[2] = ( Global.myForces [ i ].x [2] ) * Param.theDeltaTSquared;
    }
}

/**
 * compute_adjust: Either distribute the values from LOCAL dangling nodes
 *                 to LOCAL anchored nodes, or assign values from LOCAL
 *                 anchored nodes to LOCAL dangling nodes.
 *
 */
static void
compute_adjust(void *valuetable, int32_t itemsperentry, int32_t how)
{
    solver_float *vtable = (solver_float *)valuetable;
    int32_t dnindex;

    if (how == DISTRIBUTION) {
	for (dnindex = 0; dnindex < Global.myMesh->ldnnum; dnindex++) {
	    dnode_t *dnode;
	    solver_float *myvalue, *parentvalue;
	    solver_float darray[7]; /* A hack to avoid memory allocation */
	    int32link_t *int32link;
	    int32_t idx, parentlnid;
#ifdef DEBUG
	    int32_t deps = 0;
#endif /* DEBUG */

	    dnode = &Global.myMesh->dnodeTable[dnindex];
	    myvalue = vtable + dnode->ldnid * itemsperentry;

	    for (idx = 0; idx < itemsperentry; idx++) {
		darray[idx] = (*(myvalue + idx)) / dnode->deps;
	    }

	    /* Distribute my darray value to my anchors */
	    int32link = dnode->lanid;
	    while (int32link != NULL) {

#ifdef DEBUG
		deps++;
#endif

		parentlnid = int32link->id;
		parentvalue = vtable + parentlnid * itemsperentry;

		for (idx = 0; idx < itemsperentry; idx++) {
		    /* Accumulation the distributed values */
		    *(parentvalue + idx) += darray[idx];
		}

		int32link = int32link->next;
	    }

#ifdef DEBUG
	    if (deps != (int)dnode->deps) {
		fprintf(stderr, "Thread %d: compute_adjust distri: ", Global.myID);
		fprintf(stderr, "deps don't match\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
#endif /* DEBUG */
	} /* for all my LOCAL dangling nodes */

    } else {
	/* Assign the value of the anchored parents to the dangling nodes*/

	for (dnindex = 0; dnindex < Global.myMesh->ldnnum; dnindex++) {
	    dnode_t *dnode;
	    solver_float *myvalue, *parentvalue;
	    int32link_t *int32link;
	    int32_t idx, parentlnid;
#ifdef DEBUG
	    int32_t deps = 0;
#endif /* DEBUG */

	    dnode = &Global.myMesh->dnodeTable[dnindex];
	    myvalue = vtable + dnode->ldnid * itemsperentry;

	    /* Zero out the residual values the dangling node might
	       still hold */
	    memset(myvalue, 0, sizeof(solver_float) * itemsperentry);

	    /* Assign prorated anchored values to a dangling node */
	    int32link = dnode->lanid;
	    while (int32link != NULL) {

#ifdef DEBUG
		deps++;
#endif

		parentlnid = int32link->id;
		parentvalue = vtable + parentlnid * itemsperentry;

		for (idx = 0; idx < itemsperentry; idx++) {
		    *(myvalue + idx) += (*(parentvalue + idx) / dnode->deps);
		}

		int32link = int32link->next;
	    }

#ifdef DEBUG
	    if (deps != (int)dnode->deps) {
		fprintf(stderr, "Thread %d: compute_adjust assign: ", Global.myID);
		fprintf(stderr, "deps don't match\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
#endif /* DEBUG */

	} /* for all my LOCAL dangling nodes */
    }

    return;
}

static void print_timing_stat()
{

    double TotalMeshingTime;
    TotalMeshingTime = Timer_Value("Octor Newtree"          , (TimerKind)0)
                     + Timer_Value("Octor Refinetree"       , (TimerKind)0)
                     + Timer_Value("Octor Balancetree"      , (TimerKind)0)
                     + Timer_Value("Octor Partitiontree"    , (TimerKind)0)
                     + Timer_Value("Octor Extractmesh"      , (TimerKind)0)
                     + Timer_Value("Mesh correct properties", (TimerKind)0)
                     + Timer_Value("Mesh Stats Print"       , (TimerKind)0);

    if ( Param.includeBuildings == YES ) {
        TotalMeshingTime += Timer_Value("Carve Buildings", (TimerKind)0);
    }

    printf("\n\n__________________________Raw Timers__________________________\n\n");
    Timer_PrintAll(comm_solver);
    printf("\n\n\n\n\n");


    printf("\n\n__________________________Timer Statistics__________________________\n\n");


    printf("\n_____________Summary_____________\n");
    printf("Max Frequency             : %.2f\n", Param.theFreq);
    printf("Vs                        : %.2f\n", Param.theVsCut);
    printf("Total elements            : %" INT64_FMT "\n", Global.theETotal);
    printf("Elements/PE               : %" INT64_FMT "\n", Global.theETotal/Global.theGroupSize);
    printf("Simulation duration       : %.2f seconds\n", Param.theEndT - Param.theStartT);
    printf("Total steps               : %d\n", Param.theTotalSteps);
    printf("DeltaT used               : %.6f seconds\n", Param.theDeltaT);
    printf("Critical deltaT           : %.6f seconds\n", Global.theCriticalT);
    printf("\n");
    printf("Total Wall Clock          : %.2f seconds\n", Timer_Value("Total Wall Clock",(TimerKind)0));
    printf("Time/step                 : %.6f seconds\n", Timer_Value("Solver",(TimerKind)0)/Param.theTotalSteps);
    printf("Time/step/(elem/PE)       : %.6f millisec\n",Timer_Value("Solver",(TimerKind)0) * 1000.0 / Param.theTotalSteps /
	   (Global.theETotal * 1.0 / Global.theGroupSize));
    printf("Simulation Rate Variation : %.3f (Average)   %.3f (Min)   %.3f (Max)  (sec/%d timesteps)\n",
	   (Timer_Value("Solver",(TimerKind)0)/Param.theTotalSteps)*Param.monitor_stats_rate,
	   Global.fastestTimeSteps, Global.slowestTimeSteps, Param.monitor_stats_rate);
    printf("\n");


    printf("\n____________Breakdown____________\n");
    printf("TOTAL MESHING                       : %.2f seconds\n", TotalMeshingTime);
    printf("    Octor Newtree                   : %.2f seconds\n", Timer_Value("Octor Newtree",(TimerKind)0) );
    printf("    Octor Refinetree                : %.2f seconds\n", Timer_Value("Octor Refinetree",(TimerKind)0));
    printf("    Octor Balancetree               : %.2f seconds\n", Timer_Value("Octor Balancetree",(TimerKind)0));
    if ( Timer_Exists("Carve Buildings") )
        printf("    Octor Carve Buildings           : %.2f seconds\n", Timer_Value("Carve Buildings",(TimerKind)0));
    printf("    Octor Partitiontree             : %.2f seconds\n", Timer_Value("Octor Partitiontree",(TimerKind)0));
    printf("    Octor Extractmesh               : %.2f seconds\n", Timer_Value("Octor Extractmesh",(TimerKind)0));
    printf("    Mesh correct properties         : %.2f seconds\n", Timer_Value("Mesh correct properties",(TimerKind)0));
    printf("    Mesh Stats Print                : %.2f seconds\n", Timer_Value("Mesh Stats Print",(TimerKind)0));
    printf("\n");

    if(Param.drmImplement == YES) {

    	printf("DRM INIT PARAMETERS                 : %.2f (Max) %.2f (Min) seconds\n",
    			Timer_Value("Init Drm Parameters",MAX), Timer_Value("Init Drm Parameters",MIN));
    	printf("\n");

    	printf("DRM INITIALIZATION                  : %.2f (Max) %.2f (Min) seconds\n",
    			Timer_Value("Drm Init",MAX), Timer_Value("Drm Init",MIN));

    	if(Param.theDrmPart == PART2) {

    		printf("    Find Drm File To Readjust       : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Find Drm File To Readjust",AVERAGE),
    				Timer_Value("Find Drm File To Readjust",MAX),
    				Timer_Value("Find Drm File To Readjust",MIN));

    		printf("    Fill Drm Struct                 : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Fill Drm Struct",AVERAGE),
    				Timer_Value("Fill Drm Struct",MAX),
    				Timer_Value("Fill Drm Struct",MIN));

    		printf("    Comm of Drm Coordinates         : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Comm of Drm Coordinates",AVERAGE),
    				Timer_Value("Comm of Drm Coordinates",MAX),
    				Timer_Value("Comm of Drm Coordinates",MIN));

    		printf("    Find Which Drm Files To Print   : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Find Which Drm Files To Print",AVERAGE),
    				Timer_Value("Find Which Drm Files To Print",MAX),
    				Timer_Value("Find Which Drm Files To Print",MIN));

    		printf("    Read And Rearrange Drm Files    : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Read And Rearrange Drm Files",AVERAGE),
    				Timer_Value("Read And Rearrange Drm Files",MAX),
    				Timer_Value("Read And Rearrange Drm Files",MIN));

    		printf("    Find Which Drm Files To Read    : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Find Which Drm Files To Read",AVERAGE),
    				Timer_Value("Find Which Drm Files To Read",MAX),
    				Timer_Value("Find Which Drm Files To Read",MIN));

    		printf("    Locate where I am in file       : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    				Timer_Value("Locate where I am in file",AVERAGE),
    				Timer_Value("Locate where I am in file",MAX),
    				Timer_Value("Locate where I am in file",MIN));

    	}
    	printf("\n");
    }

    printf("SOURCE INITIALIZATION               : %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Source Init",MAX), Timer_Value("Source Init",MIN));
    printf("\n");
    printf("TOTAL SOLVER                        : %.2f seconds\n", Timer_Value("Solver",(TimerKind)0));
    printf("    Read My Forces                  : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Read My Forces",AVERAGE),
	   Timer_Value("Read My Forces",MAX),
	   Timer_Value("Read My Forces",MIN));
    printf("    Compute addforces s             : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Compute addforces s",AVERAGE),
	   Timer_Value("Compute addforces s",MAX),
	   Timer_Value("Compute addforces s",MIN));
    printf("    Compute addforces e             : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Compute addforces e",AVERAGE),
	   Timer_Value("Compute addforces e",MAX),
	   Timer_Value("Compute addforces e",MIN));
    printf("    Compute Damping addforce        : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Damping addforce",AVERAGE),
	   Timer_Value("Damping addforce",MAX),
	   Timer_Value("Damping addforce",MIN));
    if ( Timer_Exists("Compute Non-linear Entities") )
        printf("    Compute Non-linear Entities     : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
                Timer_Value("Compute Non-linear Entities",AVERAGE),
                Timer_Value("Compute Non-linear Entities",MAX),
                Timer_Value("Compute Non-linear Entities",MIN));
    if ( Timer_Exists("Compute addforces Non-linear") ) {
        printf("    Compute addforces Non-linear    : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
                Timer_Value("Compute addforces Non-linear",AVERAGE),
                Timer_Value("Compute addforces Non-linear",MAX),
                Timer_Value("Compute addforces Non-linear",MIN));
        printf("    Compute addforces gravity       : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
                Timer_Value("Compute addforces gravity",AVERAGE),
                Timer_Value("Compute addforces gravity",MAX),
                Timer_Value("Compute addforces gravity",MIN));
    }

    if ( Timer_Exists("Solver drm force compute") ) {
    	printf("    Solver drm force compute        : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    			Timer_Value("Solver drm force compute",AVERAGE),
    			Timer_Value("Solver drm force compute",MAX),
    			Timer_Value("Solver drm force compute",MIN));
    }
    printf("    1st schedule send data          : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("1st schedule send data (contribution)",AVERAGE),
	   Timer_Value("1st schedule send data (contribution)",MAX),
	   Timer_Value("1st schedule send data (contribution)",MIN));
    printf("    1st compute adjust              : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("1st compute adjust (distribution)",AVERAGE),
	   Timer_Value("1st compute adjust (distribution)",MAX),
	   Timer_Value("1st compute adjust (distribution)",MIN));
    printf("    2nd schedule send data          : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("2nd schedule send data (contribution)",AVERAGE),
	   Timer_Value("2nd schedule send data (contribution)",MAX),
	   Timer_Value("2nd schedule send data (contribution)",MIN));
    printf("    Compute new displacement        : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Compute new displacement",AVERAGE),
	   Timer_Value("Compute new displacement",MAX),
	   Timer_Value("Compute new displacement",MIN));
    printf("    3rd schedule send data          : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("3rd schedule send data (sharing)",AVERAGE),
	   Timer_Value("3rd schedule send data (sharing)",MAX),
	   Timer_Value("3rd schedule send data (sharing)",MIN));
    printf("    2nd compute adjust              : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("2nd compute adjust (assignment)",AVERAGE),
	   Timer_Value("2nd compute adjust (assignment)",MAX),
	   Timer_Value("2nd compute adjust (assignment)",MIN));
    printf("    4th schadule send data          : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("4th schadule send data (sharing)",AVERAGE),
	   Timer_Value("4th schadule send data (sharing)",MAX),
	   Timer_Value("4th schadule send data (sharing)",MIN));
    printf("    IO\n");

    if ( Timer_Exists("Solver drm output") ) {
    	printf("        Drm Output                  : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    			Timer_Value("Solver drm output",AVERAGE),
    			Timer_Value("Solver drm output",MAX),
    			Timer_Value("Solver drm output",MIN));
    }
    if ( Timer_Exists("Solver drm read displacements") ) {
    	printf("        Solver drm read disp        : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
    			Timer_Value("Solver drm read displacements",AVERAGE),
    			Timer_Value("Solver drm read displacements",MAX),
    			Timer_Value("Solver drm read displacements",MIN));
    }
    printf("        Solver Stats Print          : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Solver Stats Print",AVERAGE),
	   Timer_Value("Solver Stats Print",MAX),
	   Timer_Value("Solver Stats Print",MIN));
    if( Timer_Exists("Print Planes") )
	printf("        Planes                      : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	       Timer_Value("Print Planes",AVERAGE),
	       Timer_Value("Print Planes",MAX),
	       Timer_Value("Print Planes",MIN));
    if( Timer_Exists("Print Stations") )
	printf("        Stations                    : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	       Timer_Value("Print Stations",AVERAGE),
	       Timer_Value("Print Stations",MAX),
	       Timer_Value("Print Stations",MIN));
    if( Timer_Exists("Checkpoint") )
	printf("        Checkpoint                  : %.2f\n",
	       Timer_Value("Checkpoint",(TimerKind)0));
    printf("\n");
    printf("TOTAL WALL CLOCK                    : %.2f seconds\n", Timer_Value("Total Wall Clock",(TimerKind)0));
    printf("\n");
    printf("\n____________Analysis_____________\n");
    printf("Solver I/O                 : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Solver I/O",      AVERAGE),Timer_Value("Solver I/O",      MAX), Timer_Value("Solver I/O",      MIN));
    printf("Solver Compute             : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Compute Physics", AVERAGE),Timer_Value("Compute Physics", MAX), Timer_Value("Compute Physics", MIN));
    printf("Solver Communicate         : %.2f (Average)   %.2f (Max) %.2f (Min) seconds\n",
	   Timer_Value("Communication",   AVERAGE),Timer_Value("Communication",   MAX), Timer_Value("Communication",   MIN));
    printf("Compute/Communicate Ratio  : %.2f \n",
	   Timer_Value("Compute Physics", AVERAGE) / Timer_Value("Communication",   AVERAGE) );
    printf("\n\n\n\n");

    fflush (stdout);

    return;
}




/**
 * Prepare data to compute the myForce vector.  It calls compute_force.
 */
static void
source_init( const char* physicsin )
{
	if(( Param.drmImplement == NO )||( Param.drmImplement == YES && Param.theDrmPart == PART1 )){

		double globalDelayT = 0;
		double surfaceShift = 0;

		/* Load to Global.theMPIInformation */
		Global.theMPIInformation.myid      = Global.myID;
		Global.theMPIInformation.groupsize = Global.theGroupSize;

		/* Load to theNumericsInforamation */
		Global.theNumericsInformation.numberoftimesteps = Param.theTotalSteps;
		Global.theNumericsInformation.deltat	     = Param.theDeltaT;
		Global.theNumericsInformation.validfrequency    = Param.theFreq;

		Global.theNumericsInformation.xlength = Param.theDomainX;
		Global.theNumericsInformation.ylength = Param.theDomainY;
		Global.theNumericsInformation.zlength = Param.theDomainZ;

		if ( Param.includeNonlinearAnalysis == YES ) {
			globalDelayT = get_geostatic_total_time();
		}

		if ( Param.includeBuildings == YES ) {
			surfaceShift = get_surface_shift();
		}

		/* it will create the files to be read each time step to
       load (force) the mesh */
		if ( compute_print_source( physicsin, Global.myOctree, Global.myMesh,
				Global.theNumericsInformation, Global.theMPIInformation,
				globalDelayT, surfaceShift ) )
		{
			fprintf(stdout,"Err:cannot create source forces");
			MPI_Abort(MPI_COMM_WORLD, ERROR);
			exit(1);
		}

		Global.theNodesLoaded = source_get_local_loaded_nodes_count();

		if (Global.theNodesLoaded != 0) {
			size_t ret;
			int32_t node_count;

			Global.fpsource = source_open_forces_file( "r" );

			hu_fread( &node_count, sizeof(int32_t), 1, Global.fpsource );

			/* \todo add assertion for node_count == Global.theNodesLoaded */

			Global.theNodesLoadedList = (int32_t *)malloc( sizeof(int32_t) * Global.theNodesLoaded );
			Global.myForces	   = (vector3D_t *)calloc( Global.theNodesLoaded, sizeof(vector3D_t) );

			if (Global.myForces == NULL || Global.theNodesLoadedList == NULL) {
				solver_abort( "source_init", "memory allocation failed",
						"Cannot allocate memory for Global.myForces or "
						"loaded nodes list arrays\n" );
			}

			ret = hu_fread( Global.theNodesLoadedList, sizeof(int32_t), Global.theNodesLoaded,
					Global.fpsource );

			if (ret != Global.theNodesLoaded) {
				solver_abort( "source_init(", "fread failed",
						"Could not read nodal force file");
			}
		}
	}
}



/**
 * Search a point in the domain of the local mesh.
 *
 *   input: coordinates
 *  output: 0 fail 1 success
 */
int32_t search_point( vector3D_t point, octant_t **octant )
{
    tick_t  xTick, yTick, zTick;

    xTick = point.x[0] / Global.myMesh->ticksize;
    yTick = point.x[1] / Global.myMesh->ticksize;
    zTick = point.x[2] / Global.myMesh->ticksize;

    *octant = octor_searchoctant( Global.myOctree, xTick, yTick, zTick,
            PIXELLEVEL, AGGREGATE_SEARCH );

    if ( (*octant == NULL) || ((*octant)->where == REMOTE) ) {
        return 0;
    }

    return 1;
}

/**
 * \param octant where the point is located.
 * \param pointcoords coordinates of the point.
 * \param localcoords[out] the displacment.
 */
extern int
compute_csi_eta_dzeta( octant_t* octant, vector3D_t pointcoords,
		       vector3D_t* localcoords, int32_t* localNodeID )
{
    tick_t  edgeticks;
    int32_t eindex;
    double  center_x, center_y, center_z;

    /* various convienient variables */
    double xGlobal = pointcoords.x[0];
    double yGlobal = pointcoords.x[1];
    double zGlobal = pointcoords.x[2];
    double h;

    edgeticks = (tick_t)1 << (PIXELLEVEL - octant->level);
    h =  Global.myMesh->ticksize * edgeticks;

    /* Calculate the center coordinate of the element */
    center_x = Global.myMesh->ticksize * (octant->lx + edgeticks / 2);
    center_y = Global.myMesh->ticksize * (octant->ly + edgeticks / 2);
    center_z = Global.myMesh->ticksize * (octant->lz + edgeticks / 2);


    /* Go through my local elements to find which one matches the
     * containing octant. I should have a better solution than this.
     */
    for (eindex = 0; eindex < Global.myMesh->lenum; eindex++) {
        int32_t lnid0 = Global.myMesh->elemTable[eindex].lnid[0];

        if ((Global.myMesh->nodeTable[lnid0].x == octant->lx) &&
                (Global.myMesh->nodeTable[lnid0].y == octant->ly) &&
                (Global.myMesh->nodeTable[lnid0].z == octant->lz)) {

            /* Sanity check */
            if (Global.myMesh->elemTable[eindex].level != octant->level) {
                fprintf(stderr, "Thread %d: source_init: internal error\n",
                        Global.myID);
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }

            /* Fill in the local node ids of the containing element */
            memcpy( localNodeID, Global.myMesh->elemTable[eindex].lnid,
                    sizeof(int32_t) * 8 );

            break;
        }
    }  /* for all the local elements */


    if (eindex == Global.myMesh->lenum) {
        fprintf(stderr, "Thread %d: source_init: ", Global.myID);
        fprintf(stderr, "No element matches the containing octant.\n");
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    /* Derive the local coordinate of the source inside the element */
    localcoords->x[0] =  2*(xGlobal- center_x)/h;
    localcoords->x[1] =  2*(yGlobal- center_y)/h;
    localcoords->x[2] =  2*(zGlobal- center_z)/h;

    return 1;
}


/**
 * Read stations info.  This is called by PE 0.
 */
static void
read_stations_info( const char* numericalin )
{
    static const char* fname = __FUNCTION_NAME;

    int    iStation, iCorner;
    double lon, lat, depth, *auxiliar;
    FILE*  fp;

    vector3D_t coords;

    /* obtain the stations specifications */
    if ( (fp = fopen ( numericalin, "r")) == NULL ) {
	solver_abort (fname, numericalin,
		      "Error opening numerical.in configuration file");
    }

    auxiliar = (double *)malloc(sizeof(double)*8);

    if ( parsedarray( fp, "domain_surface_corners", 8, auxiliar ) != 0) {
	solver_abort( fname, NULL,
		      "Error parsing domain_surface_corners field from %s\n",
		      numericalin);
    }

    for ( iCorner = 0; iCorner < 4; iCorner++){
	Param.theSurfaceCornersLong[ iCorner ] = auxiliar [ iCorner * 2 ];
	Param.theSurfaceCornersLat [ iCorner ] = auxiliar [ iCorner * 2 +1 ];
    }
    free(auxiliar);


    if (parsetext( fp, "output_stations_print_rate", 'i',
		   &Param.theStationsPrintRate ) != 0) {
	solver_abort( fname, NULL,
		      "Error parsing output_planes_print_rate field from %s\n",
		      numericalin );
    }

    auxiliar    = (double*)malloc( sizeof(double) * Param.theNumberOfStations * 3 );
    Param.theStationX = (double*)malloc( sizeof(double) * Param.theNumberOfStations );
    Param.theStationY = (double*)malloc( sizeof(double) * Param.theNumberOfStations );
    Param.theStationZ = (double*)malloc( sizeof(double) * Param.theNumberOfStations );

    if (Param.theStationX == NULL || Param.theStationY == NULL || Param.theStationZ == NULL) {
	fprintf( stdout,
		 "Err alloc theStations arrays in output_stations_init" );
	fflush( stdout );
	MPI_Abort(MPI_COMM_WORLD, ERROR );
	exit( 1 );
    }

    if (parsedarray( fp, "output_stations", Param.theNumberOfStations * 3,
		     auxiliar ) != 0) {
	solver_abort (fname, NULL,
		      "Err parsing output_stations from %s\n", numericalin);
    }

    for (iStation = 0; iStation < Param.theNumberOfStations; iStation++) {
	lat    = auxiliar [ iStation * 3 ];
	lon    = auxiliar [ iStation * 3 +1 ];
	depth  = auxiliar [ iStation * 3 +2 ];
	coords = compute_domain_coords_linearinterp(lon,lat,
						    Param.theSurfaceCornersLong,
						    Param.theSurfaceCornersLat,
						    Param.theDomainY,Param.theDomainX);
	Param.theStationX [ iStation ] = coords.x[0];
	Param.theStationY [ iStation ] = coords.x[1];
	Param.theStationZ [ iStation ] = depth;

	if ( Param.includeBuildings == YES ) {
	    Param.theStationZ [ iStation ] += get_surface_shift();
	}
    }

    free( auxiliar );

    if ( parsetext(fp, "output_stations_directory",'s',Param.theStationsDirOut)!= 0)
	solver_abort (fname, NULL, "Error parsing fields from %s\n",
		      numericalin);

    return;
}


/**
 * Broadcast info about the stations.
 */
void
broadcast_stations_info()
{
    /*initialize the local structures */
    if ( Global.myID != 0 ) {
	Param.theStationX = (double*)malloc( sizeof(double) * Param.theNumberOfStations);
	Param.theStationY = (double*)malloc( sizeof(double) * Param.theNumberOfStations);
	Param.theStationZ = (double*)malloc( sizeof(double) * Param.theNumberOfStations);

	if (Param.theStationX == NULL ||  Param.theStationY == NULL || Param.theStationZ==NULL) {
	    solver_abort( "broadcast_stations_info", NULL,
			  "Error: Unable to create stations arrays" );
	}
    }

    MPI_Barrier( comm_solver );

    MPI_Bcast(Param.theStationX, Param.theNumberOfStations, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(Param.theStationY, Param.theNumberOfStations, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(Param.theStationZ, Param.theNumberOfStations, MPI_DOUBLE, 0, comm_solver);
    MPI_Bcast(&Param.theStationsPrintRate, 1, MPI_INT, 0, comm_solver);

    broadcast_char_array( Param.theStationsDirOut, sizeof(Param.theStationsDirOut), 0,
			  comm_solver );

    return;
}



/**
 * Prepare all the info every station needs once it is located in a processor
 */
void setup_stations_data()
{
    static char stationFile[256];

    int32_t    iStation, iLCStation = 0; /* LC local count */
    vector3D_t stationCoords;
    octant_t*  octant;

    /* look for the stations in the domain each processor has */
    for (iStation = 0; iStation < Param.theNumberOfStations; iStation++) {
        stationCoords.x[0] = Param.theStationX[iStation];
        stationCoords.x[1] = Param.theStationY[iStation];
        stationCoords.x[2] = Param.theStationZ[iStation];

        if (search_point( stationCoords, &octant ) == 1) {
            Param.myNumberOfStations++;
        }
    }

    /* allocate memory if necessary and generate the list of stations per
     * processor */
    if (Param.myNumberOfStations != 0) {
	monitor_print( "PE=%d local_stations=%d total_station_count=%d\n",
		       Global.myID, Param.myNumberOfStations, Param.theNumberOfStations );

	XMALLOC_VAR_N( Param.myStations, station_t, Param.myNumberOfStations );
    }

    for (iStation = 0; iStation < Param.theNumberOfStations; iStation++) {
	stationCoords.x[0] = Param.theStationX[iStation];
	stationCoords.x[1] = Param.theStationY[iStation];
	stationCoords.x[2] = Param.theStationZ[iStation];

	if (search_point( stationCoords, &octant ) == 1) {
	    Param.myStations[iLCStation].id = iStation;
	    Param.myStations[iLCStation].coords=stationCoords;
	    sprintf(stationFile, "%s/station.%d",Param.theStationsDirOut,iStation);
	    Param.myStations[iLCStation].fpoutputfile = hu_fopen( stationFile,"w" );
	    compute_csi_eta_dzeta( octant, Param.myStations[iLCStation].coords,
				   &(Param.myStations[iLCStation].localcoords),
				   Param.myStations[iLCStation].nodestointerpolate);

	    /*
	     * This section now enclosed in a DEBUG def because its information
	     * is only useful for debugging purposes and otherwise it just
	     * introduce noise to the post-processing.
	     */
#ifdef DEBUG
	    fprintf( Param.myStations[iLCStation].fpoutputfile,
		     "# Node identifiers:\n" );

	    int iNode;

	    for (iNode = 0; iNode < 8; iNode++) {
		fprintf( Param.myStations[iLCStation].fpoutputfile,
			 "#  %13" INT64_FMT "\n",
			 Global.myMesh->nodeTable[Param.myStations[iLCStation].nodestointerpolate[iNode]].gnid );
	    }
#endif /* DEBUG */

	    /*
	     * This is a one line heading for the station files. Spacing is
	     * such that it aligns with data.
	     *
	     * The lines after X and Y represent the actual orientation of the
	     * axes if one looks at the domain's surface on a screen. Zeta's dot
	     * means the Z axis goes in(+) and out(-) of the screen.
	     */

	    fputs( "#  Time(s)         X|(m)         Y-(m)         Z.(m)",
	            Param.myStations[iLCStation].fpoutputfile );

	    if ( ( Param.printStationVelocities    == YES ) ||
	         ( Param.printStationAccelerations == YES ) ) {
	        fputs( "       X|(m/s)       Y-(m/s)       Z.(m/s)",
	                Param.myStations[iLCStation].fpoutputfile );
	    }

            if ( Param.printStationAccelerations == YES ) {
                fputs( "      X|(m/s2)      Y-(m/s2)      Z.(m/s2)",
                        Param.myStations[iLCStation].fpoutputfile );
            }

            /*
	     * Additional headings for nonlinear data.
	     */
	    if ( Param.includeNonlinearAnalysis == YES ) {
	        fputs(  "    Epsilon_XX      Sigma_XX"
	                "    Epsilon_YY      Sigma_YY"
                        "    Epsilon_ZZ      Sigma_ZZ"
                        "    Epsilon_KK      Sigma_KK"
	                "    Epsilon_XY      Sigma_XY"
	                "    Epsilon_YZ      Sigma_YZ"
	                "    Epsilon_XZ      Sigma_XZ"
	                "        lambda            Fs"
	                "             k",
	                Param.myStations[iLCStation].fpoutputfile );
	    }

	    iLCStation += 1;
	}
    }

    free( Param.theStationX );
    free( Param.theStationY );
    free( Param.theStationZ );
}


/**
 * Interpolate the displacements for the stations.
 */
static int
interpolate_station_displacements( int32_t step )
{
    int iPhi;

    /* Auxiliary array to handle shape functions in a loop */
    double  xi[3][8]={ {-1,  1, -1,  1, -1,  1, -1, 1} ,
                       {-1, -1,  1,  1, -1, -1,  1, 1} ,
                       {-1, -1, -1, -1,  1,  1,  1, 1} };

    double     phi[8];
    double     dis_x, dis_y, dis_z;
    double     vel_x, vel_y, vel_z;
    double     acc_x, acc_y, acc_z;
    int32_t    iStation,nodesToInterpolate[8];;
    vector3D_t localCoords; /* convenient renaming */

    for (iStation=0;iStation<Param.myNumberOfStations; iStation++) {

        localCoords = Param.myStations[iStation].localcoords;

        for (iPhi=0; iPhi<8; iPhi++) {
            nodesToInterpolate[iPhi]
                    = Param.myStations[iStation].nodestointerpolate[iPhi];
        }

        /* Compute interpolation function (phi) for each node and
         * load the displacements
         */
        dis_x = 0;
        dis_y = 0;
        dis_z = 0;

        for (iPhi = 0; iPhi < 8; iPhi++) {
            phi[ iPhi ] = ( 1 + xi[0][iPhi]*localCoords.x[0] )
		                * ( 1 + xi[1][iPhi]*localCoords.x[1] )
		                * ( 1 + xi[2][iPhi]*localCoords.x[2] ) / 8;

            dis_x += phi[iPhi] * Global.mySolver->tm1[ nodesToInterpolate[iPhi] ].f[0];
            dis_y += phi[iPhi] * Global.mySolver->tm1[ nodesToInterpolate[iPhi] ].f[1];
            dis_z += phi[iPhi] * Global.mySolver->tm1[ nodesToInterpolate[iPhi] ].f[2];
        }

        double time = Param.theDeltaT * step;

        /*
         * Please DO NOT CHANGE the format for printing the displacements.
         * It has to be *this* one because it goes in hand with the printing
         * format for the nonlinear information.
         */
        fprintf( Param.myStations[iStation].fpoutputfile,
                 "\n%10.6f % 8e % 8e % 8e",
                 time, dis_x, dis_y, dis_z );

        /*
         * Addition for printing velocities on the fly
         */

        if ( ( Param.printStationVelocities    == YES ) ||
             ( Param.printStationAccelerations == YES ) ) {

            for (iPhi = 0; iPhi < 8; iPhi++) {

                phi[ iPhi ] = ( 1 + xi[0][iPhi]*localCoords.x[0] )
                                    * ( 1 + xi[1][iPhi]*localCoords.x[1] )
                                    * ( 1 + xi[2][iPhi]*localCoords.x[2] ) / 8;

                dis_x -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[0];
                dis_y -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[1];
                dis_z -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[2];
            }

            vel_x = dis_x / Param.theDeltaT;
            vel_y = dis_y / Param.theDeltaT;
            vel_z = dis_z / Param.theDeltaT;

            fprintf( Param.myStations[iStation].fpoutputfile,
                     " % 8e % 8e % 8e", vel_x, vel_y, vel_z );
        }

        /*
         * Addition for printing accelerations on the fly
         */

        if ( Param.printStationAccelerations == YES ) {

            for (iPhi = 0; iPhi < 8; iPhi++) {

                phi[ iPhi ] = ( 1 + xi[0][iPhi]*localCoords.x[0] )
                                            * ( 1 + xi[1][iPhi]*localCoords.x[1] )
                                            * ( 1 + xi[2][iPhi]*localCoords.x[2] ) / 8;

                dis_x -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[0];
                dis_y -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[1];
                dis_z -= phi[iPhi] * Global.mySolver->tm2[ nodesToInterpolate[iPhi] ].f[2];

                dis_x += phi[iPhi] * Global.mySolver->tm3[ nodesToInterpolate[iPhi] ].f[0];
                dis_y += phi[iPhi] * Global.mySolver->tm3[ nodesToInterpolate[iPhi] ].f[1];
                dis_z += phi[iPhi] * Global.mySolver->tm3[ nodesToInterpolate[iPhi] ].f[2];
            }

            acc_x = dis_x / Param.theDeltaTSquared;
            acc_y = dis_y / Param.theDeltaTSquared;
            acc_z = dis_z / Param.theDeltaTSquared;

            fprintf( Param.myStations[iStation].fpoutputfile,
                     " % 8e % 8e % 8e", acc_x, acc_y, acc_z );
        }

	/* TODO: Have this 10 as a parameter with a default value */
        if ( (step % (Param.theStationsPrintRate*10)) == 0 ) {
            fflush(Param.myStations[iStation].fpoutputfile);
        }
    }

    return 1;
}


/**
 * Init stations info and data structures
 */
void output_stations_init( const char* numericalin )
{
    if (Global.myID == 0) {
	read_stations_info( numericalin );
    }

    broadcast_stations_info();
    setup_stations_data();

    MPI_Barrier( comm_solver );

    return;
}



/**
 * \note This function should only be called by PE with rank 0.
 */
static int
load_output_parameters (const char* numericalin, output_parameters_t* params)
{
    FILE* fp;
    int   ret, value;
    char  filename[LINESIZE];

    assert (NULL != numericalin);
    assert (NULL != params);
    assert (Global.myID == 0);

     /* Read output parameters from numerical.in */
    fp = fopen (numericalin, "r");

    if (NULL == fp) {
	solver_abort ("load_output_parameters", "fopen", "numerical.in=\"%s\"",
		      numericalin);
    }

    params->do_output		= 0;
    params->parallel_output     = 0;
    params->output_displacement = 0;
    params->output_velocity     = 0;
    params->output_debug	= 0;

    params->displacement_filename = NULL;
    params->velocity_filename     = NULL;


    /* read parameters from configuration file */

    value = 0;
    ret = parsetext (fp, "output_parallel", 'i', &value);

    if (0 == ret && 0 != value) {
	value = 0;
	ret = parsetext (fp, "output_displacement", 'i', &value);

	if (0 == ret && 0 != value) { /* output_displacement = 1 in config */
	    ret = read_config_string (fp, "output_displacement_file",
				      filename, LINESIZE);

	    if (1 == ret && filename[0] != '\0') {
		params->displacement_filename = strdup (filename);
		params->output_displacement = 1;
	    } else {
		solver_abort ("load_output_parameters", NULL,
			      "Output displacement file name not specified in "
			      "numerical.in=\"%s\"",
			      numericalin);
	    }
	}

	value = 0;
	ret   = parsetext (fp, "output_velocity", 'i', &value);

	if (0 == ret && 0 != value) { /* output_displacement = 1 in config */
	    ret = read_config_string (fp, "output_velocity_file",
				      filename, LINESIZE);

	    if (1 == ret && filename[0] != '\0') {
		params->velocity_filename = strdup (filename);
		params->output_velocity = 1;
	    } else {
		solver_abort ("load_output_parameters", NULL,
			      "Output velocity file name not specified in "
			      "numerical.in=\"%s\"",
			      numericalin);
	    }
	}

	params->stats_filename = "output-stats.txt"; /* default value */

	ret = read_config_string (fp, "output_stats_file", filename, LINESIZE);

	if (1 == ret && filename[0] != '\0') {
	    params->stats_filename = strdup (filename);
	}

	params->debug_filename = "output-debug.txt"; /* default value */
	ret = read_config_string (fp, "output_debug_file", filename, LINESIZE);

	if (1 == ret && filename[0] != '\0') {
	    params->debug_filename = strdup (filename);
	}


	if (params->output_velocity || params->output_displacement) {
	    params->do_output = 1;
	}

	params->parallel_output = 1;

	ret = parsetext (fp, "output_debug", 'i', &value);
	if (0 == ret && 0 != value) { /* output_debug = 1 in config */
	    params->output_debug = 1;
	}
    }

    ret = 0;

    fclose (fp);

    return ret;
}



/**
 * Initialize output structures, including the opening of 4D output files.
 *
 * \param numericsin Name of the file with the solver and output parameters
 *	i.e., "numerical.in".
 * \param[out] params output parameters, including filenames.
 *
 * \pre The following global variables should be initialized:
 *	- Global.myID.
 *	- Global.theGroupSize.
 *
 * \post On a successful return, the output argument \c params will be
 *	properly initialized.  If the routine fails, the state of the
 *	\c param struct is undefined.
 *
 * \return 0 on success, -1 on error.
 */
static int
output_init_parameters (const char* numericalin, output_parameters_t* params)
{
    /* jc: this ugly #define here is because the catamount compiler does not
     * like static const  */
#define VALUES_COUNT    4
    int ret;
    int32_t values[VALUES_COUNT];
    off_t output_steps;

    /* sanity cleanup */
    memset (params, 0, sizeof (output_parameters_t));

    params->do_output		  = 0;
    params->displacement_filename = NULL;
    params->velocity_filename     = NULL;
    params->stats_filename	  = NULL;

    if (Global.myID == 0) {
	ret = load_output_parameters (numericalin, params);

	if (0 != ret) {
	    solver_abort ("output_init_parameters", NULL, NULL);
	    return -1;
	}
    }

    /* parameters that can be initialized from global variables */
    params->pe_id	   = Global.myID;
    params->pe_count       = Global.theGroupSize;
    params->total_nodes    = Global.theNTotal;
    params->total_elements = Global.theETotal;
    params->output_rate	   = Param.theRate;
    params->domain_x	   = Param.theDomainX;
    params->domain_y	   = Param.theDomainY;
    params->domain_z	   = Param.theDomainZ;
    params->mesh	   = Global.myMesh;
    params->solver	   = (solver_t*)Global.mySolver;
    params->delta_t	   = Param.theDeltaT;
    params->total_time_steps = Param.theTotalSteps;

    output_steps	   = (Param.theTotalSteps - 1) / Param.theRate + 1;


    values[0] = params->parallel_output;
    values[1] = params->output_displacement;
    values[2] = params->output_velocity;
    values[3] = params->output_debug;


    MPI_Bcast (values, VALUES_COUNT, MPI_INT, 0, comm_solver);
    MPI_Bcast (&params->total_nodes, 1, MPI_INT64, 0, comm_solver);

    params->parallel_output     = values[0];
    params->output_displacement = values[1];
    params->output_velocity	= values[2];
    params->output_debug	= values[3];
    params->output_size		= (output_steps * params->total_nodes
				   * sizeof(fvector_t) + sizeof(out_hdr_t));
    Global.theNTotal = params->total_nodes;


    if (params->parallel_output) {
	if (params->output_displacement) {
	    broadcast_string(&params->displacement_filename, 0,comm_solver);
	}

	if (params->output_velocity) {
	    broadcast_string( &params->velocity_filename, 0, comm_solver );
	}

	if (params->output_debug) {
	    broadcast_string( &params->debug_filename, 0, comm_solver );
	}

	/* set the expected file size */
	Param.the4DOutSize = params->output_size;
    }

    return 0;
#undef VALUES_COUNT
}


/**
 * Intialize parallel output structures according to config file
 * parameters.
 *
 * \note params parameter could be a local var instead, it doesn't need
 * to be global, since this is not really used after the initialization
 * except for the stats file name.
 *
 * \return 0 on success, -1 on error (probably aborts instead).
 */
static int
output_init( const char* numericalin, output_parameters_t* params )
{
    int ret = -1;

    assert (NULL != numericalin);
    assert (NULL != params);

    ret = output_init_parameters( numericalin, params );

    if (ret != 0) {
	return -1;
    }

    /* initialize structures, open files, etc */
    ret = output_init_state( params );

    /* these aren't needed anymore after this point */
    xfree_char( &params->displacement_filename );
    xfree_char( &params->velocity_filename );

    return ret;
}


static int
output_get_stats( void )
{
    output_stats_t disp_stats, vel_stats;
    int    ret      = 0;
    double avg_tput = 0;


    if (Param.theOutputParameters.parallel_output) {
	ret = output_collect_io_stats( Param.theOutputParameters.stats_filename,
				       &disp_stats, &vel_stats, Timer_Value("Total Wall Clock",(TimerKind)0) );

	/* magic trick so print_timing_stat() prints something sensible
	 * for the 4D output time in the parallel case.
	 *
	 * if both displacement and velocity where written out, prefer
	 * the displacement stat.
	 */
	if (Param.theOutputParameters.output_velocity) {
	    avg_tput = vel_stats.tput_avg;
	}

	if (Param.theOutputParameters.output_displacement) {
	    avg_tput = disp_stats.tput_avg;
	}
    }

    return ret;
}



/**
 * Adjust the values of the properties (Vp, Vs, Rho) for the mesh elements.
 * Initial implementation (by Leo) querying the middle point.1D.
 * Modified implementation (Ricardo) querying the 27 points and averaging.
 *
 * \param cvm the material model database (CVM etree).
 */
static void
mesh_correct_properties( etree_t* cvm )
{
    elem_t*  elemp;
    edata_t* edata;
    int32_t  eindex;
    double   east_m, north_m, depth_m, VpVsRatio, RhoVpRatio;
    int	     res, iNorth, iEast, iDepth, numPoints = 3;
    double   vs, vp, rho;
    double   points[3];
    int32_t  lnid0;

    // INTRODUCE BKT MODEL

    double Qs, Qp, Qk, L, vs_vp_Ratio, vksquared, w;
    int index_Qs, index_Qk;
    int QTable_Size = (int)(sizeof(Global.theQTABLE)/( 6 * sizeof(double)));

    points[0] = 0.005;
    points[1] = 0.5;
    points[2] = 0.995;

//    if (Global.myID == 0) {
//        fprintf( stdout,"mesh_correct_properties  ... " );
//        fflush( stdout );
//    }

    /* iterate over mesh elements */
    for (eindex = 0; eindex < Global.myMesh->lenum; eindex++) {

        elemp = &Global.myMesh->elemTable[eindex];
        edata = (edata_t*)elemp->data;
        lnid0 = Global.myMesh->elemTable[eindex].lnid[0];

        if ( Param.includeBuildings == YES ) {
            if( bldgs_correctproperties( Global.myMesh, edata, lnid0) ) {
                continue;
            }
        }

        vp  = 0;
        vs  = 0;
        rho = 0;

        for (iNorth = 0; iNorth < numPoints; iNorth++) {

        	north_m = (Global.myMesh->ticksize) * (double)Global.myMesh->nodeTable[lnid0].x
        			+ edata->edgesize * points[iNorth] + Global.theXForMeshOrigin ;

        	for (iEast = 0; iEast < numPoints; iEast++) {

        		east_m = ( (Global.myMesh->ticksize)
        				* (double)Global.myMesh->nodeTable[lnid0].y
        				+ edata->edgesize * points[iEast] + Global.theYForMeshOrigin  );

        		for (iDepth = 0; iDepth < numPoints; iDepth++) {
        			cvmpayload_t g_props; /* ground properties record */

        			depth_m = ( (Global.myMesh->ticksize)
        					* (double)Global.myMesh->nodeTable[lnid0].z
        					+ edata->edgesize * points[iDepth] + Global.theZForMeshOrigin );

        			/* NOTE: If you want to see the carving process,
        			 *       activate this and comment the query below */
        			if ( Param.includeBuildings == YES ) {
        				//                        if ( depth_m < get_surface_shift() ) {
        				//                            g_props.Vp  = NAN;
        				//                            g_props.Vs  = NAN;
        				//                            g_props.rho = NAN;
        				//                        } else {
        				depth_m -= get_surface_shift();
        				//                            res = cvm_query( Global.theCVMEp, east_m, north_m,
        				//                                             depth_m, &g_props );
        				//                        }
        			}

        			res = cvm_query( Global.theCVMEp, east_m, north_m,
        					depth_m, &g_props );

        			if (res != 0) {
        				fprintf(stderr, "Cannot find the query point\n");
        				exit(1);
        			}

        			vp  += g_props.Vp;
        			vs  += g_props.Vs;
        			rho += g_props.rho;
                }
            }
        }

        edata->Vp  =  vp / 27;
        edata->Vs  =  vs / 27;
        edata->rho = rho / 27;

        /* Auxiliary ratios for adjustments */
        VpVsRatio  = edata->Vp  / edata->Vs;
        RhoVpRatio = edata->rho / edata->Vp;

        /* Adjust material properties according to the element size and
         * softening factor.
         *
         * A factor of 1 means perfect compliance between the mesh and the
         * elements' material properties resulting in strong changes to the
         * results. A factor of 4 tends to double the simulation delta_t
         * without affecting too much the results. Testing is needed but for
         * now I recommend factors > 4.
         */
        if ( Param.theSofteningFactor > 0 ) {

            double idealVs, factoredVs;

            idealVs    = edata->edgesize * Param.theFactor;
            factoredVs = idealVs * Param.theSofteningFactor;

            if ( edata->Vs > factoredVs ) {
                edata->Vs  = factoredVs;
                edata->Vp  = factoredVs * VpVsRatio;
                edata->rho = edata->Vp  * RhoVpRatio;
            }
        }

        /* Readjust Vs, Vp and Density according to VsCut */
        if ( edata->Vs < Param.theVsCut ) {
            edata->Vs  = Param.theVsCut;
            edata->Vp  = Param.theVsCut  * VpVsRatio;
            /* edata->rho = edata->Vp * RhoVpRatio; */ /* Discuss with Jacobo */
        }


        // IMPLEMENT BKT MODEL

        /* CALCULATE QUALITY FACTOR VALUES AND READ CORRESPONDING VISCOELASTICITY COEFFICIENTS FROM THE TABLE */

        	/* L IS THE COEFFICIENT DEFINED BY "SHEARER-2009" TO RELATE QK, QS AND QP */

        if(Param.theTypeOfDamping == BKT)
        {

            vksquared = edata->Vp * edata->Vp - 4. / 3. * edata->Vs * edata->Vs;
        	vs_vp_Ratio = edata->Vs / edata->Vp;
        	vs = edata->Vs * 0.001;
        	L = 4. / 3. * vs_vp_Ratio * vs_vp_Ratio;

          	//Qs = 0.02 * edata->Vs;

        	// Ricardo's Formula based on Brocher's paper (2008) on the subject. In the paper Qp = 2*Qs is given.
        	//TODO : Make sure Qp Qs relation is correct...

        	Qs = 10.5 + vs * (-16. + vs * (153. + vs * (-103. + vs * (34.7 + vs * (-5.29 + vs * 0.31)))));
        	Qp = 2. * Qs;

        	if (Param.useInfQk == YES) {
        	    Qk = 1000;
        	} else {
                Qk = (1. - L) / (1. / Qp - L / Qs);
        	}

        	index_Qs = Search_Quality_Table(Qs, &(Global.theQTABLE[0][0]), QTable_Size);

//        	printf("Quality Factor Table\n Qs : %lf \n Vs : %lf\n",Qs,edata->Vs);

        	if(index_Qs == -2 || index_Qs >= QTable_Size)
        	{
        		fprintf(stderr,"Problem with the Quality Factor Table\n Qs : %lf \n Vs : %lf\n",Qs,edata->Vs);
        		exit(1);
        	}
        	else if(index_Qs == -1)
        	{
        		edata->a0_shear = 0;
        		edata->a1_shear = 0;
        		edata->g0_shear = 0;
        		edata->g1_shear = 0;
        		edata->b_shear  = 0;
        	}
        	else
        	{
        		edata->a0_shear = Global.theQTABLE[index_Qs][1];
        		edata->a1_shear = Global.theQTABLE[index_Qs][2];
        		edata->g0_shear = Global.theQTABLE[index_Qs][3];
        		edata->g1_shear = Global.theQTABLE[index_Qs][4];
        		edata->b_shear  = Global.theQTABLE[index_Qs][5];
        	}

        	index_Qk = Search_Quality_Table(Qk, &(Global.theQTABLE[0][0]), QTable_Size);

//        	printf("Quality Factor Table\n Qs : %lf \n Vs : %lf\n",Qs,edata->Vs);

        	if(index_Qk == -2 || index_Qk >= QTable_Size)
        	{
        		fprintf(stderr,"Problem with the Quality Factor Table\n Qk : %lf \n Vs : %lf\n",Qk,edata->Vs);
        		exit(1);
        	}
        	else if(index_Qk == -1)
        	{
        		edata->a0_kappa = 0;
        		edata->a1_kappa = 0;
        		edata->g0_kappa = 0;
        		edata->g1_kappa = 0;
        		edata->b_kappa  = 0;
        	}
        	else
        	{
        		edata->a0_kappa = Global.theQTABLE[index_Qk][1];
        		edata->a1_kappa = Global.theQTABLE[index_Qk][2];
        		edata->g0_kappa = Global.theQTABLE[index_Qk][3];
        		edata->g1_kappa = Global.theQTABLE[index_Qk][4];
        		edata->b_kappa  = Global.theQTABLE[index_Qk][5];
        	}

        	if(Param.theFreq_Vel != 0.)
        	{
        		w = Param.theFreq_Vel / Param.theFreq;

        		if ( (edata->a0_shear != 0) && (edata->a1_shear != 0) ) {
        		    double shear_vel_corr_factor;
        		    shear_vel_corr_factor = sqrt(1. - (edata->a0_shear * edata->g0_shear * edata->g0_shear / (edata->g0_shear * edata->g0_shear + w * w) + edata->a1_shear * edata->g1_shear * edata->g1_shear / (edata->g1_shear * edata->g1_shear + w * w)));
                    edata->Vs = shear_vel_corr_factor * edata->Vs;
        		}

        		if ( (edata->a0_kappa != 0) && (edata->a0_kappa != 0) ) {
        		    double kappa_vel_corr_factor;
        		    kappa_vel_corr_factor = sqrt(1. - (edata->a0_kappa * edata->g0_kappa * edata->g0_kappa / (edata->g0_kappa * edata->g0_kappa + w * w) + edata->a1_kappa * edata->g1_kappa * edata->g1_kappa / (edata->g1_kappa * edata->g1_kappa + w * w)));
                    edata->Vp = sqrt(kappa_vel_corr_factor * kappa_vel_corr_factor * vksquared + 4. / 3. * edata->Vs * edata->Vs);
        		}
        	}
        }
    }
}


/*** Program's standard entry point. ***/
int main( int argc, char** argv )
{

#ifdef DEBUG
    int32_t flag;
    MPI_Status status;
#endif /* DEBUG */

    /* Parameters init */
    Param.FourDOutFp = NULL;
    Param.theMonitorFileFp = NULL;
    Param.theMonitorFileName = NULL;
    Param.theFreq_Vel = 0;
    Param.monitor_stats_rate = 50;
    Param.theSchedulePrintErrorCheckFlag = 0;
    Param.theSchedulePrintToStdout = 0;
    Param.theSchedulePrintToFile = 0;
    strcpy(Param.theSchedulePrintFilename, "schedule_info.txt");
    Param.theScheduleStatFilename = NULL;
    Param.theMeshStatFilename = NULL;
    Param.theSofteningFactor = 0;
    Param.theStepMeshingFactor = 0;
    Param.myNumberOfStations = 0;
    Param.theUseCheckPoint = 0;
    Param.theTimingBarriersFlag = 0;
    Param.the4DOutSize = 0;
    Param.theMeshOutFlag = DO_OUTPUT;

    /* Global init */
    Global.myID = -1;
    Global.theGroupSize = -1;
    Global.myOctree = NULL;
    Global.myMesh = NULL;
    Global.theETotal = 0;
    Global.theNTotal = 0;
    Global.mySolver = NULL;
    Global.myVelocityTable = NULL;
    Global.theCriticalT = 0;
    Global.fastestTimeSteps = 10000000;
    Global.slowestTimeSteps = 0;
    Global.theNodesLoaded = -1;
    Global.theNodesLoadedList = NULL;
    Global.myForces = NULL;
    Global.fpsource = NULL;
    Global.theCVMRecordSize = sizeof(cvmrecord_t);

    /* MPI initialization */
    MPI_Init(&argc, &argv);
    Timer_Start("Total Wall Clock");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &Global.myID);
    MPI_Comm_size(MPI_COMM_WORLD, &Global.theGroupSize);

    /* Make sure using correct input arguments */
    if (argc != 2) {
        if (Global.myID == 0) {
            fputs ( "Usage: psolve <parameters.in>\n", stderr);
        }
        MPI_Finalize();
        exit(1);
    }

    /*Read in and verify IO Pool Configuration */
    char * IO_PES_ENV_VAR;
    IO_PES_ENV_VAR = getenv("IO_PES");
    if (IO_PES_ENV_VAR==NULL)
        Param.IO_pool_pe_count = 0;
    else
        Param.IO_pool_pe_count = atoi(IO_PES_ENV_VAR);
    if (Param.IO_pool_pe_count >= Global.theGroupSize)
    {
        Param.IO_pool_pe_count = 0;
        if (Global.myID==0)
            printf("Warning: IO_PES too large.  Set to 0.\n");
    }

    /* Split off PEs into IO Pool */
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_IO);
    int in_io_pool, color;
    Global.theGroupSize -= Param.IO_pool_pe_count;
    if (Global.myID >= Global.theGroupSize)
        in_io_pool = 1;
    else
        in_io_pool=0;
    if (in_io_pool)
        color = MPI_UNDEFINED;
    else
        color = 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, Global.myID, &comm_solver);
    if (in_io_pool) {
        planes_IO_PES_main(Global.myID);
        goto IO_PES_REJOIN;
    }

    if (Global.myID == 0) {
        printf( "Starting psolve $Revision: 1.166 $ on %d PEs (%d IO Pool PEs).\n\n",
                Global.theGroupSize, Param.IO_pool_pe_count );
        fflush( stdout );
    }

    /* Read input parameters from file */
    read_parameters(argc, argv);

    /* Create and open database */
    open_cvmdb();

    /* Initialize nonlinear parameters */
    if ( Param.includeNonlinearAnalysis == YES ) {
        nonlinear_init(Global.myID, Param.parameters_input_file, Param.theDeltaT, Param.theEndT);
    }

    if ( Param.includeBuildings == YES ){
        bldgs_init( Global.myID, Param.parameters_input_file );
    }

    if ( Param.drmImplement == YES ){
    	Timer_Start("Init Drm Parameters");
    	Param.theDrmPart = drm_init(Global.myID, Param.parameters_input_file , Param.includeBuildings);
    	Timer_Stop("Init Drm Parameters");
    	Timer_Reduce("Init Drm Parameters", (TimerKind)(MAX | MIN | AVERAGE) , comm_solver);
    }

    // INTRODUCE BKT MODEL
    /* Init Quality Factor Table */
    constract_Quality_Factor_Table();

    /* Generate, partition and output unstructured octree mesh */
    mesh_generate();

    if ( Param.includeBuildings == YES ){
        if ( get_fixedbase_flag() == YES ) {
            bldgs_fixedbase_init( Global.myMesh, Param.theEndT-Param.theStartT );
        }
        bldgs_finalize();
    }

    if ( Param.drmImplement == YES ) {
    	Timer_Start("Drm Init");
    	if ( Param.theDrmPart == PART0 ) {
    		find_drm_nodes(Global.myMesh, Global.myID, Param.parameters_input_file,
    				Global.myOctree->ticksize, Global.theGroupSize);
    	}
    	if (Param.theDrmPart == PART1) {
    		setup_drm_data(Global.myMesh, Global.myID, Global.theGroupSize);
    	}
    	if (Param.theDrmPart == PART2) {
    		proc_drm_elems(Global.myMesh, Global.myID, Global.theGroupSize, Param.theTotalSteps);
    	}
    	drm_stats(Global.myID, Global.theGroupSize, Global.theXForMeshOrigin,
    			 Global.theYForMeshOrigin, Global.theZForMeshOrigin);
    	Timer_Stop("Drm Init");
    	Timer_Reduce("Drm Init", (TimerKind)(MAX | MIN | AVERAGE) , comm_solver);
    }

    if (Param.theMeshOutFlag && DO_OUTPUT) {
        mesh_output();
    }

    if ( Param.storeMeshCoordinatesForMatlab == YES ) {
        saveMeshCoordinatesForMatlab( Global.myMesh, Global.myID, Param.parameters_input_file,
				      Global.myOctree->ticksize,Param.theTypeOfDamping,Global.theXForMeshOrigin,
				      Global.theYForMeshOrigin,Global.theZForMeshOrigin, Param.includeBuildings);
    }

    Timer_Start("Mesh Stats Print");
    mesh_print_stat(Global.myOctree, Global.myMesh, Global.myID, Global.theGroupSize,
		    Param.theMeshStatFilename);
    Timer_Stop("Mesh Stats Print");
    Timer_Reduce("Mesh Stats Print", (TimerKind)(MAX | MIN), comm_solver);

    /* Initialize the output planes */
    if ( Param.theNumberOfPlanes != 0 ) {
        planes_setup(Global.myID, &Param.thePlanePrintRate, Param.IO_pool_pe_count,
		     Param.theNumberOfPlanes, Param.parameters_input_file, get_surface_shift(),
		     Param.theSurfaceCornersLong, Param.theSurfaceCornersLat,
		     Param.theDomainX, Param.theDomainY, Param.theDomainZ,
		     Param.planes_input_file);
    }

    if ( Param.theNumberOfStations !=0 ){
        output_stations_init(Param.parameters_input_file);
    }

    /* Initialize the GPU */
    gpu_init();

    /* Initialize the solver, source and output structures */
    solver_init();
    Timer_Start("Solver Stats Print");
    solver_printstat( Global.mySolver );
    Timer_Stop("Solver Stats Print");
    Timer_Reduce("Solver Stats Print", (TimerKind)(MAX | MIN), comm_solver);

    /* Initialize nonlinear solver analysis structures */
    if ( Param.includeNonlinearAnalysis == YES ) {
        nonlinear_solver_init(Global.myID, Global.myMesh, Param.theDomainZ);
        if ( Param.theNumberOfStations !=0 ){
            nonlinear_stations_init(Global.myMesh, Param.myStations, Param.myNumberOfStations);
        }
        nonlinear_stats(Global.myID, Global.theGroupSize);
    }

    Timer_Start("Source Init");
    source_init(Param.parameters_input_file);
    Timer_Stop("Source Init");
    Timer_Reduce("Source Init", (TimerKind)(MAX | MIN), comm_solver);

    /* Mapping element indices for stiffness
     * This is for compatibility with nonlinear
     * \TODO a more clever way should be possible
     */
    stiffness_init(Global.myID, Global.myMesh);

    /* this is a little too late to check for output parameters,
     * but let's do this in the mean time
     */
    output_init (Param.parameters_input_file, &Param.theOutputParameters);

    /* Run the solver and output the results */
    MPI_Barrier(comm_solver);
    Timer_Start("Solver");
    solver_run();
    Timer_Stop("Solver");
    MPI_Barrier(comm_solver);

    if ( Param.includeNonlinearAnalysis == YES ) {
        nonlinear_yield_stats( Global.myMesh, Global.myID, Param.theTotalSteps,
			       Global.theGroupSize );
    }

    output_fini();

#ifdef DEBUG
    /* Does the OS page out my resident set ? */
    if ((Global.myID % PROCPERNODE) == 0) {
        /* system("ps xl"); */
    }

    /* Are there pending messages that I haven't processed */
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_solver, &flag, &status);
    if (flag != 0) {
        fprintf(stderr, "Thread %d: MPI error: unreceived incoming message\n",
                Global.myID);
    }
#endif /* DEBUG */

    local_finalize();

    output_get_stats();

    MPI_Barrier(comm_solver);
    Timer_Stop("Total Wall Clock");

    /* Print out the timing stat */
    if (Global.myID == 0) {
        print_timing_stat();
    }

    /* TODO: Think of a better place for this */
    if ( Param.includeNonlinearAnalysis == YES ) {
        if ( get_geostatic_total_time() > 0 ) {
            check_balance(Global.myID);
        }
    }

    /* Send a message to IO pool PEs to close output files and goto here */
    if (Param.theNumberOfPlanes != 0) {
        planes_close(Global.myID, Param.IO_pool_pe_count, Param.theNumberOfPlanes);
    }
    IO_PES_REJOIN:

    MPI_Finalize();

    return 0;
}
