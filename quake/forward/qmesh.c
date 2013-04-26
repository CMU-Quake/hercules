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

/**
 * qmesh.c: Generate an unstructured octree mesh (in parallel).
 *
 * Input:  material database (cvm etree), physics.in, numerical.in.
 * Output: mesh database (mesh.e)
 *
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

#include "octor.h"
#include "etree.h"

/* Static global variables */
static int32_t myID, theGroupSize;
static float theFreq;
static octree_t *myOctree;

#define ERROR		-100
#define LINESIZE	512
#define FILEBUFSIZE	(1 << 26)

#define GOAHEAD_MSG	100
#define MESH_MSG	101
#define STAT_MSG	102
#define CVMRECORD_MSG	103

#define BATCH		(1 << 20)

/**
 * edata_t: Mesh element data fields
 *
 */
typedef struct edata_t {
    float edgesize, Vp, Vs, rho;
} edata_t;


/**
 * mdata_t: Mesh database record payload (excluing the locational code).
 *
 */
typedef struct mdata_t {
    int64_t nid[8];
    float edgesize, Vp, Vs, rho;
} mdata_t;


/**
 * mrecord_t: Complete mesh database record
 *
 */
typedef struct mrecord_t {
    etree_addr_t addr;
    mdata_t mdata;
} mrecord_t;



/**
 * Static global variables that will be accessed by setrec().
 *
 */
static double theVsCut, theFactor;
static double theNorth_m, theEast_m, theDepth_m;




#ifdef USECVMDB

#include "cvm.h"

#ifndef CVMBUFSIZE
#define CVMBUFSIZE 100
#endif  /* CVMBUFSIZE */

static etree_t *theCVMEp;
static double theXForMeshOrigin, theYForMeshOrigin, theZForMeshOrigin;

/**
 * setrec: Query the CVM database to obtain values (material properties)
 *         for a leaf octant.
 *
 */
void setrec(octant_t *leaf, double ticksize, void *data)
{
    edata_t*	 edata;
    double	 east_m, north_m, depth_m;
    tick_t	 halfticks;
    int32_t	 res;
    cvmpayload_t payload;

    edata = (edata_t*)data;

    halfticks = (tick_t)1 << (PIXELLEVEL - leaf->level - 1);

    edata->edgesize = ticksize * halfticks * 2;

    north_m = theXForMeshOrigin + (leaf->lx + halfticks) * ticksize;
    east_m  = theYForMeshOrigin + (leaf->ly + halfticks) * ticksize;
    depth_m = theZForMeshOrigin + (leaf->lz + halfticks) * ticksize;

    res = cvm_query(theCVMEp, east_m, north_m, depth_m, &payload);

    if (res != 0) {
	/* Center point out the bound. Set Vs to force split */
	edata->Vs = theFactor * edata->edgesize / 2;
    } else {
	/* Adjust the Vs */
	edata->Vs  = (payload.Vs < theVsCut) ? theVsCut : payload.Vs;
	edata->Vp  = payload.Vp;
	edata->rho = payload.rho;
    }

    return;
}

#else

/**
 * cvmrecord_t: cvm record.
 *
 */
typedef struct cvmrecord_t {
    char key[12];
    float Vp, Vs, density;
} cvmrecord_t;


static const int theCVMRecordSize = sizeof(cvmrecord_t);
static int theCVMRecordCount;
static cvmrecord_t * theCVMRecord;

static int32_t
zsearch(void *base, int32_t count, int32_t recordsize,
	const point_t *searchpt)
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
    if (myID == theGroupSize - 1) {
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
	    fprintf(stderr, "Thread %d: Cannot open flat CVM file\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	fd = fileno(fp);
	if (fstat(fd, &statbuf) != 0) {
	    fprintf(stderr, "Thread %d: Cannot get the status of CVM file\n",
		    myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	intervaltable = octor_getintervaltable(myOctree);

	/*
	  for (procid = 0; procid <= myID; procid++) {
	  fprintf(stderr, "interval[%d] = {%d, %d, %d}\n", procid,
	  intervaltable[procid].x << 1, intervaltable[procid].y << 1,
	  intervaltable[procid].z << 1);
	  }
	*/

	bytesent = 0;
	maxbuf = malloc(maxbufsize) ;
	if (maxbuf == NULL) {
	    fprintf(stderr, "Thread %d: Cannot allocate send buffer\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	isendreqs = (MPI_Request *)malloc(sizeof(MPI_Request) * theGroupSize);
	isendstats = (MPI_Status *)malloc(sizeof(MPI_Status) * theGroupSize);
	if ((isendreqs == NULL) || (isendstats == NULL)) {
	    fprintf(stderr, "Thread %d: Cannot allocate isend controls\n",
		    myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Try to read max number of CVM records as allowed */
	gettimeofday(&starttime, NULL);
	recordcount = fread(maxbuf, sizeof(cvmrecord_t),
			    maxbufsize / sizeof(cvmrecord_t), fp);
	gettimeofday(&endtime, NULL);

	iotime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
	    + (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	if (recordcount != maxbufsize / sizeof(cvmrecord_t)) {
	    fprintf(stderr, "Thread %d: Cannot read-init buffer\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* start with proc 0 */
	procid = 0;

	while (procid < myID) { /* repeatedly fill the buffer */
	    point_t searchpoint, *point;
	    int newreads;
	    int isendcount = 0;

	    /* we have recordcount to work with */
	    cvmrecord = (cvmrecord_t *)maxbuf;

	    while (procid < myID) { /* repeatedly send out data */

		searchpoint.x = intervaltable[procid + 1].x << 1;
		searchpoint.y = intervaltable[procid + 1].y << 1;
		searchpoint.z = intervaltable[procid + 1].z << 1;

		offset = zsearch(cvmrecord, recordcount, theCVMRecordSize,
				 &searchpoint);

		point = (point_t *)(cvmrecord + offset);

		if ((point->x != searchpoint.x) ||
		    (point->y != searchpoint.y) ||
		    (point->z != searchpoint.z)) {
		    break;
		} else {
		    bytecount = offset * sizeof(cvmrecord_t);
		    MPI_Isend(cvmrecord, bytecount, MPI_CHAR, procid,
			      CVMRECORD_MSG, MPI_COMM_WORLD,
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

	    gettimeofday(&starttime, NULL);
	    memmove(maxbuf, cvmrecord, bufferedbytes);
	    gettimeofday(&endtime, NULL);
	    memmovetime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
		+ (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	    gettimeofday(&starttime, NULL);
	    newreads = fread((char *)maxbuf + bufferedbytes,
			     sizeof(cvmrecord_t), maxcount - recordcount, fp);
	    gettimeofday(&endtime, NULL);
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
	    fprintf(stderr, "Thread %d: out of memory\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* fseek exiting the for loop has file cursor propertly */
#ifdef BIGBEN
	if (fseek(fp, bytesent, SEEK_SET) != 0) {
	    fprintf(stderr, "Thread %d: fseeko failed\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
#else
	if (fseeko(fp, bytesent, SEEK_SET) != 0) {
	    fprintf(stderr, "Thread %d: fseeko failed\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
#endif

	gettimeofday(&starttime, NULL);
	if (fread(cvmrecord, 1, bytecount, fp) != (size_t)bytecount) {
	    fprintf(stderr, "Thread %d: fail to read the last chunk\n",
		    myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}
	gettimeofday(&endtime, NULL);
	iotime += (endtime.tv_sec - starttime.tv_sec) * 1000.0
	    + (endtime.tv_usec - starttime.tv_usec) / 1000.0;

	/*
	  fprintf(stderr, "Procid = %d offset = %qd bytecount = %d\n",
	  myID, (int64_t)bytesent, bytecount);
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

	MPI_Probe(theGroupSize - 1, CVMRECORD_MSG, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_CHAR, &bytecount);

	cvmrecord = (cvmrecord_t *)malloc(bytecount);
	if (cvmrecord == NULL) {
	    fprintf(stderr, "Thread %d: out of memory\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	MPI_Recv(cvmrecord, bytecount, MPI_CHAR, theGroupSize - 1,
		 CVMRECORD_MSG, MPI_COMM_WORLD,  &status);

    }

    /* Every processor should set these parameters correctly */
    theCVMRecordCount = bytecount / sizeof(cvmrecord_t);
    if (theCVMRecordCount * sizeof(cvmrecord_t) != (size_t)bytecount) {
	fprintf(stderr, "Thread %d: received corrupted CVM data\n",
		myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    return cvmrecord;
}


/**
 * setrec: Search the CVM record array to obtain the material property of
 *         a leaf octant.
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

    if ((x * ticksize >= theNorth_m) ||
	(y * ticksize >= theEast_m) ||
	(z * ticksize >= theDepth_m)) {
	/* Center point out the bound. Set Vs to force split */
	edata->Vs = theFactor * edata->edgesize / 2;
    } else {
	int offset;

	/* map the coordinate from the octor address space to the
	 * etree address space
	 */
	searchpoint.x = x << 1;
	searchpoint.y = y << 1;
	searchpoint.z = z << 1;

	/* Inbound */
	offset = zsearch(theCVMRecord, theCVMRecordCount, theCVMRecordSize,
			 &searchpoint);
	if (offset < 0) {
	    fprintf(stderr, "setrec: fatal error\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	agghit = theCVMRecord + offset;
	edata->Vs = agghit->Vs;
	edata->Vp = agghit->Vp;
	edata->rho = agghit->density;

	/* Adjust the Vs */
	edata->Vs = (edata->Vs < theVsCut) ? theVsCut : edata->Vs;
    }

    return;
}

#endif  /*USECVMDB */




/**
 * toexpand: Instruct the Octor library whether a leaf octant needs to
 *           be expanded or not. Return 1 if true, 0 otherwise.
 *
 */
int32_t toexpand(octant_t *leaf, double ticksize, const void *data)
{

    if (data == NULL)
	return 1;
    else {
	const edata_t *edata;

	edata = (edata_t *)data;
	if (edata->edgesize <= edata->Vs / theFactor)
	    return 0;
	else
	    return 1;
    }
}



/**
 * parsetext: Parse a text file and return the value of a match string.
 *
 * - return 0 if OK, -1 on error
 *
 */
int parsetext( FILE *fp, const char *querystring, const char type,
	       void *result )
{
    int32_t res = 0, found = 0;

    /* Start from the beginning */
    rewind(fp);

    /* Look for the string until found */
    while (!found) {
	char line[LINESIZE];
	char delimiters[] = " =\n";
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
 * initparameters: Open material database and initialize various
 *                 static global variables. Return 0 if OK, -1 on error.
 */
int32_t initparameters(const char *physicsin, const char *numericalin,
		       double *px, double *py, double *pz)
{
    FILE *fp;
    double freq;
    int32_t samples;
    double region_origin_latitude_deg, region_origin_longitude_deg;
    double region_depth_shallow_m, region_length_east_m;
    double region_length_north_m, region_depth_deep_m;

#ifdef USECVMDB
    dbctl_t *myctl;
#endif

    /* Obtain the specficiation of the simulation */
    if ((fp = fopen(physicsin, "r")) == NULL) {
	fprintf(stderr, "Error opening %s\n", physicsin);
	return -1;
    }

    if ((parsetext(fp, "region_origin_latitude_deg", 'd',
		   &region_origin_latitude_deg) != 0) ||
	(parsetext(fp, "region_origin_longitude_deg", 'd',
		   &region_origin_longitude_deg) != 0) ||
	(parsetext(fp, "region_depth_shallow_m", 'd',
		   &region_depth_shallow_m) != 0) ||
	(parsetext(fp, "region_length_east_m", 'd',
		   &region_length_east_m) != 0) ||
	(parsetext(fp, "region_length_north_m", 'd',
		   &region_length_north_m) != 0) ||
	(parsetext(fp, "region_depth_deep_m", 'd',
		   &region_depth_deep_m) != 0)) {
	fprintf(stderr, "Error parsing fields from %s\n", physicsin);
	return -1;
    }
    fclose(fp);

    if ((fp = fopen(numericalin, "r")) == NULL) {
	fprintf(stderr, "Error opening %s\n", numericalin);
	return -1;
    }

    if ((parsetext(fp, "simulation_wave_max_freq_hz", 'd', &freq) != 0) ||
	(parsetext(fp, "simulation_node_per_wavelength", 'i', &samples) != 0)||
	(parsetext(fp, "simulation_shear_velocity_min", 'd', &theVsCut) != 0)){
	fprintf(stderr, "Error parsing fields from %s\n", numericalin);
	return -1;
    }
    fclose(fp);

    theFreq = freq;
    theFactor = freq * samples;

#ifdef USECVMDB

    /* Obtain the material database application control/meta data */
    if ((myctl = cvm_getdbctl(theCVMEp)) == NULL) {
	fprintf(stderr, "Error reading CVM etree control data\n");
	return -1;
    }

    /* Check the ranges of the mesh and the scope of the CVM etree */
    if ((region_origin_latitude_deg < myctl->region_origin_latitude_deg) ||
	(region_origin_longitude_deg < myctl->region_origin_longitude_deg) ||
	(region_depth_shallow_m < myctl->region_depth_shallow_m) ||
	(region_depth_deep_m > myctl->region_depth_deep_m) ||
	(region_origin_latitude_deg + region_length_north_m / DIST1LAT
	 > myctl->region_origin_latitude_deg
	 + myctl->region_length_north_m / DIST1LAT) ||
	(region_origin_longitude_deg + region_length_east_m / DIST1LON
	 > myctl->region_origin_longitude_deg +
	 myctl->region_length_north_m / DIST1LON)) {
	fprintf(stderr, "Mesh area out of the CVM etree\n");
	return -1;
    }

    /* Compute the coordinates of the origin of the mesh coordinate
       system in the CVM etree domain coordinate system */
    theXForMeshOrigin = (region_origin_latitude_deg
			 - myctl->region_origin_latitude_deg) * DIST1LAT;
    theYForMeshOrigin = (region_origin_longitude_deg
			 - myctl->region_origin_longitude_deg) * DIST1LON;
    theZForMeshOrigin = region_depth_shallow_m - myctl->region_depth_shallow_m;

    /* Free memory used by myctl */
    cvm_freedbctl(myctl);

#endif

    *px = region_length_north_m;
    *py = region_length_east_m;
    *pz = region_depth_deep_m - region_depth_shallow_m;

    return 0;
}


/**
 * bulkload: Append the data to the end of the mesh database. Return 0 if OK,
 *           -1 on error.
 *
 */
int32_t bulkload(etree_t *mep, mrecord_t *partTable, int32_t count)
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


/**
 * main
 *
 */

int main(int argc, char **argv)
{
    mesh_t *mesh;
    double double_message[5];
    double x, y, z, factor = 0, vscut = 0;
    double elapsedtime;
#ifndef NO_OUTPUT
    int32_t eindex;
    int32_t remains, batch, idx;
    mrecord_t *partTable;
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);
    MPI_Comm_size(MPI_COMM_WORLD, &theGroupSize);

    /* Read commandline arguments  */
    if (argc != 5) {
	if (myID == 0) {
	    fprintf(stderr, "usage: qmesh cvmdb physics.in numerical.in ");
	    fprintf(stderr, "meshdb\n");
	    fprintf(stderr,
		    "cvmdb: path to an etree database or a flat file.\n");
	    fprintf(stderr, "physics.in: path to physics.in.\n");
	    fprintf(stderr, "numerical.in: path to numerical.in.\n");
	    fprintf(stderr, "meshetree: path to the output mesh etree.\n");
	    fprintf(stderr, "\n");
	}
	MPI_Finalize();
	return -1;
    }

    /* Change the working directory to $LOCAL */
    /*
    localpath = getenv("LOCAL");
    if (localpath == NULL) {
        fprintf(stderr, "Thread %d: Cannot get $LOCAL value\n", myID);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }

    if (chdir(localpath) != 0) {
        fprintf(stderr, "Thread %d: Cannot chdir to %s\n", myID, localpath);
        MPI_Abort(MPI_COMM_WORLD, ERROR);
        exit(1);
    }
    */

    /* Replicate the material database among the processors */
    /*

    if ((theGroupSize - 1) / PROCPERNODE >= 1) {

        MPI_Comm replica_comm;

        if (myID % PROCPERNODE != 0) {
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, myID, &replica_comm);

        } else {
            int replica_id;
            off_t filesize, remains, batchsize;
            void *filebuf;
            int fd;

            MPI_Comm_split(MPI_COMM_WORLD, 0, myID, &replica_comm);
            MPI_Comm_rank(replica_comm, &replica_id);

            if (replica_id == 0) {

                struct stat statbuf;


                if (stat(argv[1], &statbuf) != 0) {
                    fprintf(stderr, "Thread 0: Cannot get stat of %s\n",
                            argv[1]);
                    MPI_Abort(MPI_COMM_WORLD, ERROR);
                    exit(1);
                }

                filesize = statbuf.st_size;
            }

            MPI_Bcast(&filesize, sizeof(off_t), MPI_CHAR, 0, replica_comm);

            if ((filebuf = malloc(FILEBUFSIZE)) == NULL) {
                fprintf(stderr, "Thread %d: run out of memory while ", myID);
                fprintf(stderr, "preparing to receive material database\n");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }

	    fd = (replica_id == 0) ?
	    open(argv[1], O_RDONLY) :
                open(argv[1], O_CREAT|O_TRUNC|O_WRONLY, S_IRUSR|S_IWUSR);

            if (fd == -1) {
                fprintf(stderr, "Thread %d: Cannot create replica database\n",
                        myID);
                perror("open");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }

            remains = filesize;
            while (remains > 0) {
                batchsize = (remains > FILEBUFSIZE) ? FILEBUFSIZE : remains;

                if (replica_id == 0) {
                    if (read(fd, filebuf, batchsize) !=  batchsize) {
                        fprintf(stderr, "Thread 0: Cannot read database\n");
                        perror("read");
                        MPI_Abort(MPI_COMM_WORLD, ERROR);
                        exit(1);
                    }
                }

                MPI_Bcast(filebuf, batchsize, MPI_CHAR, 0, replica_comm);

                if (replica_id != 0) {
                    if (write(fd, filebuf, batchsize) != batchsize) {
                        fprintf(stderr, "Thread %d: Cannot write replica ",
                                myID);
                        fprintf(stderr, "database\n");
                        MPI_Abort(MPI_COMM_WORLD, ERROR);
                        exit(1);
                    }
                }

                remains -= batchsize;
		}

            if (close(fd) != 0) {
                fprintf(stderr, "Thread %d: cannot close replica database\n",
                        myID);
                perror("close");
                MPI_Abort(MPI_COMM_WORLD, ERROR);
                exit(1);
            }
	    }

        MPI_Barrier(MPI_COMM_WORLD);

	}

    */



    /* Initialize static global varialbes */
    if (myID == 0) {
	/* Processor 0 reads the parameters */
	if (initparameters(argv[2], argv[3], &x, &y, &z) != 0) {
	    fprintf(stderr, "Thread %d: Cannot init parameters\n", myID);
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	factor = theFactor;
	vscut = theVsCut;
	double_message[0] = x;
	double_message[1] = y;
	double_message[2] = z;
	double_message[3] = factor;
	double_message[4] = vscut;

	/*
          fprintf(stderr, "&double_message[0] = %p\n", &double_message[0]);
          fprintf(stderr, "&double_message[4] = %p\n", &double_message[4]);
	  fprintf(stderr, "Thread 0: %f %f %f %f %f\n", x, y, z, factor, vscut);
        */
    }

    MPI_Bcast(double_message, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    x = double_message[0];
    y = double_message[1];
    z = double_message[2];
    factor = double_message[3];
    vscut  = double_message[4];

    theNorth_m = x;
    theEast_m = y;
    theDepth_m = z;

    /*
      printf("Thread %d: %f %f %f %f %f\n", myID, x, y, z, factor, vscut);
    */

    theFactor = factor;
    theVsCut = vscut;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsedtime = -MPI_Wtime();


    if (myID == 0) {
	fprintf(stdout, "PE = %d, Freq = %.2f\n", theGroupSize, theFreq);
	fprintf(stdout, "-----------------------------------------------\n");
    }


    /*----  Generate and partition an unstructured octree mesh ----*/
    if (myID == 0) {
	fprintf(stdout, "octor_newtree ... ");
    }

    /*
     * RICARDO: Carful with new_octree parameters (cutoff_depth)
     */

    myOctree = octor_newtree(x, y, z, sizeof(edata_t), myID, theGroupSize, MPI_COMM_WORLD, 0);
    if (myOctree == NULL) {
	fprintf(stderr, "Thread %d: fail to create octree\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    elapsedtime += MPI_Wtime();

    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }

#ifdef USECVMDB
    /* Open my copy of the material database */
    theCVMEp = etree_open(argv[1], O_RDONLY, CVMBUFSIZE, 0, 0);
    if (!theCVMEp) {
	fprintf(stderr, "Thread %d: Cannot open CVM etree database %s\n",
		myID, argv[1]);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

#else

    /* Use flat data record file and distibute the data in memories */
    elapsedtime = -MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "slicing CVM database ...");
    }

    theCVMRecord = sliceCVM(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsedtime += MPI_Wtime();
    if (theCVMRecord == NULL) {
	fprintf(stderr, "Thread %d: Error obtaining the CVM records from %s\n",
		myID, argv[1]);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    };
    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }

#endif

    elapsedtime = -MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "octor_refinetree ...");
    }
    if (octor_refinetree(myOctree, toexpand, setrec) != 0) {
	fprintf(stderr, "Thread %d: fail to refine octree\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    elapsedtime += MPI_Wtime();

    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }


    elapsedtime = -MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "octor_balancetree ... ");
    }
    if (octor_balancetree(myOctree, setrec, 0) != 0) { /* no progressive meshing (ricardo) */
	fprintf(stderr, "Thread %d: fail to balance octree\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    elapsedtime += MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }

#ifdef USECVMDB
    /* Close the material database */
    etree_close(theCVMEp);
#else
    free(theCVMRecord);

#endif /* USECVMDB */

    elapsedtime = -MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "octor_partitiontree ...");
    }
    if (octor_partitiontree(myOctree, NULL) != 0) {
	fprintf(stderr, "Thread %d: fail to balance load\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    elapsedtime +=  MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }

    elapsedtime = - MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "octor_extractmesh ... ");
    }
    mesh = octor_extractmesh(myOctree, NULL);
    if (mesh == NULL) {
	fprintf(stderr, "Thread %d: fail to extract mesh\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    elapsedtime += MPI_Wtime();
    if (myID == 0) {
	fprintf(stdout, "done.... %.2f seconds\n", elapsedtime);
    }

    /* We can do without the octree now */
    octor_deletetree(myOctree);


    /*---- Obtain and print the statistics of the mesh ----*/
    if (myID == 0) {
	int64_t etotal, ntotal, dntotal;
	int32_t received, procid;
	int32_t *enumTable, *nnumTable, *dnnumTable;
	int32_t rcvtrio[3];

	/* Allocate the arrays to hold the statistics */
	enumTable = (int32_t *)malloc(sizeof(int32_t) * theGroupSize);
	nnumTable = (int32_t *)malloc(sizeof(int32_t) * theGroupSize);
	dnnumTable = (int32_t *)malloc(sizeof(int32_t) * theGroupSize);

	if ((enumTable == NULL) ||
	    (nnumTable == NULL) ||
	    (dnnumTable == NULL)) {
	    fprintf(stderr, "Thread 0: out of memory\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Fill in my counts */
	enumTable[0] = mesh->lenum;
	nnumTable[0] = mesh->lnnum;
	dnnumTable[0] = mesh->ldnnum;

	/* Initialize sums */
	etotal = mesh->lenum;
	ntotal = mesh->lnnum;
	dntotal = mesh->ldnnum;

	/* Fill in the rest of the tables */
	received = 0;
	while (received < theGroupSize - 1) {
	    int32_t fromwhom;
	    MPI_Status status;

	    MPI_Probe(MPI_ANY_SOURCE, STAT_MSG, MPI_COMM_WORLD, &status);

	    fromwhom = status.MPI_SOURCE;

	    MPI_Recv(rcvtrio, 3, MPI_INT, fromwhom, STAT_MSG, MPI_COMM_WORLD,
		     &status);

	    enumTable[fromwhom] = rcvtrio[0];
	    nnumTable[fromwhom] = rcvtrio[1];
	    dnnumTable[fromwhom] = rcvtrio[2];

	    etotal += rcvtrio[0];
	    ntotal += rcvtrio[1];
	    dntotal += rcvtrio[2];

	    received++;
	}

	fprintf(stdout, "Mesh statistics:\n");
	fprintf(stdout, "                 Elements     Nodes    Danglings\n");
#ifdef ALPHA_TRU64UNIX_CC
	fprintf(stdout, "Total     :   %10ld%10ld   %10ld\n\n",
		etotal, ntotal, dntotal);
	for (procid = 0; procid < theGroupSize; procid++) {
	    fprintf(stdout, "Proc %5d:   %10d%10d   %10d\n", procid,
		    enumTable[procid], nnumTable[procid], dnnumTable[procid]);
	}

#else
	fprintf(stdout, "Total      :    %10qd%10qd   %10qd\n\n",
		etotal, ntotal, dntotal);
	for (procid = 0; procid < theGroupSize; procid++) {
	    fprintf(stdout, "Proc %5d:   %10d%10d   %10d\n", procid,
		    enumTable[procid], nnumTable[procid], dnnumTable[procid]);
	}
#endif

	free(enumTable);
	free(nnumTable);
	free(dnnumTable);

    } else {
	int32_t sndtrio[3];

	sndtrio[0] = mesh->lenum;
	sndtrio[1] = mesh->lnnum;
	sndtrio[2] = mesh->ldnnum;

	MPI_Send(sndtrio, 3, MPI_INT, 0, STAT_MSG, MPI_COMM_WORLD);
    }

#ifndef NO_OUTPUT

    /*---- Join elements and nodes, and send to Thread 0 for output */

    /* Allocate a fixed size buffer space to store the join results */
    partTable = (mrecord_t *)calloc(BATCH, sizeof(mrecord_t));
    if (partTable == NULL) {
	fprintf(stderr,	 "Thread %d: out of memory\n", myID);
	MPI_Abort(MPI_COMM_WORLD, ERROR);
	exit(1);
    }

    if (myID == 0) {
	char *mEtree;
	etree_t *mep;
	int32_t procid;

	mEtree = argv[4];
	mep = etree_open(mEtree, O_CREAT|O_RDWR|O_TRUNC, 0, sizeof(mdata_t),3);
	if (mep == NULL) {
	    fprintf(stderr, "Thread 0: cannot create mesh etree\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	/* Begin an appending operation */
	if (etree_beginappend(mep, 1) != 0) {
	    fprintf(stderr, "Thread 0: cannot begin an append operation\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

	eindex = 0;
	while (eindex < mesh->lenum) {
	    remains = mesh->lenum - eindex;
	    batch = (remains < BATCH) ? remains : BATCH;

	    for (idx = 0; idx < batch; idx++) {
		mrecord_t *mrecord;
		int32_t whichnode;
		int32_t localnid0;

		mrecord = &partTable[idx];

		/* Fill the address field */
		localnid0 = mesh->elemTable[eindex].lnid[0];

		mrecord->addr.x = mesh->nodeTable[localnid0].x;
		mrecord->addr.y = mesh->nodeTable[localnid0].y;
		mrecord->addr.z = mesh->nodeTable[localnid0].z;
		mrecord->addr.level = mesh->elemTable[eindex].level;
		mrecord->addr.type = ETREE_LEAF;

		/* Find the global node ids for the vertices */
		for (whichnode = 0; whichnode < 8; whichnode++) {
		    int32_t localnid;
		    int64_t globalnid;

		    localnid = mesh->elemTable[eindex].lnid[whichnode];
		    globalnid = mesh->nodeTable[localnid].gnid;

		    mrecord->mdata.nid[whichnode] = globalnid;
		}

		memcpy(&mrecord->mdata.edgesize, mesh->elemTable[eindex].data,
		       sizeof(edata_t));

		eindex++;
	    } /* for a batch */

	    if (bulkload(mep, partTable, batch) != 0) {
		fprintf(stderr, "Thread 0: Error bulk-loading data\n");
		MPI_Abort(MPI_COMM_WORLD, ERROR);
		exit(1);
	    }
	} /* for all the elements Thread 0 has */

	/* Receive data from other processors */
	for (procid = 1; procid < theGroupSize; procid++) {
	    MPI_Status status;
	    int32_t rcvbytecount;

	    /* Signal the next processor to go ahead */
	    MPI_Send(NULL, 0, MPI_CHAR, procid, GOAHEAD_MSG, MPI_COMM_WORLD);

	    while (1) {
		MPI_Probe(procid, MESH_MSG, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_CHAR, &rcvbytecount);

		batch = rcvbytecount / sizeof(mrecord_t);
		if (batch == 0) {
		    /* Done */
		    break;
		}

		MPI_Recv(partTable, rcvbytecount, MPI_CHAR, procid,
			 MESH_MSG, MPI_COMM_WORLD, &status);

		if (bulkload(mep, partTable, batch) != 0) {
		    fprintf(stderr, "Thread 0: Cannot bulk-load data from ");
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
	    fprintf(stderr, "Thread 0: Cannot close the etree database\n");
	    MPI_Abort(MPI_COMM_WORLD, ERROR);
	    exit(1);
	}

    } else {
	/* Processors other than 0 needs to send data to 0 */
	int32_t sndbytecount;
	MPI_Status status;

	/* Wait for my turn */
	MPI_Recv(NULL, 0, MPI_CHAR, 0, GOAHEAD_MSG, MPI_COMM_WORLD, &status);

	eindex = 0;
	while (eindex < mesh->lenum) {
	    remains = mesh->lenum - eindex;
	    batch = (remains < BATCH) ? remains : BATCH;

	    for (idx = 0; idx < batch; idx++) {
		mrecord_t *mrecord;
		int32_t whichnode;
		int32_t localnid0;

		mrecord = &partTable[idx];

		/* Fill the address field */
		localnid0 = mesh->elemTable[eindex].lnid[0];

		mrecord->addr.x = mesh->nodeTable[localnid0].x;
		mrecord->addr.y = mesh->nodeTable[localnid0].y;
		mrecord->addr.z = mesh->nodeTable[localnid0].z;
		mrecord->addr.level = mesh->elemTable[eindex].level;
		mrecord->addr.type = ETREE_LEAF;

		/* Find the global node ids for the vertices */
		for (whichnode = 0; whichnode < 8; whichnode++) {
		    int32_t localnid;
		    int64_t globalnid;

		    localnid = mesh->elemTable[eindex].lnid[whichnode];
		    globalnid = mesh->nodeTable[localnid].gnid;

		    mrecord->mdata.nid[whichnode] = globalnid;
		}

		memcpy(&mrecord->mdata.edgesize, mesh->elemTable[eindex].data,
		       sizeof(edata_t));

		eindex++;
	    } /* for a batch */

	    /* Send data to proc 0 */
	    sndbytecount = batch * sizeof(mrecord_t);
	    MPI_Send(partTable, sndbytecount, MPI_CHAR, 0, MESH_MSG,
		     MPI_COMM_WORLD);
	} /* While there is data left to be sent */

	/* Send an empty message to indicate the end of my transfer */
	MPI_Send(NULL, 0, MPI_CHAR, 0, MESH_MSG, MPI_COMM_WORLD);
    }

    /* Free the memory for the partial join results */
    free(partTable);

#endif

    octor_deletemesh(mesh);

    MPI_Finalize();

    return 0;
}
