#ifdef USE_PERF
/* Performance tracking enabled */

/* Include files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "mpi.h"
#include "myperf.h"

/* Maximum length of a string */
#define PERF_MAX_STR 256

/* Maximum number of timers */
#define PERF_MAX_TIMERS 16

/* Performance module run states */
typedef enum {PERF_STATE_UNINIT, PERF_STATE_OFF, PERF_STATE_ON} pstate_t;

/* Data structure representing a wallclock timer */
typedef struct {
  long long start;
  long long elapsed;
} ptimer_t;

/* Data structure for statistic gathering */
typedef struct {
  uint64_t cpu_flops;
  double cpu_elapsed;
  uint64_t gpu_flops;
  double gpu_elapsed;
  uint64_t mpi_sent;
  uint64_t mpi_recv;
  double mpi_elapsed;
  double timer_elapsed[PERF_MAX_TIMERS];
} pstat_t;

/* Performance module state */
static pstate_t perf_state;
MPI_Datatype PERF_DATATYPE_STAT;

/* Timer state information */
static uint32_t perf_timer_count = 0;
static ptimer_t perf_timers[PERF_MAX_TIMERS];

/* PAPI state information */
static pstate_t perf_flops_cpu_state;
static int perf_papi_events = PAPI_NULL;
static long_long perf_papi_values;
static ptimer_t perf_papi_timer;

/* MPI state information */
static pstate_t perf_mpi_state;
static uint64_t perf_mpi_sent_bytes = 0;
static uint64_t perf_mpi_recv_bytes = 0;
static double perf_mpi_elapsed = 0.0;


int perf_create_datatype_stats(MPI_Datatype *dt)
{
  int i;
  pstat_t stat;
  MPI_Datatype types[8];
  int blocklen[8];
  MPI_Aint disp[8];
  
  types[0] = MPI_UNSIGNED_LONG_LONG;
  types[1] = MPI_DOUBLE;
  types[2] = MPI_UNSIGNED_LONG_LONG;
  types[3] = MPI_DOUBLE;
  types[4] = MPI_UNSIGNED_LONG_LONG;
  types[5] = MPI_UNSIGNED_LONG_LONG;
  types[6] = MPI_DOUBLE;
  types[7] = MPI_DOUBLE;

  disp[0] = (char*)&stat.cpu_flops - (char*)&stat.cpu_flops;
  disp[1] = (char*)&stat.cpu_elapsed - (char*)&stat.cpu_flops;
  disp[2] = (char*)&stat.gpu_flops - (char*)&stat.cpu_flops;
  disp[3] = (char*)&stat.gpu_elapsed - (char*)&stat.cpu_flops;
  disp[4] = (char*)&stat.mpi_sent - (char*)&stat.cpu_flops;
  disp[5] = (char*)&stat.mpi_recv - (char*)&stat.cpu_flops;
  disp[6] = (char*)&stat.mpi_elapsed - (char*)&stat.cpu_flops;
  disp[7] = (char*)&stat.timer_elapsed[0] - (char*)&stat.cpu_flops;

  for (i = 0; i < 7; i++) {
    blocklen[i] = 1;
  }
  blocklen[7] = PERF_MAX_TIMERS;

  MPI_Type_create_struct(8, blocklen, disp, types, dt);
  MPI_Type_commit(dt);
  
  return(0);
}

int perf_init()
{
  int retval;

  perf_state = PERF_STATE_UNINIT;
  memset(&perf_timers[0], 0, PERF_MAX_TIMERS * sizeof(ptimer_t));

  perf_timer_count = 0;
  perf_flops_cpu_state = PERF_STATE_UNINIT;
  perf_papi_events = PAPI_NULL;
  memset(&perf_papi_timer, 0, sizeof(ptimer_t));
  perf_mpi_state = PERF_STATE_UNINIT;
  perf_mpi_sent_bytes = 0;
  perf_mpi_recv_bytes = 0;
  perf_mpi_elapsed = 0.0;

  /* Initialize PAPI library */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    fprintf(stderr,"PAPI library version mismatch!\n");
    return(-1); 
  }
  if (retval < 0) {
    return(-2);
  }
  retval = PAPI_is_initialized();
  if (retval != PAPI_LOW_LEVEL_INITED) {
    return(-3);
  }

  /* Create empty PAPI event set */
  if (PAPI_create_eventset(&perf_papi_events) != PAPI_OK) {
    fprintf(stderr, "Failed to create event set\n");
    return(-4);
  }

  /* Add metric to the PAPI eventset */
  if (PAPI_add_event(perf_papi_events, PAPI_FP_OPS) != PAPI_OK) {
    fprintf(stderr, "Failed to add PAPI_FP_INS event to PAPI event set\n");
    return(-5);
  }

  /* Create PAPI values buffer */
  memset(&perf_papi_values, 0, 1*sizeof(long_long));
  
  perf_state = PERF_STATE_ON;
  perf_flops_cpu_state = PERF_STATE_OFF;
  perf_mpi_state = PERF_STATE_OFF;

  /* Create statistics datatype */
  perf_create_datatype_stats(&PERF_DATATYPE_STAT);

  return(0);
}


int perf_finalize()
{
  if (perf_state != PERF_STATE_ON) {
    fprintf(stderr, "Performance module is uninitialized\n");
    return(-1);
  }

  /* Remove all events in the eventset */
  if (PAPI_cleanup_eventset(perf_papi_events) != PAPI_OK) {
    fprintf(stderr, "Failed to cleanup papi eventset\n");
    return(-2);
  }
  
  /* Free all memory and data structures, EventSet must be empty. */
  if (PAPI_destroy_eventset(&perf_papi_events) != PAPI_OK) {
    fprintf(stderr, "Failed to destroy papi eventset\n");
    return(-3);
  }

  PAPI_shutdown();

  perf_timer_count = 0;
  memset(&perf_timers[0], 0, PERF_MAX_TIMERS * sizeof(ptimer_t));
  perf_papi_events = PAPI_NULL;
  memset(&perf_papi_timer, 0, sizeof(ptimer_t));
  perf_mpi_sent_bytes = 0;
  perf_mpi_recv_bytes = 0;
  perf_mpi_elapsed = 0.0;

  perf_state = PERF_STATE_UNINIT;
  perf_flops_cpu_state = PERF_STATE_UNINIT;
  perf_mpi_state = PERF_STATE_UNINIT;

  return(0);
}


int perf_metrics_collect(MPI_Comm comm, const char *outfile)
{
  int i, p, myid, nproc;
  pstat_t stats, *statbuf = NULL;
  FILE *fp = NULL;

  /* Get number of of procs in the job */
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &nproc);

  /* Stop collection of metrics */
  for (i = 0; i < perf_timer_count; i++) {
    perf_timer_stop(i);
  }
  perf_flops_cpu_stop();
  perf_mpi_stop();

  /* Collect my statistics */
  for (i = 0; i < perf_timer_count; i++) {
    perf_timer_values(i, &(stats.timer_elapsed[i]));
  }
  perf_flops_cpu_values(&(stats.cpu_flops), &(stats.cpu_elapsed));
  perf_mpi_values(&(stats.mpi_sent), &(stats.mpi_recv), &(stats.mpi_elapsed));

  if (myid == 0) {
    /* Allocate buffer for statistics gathering */
    statbuf = (pstat_t *)malloc(nproc * sizeof(pstat_t));
    if (statbuf == NULL) {
      fprintf(stderr, "Failed to allocate statistics buffer\n");
      return(-1);
    }
    memset(statbuf, 0, nproc * sizeof(pstat_t));
  }

  /* Gather metrics */
  MPI_Gather(&stats, 1, PERF_DATATYPE_STAT,
  	     statbuf, 1, PERF_DATATYPE_STAT, 0, comm);

  if (myid == 0) {
    /* Write metrics to file */
    fp = fopen(outfile, "w");
    if (fp == NULL) {
      fprintf(stderr, "Failed to open statistics output file\n");
      free(statbuf);
      return(-2);
    }

    fprintf(fp, "# PROC CPU_FLOPS CPU_ELAPSED MPI_SENT MPI_RECV MPI_ELAPSED [TIMERS_0-%d]\n", perf_timer_count);
    for (p = 0; p < nproc; p++) {
      fprintf(fp, "%d\t%lu\t%e\t%lu\t%lu\t%e", p,
	      statbuf[p].cpu_flops, statbuf[p].cpu_elapsed, 
	      statbuf[p].mpi_sent, statbuf[p].mpi_recv, statbuf[p].mpi_elapsed);
      for (i = 0; i < perf_timer_count; i++) {
	fprintf(fp, "\t%e", statbuf[p].timer_elapsed[i]);
      }
      fprintf(fp, "\n");
    }
    
    fclose(fp);
    free(statbuf);
  }

  return(0);
}

int perf_timer_new(uint32_t *tid)
{
  if (perf_state != PERF_STATE_ON) {
    return(-1);
  }

  if (perf_timer_count >= PERF_MAX_TIMERS) {
    return(-2);
  }

  *tid = perf_timer_count++;

  return(0);
}


int perf_timer_start(uint32_t tid)
{
  if (perf_state != PERF_STATE_ON) {
    return(-1);
  }

  if (tid >= perf_timer_count) {
    return(-2);
  }

  perf_timers[tid].start = PAPI_get_real_usec();

  return(0);
}


int perf_timer_stop(uint32_t tid)\
{

  if (perf_state != PERF_STATE_ON) {
    return(-1);
  }

  if (tid >= perf_timer_count) {
    return(-2);
  }

  perf_timers[tid].elapsed += PAPI_get_real_usec() - perf_timers[tid].start;

  return(0);
}


int perf_timer_values(uint32_t tid, double *elapsed)
{
  *elapsed = ((double)(perf_timers[tid].elapsed)) * .000001;
  return(0);
}


int perf_flops_cpu_start()
{
  if (perf_flops_cpu_state != PERF_STATE_OFF) {
    return(-1);
  }

  if (PAPI_start(perf_papi_events) != PAPI_OK) {
    return(-2);
  }

  //perf_papi_timer_start = PAPI_get_virt_usec();
  perf_papi_timer.start = PAPI_get_real_usec();

  perf_flops_cpu_state = PERF_STATE_ON;
    
  return(0);
}


int perf_flops_cpu_stop()
{
  long_long new_papi_values;

  if (perf_flops_cpu_state != PERF_STATE_ON) {
    return(-1);
  }

  //perf_papi_elapsed += PAPI_get_virt_usec() - perf_papi_timer_start;
  perf_papi_timer.elapsed += PAPI_get_real_usec() - perf_papi_timer.start;
  
  if (PAPI_stop(perf_papi_events, 
		&new_papi_values) != PAPI_OK) {
    return(-1);
  }

  /* Accumulate counters */
  perf_papi_values += new_papi_values;

  perf_flops_cpu_state = PERF_STATE_OFF;

  return(0);
}


int perf_flops_cpu_values(uint64_t *ops, double *elapsed)
{
  *ops = perf_papi_values;
  *elapsed = ((double)(perf_papi_timer.elapsed)) * .000001;
  
  return(0);
}


int perf_mpi_start()
{
  if (perf_mpi_state != PERF_STATE_OFF) {
    return(-1);
  }

  perf_mpi_state = PERF_STATE_ON;

  return(0);
}


int perf_mpi_stop()
{
  if (perf_mpi_state != PERF_STATE_ON) {
    return(-1);
  }

  perf_mpi_state = PERF_STATE_OFF;

  return(0);
}


int perf_mpi_values(uint64_t *sent_bytes, uint64_t *recv_bytes,
		    double *elapsed)
{
  *sent_bytes = perf_mpi_sent_bytes;
  *recv_bytes = perf_mpi_recv_bytes;
  *elapsed = perf_mpi_elapsed;
  return(0);
}


#ifdef PERF_MPI_MPICH2
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
	     int dest, int tag, MPI_Comm comm)
#else
int MPI_Send(void *buf, int count, MPI_Datatype datatype,
	     int dest, int tag, MPI_Comm comm)
#endif
{
  double tstart;
  int size, retval;

  if (perf_mpi_state != PERF_STATE_ON) {
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
  }

  MPI_Type_size(datatype, &size);
  perf_mpi_sent_bytes += count * size;

  tstart = MPI_Wtime();
  retval = PMPI_Send(buf, count, datatype, dest, tag, comm);
  perf_mpi_elapsed += MPI_Wtime() - tstart;

  return(retval);
}


int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
{
  double tstart;
  int size, retval;

  if (perf_mpi_state != PERF_STATE_ON) {
    return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  }

  MPI_Type_size(datatype, &size);
  perf_mpi_recv_bytes += count * size; 

  tstart = MPI_Wtime();
  retval = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  perf_mpi_elapsed += MPI_Wtime() - tstart;

  return(retval);
}


int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request)
{
  double tstart;
  int size, retval;

  if (perf_mpi_state != PERF_STATE_ON) {
    return PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  }

  MPI_Type_size(datatype, &size);
  perf_mpi_recv_bytes += count * size;

  tstart = MPI_Wtime();
  retval = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
  perf_mpi_elapsed += MPI_Wtime() - tstart;

  return(retval);
}


int MPI_Waitall(int count, MPI_Request array_of_requests[], 
		MPI_Status array_of_statuses[])
{
  double tstart;
  int retval;

  if (perf_mpi_state != PERF_STATE_ON) {
    return PMPI_Waitall(count, array_of_requests, array_of_statuses);
  }

  tstart = MPI_Wtime();
  retval = PMPI_Waitall(count, array_of_requests, array_of_statuses);
  perf_mpi_elapsed += MPI_Wtime() - tstart;

  return(retval);
}


int MPI_Barrier(MPI_Comm comm)
{
  double tstart;
  int retval;

  if (perf_mpi_state != PERF_STATE_ON) {
    return PMPI_Barrier(comm);
  }

  tstart = MPI_Wtime();
  retval = PMPI_Barrier(comm);
  perf_mpi_elapsed += MPI_Wtime() - tstart;

  return(retval);
}

#else
/* Performance tracking disabled - NO-OP the api */

/* Include files */
#include "myperf.h"

int perf_init(int devicenum)
{
  return(0);
}

int perf_finalize()
{
  return(0);
}

int perf_metrics_collect(MPI_Comm comm, const char *outfile)
{
  return(0);
}

int perf_timer_new(uint32_t *tid)
{
  return(0);
}

int perf_timer_start(uint32_t tid)
{
  return(0);
}

int perf_timer_stop(uint32_t tid)
{
  return(0);
}

int perf_timer_values(uint32_t tid, double *elapsed)
{
  return(0);
}

int perf_flops_cpu_start()
{
  return(0);
}

int perf_flops_cpu_stop()
{
  return(0);
}

int perf_flops_cpu_values(uint64_t *ops, double *elapsed)
{
  return(0);
}

int perf_mpi_start()
{
  return(0);
}

int perf_mpi_stop()
{
  return(0);
}

int perf_mpi_values(uint64_t *sent_bytes, uint64_t *recv_bytes, 
		    double *elapsed)
{
  return(0);
}

#endif
