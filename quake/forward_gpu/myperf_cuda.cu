#ifdef USE_PERF
/* Performance tracking enabled */

/* Include files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "mpi.h"
#include <cuda.h>
#include <cupti.h>
#include "myperf_cuda.h"

/* Maximum length of a string */
#define PERF_MAX_STR 256

/* Maximum number of timers */
#define PERF_MAX_TIMERS 16

/* CUDA flops metric */
#define PERF_CUDA_FLOPS_METRIC "flops_dp"
//#define PERF_CUDA_FLOPS_METRIC "flop_count_dp"

/* Performance module run states */
typedef enum {PERF_STATE_UNINIT, PERF_STATE_OFF, PERF_STATE_ON} pstate_t;

/* Data structure representing a wallclock timer */
typedef struct {
  long long start;
  long long elapsed;
} ptimer_t;

/* User data for event collection callback */
typedef struct MetricData_t {
  // the device where metric is being collected
  CUdevice device;
  // the set of event groups to collect for a pass
  CUpti_EventGroupSet *eventGroups;
  // the current number of events collected in eventIdArray and                  // eventValueArray
  uint32_t eventIdx;
  // the number of entries in eventIdArray and eventValueArray
  uint32_t numEvents;
  // array of event ids
  CUpti_EventID *eventIdArray;
  // array of event values
  uint64_t *eventValueArray;
} MetricData_t;

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

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
  CUptiResult _status = call;                                           \
  if (_status != CUPTI_SUCCESS) {                                       \
  const char *errstr;                                                   \
  cuptiGetResultString(_status, &errstr);                               \
  fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
	  __FILE__, __LINE__, #call, errstr);                           \
  exit(-1);                                                             \
  }                                                                     \
  } while (0)

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
//static long long perf_papi_elapsed;
//static long long perf_papi_timer_start;

/* MPI state information */
static pstate_t perf_mpi_state;
static uint64_t perf_mpi_sent_bytes = 0;
static uint64_t perf_mpi_recv_bytes = 0;
static double perf_mpi_elapsed = 0.0;

/* CUDA state information */
static pstate_t perf_flops_gpu_state;
static ptimer_t perf_cuda_timer;
//static long long perf_cuda_elapsed;
//static long long perf_cuda_timer_start;
static CUdevice perf_device;
static CUcontext perf_context = 0;
static int perf_context_is_new = 0;
static CUpti_MetricID perf_metricId;
static MetricData_t perf_metricData;


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

int perf_init(int devicenum)
{
  int retval;
  CUptiResult cupti_retval;
  CUresult cuda_retval;
  CUpti_EventGroupSets *passData;
  struct cudaDeviceProp deviceprops;

  perf_state = PERF_STATE_UNINIT;
  memset(&perf_timers[0], 0, PERF_MAX_TIMERS * sizeof(ptimer_t));

  perf_timer_count = 0;
  perf_flops_cpu_state = PERF_STATE_UNINIT;
  perf_papi_events = PAPI_NULL;
  //perf_papi_elapsed = 0;
  memset(&perf_papi_timer, 0, sizeof(ptimer_t));
  perf_flops_gpu_state = PERF_STATE_UNINIT;
  //perf_cuda_elapsed = 0;
  memset(&perf_cuda_timer, 0, sizeof(ptimer_t));
  perf_mpi_state = PERF_STATE_UNINIT;
  perf_mpi_sent_bytes = 0;
  perf_mpi_recv_bytes = 0;
  perf_mpi_elapsed = 0.0;

  /* Create statistics datatype */
  perf_create_datatype_stats(&PERF_DATATYPE_STAT);

  /* Initialize PAPI library */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    //fprintf(stderr,"PAPI library version mismatch!\n");
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
    //fprintf(stderr, "Failed to create event set\n");
    return(-4);
  }

  /* Add metric to the PAPI eventset */
  if (PAPI_add_event(perf_papi_events, PAPI_FP_OPS) != PAPI_OK) {
    //fprintf(stderr, "Failed to add PAPI_FP_INS event to PAPI event set\n");
    return(-5);
  }

  /* Create PAPI values buffer */
  memset(&perf_papi_values, 0, 1*sizeof(long_long));

  /* Initialize CUDA API driver */
  if (cuInit(0) != CUDA_SUCCESS) {
    //fprintf(stderr, "Failed to initialize CUDA API driver\n");
    return(-6);
  }

  if (cuDeviceGet(&perf_device, devicenum) != CUDA_SUCCESS) {
    //fprintf(stderr, "Failed to get device handle\n");
    return(-7);
  }

  /* Create new context only if not in exclusive mode */
  if (cudaGetDeviceProperties(&deviceprops, devicenum) != cudaSuccess) {
    //fprintf(stderr, "Failed to get device propertiesn");
    return(-1);
  }

  if ((deviceprops.computeMode == cudaComputeModeExclusive) ||
      (deviceprops.computeMode == cudaComputeModeExclusiveProcess)) {
    cuda_retval = cuCtxGetCurrent(&perf_context);
    if (cuda_retval == CUDA_SUCCESS) {
      if (perf_context == NULL) {
	if (cuCtxCreate(&perf_context, 0, perf_device) != CUDA_SUCCESS) {
	  //fprintf(stderr, "Failed to create new device context\n");
	  return(-8);
	}
	perf_context_is_new = 1;
      }
    } else {
      //fprintf(stderr, "Failed to query current context - error %d\n", 
      //	      (int)cuda_retval);
      return(-8);
    }
  } else {
    if (cuCtxCreate(&perf_context, 0, perf_device) != CUDA_SUCCESS) {
      //fprintf(stderr, "Failed to create new device context\n");
      return(-8);
    }
    perf_context_is_new = 1;
  }

  /* allocate space to hold all the events needed for the metric */
  cupti_retval = cuptiMetricGetIdFromName(perf_device, 
					  PERF_CUDA_FLOPS_METRIC, 
					  &perf_metricId);
  if (cupti_retval != CUPTI_SUCCESS) {
    //const char *errstr;
    //cuptiGetResultString(cupti_retval, &errstr);
    //fprintf(stderr, "CUPTI ERROR: %s\n", errstr);
    return(-9);
  }

  cupti_retval = cuptiMetricGetNumEvents(perf_metricId, 
					 &perf_metricData.numEvents);
  if (cupti_retval != CUPTI_SUCCESS) {
    //const char *errstr;
    //cuptiGetResultString(cupti_retval, &errstr);
    //fprintf(stderr, "CUPTI ERROR: %s\n", errstr);
    return(-9);
  }

  perf_metricData.device = perf_device;
  perf_metricData.eventIdArray = (CUpti_EventID *)malloc(perf_metricData.numEvents * sizeof(CUpti_EventID));
  perf_metricData.eventValueArray = (uint64_t *)malloc(perf_metricData.numEvents * sizeof(uint64_t));
  memset(perf_metricData.eventValueArray, 0, perf_metricData.numEvents * sizeof(uint64_t));
  perf_metricData.eventIdx = 0;

  // get the number of passes required to collect all the events                 // needed for the metric and the event groups for each pass
  cupti_retval = cuptiMetricCreateEventGroupSets(perf_context, 
					   sizeof(perf_metricId),
					   &perf_metricId, 
					   &passData);
  if (cupti_retval != CUPTI_SUCCESS) {
    //const char *errstr;
    //cuptiGetResultString(cupti_retval, &errstr);
    //fprintf(stderr, "CUPTI ERROR: %s\n", errstr);
    return(-9);
  }

  if (passData->numSets != 1) {
    //fprintf(stderr, "Too many passes required to compute metric (%d)\n", 
    //	    passData->numSets);
    return(-9);
  }
  perf_metricData.eventGroups = passData->sets;

  // Collect continuous versus kernel duration only
  //CUPTI_CALL(cuptiSetEventCollectionMode(perf_context, 
  //					 CUPTI_EVENT_COLLECTION_MODE_KERNEL));
  CUPTI_CALL(cuptiSetEventCollectionMode(perf_context, 
  					 CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));
  
  perf_state = PERF_STATE_ON;
  perf_flops_cpu_state = PERF_STATE_OFF;
  perf_flops_gpu_state = PERF_STATE_OFF;
  perf_mpi_state = PERF_STATE_OFF;

  return(0);
}


int perf_finalize()
{
  //  static CUcontext context = 0;

  if (perf_state != PERF_STATE_ON) {
    //fprintf(stderr, "Performance module is uninitialized\n");
    return(-1);
  }

  /* Remove all events in the eventset */
  if (PAPI_cleanup_eventset(perf_papi_events) != PAPI_OK) {
    //fprintf(stderr, "Failed to cleanup papi eventset\n");
    return(-2);
  }
  
  /* Free all memory and data structures, EventSet must be empty. */
  if (PAPI_destroy_eventset(&perf_papi_events) != PAPI_OK) {
    //fprintf(stderr, "Failed to destroy papi eventset\n");
    return(-3);
  }

  PAPI_shutdown();

  /* Destroy the device context we created */
  if (perf_context_is_new) {
    if (cuCtxDetach(perf_context) != CUDA_SUCCESS) {
      //fprintf(stderr, "Failed to detach and destroy context\n");
      return(-4);
    }
  }

  free(perf_metricData.eventIdArray);
  free(perf_metricData.eventValueArray);

  perf_timer_count = 0;
  memset(&perf_timers[0], 0, PERF_MAX_TIMERS * sizeof(ptimer_t));
  perf_papi_events = PAPI_NULL;
  //perf_papi_elapsed = 0;
  //perf_cuda_elapsed = 0;
  memset(&perf_papi_timer, 0, sizeof(ptimer_t));
  memset(&perf_cuda_timer, 0, sizeof(ptimer_t));
  perf_mpi_sent_bytes = 0;
  perf_mpi_recv_bytes = 0;
  perf_mpi_elapsed = 0.0;

  perf_state = PERF_STATE_UNINIT;
  perf_flops_cpu_state = PERF_STATE_UNINIT;
  perf_flops_gpu_state = PERF_STATE_UNINIT;
  perf_mpi_state = PERF_STATE_UNINIT;

  return(0);
}


int perf_metrics_gpu_list(int devicenum)
{
  int i;
  struct cudaDeviceProp deviceprops;
  CUdevice d;
  uint32_t nummetrics;

  size_t bufsize;
  CUpti_MetricID *buf;
  size_t namesize;
  char name[PERF_MAX_STR];

  if (cudaGetDeviceProperties(&deviceprops, devicenum) != cudaSuccess) {
    fprintf(stderr, "Failed to get device propertiesn");
    return(-1);
  }
  
  if (cuDeviceGet(&d, devicenum) != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to get device handle\n");
    return(-2);
  }

  cuptiDeviceGetNumMetrics(d, &nummetrics);

  bufsize = nummetrics * sizeof(CUpti_MetricID);
  buf = (CUpti_MetricID*)malloc(bufsize);
  cuptiDeviceEnumMetrics (d, &bufsize, buf);

  for (i = 0; i < nummetrics; i++) {
    namesize = PERF_MAX_STR;
    cuptiMetricGetAttribute (buf[i], CUPTI_METRIC_ATTR_NAME, 
			     &namesize, (void *)&name[0]);
    fprintf(stdout, "%s\n", name);
  }

  free(buf);

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
  perf_flops_gpu_stop();
  perf_mpi_stop();

  /* Collect my statistics */
  for (i = 0; i < perf_timer_count; i++) {
    perf_timer_values(i, &(stats.timer_elapsed[i]));
  }
  perf_flops_cpu_values(&(stats.cpu_flops), &(stats.cpu_elapsed));
  perf_flops_gpu_values(&(stats.gpu_flops), &(stats.gpu_elapsed));
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

    //fprintf(stderr, "\n%d\t%lu\t%e\t%lu\t%e\t%lu\t%lu\t%e\n", myid,
    //    stats.cpu_flops, stats.cpu_elapsed, 
    //    stats.gpu_flops, stats.gpu_elapsed, 
    //    stats.mpi_sent, stats.mpi_recv, stats.mpi_elapsed);

    /* Write metrics to file */
    fp = fopen(outfile, "w");
    if (fp == NULL) {
      fprintf(stderr, "Failed to open statistics output file\n");
      free(statbuf);
      return(-2);
    }

    fprintf(fp, "# PROC CPU_FLOPS CPU_ELAPSED GPU_FLOPS GPU_ELAPSED MPI_SENT MPI_RECV MPI_ELAPSED [TIMERS_0-%d]\n", perf_timer_count);
    for (p = 0; p < nproc; p++) {
      fprintf(fp, "%d\t%lu\t%e\t%lu\t%e\t%lu\t%lu\t%e", p,
	      statbuf[p].cpu_flops, statbuf[p].cpu_elapsed, 
	      statbuf[p].gpu_flops, statbuf[p].gpu_elapsed, 
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
    //handle_error(1);
  }

  //perf_papi_timer_start = PAPI_get_virt_usec();
  perf_papi_timer.start = PAPI_get_real_usec();

  perf_flops_cpu_state = PERF_STATE_ON;
    
  return(0);
}


int perf_flops_gpu_start()
{
  if (perf_flops_gpu_state != PERF_STATE_OFF) {
    return(-1);
  }

  cudaDeviceSynchronize();

  /* Enable collection all event groups */
  for (int i = 0; i < perf_metricData.eventGroups->numEventGroups; i++) {
    uint32_t all = 1;
    
    CUPTI_CALL(cuptiEventGroupSetAttribute(perf_metricData.eventGroups->eventGroups[i],
					   CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
					   sizeof(all), &all));
    
    CUPTI_CALL(cuptiEventGroupEnable(perf_metricData.eventGroups->eventGroups[i]));
  }

  // Reset event counter
  perf_metricData.eventIdx = 0;  

  //perf_papi_timer_start = PAPI_get_virt_usec();
  perf_cuda_timer.start = PAPI_get_real_usec();

  perf_flops_gpu_state = PERF_STATE_ON;
    
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
    // handle_error(1);
  }

  /* Accumulate counters */
  perf_papi_values += new_papi_values;

  perf_flops_cpu_state = PERF_STATE_OFF;

  return(0);
}


int perf_flops_gpu_stop()
{
  if (perf_flops_gpu_state != PERF_STATE_ON) {
    return(-1);
  }

  cudaDeviceSynchronize();

  //perf_papi_elapsed += PAPI_get_virt_usec() - perf_papi_timer_start;
  perf_cuda_timer.elapsed += PAPI_get_real_usec() - perf_cuda_timer.start;
  
  // for each group, read the event values from the group and record
  // in metricData
  for (int i = 0; i < perf_metricData.eventGroups->numEventGroups; i++) {
    CUpti_EventGroup group = perf_metricData.eventGroups->eventGroups[i];
    CUpti_EventDomainID groupDomain;
    uint32_t numEvents, numInstances, numTotalInstances;
    CUpti_EventID *eventIds;
    size_t groupDomainSize = sizeof(groupDomain);
    size_t numEventsSize = sizeof(numEvents);
    size_t numInstancesSize = sizeof(numInstances);
    size_t numTotalInstancesSize = sizeof(numTotalInstances);
    uint64_t *values, normalized, sum;
    size_t valuesSize, eventIdsSize;
    size_t numEventIdsRead;

    CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					   CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
					   &groupDomainSize, &groupDomain));
    CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(perf_metricData.device, groupDomain, 
						  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
						  &numTotalInstancesSize, &numTotalInstances));
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
					   &numInstancesSize, &numInstances));
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					   CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
					   &numEventsSize, &numEvents));

    eventIdsSize = numEvents * sizeof(CUpti_EventID);
    eventIds = (CUpti_EventID *)malloc(eventIdsSize);
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, 
					   CUPTI_EVENT_GROUP_ATTR_EVENTS,
					   &eventIdsSize, eventIds));
    
    //valuesSize = sizeof(uint64_t) * numInstances;
    valuesSize = sizeof(uint64_t) * numInstances * numEvents;
    values = (uint64_t *)malloc(valuesSize);

    //fprintf(stdout, "Number of events = %d\n", numEvents);
    //fprintf(stdout, "Number of total instances = %d\n", numTotalInstances);
    //fprintf(stdout, "Number of instances = %d\n", numInstances);
    
    CUPTI_CALL(cuptiEventGroupReadAllEvents(group, CUPTI_EVENT_READ_FLAG_NONE, 
    					    &valuesSize, values,
    					    &eventIdsSize, eventIds, 
    					    &numEventIdsRead));

    if (numEventIdsRead != numEvents) {
      fprintf(stderr, "Failed to read event group counters\n");
    }

    for (int j = 0; j < numEvents; j++) {

      // sum collect event values from all instances
      sum = 0;
      for (int k = 0; k < numInstances; k++)
	sum += values[k*numEvents + j];
      
      // normalize the event value to represent the total number of
      // domain instances on the device
      normalized = (sum * numTotalInstances) / numInstances;
      
      perf_metricData.eventIdArray[perf_metricData.eventIdx] = eventIds[j];
      perf_metricData.eventValueArray[perf_metricData.eventIdx] += normalized;
      perf_metricData.eventIdx++;
    }

    free(eventIds);
    free(values);
  }

  for (int i = 0; i < perf_metricData.eventGroups->numEventGroups; i++) {
    CUPTI_CALL(cuptiEventGroupDisable(perf_metricData.eventGroups->eventGroups[i]));
  }

  // Reset event counter
  //perf_metricData.eventIdx = 0;  

  perf_flops_gpu_state = PERF_STATE_OFF;

  return(0);
}

    
int perf_flops_cpu_values(uint64_t *ops, double *elapsed)
{
  *ops = perf_papi_values;
  *elapsed = ((double)(perf_papi_timer.elapsed)) * .000001;
  
  return(0);
}


int perf_flops_gpu_values(uint64_t *ops, double *elapsed)
{
  CUpti_MetricValue metricValue;
  CUpti_MetricValueKind valueKind;
  size_t valueKindSize = sizeof(valueKind);
  uint64_t duration = 1;

  *ops = 0;
  *elapsed = ((double)(perf_cuda_timer.elapsed)) * .000001;

  /* No events collected */
  if (perf_metricData.eventIdx == 0) {
    return(-1);
  }

  if (perf_metricData.eventIdx != perf_metricData.numEvents) {
    fprintf(stderr, "error: expected %u metric events, got %u\n",
            perf_metricData.numEvents, perf_metricData.eventIdx);
     return(-1);
   }

  // use all the collected events to calculate the metric value
  cuptiMetricGetValue(perf_device, perf_metricId,
		      perf_metricData.numEvents * sizeof(CUpti_EventID),
		      perf_metricData.eventIdArray,
		      perf_metricData.numEvents * sizeof(uint64_t),
		      perf_metricData.eventValueArray,
		      duration, &metricValue);
  
  cuptiMetricGetAttribute(perf_metricId, CUPTI_METRIC_ATTR_VALUE_KIND, 
			  &valueKindSize, &valueKind);
  switch (valueKind) {
  case CUPTI_METRIC_VALUE_KIND_DOUBLE:
    //printf("Metric %s = %f\n", metricName, metricValue.metricValueDouble);
    break;
  case CUPTI_METRIC_VALUE_KIND_UINT64:
    //printf("Metric %s = %llu\n", metricName,
    //	   (unsigned long long)metricValue.metricValueUint64);
    *ops = metricValue.metricValueUint64;
    break;
  case CUPTI_METRIC_VALUE_KIND_INT64:
    //printf("Metric %s = %lld\n", metricName,
    //	   (long long)metricValue.metricValueInt64);
    break;
  case CUPTI_METRIC_VALUE_KIND_PERCENT:
    //printf("Metric %s = %f%%\n", metricName, metricValue.metricValuePercent);
    break;
  case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
    //printf("Metric %s = %llu bytes/sec\n", metricName, 
    //	   (unsigned long long)metricValue.metricValueThroughput);
    break;
  case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
    // printf("Metric %s = utilization level %u\n", metricName, 
    //   (unsigned int)metricValue.metricValueUtilizationLevel);
    break;
  default:
    fprintf(stderr, "error: unknown value kind\n");
    return(-2);
    }
  
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

  //MPI_Type_size(datatype, &size);
  //perf_mpi_recv_bytes += count * size;

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
#include "myperf_cuda.h"

int perf_init(int devicenum)
{
  return(0);
}

int perf_finalize()
{
  return(0);
}

int perf_metrics_gpu_list(int devicenum)
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

int perf_flops_gpu_start()
{
  return(0);
}

int perf_flops_gpu_stop()
{
  return(0);
}

int perf_flops_gpu_values(uint64_t *ops, double *elapsed)
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
