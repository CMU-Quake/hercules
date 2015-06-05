#ifndef MYPERF_H
#define MYPERF_H

#include <stdint.h>
#include "mpi.h"

/* Initialization and finalization */
int perf_init();
int perf_finalize();

/* Collect metrics into a file */
int perf_metrics_collect(MPI_Comm comm, const char *outfile);

/* Wallclock timers */
int perf_timer_new(uint32_t *tid);
int perf_timer_start(uint32_t tid);
int perf_timer_stop(uint32_t tid);
int perf_timer_values(uint32_t tid, double *elapsed);

/* CPU FLOPS Performance tracking */
int perf_flops_cpu_start();
int perf_flops_cpu_stop();
int perf_flops_cpu_values(uint64_t *ops, double *elapsed);

/* MPI Performance tracking */
int perf_mpi_start();
int perf_mpi_stop();
int perf_mpi_values(uint64_t *sent_bytes, uint64_t *recv_bytes, 
		    double *elapsed);

/* Only override MPI functions if performace tracking unabled */
#ifdef USE_PERF

/* MPI function over-rides */
#ifdef PERF_MPI_MPICH2
int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
	     int dest, int tag, MPI_Comm comm);
#else
int MPI_Send(void *buf, int count, MPI_Datatype datatype,
	     int dest, int tag, MPI_Comm comm);
#endif
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, 
	     int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
	      int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Waitall(int count, MPI_Request array_of_requests[], 
		MPI_Status array_of_statuses[]);
int MPI_Barrier(MPI_Comm comm);

#endif

#endif

