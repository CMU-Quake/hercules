#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "timers.h"


#define MAXTIMERS 100 /*Maximim number of usable timers*/

struct timer {

    char             name[128];
    double           starttime;
    double           elapsed;
    double           max;
    double           min;
    double           average;
    int              running;
    enum TimerKind   flags;
};


static int FindTimer(char* TimerName);

struct timer Timers[MAXTIMERS];
static int   NumberOfTimersUsed = 0;


void  Timer_Start(char* TimerName){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	if (NumberOfTimersUsed == MAXTIMERS){
	    fprintf(stderr, "Too many timers (%d) allocated!\n", MAXTIMERS);
	    MPI_Abort(MPI_COMM_WORLD, -1);
	    exit(1);
	}
	strcpy(Timers[NumberOfTimersUsed].name, TimerName);
	Timers[NumberOfTimersUsed].elapsed = 0;
	Timers[NumberOfTimersUsed].max     = 0;
	Timers[NumberOfTimersUsed].min     = 0;
	Timers[NumberOfTimersUsed].average = 0;
	Timers[NumberOfTimersUsed].flags   = 0;
	Timers[NumberOfTimersUsed].running = 1;
	Timers[NumberOfTimersUsed].starttime = MPI_Wtime();
	NumberOfTimersUsed++;
    }
    else{
	Timers[timernum].starttime= MPI_Wtime();
    }

}



void     Timer_Stop(char* TimerName){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	fprintf(stderr, "Timer_Stop called for nonexistant timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }
    Timers[timernum].running = 0;
    if (Timers[timernum].flags & NON_CUMULATIVE)
	Timers[timernum].elapsed =   MPI_Wtime() - Timers[timernum].starttime;
    else
	Timers[timernum].elapsed +=  MPI_Wtime() - Timers[timernum].starttime;
}



void     Timer_Reset(char* TimerName){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	fprintf(stderr, "Timer_Reset called for nonexistant timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }
    Timers[timernum].elapsed = 0;
    Timers[timernum].max     = 0;
    Timers[timernum].min     = 0;
    Timers[timernum].average = 0;
    Timers[timernum].flags   = 0;
    Timers[timernum].running = 0;
    Timers[timernum].starttime=0;
}


/*Only current use is to set non-cumlative timers */
void     Timer_Config(char* TimerName, enum TimerKind kind){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	fprintf(stderr, "Timer_Config called for nonexistant timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }

    Timers[timernum].flags = kind;

}



double   Timer_Value(char* TimerName, enum TimerKind kind){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	fprintf(stderr, "Timer_Value called for nonexistant timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }
    if (Timers[timernum].running){
	if (Timers[timernum].flags & NON_CUMULATIVE)
	    return (MPI_Wtime() - Timers[timernum].starttime);
	else
	    return (Timers[timernum].elapsed + MPI_Wtime() - Timers[timernum].starttime);
    }

    if (kind & MAX)
	return Timers[timernum].max;
    if (kind & MIN)
	return Timers[timernum].min;
    if (kind & AVERAGE)
	return Timers[timernum].average;

    return Timers[timernum].elapsed;
}



/*Used to see if a timer has been instantiated */
int      Timer_Exists(char* TimerName){

    int timernum;
 
    timernum = FindTimer(TimerName);
 
    if (timernum== -1)
	return 0;
    else
	return 1;
}




void     Timer_PrintAll(MPI_Comm communicator){

    int PENum;
    MPI_Comm_rank(communicator, &PENum);
    if (PENum != 0){
	fprintf(stderr, "PrintAll() called on non-zero PE# %d!\n", PENum);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }

    int index;
    for( index=0; index<NumberOfTimersUsed; index++){
	printf("Timer: %s: ",Timers[index].name);
	if (Timers[index].running) printf("This timer was never stopped!");
	if (Timers[index].flags & NON_CUMULATIVE)
	    printf("(Non-Cumulative) ");
	printf("%f  ", Timers[index].elapsed);
	if (Timers[index].flags & MIN)
	    printf("(Min %f)   ", Timers[index].min);
	if (Timers[index].flags & MAX)
	    printf("(Max %f)   ", Timers[index].max);
	if (Timers[index].flags & AVERAGE)
	    printf("(Avg %f)   ", Timers[index].average);
	printf("\n");
    }
}



void     Timer_Reduce(char* TimerName, enum TimerKind kind, MPI_Comm Communicator){

    int timernum;

    timernum = FindTimer(TimerName);
    if (timernum== -1){
	fprintf(stderr, "Timer_Reduce called for nonexistant timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }

    if (Timers[timernum].running){
	fprintf(stderr, "Timer_Reduce called on unstopped timer %s!\n", TimerName);
	MPI_Abort(MPI_COMM_WORLD, -1);
	exit(1);
    }

    if (kind & MIN){
	MPI_Reduce (&Timers[timernum].elapsed, &Timers[timernum].min, 1, MPI_DOUBLE, MPI_MIN, 0, Communicator);
	Timers[timernum].flags = Timers[timernum].flags | MIN;
    }

    if (kind & MAX){
	MPI_Reduce (&Timers[timernum].elapsed, &Timers[timernum].max, 1, MPI_DOUBLE, MPI_MAX, 0, Communicator);
	Timers[timernum].flags = Timers[timernum].flags | MAX;
    }

    if (kind & AVERAGE){
	int size;
	MPI_Comm_size(Communicator, &size);
	MPI_Reduce (&Timers[timernum].elapsed, &Timers[timernum].average, 1, MPI_DOUBLE, MPI_SUM, 0, Communicator);
	Timers[timernum].average = Timers[timernum].average / size;
	Timers[timernum].flags = Timers[timernum].flags | AVERAGE;
    }

}



static int FindTimer(char* TimerName){

    int index;
    for( index=0; index<NumberOfTimersUsed; index++)
	if ( !strcmp(TimerName, Timers[index].name) )
	    return index;

    return -1;
}


