#include <mpi.h>

enum TimerKind {NO_TIMER=0, NON_CUMULATIVE=1, MAX=2, MIN=4, AVERAGE=8};

extern void     Timer_Start(char* TimerName);
extern void     Timer_Stop(char* TimerName);
extern void     Timer_Reset(char* TimerName);
extern int      Timer_Exists(char* TimerName);
extern void     Timer_Config(char* TimerName, enum TimerKind kind);
extern double   Timer_Value(char* TimerName, enum TimerKind kind);
extern void     Timer_PrintAll(MPI_Comm Communicator);
extern void     Timer_Reduce(char* TimerName, enum TimerKind kind, MPI_Comm Communicator);

