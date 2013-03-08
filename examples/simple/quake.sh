#!/bin/bash

# Turn echo on
set -v

#MPIFLAGS="-gdb"
#DEBUG_FLAGS="-debug -debugger gdb"
#NUMBER_PROCESSORS=2
NUMBER_PROCESSORS=1

RUNDIR=`pwd`
JOBDIR=${RUNDIR}
MPIDIR=/usr/bin
QUAKEDIR=${HOME}/v/quake/hercules

PSOLVE_EXE=${QUAKEDIR}/quake/forward/psolve

# Source CVM database

#CVM_DBNAME=${RUNDIR}/../cvmdb/labase.e
CVM_DBNAME=${JOBDIR}/simple_case.e
PHYSICS_IN=${JOBDIR}/in/physics.in
NUMERICAL_IN=${JOBDIR}/in/numerical.in
VIS_CFG=${JOBDIR}/vis.cfg
MESH_OUT=${JOBDIR}/out/mesh.e
SOLVER_OUT=${JOBDIR}/out/disp-out.q4d

set +v

function exec_echo() {
    local exe_name=$1
    shift
    echo ${exe_name} $*
    ${exe_name} $*
}

exec_echo \
    ${MPIDIR}/mpiexec \
    ${*} \
    ${MPIFLAGS} \
    ${DEBUG_FLAGS} \
    -np ${NUMBER_PROCESSORS} \
    ${PSOLVE_EXE} \
    ${CVM_DBNAME} \
    ${PHYSICS_IN} \
    ${NUMERICAL_IN} \
    ${MESH_OUT} \
    ${SOLVER_OUT}

