#!/bin/bash

# Turn echo on
set -v

#MPIFLAGS="-gdb"
#MPIFLAGS="--debug"
NUMBER_PROCESSORS=2
#NUMBER_PROCESSORS=1

RUNDIR=`pwd`

# JOBDIR is by default the same as RUNDIR
JOBDIR=${RUNDIR}
MPIDIR=/usr/bin
QUAKEDIR=../..

PSOLVE_EXE=${QUAKEDIR}/quake/forward/psolve

# Source CVM database

#CVM_DBNAME=${RUNDIR}/../cvmdb/labase.e
CVM_DBNAME=${JOBDIR}/cvmdb/labase.e
PHYSICS_IN=${JOBDIR}/physics.in
NUMERICAL_IN=${JOBDIR}/numerical.in
VIS_CFG=${JOBDIR}/vis.cfg
MESH_OUT=${JOBDIR}/output/mesh.e
SOLVER_OUT=${JOBDIR}/output/wavefield/disp-out.q4d
WF_OUTDIR=`dirname ${SOLVER_OUT}`


set +v

function exec_echo() {
    local exe_name=$1
    shift
    echo ${exe_name} $*
    ${exe_name} $*
}

function print_var() {
    local varname=${1}
    echo ${varname}=${!varname}
}

print_var RUNDIR
print_var JOBDIR
print_var QUAKEDIR
print_var WF_OUTDIR

mkdir -p ${WF_OUTDIR}
mkdir checkpoints

exec_echo \
    ${MPIDIR}/mpiexec \
    ${*} \
    ${MPIFLAGS} \
    -np ${NUMBER_PROCESSORS} \
    ${PSOLVE_EXE} \
    ${CVM_DBNAME} \
    ${PHYSICS_IN} \
    ${NUMERICAL_IN} \
    ${MESH_OUT} \
    ${SOLVER_OUT}

