#!/bin/sh
#PBS -l size=4
#PBS -l walltime=00:30:00
#PBS -j oe
# PBS -m aeb
#PBS -q debug

# Model:        ShakeOut or TeraShake Box: Tera1Hz-new.e
# Sim freq:     0.5 Hz
# Sim Vsmin:    500 m/s
# Sim time:     250 s
# Point per WL: 8
# delta t:      0.0125

# Old iobuf params
# IOBUF_PARAMS="*mesh.e:size=1M:count=64:ignoreflush:lazyclose,*.e:size=2M,%stdout,%stderr"

# New flags for new planes (04/08)
export MPICH_PTL_SEND_CREDITS=-1
export MPICH_PTL_UNEX_EVENTS=20000

# New iobuf params, got from Leo + my little modifications for no verbose in all but planes and correct cvmetree *.e file
IOBUF_PARAMS="*planedisplacements.*:size=2M:count=1:prefetch=1,*mesh.e:size= 1M:count=1:prefetch=1,*force_process.*:size=1M:count=1:prefetch=1,*.e:size=1M:count=32:prefetch=0,*station.*:size=20K:count=32:prefetch=0,%stdout,%stderr"

# export the iobuf params but not the verbose
export IOBUF_PARAMS
# IOBUF_VERBOSE=1
# export IOBUF_PARAMS IOBUF_VERBOSE

#JOB_NAME=`basename ${PBS_O_WORKDIR}`

CVM_DBNAME=${SCRATCH}/cvmetrees/simple_case.e
MY_JOB_DIR=${SCRATCH}/simple_case_MAY
CVM_DESTDIR=${MY_JOB_DIR}

#
# print PBS parameters for logging purpose
#
echo PBS_O_WORKDIR = ${PBS_O_WORKDIR}
echo PBS_O_SIZE    = ${PBS_O_SIZE}
echo PBS_NPROCS    = ${PBS_NPROCS}
echo PBS_NNODES    = ${PBS_NNODES}

echo "Initial dir ..."
pwd

echo "Changing to PBS_O_WORKDIR ..."
cd ${PBS_O_WORKDIR}
pwd

export CVM_DESTDIR

#
# print out some simulation parameters of interest:
#
echo
echo Material model: ${CVM_DBNAME}

#
# print some of the simulation parameters from numerical.in
#
grep -i simulation_ inputfiles/numerical.in


cat <<EOF
psolve command line:

    ${MY_JOB_DIR}/psolve
    ${CVM_DBNAME}
    ${MY_JOB_DIR}/inputfiles/physics.in
    ${MY_JOB_DIR}/inputfiles/numerical.in
    ${MY_JOB_DIR}/mesh.e
    ${MY_JOB_DIR}/disp.out

EOF


# run psolve
pbsyod ${MY_JOB_DIR}/psolve \
    ${CVM_DBNAME} \
    ${MY_JOB_DIR}/inputfiles/physics.in \
    ${MY_JOB_DIR}/inputfiles/numerical.in \
    ${MY_JOB_DIR}/mesh.e \
    ${MY_JOB_DIR}/disp.out
