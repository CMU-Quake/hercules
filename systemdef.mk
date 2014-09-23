# -*- Makefile -*-
#
# Override SYSTEM and other variable definition in user.mk
#
# In order to switch between a debug and an optimized executable, set CFLAGS
# in your 'user.mk' file as follows:
#
# * For an executable with debug information:
#	CFLAGS = -g -DDEBUG -O0
#
# * For an optimized executable:
#	CFLAGS = -O2
#
# check other platform specific flags below.
#
-include $(WORKDIR)/user.mk

ifndef SYSTEM
	SYSTEM = XT5
endif

ifeq ($(SYSTEM), XT5)
        CC      = cc
        CXX     = CC
        LD      = CC
        CFLAGS  += -DBIGBEN 
        LDFLAGS += 
        ifdef IOBUF_INC
            CPPFLAGS += -I${IOBUF_INC}
        endif        
        CPPFLAGS    += -D_USE_FILE_OFFSET64 -D_FILE_OFFSET_BITS=64 -D_USE_LARGEFILE64       
        
endif


ifeq ($(SYSTEM), BGW)
	MPI_DIR ?= /bgl/BlueLight/ppcfloor/bglsys
	CC	 = mpcc
	CXX	 = mpCC
	LD	 =  mpCC
	CFLAGS  += -qmaxmem=-1 -qarch=440
	CFLAGS  += -DBGL 
	CFLAGS  += -D_FILE_OFFSET_BITS=64 
endif


ifeq ($(SYSTEM), BGL)
	MPI_DIR ?= /bgl/BlueLight/ppcfloor/bglsys
	CC	 = mpicc
	CXX	 = mpicxx
	LD	 = mpicxx
	CFLAGS  += -DBGL 
	CFLAGS  += -D_FILE_OFFSET_BITS=64 -Wall -D_LARGEFILE_SOURCE 
endif

#
# Use the following settings for IBM compilers
# CC     = /bgsys/drivers/ppcfloor/comm/bin/mpixlc
# CFLAGS = -O -qarch=450d -qtune=450
#
ifeq ($(SYSTEM), BGP)
	MPI_DIR  ?= /bgsys/drivers/ppcfloor/comm
	CC	  = mpicc
	CXX	  = mpicxx
	LD	  = mpicxx
	CPPFLAGS += -DBGP -DMPICH_IGNORE_CXX_SEEK -D_LARGEFILE_SOURCE  \
		    -D_FILE_OFFSET_BITS=64
	CFLAGS   += -Wall
endif


#
# Note: Add the following flags ( -fastsse -O3 ) to the CFLAGS variable
# in order to create an optimized executable.
#
# Also define USE_IOBUF=1 in order to use the io_buf library and remember
# to load the io_buf module before compiling and launching your job.
#
ifeq ($(SYSTEM), BIGBEN)
	CC 	= cc
	CXX 	= CC
	LD 	= CC
	CFLAGS  += -DBIGBEN -target=catamount
	LDFLAGS += -target=catamount
	ifdef IOBUF_INC
	    CPPFLAGS += -I${IOBUF_INC}
	endif
endif


#
# optimization flags: -fast -O
#
ifeq ($(SYSTEM), LEMIEUX)
	CC	 = cc
	CXX	 = cxx
	LD	 = cxx
	CFLAGS  += -DPROCPERNODE=4
	CFLAGS  += -trapuv -check_bounds -DALPHA_TRU64UNIX_CC -DALIGNMENT
	LDFLAGS += -lelan -lmpi
endif


ifeq ($(SYSTEM), MANTEO)
	MPI_DIR	 ?= /usr/local/mpich-1.2.6
	CC	  = $(MPI_DIR)/bin/mpicc
	CXX	  = $(MPI_DIR)/bin/mpicxx
	LD	  = $(MPI_DIR)/bin/mpicxx
	CFLAGS   += -Wall
	CPPFLAGS +=  -DPROCPERNODE=64 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif


ifeq ($(SYSTEM), USCHPC)
	MPI_DIR	 ?= /usr/usc/mpich/1.2.6..13b/gm-gnu32
	CC	  = $(MPI_DIR)/bin/mpicxx
	LD	  = $(MPI_DIR)/bin/mpicxx
	CFLAGS   += -Wall
	CPPFLAGS += -DPROCPERNODE=2 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif


ifeq ($(SYSTEM), SCEC)
	MPI_DIR	 ?= /opt/mpich/gnu
	CC	  = $(MPI_DIR)/bin/mpicxx
	LD	  = $(MPI_DIR)/bin/mpicxx
	CPPFLAGS += -DPROCPERNODE=1 -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
		    -DSCEC
	CFLAGS   += -Wall
endif

#
# Add, define or override the following options in $(WORKDIR)/user.mk:
#   RUN_DIR = /path/to/your/run/time/directory
#   CFLAGS += -O2
#   CFLAGS += -g -DDEBUG
#   CFLAGS += -DPROCPERNODE=2
#   CFLAGS += -DCVM_SRCPATH=\"/path/to/your/cvm/database/file\"
#   CFLAGS += -DCVM_DESTDIR=\"$(RUN_DIR)/destination/directory\"
#
ifeq ($(SYSTEM), LINUX-MPICH)
	MPI_DIR	    ?= /usr
	MPI_INCLUDE ?= $(MPI_DIR)/include
	CC	     = $(MPI_DIR)/bin/mpicc
	CXX          = $(MPI_DIR)/bin/mpicxx
	LD	     = $(MPI_DIR)/bin/mpicxx
	CXXFLAGS    += -DMPICH_IGNORE_CXX_SEEK
	CFLAGS      += -Wall
	CPPFLAGS    += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif

# This is the configuration to work locally in hooke.cml.cs.cmu.edu
# Prepared by: Ricardo, 2008--2009

ifeq ($(SYSTEM), HOOKE)
	MPI_INCLUDE  = /usr0/local/mpich2-1.0.3/src/include/
    MPI_DIR      = /usr0/local/mpich2-1.0.3/
    CC           = $(MPI_DIR)/bin/mpicc
    CXX          = $(MPI_DIR)/bin/mpicxx
    LD           = $(MPI_DIR)/bin/mpicxx
    CPPFLAGS    += -DMPICH_IGNORE_CXX_SEEK
    CFLAGS      += -Wall
    CPPFLAGS    += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif

ifeq ($(SYSTEM), MACBOOK)
	MPI_DIR      = /usr/
        MPI_INCLUDE  = $(MPI_DIR)/include/openmpi/ompi/mpi/cxx
        CC           = $(MPI_DIR)/local/bin/mpicc
        CXX          = $(MPI_DIR)/local/bin/mpicxx
        LD           = $(MPI_DIR)/local/bin/mpicxx
        CXXFLAGS    += -DMPICH_IGNORE_CXX_SEEK
        CFLAGS      += -Wall
        CPPFLAGS    += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
endif
 
