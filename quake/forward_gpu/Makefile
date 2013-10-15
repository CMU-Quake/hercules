# -*- Makefile -*-
# 
#
ifndef WORKDIR
  WORKDIR = $(PWD)/../..
endif

include $(WORKDIR)/systemdef.mk
include $(WORKDIR)/common.mk

# Earthquake related, set these in user.mk
#
# -DDAMPING
# -DDEBUG_OVERFLOW
# -DAUTO_DELTA_T
# -DVALIDATION
# -DHALFSPACE
# -DBOUNDARY
SOLVE_CPPFLAGS ?= -DHALFSPACE -DBOUNDARY  


LOADLIBES = $(CVM_DIR)/cvm.o $(OCTOR_LIB) $(ETREE_LIB) 

#
# Is vis enabled?
#
ifeq (${ENABLE_VIZ},1)
  VIS_CPPFLAGS ?= -DVISINTERVAL=10 
  VIS_CPPFLAGS += -DVIS -I${VIS_DIR}
  LOADLIBES += $(VIS_LIB) $(JPEG_LIB)
endif

# I/O
# -DNO_OUTPUT
# -DUSERCVMDB
IO_CPPFLAGS ?= -DUSECVMDB 


CPPFLAGS += -I$(ETREE_DIR) -I$(CVM_DIR) -I$(OCTOR_DIR) -I. -I$(MPI_DIR)/include
CPPFLAGS += $(SOLVE_CPPFLAGS) $(IO_CPPFLAGS) $(VIS_CPPFLAGS)

LDFLAGS += -lm 

ifeq ($(SYSTEM), BGW)
  LOADLIBES += /bgl/local/lib/libmpihpm_c.a /bgl/BlueLight/ppcfloor/bglsys/lib/libbgl_perfctr.rts.a
endif

#
# Object modules 
#
PSOLVE_OBJECTS = psolve.o nrutila.o quakesource.o geometrics.o nonlinear.o \
	commutil.o util.o output.o io_planes.o io_checkpoint.o stiffness.o damping.o quake_util.o kernel.o timers.o buildings.o meshformatlab.o drm.o 

OBJECTS = ${PSOLVE_OBJECTS}

.PHONY: all tags clean cleanall move_cxx_repository

all: move_cxx_repository $(TARGET)

TARGET = psolve


psolve: ${PSOLVE_OBJECTS} ${OCTOR_LIB}
	$(LD) $^ \
	$(LOADLIBES) \
	$(LDFLAGS) \
	-o $@

psolve_debug: psolve_debug.o 
	$(LD) $^ \
	$(LOADLIBES) \
	$(LDFLAGS) \
	-o $@

psolve_debug.o: psolve.o nrutila.o quakesource.o geometrics.o io_planes.o io_checkpoint.o
	cp -f psolve.o psolve_debug.o

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVFLAGS) -c $^ -o $@

all: $(TARGET)

clean:
	rm -f $(OBJECTS)  core *~ 

cleanall:
	rm -f $(OBJECTS)  $(TARGET) core *~

TAGS:
	etags *.c *.h

tags:	TAGS

# $Id: Makefile,v 1.20 2011/10/07 15:00:10 yisbilir Exp $
