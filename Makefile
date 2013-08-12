# -*- Makefile -*-
#
# Description:	Quake's Hercules toolchain top directory Makefile
#
-include user.mk

WORKDIR = $(CURDIR)

include systemdef.mk
include common.mk


# Run make in a directory for the given target
#
# param ${1}: dir to run make on
# param ${2}: target
define run_make_in_dir
	@echo Building TARGET=${2} in DIR=${1}
	${MAKE} -C ${1} SYSTEM=${SYSTEM} WORKDIR=${WORKDIR} ${2}

endef

make_target_in_viz_dir = $(if $(ENABLE_VIZ),$(call run_make_in_dir,$(VIS_DIR),${1}))

.PHONY: all clean cleanall etree octor cvm forward forward_gpu

#	$(MAKE) -C $(VIS_DIR)     SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)

all:	etree octor cvm forward forward_gpu
	$(call make_target_in_viz_dir,all)


etree:
	$(MAKE) -C $(ETREE_DIR)   SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)

octor:
	$(MAKE) -C $(OCTOR_DIR)   SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)

cvm:
	$(MAKE) -C $(CVM_DIR)     SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)

forward:
	$(MAKE) -C $(FORWARD_DIR) SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)

forward_gpu:
	$(MAKE) -C $(FORWARD_GPU_DIR) SYSTEM=$(SYSTEM) WORKDIR=$(WORKDIR)


clean:
	$(call make_target_in_viz_dir,clean)
	$(MAKE) -C $(ETREE_DIR)   WORKDIR=$(WORKDIR) clean
	$(MAKE) -C $(OCTOR_DIR)   WORKDIR=$(WORKDIR) clean
	$(MAKE) -C $(CVM_DIR)     WORKDIR=$(WORKDIR) clean
	$(MAKE) -C $(FORWARD_DIR) WORKDIR=$(WORKDIR) clean
	$(MAKE) -C $(FORWARD_GPU_DIR) WORKDIR=$(WORKDIR) clean


MY_DIRS := $(ETREE_DIR) $(OCTOR_DIR) $(CVM_DIR) $(FORWARD_DIR) $(FORWARD_GPU_DIR)


# Call run_make_in_dir for each directory in a directory list.
#
# param ${1}: dir list
# param ${2}: target
#
run_make_for_dirs = $(foreach MY_DIR,${1},$(call run_make_in_dir,${MY_DIR},${2}))

cleanall: clean
	@echo MY_DIRS="${MY_DIRS}"
	$(call run_make_for_dirs,${MY_DIRS},cleanall)



# $Id: Makefile,v 1.11 2010/07/13 20:34:21 rtaborda Exp $
