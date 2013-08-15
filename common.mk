# Library paths
OCTOR_DIR = $(WORKDIR)/octor
OCTOR_LIB = $(OCTOR_DIR)/liboctor.a

ETREE_DIR = $(WORKDIR)/etree
ETREE_LIB = $(ETREE_DIR)/libetree.a

VIS_DIR = $(WORKDIR)/visualize
VIS_LIB = $(VIS_DIR)/libviz.a

JPEG_DIR = $(VIS_DIR)/jpeg-6b
JPEG_LIB = $(JPEG_DIR)/libjpeg.a


CVM_DIR = $(WORKDIR)/quake/cvm

FORWARD_DIR = $(WORKDIR)/quake/forward
FORWARD_GPU_DIR = $(WORKDIR)/quake/forward_gpu

-include $(WORKDIR)/user.mk
