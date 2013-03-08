#ifndef WORKDIR
	WORKDIR = $(PWD)/../..
#endif

include $(WORKDIR)/systemdef.mk
include $(WORKDIR)/common.mk

CFLAGS += -I$(ETREE_DIR) 
LOADLIBES += $(ETREE_DIR)/libetree.a

#
# Object modules 
#
OBJECTS = cvm.o .setdbctl.o showdbctl.o

TARGET = showdbctl querycvm scancvm dumpcvm pickrecord

all: $(TARGET)

.setdbctl: cvm.o .setdbctl.o 
showdbctl: cvm.o showdbctl.o 
querycvm: cvm.o querycvm.o 
scancvm: scancvm.o
dumpcvm: dumpcvm.o 
pickrecord: pickrecord.o 

clean:
	rm -f $(OBJECTS)  core *.o *~

cleanall:
	rm -f $(OBJECTS)  $(TARGET) core *~
