#ifndef WORKDIR
	WORKDIR = $(PWD)/..
#endif

include $(WORKDIR)/systemdef.mk
include $(WORKDIR)/common.mk

OBJECTS = octor.o

TARGET = liboctor.a

all: $(TARGET)

liboctor.a: $(OBJECTS)
	ar rcs liboctor.a $(OBJECTS) 
	ranlib liboctor.a

.PHONY: clean
clean:
	rm -f $(OBJECTS) *~ 

.PHONY: cleanall
cleanall:
	rm -f $(TARGET) $(OBJECTS) *~	

