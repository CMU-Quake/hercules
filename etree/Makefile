#ifndef WORKDIR
	WORKDIR = $(PWD)/..
#endif

-include $(WORKDIR)/systemdef.mk
-include $(WORKDIR)/common.mk

#
# Object modules of the library
#
OBJECTS = dlink.o code.o  buffer.o schema.o xplatform.o btree.o etree.o wrapper.o

TARGET = libetree.a

all: $(TARGET)

libetree.a: $(OBJECTS)
	ar rcs libetree.a $(OBJECTS) 
	ranlib libetree.a

.PHONY: clean
clean:
	rm -f $(OBJECTS) *~ 

.PHONY: cleanll
cleanall:
	rm -f $(TARGET) $(OBJECTS) *~	

