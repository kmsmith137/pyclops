# Makefile.local must define the following variables
#   LIBDIR      install dir for C++ libraries
#   INCDIR      install dir for C++ headers
#   CPP         C++ compiler command line
#
# Some optional variables which I only use for osx/clang:
#   CPP_LFLAGS      extra linker flags when creating a .so or executable file from .o files
#   LIBS_PYMODULE   any extra libraries needed to link a python extension module (osx needs -lPython)
#
# See site/Makefile.local.* for examples.


INCFILES = pyclops.hpp \
  pyclops/py_object.hpp \
  pyclops/py_tuple.hpp \
  pyclops/py_dict.hpp \
  pyclops/cmodule.hpp \
  pyclops/to_python.hpp \
  pyclops/from_python.hpp \
  pyclops/functional_wrappers.hpp \
  pyclops/internals.hpp

OFILES = cfunction_table.o \
  cmodule.o \
  numpy_array.o \
  exceptions.o


####################################################################################################


include Makefile.local

ifndef CPP
$(error Fatal: Makefile.local must define CPP variable)
endif

ifndef INCDIR
$(error Fatal: Makefile.local must define INCDIR variable)
endif

ifndef LIBDIR
$(error Fatal: Makefile.local must define LIBDIR variable)
endif


####################################################################################################


all: libpyclops.so the_greatest_module.so

install: libpyclops.so
	mkdir -p $(INCDIR)/pyclops $(LIBDIR)/
	cp -f $(INCFILES) $(INCDIR)/
	cp -f libpyclops.so $(LIBDIR)/

uninstall:
	rm -f $(INCDIR)/pyclops.hpp $(INCDIR)/pyclops/ $(LIBDIR)/libpyclops.so

clean:
	rm -f *~ *.o *.so *.pyc pyclops/*~


####################################################################################################


%.o: %.cpp $(INCFILES)
	$(CPP) -c -o $@ $<

libpyclops.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -Wno-strict-aliasing -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -shared -o $@ $^ $(LIBS_PYMODULE)

the_greatest_module.so: the_greatest_module.cpp libpyclops.so
	$(CPP) $(CPP_LFLAGS) -L. -Wno-strict-aliasing -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -shared -o $@ $< -lpyclops $(LIBS_PYMODULE)
