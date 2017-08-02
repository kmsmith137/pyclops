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
  pyclops/py_array.hpp \
  pyclops/converters.hpp \
  pyclops/extension_module.hpp \
  pyclops/functional_wrappers.hpp \
  pyclops/internals.hpp

OFILES = cfunction_table.o \
  extension_module.o \
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

ifndef PYDIR
$(error Fatal: Makefile.local must define PYDIR variable)
endif


####################################################################################################


all: libpyclops.so pyclops.so the_greatest_module.so

install: libpyclops.so pyclops.so
	mkdir -p $(INCDIR)/pyclops $(LIBDIR)/ $(PYDIR)/
	cp -f $(INCFILES) $(INCDIR)/
	cp -f libpyclops.so $(LIBDIR)/
	cp -f pyclops.so $(PYDIR)/

uninstall:
	rm -rf $(INCDIR)/pyclops.hpp $(INCDIR)/pyclops/ $(LIBDIR)/libpyclops.so $(PYDIR)/pyclops.so

clean:
	rm -f *~ *.o *.so *.pyc pyclops/*~


####################################################################################################


%.o: %.cpp $(INCFILES)
	$(CPP) -c -o $@ $<

libpyclops.so: $(OFILES)
	$(CPP) $(CPP_LFLAGS) -Wno-strict-aliasing -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -shared -o $@ $^ $(LIBS_PYMODULE)

pyclops.so: pyclops.o libpyclops.so
	$(CPP) $(CPP_LFLAGS) -Wno-strict-aliasing -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -shared -o $@ $^ -lpyclops $(LIBS_PYMODULE)

the_greatest_module.so: the_greatest_module.cpp libpyclops.so
	$(CPP) $(CPP_LFLAGS) -L. -Wno-strict-aliasing -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -shared -o $@ $< -lpyclops $(LIBS_PYMODULE)
