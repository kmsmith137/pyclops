# Makefile.local for cedar.computecanada.ca

# Directory where C++ libraries will be installed
LIBDIR=$(HOME)/lib

# Directory where C++ header files will be installed
INCDIR=$(HOME)/include

# Directory where Python modules will be installed
PYDIR=$(HOME)/lib/python2.7/site-packages

# Read-only location of python header files
# Hint for PYTHON_INCDIR: import distutils.sysconfig; distutils.sysconfig.get_python_inc()
PYTHON_INCDIR=/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/python/2.7.14/include/python2.7

# Hint for NUMPY_INCDIR: import numpy; numpy.get_include()
# On cedar, I'm assuming numpy has been installed with 'pip install --user'.
NUMPY_INCDIR=$(HOME)/.local/lib/python2.7/site-packages/numpy/core/include

# C++ compiler command, including flags
# Must support c++11
CPP=g++ -std=c++11 -fPIC -Wall -O3 -march=native -ffast-math -funroll-loops -I. -I$(INCDIR) -I$(PYTHON_INCDIR) -I$(NUMPY_INCDIR)

# Extra linker flags when creating a shared library or exectutable from .o files
# Don't forget to add . and $(LIBDIR) in your LD_LIBRARY_PATH environment variable (in this order)
CPP_LFLAGS=-L. -L$(LIBDIR)
