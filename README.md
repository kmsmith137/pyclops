## pyclops

Some hacks for writing hybrid C++/python code.

Currently unreadable and documented, but it will be cleaned up some day...

Installation instructions follow.

### INSTALLATION

  - The pyclops Makefile assumes the existence of a file `Makefile.local` which defines
    a few machine-dependent Makefile variables:
    ```
      INCDIR     Installation directory for C++ header files
      LIBDIR     Installation directory for libraries
      CPP        C++ compiler executable + flags, see below for tips!
        etc.
    ```

    For a complete list of variables which must be defined, see comments at the top of ./Makefile.

    Rather than write a Makefile.local from scratch, I recommend that you start with one of the
    examples in the site/ directory, which contains Makefile.locals for a few frequently-used
    CHIME machines.  In particular, site/Makefile.local.kms_laptop16 is a recent osx machine,
    and site/Makefile.local.frb1 is a recent CentOS Linux machine.  (If you're a member of
    CHIME and you're using one of these machines, you can just symlink the appropriate file in
    site/ to ./Makefile.local)

  - Do `make all install` to build.

  - If you have trouble getting pyclops to build/work, then the problem probably has
    something to do with your compiler flags (specified as part of CPP) or environment 
    variables.  Here are a few hints:

      - You probably need `-std=c++11` in your compiler flags, for C++11 support
      - I usually use optimization flags `-O3 -march=native -ffast-math -funroll-loops`.
      - You probably want `-Wall -fPIC` in your compiler flags on general principle.
      - The pyclops build procedure assumes that the current directory is searched for header
        files and libraries, i.e. you should have `-I. -L.` in your compiler flags.
      - You also probably want `-I$(INCDIR) -L$(LIBDIR)` in your compiler flags, so that
        these install dirs are also searched for headers/libraries (e.g. simpulse)
      - You may need more -I and -L flags to find all necessary headers/libraries.
      - In particular, if you get the error message "Python.h not found", then you
        probably need something like -I/usr/include/python2.7.  You can get the header
	directory for your version of python with `distutils.sysconfig.get_python_inc()`
      - If you get the error message "numpy/arrayobject.h not found", then you probably 
        need something like -I/usr/lib64/python2.7/site-packages/numpy/core/include.
        You can get the header directory for your numpy installation with 
	`numpy.get_include()`.
      - If everything compiles but libraries are not being found at runtime, then you
        probably need to add `.` or LIBDIR to the appropriate environment variable
        ($LD_LIBRARY_PATH in Linux, or $DYLD_LIBRARY_PATH in osx)

    Feel free to email me if you have trouble!
