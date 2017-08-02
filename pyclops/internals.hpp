#ifndef _PYCLOPS_INTERNALS_HPP
#define _PYCLOPS_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with -std=c++11"
#endif

#include <sstream>
#include <iostream>

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"

#include "converters.hpp"
#include "extension_module.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// This is called whenever we want to "swallow" a C++ exception, but propagate it into the python error indicator.
extern void set_python_error(const std::exception &e) noexcept;

// These functions convert a C++ std::function to a C-style function pointer.
// The return types are typedef's defined in Python.h.
//
// There is currently a hardcoded limit on the number of functions which can be converted,
// but this should be fixable with some hackery.  

extern PyCFunction make_kwargs_cfunction(std::function<py_object(py_tuple,py_dict)> f);
extern PyCFunction make_kwargs_cmethod(std::function<py_object(py_object,py_tuple,py_dict)> f);
extern newfunc make_kwargs_tp_new(std::function<PyObject* (PyTypeObject *, py_tuple, py_dict)> f);

// Used to manage addition of the 'pyclops.mcpp_pybase' class to the pyclops.so extension module.
extern void _add_reaper_type(PyObject *module);
extern bool _reaper_type_ready();


}  // namespace pyclops

#endif  // _PYCLOPS_INTERNALS_HPP
