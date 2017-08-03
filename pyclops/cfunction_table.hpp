#ifndef _PYCLOPS_CFUNCTION_TABLE_HPP
#define _PYCLOPS_CFUNCTION_TABLE_HPP

#include <sstream>
#include <iostream>

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// These functions convert a C++ std::function to a C-style function pointer.
// The return types are typedef's defined in Python.h.
//
// There is currently a hardcoded limit on the number of functions which can be converted,
// but this should be fixable with some hackery.  

extern PyCFunction make_kwargs_cfunction(std::function<py_object(py_tuple,py_dict)> f);
extern PyCFunction make_kwargs_cmethod(std::function<py_object(py_object,py_tuple,py_dict)> f);
extern newfunc make_kwargs_newfunc(std::function<PyObject* (PyTypeObject *, py_tuple, py_dict)> f);


}  // namespace pyclops

#endif  // _PYCLOPS_CFUNCTION_TABLE_HPP