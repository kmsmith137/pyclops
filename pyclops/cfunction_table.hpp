#ifndef _PYCLOPS_CFUNCTION_TABLE_HPP
#define _PYCLOPS_CFUNCTION_TABLE_HPP

#include <sstream>
#include <iostream>
#include <functional>

#include "core.hpp"

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
extern initproc make_kwargs_initproc(std::function<void(py_object, py_tuple, py_dict)> f);


// -------------------------------------------------------------------------------------------------
//
// Because property getters/setters have a "closure", we can simplify by using a single cfunction,
// rather than a cfunction_table.


struct property_closure {
    std::function<py_object(py_object)> f_get;
    std::function<void(py_object,py_object)> f_set;
};

extern PyObject *pyclops_getter(PyObject *self, void *closure);
extern int pyclops_setter(PyObject *self, PyObject *value, void *closure);


}  // namespace pyclops

#endif  // _PYCLOPS_CFUNCTION_TABLE_HPP
