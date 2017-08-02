#ifndef _PYCLOPS_FROM_PYTHON_HPP
#define _PYCLOPS_FROM_PYTHON_HPP

#include <mcpp_arrays.hpp>

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"
#include "py_array.hpp"


namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// from_python()


template<typename T> inline T from_python(const py_object &x, const char *where=nullptr);


// -------------------------------------------------------------------------------------------------
//
// Implementations


template<> inline py_object from_python(const py_object &x, const char *where) { return x; }
template<> inline py_tuple from_python(const py_object &x, const char *where) { return py_tuple(x,where); }
template<> inline py_dict from_python(const py_object &x, const char *where) { return py_dict(x,where); }
template<> inline py_array from_python(const py_object &x, const char *where) { return py_array(x,where); }


template<> inline ssize_t from_python(const py_object &x, const char *where)
{
    ssize_t n = PyInt_AsSsize_t(x.ptr);
    if ((n == -1) && PyErr_Occurred())
	throw pyerr_occurred(where);
    return n;
}

template<> inline double from_python(const py_object &x, const char *where)
{
    double ret = PyFloat_AsDouble(x.ptr);
    if ((ret == -1.0) && PyErr_Occurred())
	throw pyerr_occurred(where);
    return ret;
}

template<> inline std::string from_python(const py_object &x, const char *where)
{
    char *ret = PyString_AsString(x.ptr);
    if (!ret)
	throw pyerr_occurred(where);
    return ret;
}

template<typename T> 
inline mcpp_arrays::rs_array<T> from_python(const py_object &x, const char *where)
{
    py_array a(x);

    int ndim = a.ndim();
    npy_intp *shape = a.shape();
    npy_intp *strides = a.strides();
    npy_intp itemsize = a.itemsize();
    T *data = reinterpret_cast<T *> (a.data());

    // FIXME how to handle case where numpy array is read-only,
    // and T is a non-const type?

    auto dtype = mcpp_typeid_from_npy_type(a.npy_type(), where);
    auto reaper = make_mcpp_reaper_from_pybase(a.base());

    // Note: rs_array constructor will throw an exception if 'dtype' doesn't match T.
    // Note: this is an "incomplete" constructor, still need to set shape/strides and call _finalize_shape_and_strides().
    mcpp_arrays::rs_array<T> ret(ndim, data, dtype, dtype, reaper, where);

    // This check should never fail, but seemed like a good idea.
    if (ret.itemsize != itemsize)
	throw std::runtime_error("pyclops internal error: itemsize mismatch in rs_array from_python converter");

    for (int i = 0; i < ndim; i++) {
	// FIXME how to handle case where stride is not divisible by itemsize?
	if (strides[i] % itemsize != 0)
	    throw std::runtime_error("pyclops internal error: can't divide stride by itemsize");
	ret.shape[i] = shape[i];
	ret.strides[i] = strides[i] / itemsize;
    }

    // Finishes construction of rs_array, as noted above.
    ret._finalize_shape_and_strides(where);

    return ret;
}


}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
