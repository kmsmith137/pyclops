#ifndef _PYCLOPS_FROM_PYTHON_HPP
#define _PYCLOPS_FROM_PYTHON_HPP

#include <mccp_arrays.hpp>

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
    
    // mcpp_typeid_from_npy_type<>() has the following semantics:
    //
    //  - If (T != void), then there is a unique npy_type which is expected.
    //    We check that the actual npy_type matches, and return the associated mcpp_typeid.
    //
    //  - If (T == void), then we return the mcpp_typeid associated to the npy_type,
    //    or throw an exception if no such mcpp_typeid exists (e.g. 
    //    and 
    mcpp_arrays::mcpp_typeid dtype = mcpp_typeid_from_npy_type<T> (a.npy_type(), where);

    rs_array<T> ret();
    
    ret._alloc_sbuf(ndim);

    for (int i = 0; i < ndim; i++) {
	ret.shape[i] = shape[i];
	ret.strides[i] = strides[i] / itemsize;
    }

    ret.itemsize = itemsize;
    ret.dtype = rstype_from_npytype(a.npy_type());
    ret.data = a.data();

    ret._set_ndim(ndim);
    
    ret._set_ncontig();
    ret._reaper = make_rs_array_reaper(a.base());
}


}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
