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


// For each type T, we define a static member function
//
// struct from_python<T> {
//    static T convert(const py_object &x, const char *where=nullptr);
// };
//
// FIXME: this API can probably be improved!


template<typename T> struct from_python;


// -------------------------------------------------------------------------------------------------
//
// Built-in py_object subclasses.
// FIXME any way to use template magic to improve the declarations below?
//
// E.g., if a class defines a constructor with args (const py_object &x, const char *where),
// can we arrange things so that it is automatically used as the converter?


template<> struct from_python<py_object> {
    static py_object convert(const py_object &x, const char *where=nullptr) { return x; }
};

template<> struct from_python<py_tuple> {
    static py_tuple convert(const py_object &x, const char *where=nullptr) { return py_tuple(x,where); }
};

template<> struct from_python<py_dict> {
    static py_dict convert(const py_object &x, const char *where=nullptr) { return py_dict(x,where); }
};

template<> struct from_python<py_array> {
    static py_array convert(const py_object &x, const char *where=nullptr) { return py_array(x,where); }
};


// -------------------------------------------------------------------------------------------------
//
// Some fundamental types (integers, floating-point, strings, etc.)


template<> struct from_python<ssize_t> {
    static ssize_t convert(const py_object &x, const char *where=nullptr)
    {
	ssize_t n = PyInt_AsSsize_t(x.ptr);
	if ((n == -1) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return n;
    }
};


template<> struct from_python<double> {
    static double convert(const py_object &x, const char *where=nullptr)
    {
	double ret = PyFloat_AsDouble(x.ptr);
	if ((ret == -1.0) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return ret;
    }
};


template<> struct from_python<std::string> {
    static std::string convert(const py_object &x, const char *where=nullptr)
    {
	char *ret = PyString_AsString(x.ptr);
	if (!ret)
	    throw pyerr_occurred(where);
	return ret;
    }
};


// -------------------------------------------------------------------------------------------------
//
// mcpp_arrays


template<typename T> 
struct from_python<mcpp_arrays::rs_array<T>> {
    static mcpp_arrays::rs_array<T> convert(const py_object &x, const char *where=nullptr)
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
	mcpp_arrays::rs_array<T> ret(ndim, data, dtype, reaper, where);

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
};


}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
