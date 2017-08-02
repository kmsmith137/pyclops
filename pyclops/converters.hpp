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


// For each type T, we define static member functions
//
// struct converter<T> {
//    static T from_python(const py_object &x, const char *where=nullptr);
//    static py_object to_python(const T & x);
// };
//
// FIXME: this API can probably be improved!


template<typename T> struct converter;


// -------------------------------------------------------------------------------------------------
//
// Built-in py_object subclasses.
// FIXME any way to use template magic to improve the declarations below?
//
// E.g., if a class defines a constructor with args (const py_object &x, const char *where),
// can we arrange things so that it is automatically used as the converter?


template<> struct converter<py_object> {
    static py_object from_python(const py_object &x, const char *where=nullptr) { return x; }
    static py_object to_python(const py_object &x) { return x; }
};

template<> struct converter<py_tuple> {
    static py_tuple from_python(const py_object &x, const char *where=nullptr) { return py_tuple(x,where); }
    static py_object to_python(const py_tuple &x) { return x; }
};

template<> struct converter<py_dict> {
    static py_dict from_python(const py_object &x, const char *where=nullptr) { return py_dict(x,where); }
    static py_object to_python(const py_dict &x) { return x; }
};

template<> struct converter<py_array> {
    static py_array from_python(const py_object &x, const char *where=nullptr) { return py_array(x,where); }
    static py_object to_python(const py_array &x) { return x; }
};


// -------------------------------------------------------------------------------------------------
//
// Some fundamental types (integers, floating-point, strings, etc.)


template<> struct converter<ssize_t> {
    static ssize_t from_python(const py_object &x, const char *where=nullptr)
    {
	ssize_t n = PyInt_AsSsize_t(x.ptr);
	if ((n == -1) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return n;
    }

    static py_object to_python(const ssize_t &x) 
    {    
	return py_object::new_reference(PyInt_FromSsize_t(x));
    }
};


template<> struct converter<double> {
    static double from_python(const py_object &x, const char *where=nullptr)
    {
	double ret = PyFloat_AsDouble(x.ptr);
	if ((ret == -1.0) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return ret;
    }

    static py_object to_python(const double &x)
    {
	return py_object::new_reference(PyFloat_FromDouble(x));
    }
};


template<> struct converter<std::string> {
    static std::string from_python(const py_object &x, const char *where=nullptr)
    {
	char *ret = PyString_AsString(x.ptr);
	if (!ret)
	    throw pyerr_occurred(where);
	return ret;
    }

    static py_object to_python(const std::string &x)
    {
	return py_object::new_reference(PyString_FromString(x.c_str()));
    }
};


// -------------------------------------------------------------------------------------------------
//
// mcpp_arrays


template<typename T> 
struct converter<mcpp_arrays::rs_array<T>> {
    static mcpp_arrays::rs_array<T> from_python(const py_object &x, const char *where=nullptr)
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
	auto reaper = make_mcpp_reaper_from_pybase(a._base());
	
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

    // FIXME: here is one case that could be handled better by our rs_array converters.
    // Suppose we python-wrap a function which accepts an rs_array, and returns it:
    //
    //   rs_array<T> f(rs_array<T> x) { return x; }
    //
    // The from_python converter will be called, followed by the to_python converter.
    // This has the effect of returning a new numpy array which points to the same data
    // as the input array.  It would be better to return a reference to the input array!

    static py_object to_python(const mcpp_arrays::rs_array<T> &x)
    {
	py_object pybase = make_pybase_from_mcpp_reaper(x.reaper);
	int npy_type = npy_type_from_mcpp_typeid(x.dtype, "rs_array to-python converter");

	std::vector<npy_intp> v(2 * x.ndim);
	npy_intp *shape = &v[0];
	npy_intp *strides = &v[x.ndim];

	for (int i = 0; i < x.ndim; i++) {
	    shape[i] = x.shape[i];
	    strides[i] = x.strides[i] * x.itemsize;    // Note factor of 'itemsize' here
	}

	// FIXME: figure out with 100% certainty which flags we should set here.
	int flags = NPY_ARRAY_NOTSWAPPED;

	return py_array::from_pointer(x.ndim, shape, strides, x.itemsize, x.data, npy_type, flags, pybase);
    }
};


}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
