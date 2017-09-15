#ifndef _PYCLOPS_FROM_PYTHON_HPP
#define _PYCLOPS_FROM_PYTHON_HPP

#include <vector>

#include "core.hpp"
#include "py_array.hpp"
#include "py_type.hpp"


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
//
// Note: the primary template converter<T> is declared in pyclops/core.hpp.


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

template<> struct converter<py_type> {
    static py_type from_python(const py_object &x, const char *where=nullptr) { return py_type(x,where); }
    static py_object to_python(const py_type &x) { return x; }
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


// By default the bool from-python converter is "strict", i.e. it expects either True or False.
// FIXME define "relaxed_bool", which evaluates its argument to True/False.
template<> struct converter<bool> {
    static bool from_python(const py_object &x, const char *where=nullptr)
    {
	if (x.ptr == Py_True) return true;
	if (x.ptr == Py_False) return false;
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": expected True or False");
    }

    static py_object to_python(bool x)
    {
	return x ? py_object::borrowed_reference(Py_True) : py_object::borrowed_reference(Py_False);
    }
};


}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
