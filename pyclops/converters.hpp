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


// For each type T, we define:
//
//   struct converter<T> {
//      static inline T from_python(const py_object &x, const char *where=nullptr);
//      static inline py_object to_python(const T & x);
//   };
//
// It is also useful to define "predicated" converters which apply to all types T which
// satisfy a boolean condition.  For example, a converter which applies to all integral
// types.  This can be done with the following awkward and not-very-intuitive boilerplate:
//
//   template<typename T>
//   struct predicated_converter<T, typename std::enable_if<P,int>::type>
//
// where P is a boolean predicate (for example std::is_integral<T>::value).
// FIXME: is there a more intuitive way to define predicated_converter?
//
// Note: the 'converter' and 'predicated_converter' primary templates are defined in pyclops/core.hpp.


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


template<typename T>
struct predicated_converter<T, typename std::enable_if<std::is_integral<T>::value,int>::type>
{
    static inline T from_python(const py_object &x, const char *where=nullptr)
    {
	ssize_t n = PyInt_AsSsize_t(x.ptr);
	if ((n == -1) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return n;
    }

    static inline py_object to_python(const T &x) 
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
