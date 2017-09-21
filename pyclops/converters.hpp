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
//   struct predicated_converter<T, typename std::enable_if<P>::type>
//
// where P is a boolean predicate (for example std::is_integral<T>::value).
// FIXME: is there a more intuitive way to define predicated_converter?
//
// Note: the 'converter' and 'predicated_converter' primary templates are defined in pyclops/core.hpp.


// -------------------------------------------------------------------------------------------------
//
// These type_traits can be used to test whether a converter exists.


template<typename T, typename = T>
struct converts_from_python : std::false_type { };

template<typename T>
struct converts_from_python<T, decltype(converter<T>::from_python(std::declval<py_object>()))> : std::true_type { };


template<typename T, typename = py_object>
struct converts_to_python : std::false_type { };

template<typename T>
struct converts_to_python<T, decltype(converter<T>::to_python(std::declval<const T>()))> : std::true_type { };


// -------------------------------------------------------------------------------------------------
//
// Trivial "converters" which operate on subclasses of py_object.
//
// It is assumed that each such subclass T has a special constructor of the form
//   T(const py_object &x, const char *where);


template<typename T>
struct predicated_converter<T, typename std::enable_if<std::is_base_of<py_object,T>::value>::type>
{
    static inline T from_python(const py_object &x, const char *where=nullptr) { return T(x,where); }
    static inline py_object to_python(const T &x) { return x; }
};

// py_object is a special case, since it doesn't define the "special" contructor described above.
template<> struct converter<py_object> {
    static py_object from_python(const py_object &x, const char *where=nullptr) { return x; }
    static py_object to_python(const py_object &x) { return x; }
};


// -------------------------------------------------------------------------------------------------
//
// Some fundamental types (integers, floating-point, strings, etc.)


// predicated_converter for integral types, except 'bool' which is a special case below.
// FIXME should also have special case for char (to convert from python length-1 string)?
// FIXME incorporate climits here.

template<typename T>
struct predicated_converter<T, typename std::enable_if<std::is_integral<T>::value>::type>
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


// bool converter.
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


// predicated_converter for floating-point types
template<typename T>
struct predicated_converter<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{
    static inline T from_python(const py_object &x, const char *where=nullptr)
    {
	double ret = PyFloat_AsDouble(x.ptr);
	if ((ret == -1.0) && PyErr_Occurred())
	    throw pyerr_occurred(where);
	return ret;
    }

    static inline py_object to_python(const T &x)
    {
	return py_object::new_reference(PyFloat_FromDouble(x));
    }
};


// string converter
// FIXME: write a converter so that functions with (const char *) args are wrappable.
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



}  // namespace pyclops

#endif  // _PYCLOPS_FROM_PYTHON_HPP
