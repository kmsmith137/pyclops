#ifndef _PYCLOPS_OBJECT_HPP
#define _PYCLOPS_OBJECT_HPP

#include <iostream>
#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// struct py_object


struct py_object {
    // Holds reference, can never be NULL.  (If there is an attempt to construct a py_object
    // with a null pointer, then the pyerr_occurred exception will be thrown, see below.)
    PyObject *ptr = nullptr;

    py_object();   // default constructor produces Py_None
    ~py_object();

    py_object(py_object &&x);
    py_object(const py_object &x);
    py_object &operator=(const py_object &x);
    py_object &operator=(py_object &&x);
    
    // Note: instead of using this constructor...
    py_object(PyObject *x, bool increment_refcount);
    
    // ...I prefer to use these constructor-like functions.
    static py_object borrowed_reference(PyObject *x) { return py_object(x, true); }  // increment refcount
    static py_object new_reference(PyObject *x) { return py_object(x, false); }      // don't increment refcount

    inline bool is_none() const { return ptr == Py_None; }
    inline bool is_tuple() const { return PyTuple_Check(ptr); }
    inline bool is_dict() const { return PyDict_Check(ptr); }
    inline bool is_array() const { return PyArray_Check(ptr); }
    inline bool is_callable() const { return PyCallable_Check(ptr); }
    inline ssize_t get_refcount() const { return Py_REFCNT(ptr); }

    // Note: to further convert to a C++ string, wrap return value in "from_python<std::string> ()".
    py_object str() const  { return py_object::new_reference(PyObject_Str(ptr)); }
    py_object repr() const { return py_object::new_reference(PyObject_Repr(ptr)); }
};


// See pyclops/converters.hpp.
template<typename T> struct converter;


inline std::ostream &operator<<(std::ostream &os, const py_object &x);


// This exception is thrown whenever we "discover" in C++ code that a python exception
// has occurred (either because PyErr_Occurred() returned true, or (PyObject *) is NULL).
struct pyerr_occurred : std::exception
{
    const char *where = nullptr;
    pyerr_occurred(const char *where=nullptr);
    virtual char const *what() const noexcept;
};


// -------------------------------------------------------------------------------------------------
//
// Implementation follows.


inline py_object::py_object() : 
    py_object(Py_None, true)
{ }

inline py_object::~py_object()
{
    Py_XDECREF(ptr);
    this->ptr = nullptr;
}

inline py_object::py_object(const py_object &x) : 
    py_object(x.ptr, true)
{ }

inline py_object::py_object(py_object &&x) :
    ptr(x.ptr)
{
    x.ptr = nullptr;
}

inline py_object::py_object(PyObject *x, bool increment_refcount) :
    ptr(x)
{
    if (!x)
	throw pyerr_occurred();
    if (increment_refcount)
	Py_INCREF(x);
}

inline py_object &py_object::operator=(const py_object &x)
{ 
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    return *this; 
}

inline py_object &py_object::operator=(py_object &&x)
{ 
    this->ptr = x.ptr;
    x.ptr = NULL;
    return *this; 
}


inline std::ostream &operator<<(std::ostream &os, const py_object &x)
{
    py_object s = x.str();
    char *p = PyString_AsString(s.ptr);
    if (!p)
	throw pyerr_occurred();
    os << p;
    return os;
}


}  // namespace pyclops

#endif  // _PYCLOPS_OBJECT_HPP
