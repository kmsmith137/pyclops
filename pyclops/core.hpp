#ifndef _PYCLOPS_CORE_HPP
#define _PYCLOPS_CORE_HPP

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <memory>
#include <iostream>
#include <stdexcept>

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// See pyclops/converters.hpp.
template<typename T, typename=void> struct predicated_converter { };
template<typename T> struct converter : predicated_converter<T> { };

// Forward declarations needed below.
struct py_tuple;
struct py_dict;


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
    inline bool is_list() const { return PyList_Check(ptr); }
    inline bool is_dict() const { return PyDict_Check(ptr); }
    inline bool is_array() const { return PyArray_Check(ptr); }
    inline bool is_callable() const { return PyCallable_Check(ptr); }
    inline bool is_integer() const { return PyInt_Check(ptr); }
    inline bool is_floating_point() const { return PyFloat_Check(ptr); }
    inline bool is_string() const { return PyString_Check(ptr) || PyUnicode_Check(ptr); }
    inline ssize_t get_refcount() const { return ptr->ob_refcnt; }

    // These are safe to call without checking is_callable().
    py_object call(const py_tuple &args) const;
    py_object call(const py_tuple &args, const py_dict &kwds) const;

    // Note: to further convert to a C++ string, wrap return value in "from_python<std::string> ()".
    py_object str() const  { return py_object::new_reference(PyObject_Str(ptr)); }
    py_object repr() const { return py_object::new_reference(PyObject_Repr(ptr)); }

    const char *type_name() const { return this->ptr->ob_type->tp_name; }
};


// -------------------------------------------------------------------------------------------------
//
// struct py_tuple
//
// Reference: https://docs.python.org/2/c-api/tuple.html


struct py_tuple : public py_object {
    // Note: no default constructor.  To create a length-zero tuple, use make() or make_empty(0).
    py_tuple(const py_object &x, const char *where=NULL);
    py_tuple(py_object &&x, const char *where=NULL);
    py_tuple &operator=(const py_object &x);
    py_tuple &operator=(py_object &&x);

    ssize_t size() const { return PyTuple_Size(ptr); }

    inline py_object get_item(ssize_t pos) const;
    inline void set_item(ssize_t pos, const py_object &x);

    // make_empty(): constructor-like function which makes "empty" tuple containing None values.
    static inline py_tuple make_empty(ssize_t n);

    // make(): constructor-like function which makes python tuple from arbitrary C++ args.
    template<typename... Ts>
    static inline py_tuple make(const Ts & ... args);

    inline void _check(const char *where=NULL);
    static void _throw(const char *where);   // non-inline, defined in exceptions.cpp

    // For convenience in pyclops/functional_wrappers.hpp.
    // Returns borrowed reference.
    inline PyObject *_get_item(ssize_t pos) const;
};


// -------------------------------------------------------------------------------------------------
//
// struct py_dict
//
// Reference: https://docs.python.org/2/c-api/dict.html


struct py_dict : public py_object {
    py_dict();
    py_dict(const py_object &x, const char *where=NULL);
    py_dict(py_object &&x, const char *where=NULL);
    py_dict &operator=(const py_object &x);
    py_dict &operator=(py_object &&x);

    ssize_t size() const { return PyDict_Size(ptr); }

    inline py_object get_item(const char *key) const;
    inline py_object get_item(const std::string &key) const;
    inline py_object get_item(const py_object &key) const;
    
    inline void set_item(const char *key, const py_object &val);
    inline void set_item(const std::string &key, const py_object &val);
    inline void set_item(const py_object &key, const py_object &val);

    inline void _check(const char *where=NULL);
    static void _throw(const char *where);   // non-inline, defined in exceptions.cpp

    // For convenience in pyclops/functional_wrappers.hpp.
    // Returns borrowed reference.  If key is not found, returns NULL without setting an exception.
    inline PyObject *_get_item(const char *key) const;

    // The boilerplate below defines a range-based for-loop for looping over (key,value) pairs.
    // Usage:
    //
    //   for (const auto &p: dict) {
    //        const py_object &key = p.first;
    //        const py_object &val = p.second;
    //        ...
    //   }

    struct iterator {
	// Note: all (PyObject *) pointers are borrowed references.
	PyObject *dict = nullptr;
	PyObject *key = nullptr;
	PyObject *val = nullptr;
	Py_ssize_t pos = 0;

	inline iterator(PyObject *dict);
	inline bool operator!=(const iterator &it) const;
	inline iterator& operator++();
	inline std::pair<py_object,py_object> operator* () const;
    };

    inline iterator begin() const;
    inline iterator end() const;
};


// -------------------------------------------------------------------------------------------------
//
// Exceptions.


// This exception is thrown whenever we "discover" in C++ code that a python exception
// has occurred (either because PyErr_Occurred() returned true, or (PyObject *) is NULL).
struct pyerr_occurred : std::exception
{
    std::shared_ptr<const char> msg;

    pyerr_occurred(const char *where=nullptr);

    virtual char const *what() const noexcept;
};


// -------------------------------------------------------------------------------------------------
//
// Master hash table.


extern void master_hash_table_add(const void *cptr, PyObject *pptr);
extern void master_hash_table_remove(const void *cptr, PyObject *pptr);

// Returns borrowed reference.
extern PyObject *master_hash_table_query(const void *cptr);

// Custom shared_ptr<> deleter.
extern void master_hash_table_deleter(const void *p);

// Intended for debugging.
extern void master_hash_table_print();


// -------------------------------------------------------------------------------------------------
//
// py_object implementation.


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
    if (x->ob_refcnt <= 0)
	throw std::runtime_error("pyclops internal error: py_object constructor invoked on object with refcount <= 0");
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

// Note: PyObject_Call() is always fastest, if args/kwds are known to be a tuple/dict.
inline py_object py_object::call(const py_tuple &args) const
{
    PyObject *p = PyObject_Call(ptr, args.ptr, NULL);
    return new_reference(p);
}

inline py_object py_object::call(const py_tuple &args, const py_dict &kwds) const
{
    PyObject *p = PyObject_Call(ptr, args.ptr, kwds.ptr);
    return new_reference(p);
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


// -------------------------------------------------------------------------------------------------
//
// py_tuple implementation



inline py_tuple::py_tuple(const py_object &x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_tuple::py_tuple(py_object &&x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}

inline py_tuple &py_tuple::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_tuple &py_tuple::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}

inline void py_tuple::_check(const char *where)
{
    if (!PyTuple_Check(this->ptr))
	_throw(where);
}

inline py_object py_tuple::get_item(ssize_t pos) const
{
    return py_object::borrowed_reference(PyTuple_GetItem(this->ptr, pos));
}

inline PyObject *py_tuple::_get_item(ssize_t pos) const
{
    return PyTuple_GetItem(this->ptr, pos);
}

inline void py_tuple::set_item(ssize_t pos, const py_object &x)
{
    int err = PyTuple_SetItem(this->ptr, pos, x.ptr);

    // TODO: check that my understanding of refcounting and error indicator is correct here

    if (!err)
	Py_INCREF(x.ptr);   // success
    else
	throw pyerr_occurred();  // failure
}


// Static constructor-like member function 
inline py_tuple py_tuple::make_empty(ssize_t n)
{
    // Note: if n < 0, then PyTuple_New() sets the python global error appropriately.
    return py_tuple::new_reference(PyTuple_New(n));
}


// _set_tuple(): helper for py_tuple::make() below.
template<typename... Ts>
inline void _set_tuple(py_tuple &t, int pos, const Ts & ... args);

template<> inline void _set_tuple(py_tuple &t, int pos) { }

template<typename T, typename... Ts>
inline void _set_tuple(py_tuple &t, int pos, const T &a, const Ts & ... ap)
{
    t.set_item(pos, converter<T>::to_python(a));
    _set_tuple(t, pos+1, ap...);
}

// Static constructor-like member function.
// FIXME: improve using the new fancy templates in functional_wrappers.hpp (maybe
// py_tuple::make() should move to this source file?)
template<typename... Ts>
inline py_tuple py_tuple::make(const Ts & ... args)
{
    py_tuple ret = make_empty(sizeof...(Ts));
    _set_tuple(ret, 0, args...);
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// py_dict implementation.


inline py_dict::py_dict() :
    py_object(PyDict_New(), false)
{ }

inline py_dict::py_dict(const py_object &x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_dict::py_dict(py_object &&x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}

inline py_dict &py_dict::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_dict &py_dict::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}

inline void py_dict::_check(const char *where)
{
    if (!PyDict_Check(this->ptr))
	_throw(where);
}

inline PyObject *py_dict::_get_item(const char *key) const
{
    return PyDict_GetItemString(this->ptr, key);
}

inline py_object py_dict::get_item(const char *key) const
{
    return py_object::borrowed_reference(PyDict_GetItemString(this->ptr, key));
}

inline py_object py_dict::get_item(const std::string &key) const
{
    return get_item(key.c_str()); 
}

inline py_object py_dict::get_item(const py_object &key) const
{
    return py_object::borrowed_reference(PyDict_GetItem(this->ptr, key.ptr));
}

inline void py_dict::set_item(const char *key, const py_object &val)
{
    // FIXME I assume that if PyDict_SetItemString() fails, it sets a python exception,
    // but the docs don't actually say this.
    int err = PyDict_SetItemString(this->ptr, key, val.ptr);
    if (err != 0)
	throw pyerr_occurred();
}

inline void py_dict::set_item(const std::string &key, const py_object &val)
{
    set_item(key.c_str(), val);
}

inline void py_dict::set_item(const py_object &key, const py_object &val)
{
    int err = PyDict_SetItem(this->ptr, key.ptr, val.ptr);
    if (err != 0)
	throw pyerr_occurred();
}


inline py_dict::iterator::iterator(PyObject *dict_) :
    dict(dict_)
{
    if (dict == NULL)
	return;
    if (!PyDict_Next(dict, &pos, &key, &val))
	dict = NULL;
}

inline bool py_dict::iterator::operator!=(const py_dict::iterator &it) const
{
    bool null1 = (this->dict == NULL);
    bool null2 = (it.dict == NULL);

    if (null1 && null2)
	return false;  // equal
    if (null1 || null2)
	return true;   // unequal
    return (this->pos != it.pos);
}

inline py_dict::iterator &py_dict::iterator::operator++()
{
    if (dict && !PyDict_Next(dict, &pos, &key, &val))
	dict = NULL;
    return *this;
}

inline std::pair<py_object,py_object> py_dict::iterator::operator*() const
{
    return std::make_pair(py_object::borrowed_reference(this->key), py_object::borrowed_reference(this->val));
}

inline py_dict::iterator py_dict::begin() const
{
    return py_dict::iterator(this->ptr);
}

inline py_dict::iterator py_dict::end() const
{
    return py_dict::iterator(NULL);
}


}  // namespace pyclops

#endif  // _PYCLOPS_OBJECT_HPP
