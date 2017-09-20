#ifndef _PYCLOPS_LIST_HPP
#define _PYCLOPS_LIST_HPP

#include <complex>
#include "core.hpp"


namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// Reference: https://docs.python.org/2/c-api/list.html
struct py_list : public py_object {
    py_list();  // Default constructor makes empty list.
    py_list(const py_object &x, const char *loc=NULL);
    py_list(py_object &&x, const char *loc=NULL);
    py_list &operator=(const py_object &x);
    py_list &operator=(py_object &&x);

    inline ssize_t size() const { return PyList_Size(this->ptr); }
    inline py_object get_item(ssize_t i) const;
    inline void set_item(ssize_t i, const py_object &x);
    inline void append(const py_object &x);

    inline void _check(const char *loc=NULL);
    static void _throw(const char *loc);  // non-inline, defined in exceptions.cpp
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


inline py_list::py_list() :
    // Note: py_object constructor will throw exception if PyList_New() returns NULL.
    py_object(PyList_New(0), false)   // increment_refcount=false (i.e. new reference)
{ }

inline py_list::py_list(const py_object &x, const char *loc) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_list::py_list(py_object &&x, const char *loc) :
    py_object(x) 
{ 
    this->_check();
}

inline py_list &py_list::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_list &py_list::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}

inline void py_list::_check(const char *loc)
{
    if (!PyList_Check(this->ptr))
	_throw(loc);
}


inline py_object py_list::get_item(ssize_t i) const
{
    return py_object::borrowed_reference(PyList_GetItem(this->ptr, i));
}

inline void py_list::set_item(ssize_t i, const py_object &x)
{
    // FIXME I assume that PyList_SetItem() sets a python exception on failure, 
    // but the docs don't actually say this.
    int err = PyList_SetItem(this->ptr, i, x.ptr);
    if (err != 0)
	throw pyerr_occurred();
}

inline void py_list::append(const py_object &x)
{
    int err = PyList_Append(this->ptr, x.ptr);
    if (err != 0)
	throw pyerr_occurred();
}


}  // namespace pyclops

#endif
