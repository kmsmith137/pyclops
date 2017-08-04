#ifndef _PYCLOPS_TYPE_HPP
#define _PYCLOPS_TYPE_HPP

#include "core.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// py_type
//
// API reference: https://docs.python.org/2/c-api/type.html
// Note: PyTypeObject is defined in Python-2.x.x./Include/object.h


struct py_type : public py_object {
    py_type(const py_object &x, const char *where=NULL);
    py_type(py_object &&x, const char *where=NULL);
    py_type &operator=(const py_object &x);
    py_type &operator=(py_object &&x);

    inline PyTypeObject *tptr() const { return (PyTypeObject *) ptr; }
    inline ssize_t get_basicsize() const { return tptr()->tp_basicsize; }

    inline void _check(const char *where=NULL);
    static void _throw(const char *where);   // non-inline, defined in exceptions.cpp
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


inline py_type::py_type(const py_object &x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_type::py_type(py_object &&x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}

inline py_type &py_type::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_type &py_type::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}

inline void py_type::_check(const char *where)
{
    if (!PyType_Check(this->ptr))
	_throw(where);
}


}  // namespace pyclops

#endif  // _PYCLOPS_TYPE_HPP
