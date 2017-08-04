#ifndef _PYCLOPS_WEAKREF_HPP
#define _PYCLOPS_WEAKREF_HPP

#include "core.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// py_weakref
// Reference: https://docs.python.org/2/c-api/weakref.html


struct py_weakref : public py_object {
    // Note: copy constructors and assigment operators do not create a new weak reference to their argument!
    // To create a new weak reference, use py_weakref::make() below.
    py_weakref(const py_object &x, const char *where=NULL);
    py_weakref(py_object &&x, const char *where=NULL);
    py_weakref &operator=(const py_object &x);
    py_weakref &operator=(py_object &&x);

    // Returns None if weak reference has expired.
    inline py_object dereference(const char *where=nullptr);

    // This constructor-like function returns a new weak reference.
    static inline py_weakref make(const py_object &x);

    inline void _check(const char *where=NULL);
    static void _throw(const char *where);   // non-inline, defined in exceptions.cpp
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


inline py_weakref::py_weakref(const py_object &x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_weakref::py_weakref(py_object &&x, const char *where) :
    py_object(x) 
{ 
    this->_check();
}

inline py_weakref &py_weakref::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_weakref &py_weakref::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}

inline void py_weakref::_check(const char *where)
{
    // FIXME: do I want PyWeakref_Check() or PyWeakref_CheckRef() here?
    if (!PyWeakref_Check(this->ptr))
	_throw(where);
}

inline py_object py_weakref::dereference(const char *where)
{
    PyObject *p = PyWeakref_GetObject(ptr);
    return py_object::borrowed_reference(p);
}

// Static constructor-like member function
inline py_weakref py_weakref::make(const py_object &x)
{
    // FIXME: do I want PyWeakref_NewRef() or PyWeakref_NewProxy() here?
    PyObject *p = PyWeakref_NewRef(x.ptr, NULL);
    return py_object::new_reference(p);
}


}  // namespace pyclops

#endif  // _PYCLOPS_WEAKREF_HPP
