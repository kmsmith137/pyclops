#ifndef _PYCLOPS_TUPLE_HPP
#define _PYCLOPS_TUPLE_HPP

#include "py_object.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// py_tuple
// Reference: https://docs.python.org/2/c-api/tuple.html


struct py_tuple : public py_object {
    py_tuple(const py_object &x, const char *where=NULL);
    py_tuple(py_object &&x, const char *where=NULL);
    py_tuple &operator=(const py_object &x);
    py_tuple &operator=(py_object &&x);

    ssize_t size() const { return PyTuple_Size(ptr); }

    inline py_object get_item(ssize_t pos) const;
    inline void set_item(ssize_t pos, const py_object &x);

    inline void _check(const char *where=NULL)
    {
	if (!PyTuple_Check(this->ptr))
	    _throw(where);
    }

    static void _throw(const char *where);
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


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


inline py_object py_tuple::get_item(ssize_t pos) const
{
    return py_object::borrowed_reference(PyTuple_GetItem(this->ptr, pos));
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


}  // namespace pyclops

#endif  // _PYCLOPS_TUPLE_HPP
