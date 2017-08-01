#ifndef _PYCLOPS_DICT_HPP
#define _PYCLOPS_DICT_HPP

#include "py_object.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// py_dict
// Reference: https://docs.python.org/2/c-api/dict.html


struct py_dict : public py_object {
    py_dict();
    py_dict(const py_object &x, const char *where=NULL);
    py_dict(py_object &&x, const char *where=NULL);
    py_dict &operator=(const py_object &x);
    py_dict &operator=(py_object &&x);

    ssize_t size() const { return PyDict_Size(ptr); }

    inline void _check(const char *where=NULL)
    {
	if (!PyDict_Check(this->ptr))
	    _throw(where);
    }

    static void _throw(const char *where);
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


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


}  // namespace pyclops

#endif  // _PYCLOPS_DICT_HPP
