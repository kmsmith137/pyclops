#ifndef _PYCLOPS_ARRAY_HPP
#define _PYCLOPS_ARRAY_HPP

#include "py_object.hpp"
#include <mcpp_arrays.hpp>


namespace pyclops {
#if 0
}  // emacs pacifier
#endif


struct py_array : public py_object {
    py_array(const py_object &x, const char *loc=NULL);
    py_array(py_object &&x, const char *loc=NULL);
    py_array &operator=(const py_object &x);
    py_array &operator=(py_object &&x);

    inline PyArrayObject *aptr() const { return reinterpret_cast<PyArrayObject *> (this->ptr); }

    int ndim() const           { return PyArray_NDIM(aptr()); }
    npy_intp *shape() const    { return PyArray_SHAPE(aptr()); }
    npy_intp *strides() const  { return PyArray_STRIDES(aptr()); }
    npy_intp itemsize() const  { return PyArray_ITEMSIZE(aptr()); }
    void *data() const         { return PyArray_DATA(aptr()); }
    int npy_type() const       { return PyArray_TYPE(aptr()); }

    py_object _base() const
    {
	PyObject *b = PyArray_BASE(aptr());
	if (b)
	    return py_object::borrowed_reference(b);
	return (*this);
    }

    inline void _check(const char *loc=NULL)
    {
	if (!PyArray_Check(this->ptr))
	    _throw(loc);
    }

    static void _throw(const char *loc);
};


extern const char *npy_typestr(int npy_type);

extern int npy_type_from_mccp_typeid(mcpp_arrays::mcpp_typeid mcpp_type, const char *where=nullptr);

extern mcpp_arrays::mcpp_typeid mcpp_typeid_from_npy_type(int npy_type, const char *where=nullptr);

extern std::shared_ptr<mcpp_arrays::mcpp_reaper> make_mcpp_reaper_from_pybase(const py_object &x);

extern py_object make_pybase_from_mcpp_reaper(const std::shared_ptr<mcpp_arrays::mcpp_reaper> &reaper);


// -------------------------------------------------------------------------------------------------
//
// Implementation.


inline py_array::py_array(const py_object &x, const char *loc) :
    py_object(x) 
{ 
    this->_check();
}
    
inline py_array::py_array(py_object &&x, const char *loc) :
    py_object(x) 
{ 
    this->_check();
}

inline py_array &py_array::operator=(const py_object &x)
{
    // this ordering handles the self-assignment case correctly
    Py_XINCREF(x.ptr);
    Py_XDECREF(this->ptr);
    this->ptr = x.ptr;
    this->_check();
    return *this;
}

inline py_array &py_array::operator=(py_object &&x)
{
    this->ptr = x.ptr;
    x.ptr = NULL;
    this->_check();
    return *this;
}


}  // namespace pyclops

#endif
