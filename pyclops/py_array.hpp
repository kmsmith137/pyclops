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

    inline int ndim() const           { return PyArray_NDIM(aptr()); }
    inline npy_intp *shape() const    { return PyArray_SHAPE(aptr()); }
    inline npy_intp *strides() const  { return PyArray_STRIDES(aptr()); }
    inline npy_intp itemsize() const  { return PyArray_ITEMSIZE(aptr()); }
    inline void *data() const         { return PyArray_DATA(aptr()); }
    inline int npy_type() const       { return PyArray_TYPE(aptr()); }

    // Reminder: the numpy flags are as follows:
    //
    //   NPY_ARRAY_C_CONTIGUOUS
    //   NPY_ARRAY_F_CONTIGUOUS
    //   NPY_ARRAY_OWNDATA           free 'data' pointer when array object is deallocated
    //   NPY_ARRAY_ALIGNED           aligned in multiples of sizeof(T)
    //   NPY_ARRAY_NOTSWAPPED        data has native endianness
    //   NPY_ARRAY_UPDATEIFCOPY      write to base->data when array object is deallocated
    //   NPY_ARRAY_WRITEABLE
    //
    // The following flags aren't stored in the PyArray_FLAGS, but can be specified in
    // constructor functions such as PyArray_New().
    //
    //   NPY_ARRAY_FORCECAST         cause a cast to occur regardless of whether or not it is safe
    //   NPY_ARRAY_ENSURECOPY        always copy the array (returned array is contiguous and writeable)
    //   NPY_ARRAY_ENSUREARRAY       make sure the returned array is a base-class ndarray
    //   NPY_ARRAY_ELEMENTSTRIDES    make sure that the strides are in units of the element size

    inline int flags() const { return PyArray_FLAGS(aptr()); }

    // py_array::_base() wraps PyArray_BASE(), but this can return NULL if the array owns its
    // own memory.  In this case, we just return a reference to the array.
    // 
    // From the numpy documentation: 
    // 
    //   In most cases, 'base' is the object which owns the memory the array is pointing at.  
    //   If the NPY_ARRAY_UPDATEIFCOPY flag is set, it has a different  meaning, namely 'base'
    //   is the array into which the current array will be copied upon destruction. 
    //   This overloading of 'base' is likely to change in a future version of NumPy. (!!)

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
