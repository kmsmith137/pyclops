#ifndef _PYCLOPS_ARRAY_HPP
#define _PYCLOPS_ARRAY_HPP

#include <complex>
#include "core.hpp"


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
    inline npy_intp size() const      { return PyArray_SIZE(aptr()); }
    inline npy_intp itemsize() const  { return PyArray_ITEMSIZE(aptr()); }
    inline void *data() const         { return PyArray_DATA(aptr()); }
    inline int type() const           { return PyArray_TYPE(aptr()); }

    inline int ncontig() const;
    
    inline npy_intp shape(int i) const  { return shape()[i]; }
    inline npy_intp stride(int i) const { return strides()[i]; }

    // Note that the 'type' argument can be npy_type<T>::id.
    static inline py_array make(int ndim, const npy_intp *shape, int type);

    // The 'req' argument is a bitwise-or of required numpy flags (see below for list of all flags).
    // If 'min_ndim' and/or 'max_ndim' arguments are zero, they will be ignored.
    static inline py_array from_sequence(const py_object &seq, int type, int req, int min_ndim=0, int max_ndim=0);

    // Two versions of py_array::from_pointer(), with and without a base object.
    // Reminder: following numpy conventions, the strides should include a factor of 'itemsize'!

    static inline py_array from_pointer(int ndim, const npy_intp *shape, const npy_intp *strides,
					int itemsize, void *data, int npy_type, int flags);

    static inline py_array from_pointer(int ndim, const npy_intp *shape, const npy_intp *strides,
					int itemsize, void *data, int npy_type, int flags, const py_object &base);

    
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
    //
    //   This overloading of 'base' is likely to change in a future version of NumPy. (!!)
    //
    //   Once the base is set, it may not be changed to another value.

    py_object _base() const
    {
	PyObject *b = PyArray_BASE(aptr());
	if (b)
	    return py_object::borrowed_reference(b);
	return (*this);
    }

    inline void _check(const char *loc=NULL);
    static void _throw(const char *loc);  // non-inline, defined in exceptions.cpp
};


// Externally-visible functions defined in numpy_arrays.cpp
extern const char *npy_typestr(int npy_type);


// -------------------------------------------------------------------------------------------------
//
// npy_type<T>::id = numpy typenum corresponding to specified C++ type T (determined at compile time)


template<typename T, int dummy=0> struct npy_type;

// The "dummy" argument shouldn't be specified, and only exists so that a human-friendly error message may be prined.
template<typename T, int dummy>
struct npy_type {
    static_assert(dummy, "npy_type<T>: no numpy array type could be found for requested type T");
};

// Handle 'const T'.
template<typename T> 
struct npy_type<const T, 0> 
{ 
    static constexpr int id = npy_type<T>::id; 
};

// Specific types follow.
template<> struct npy_type<char,0> { static constexpr int id = NPY_BYTE; };
template<> struct npy_type<int,0> { static constexpr int id = NPY_INT; };
template<> struct npy_type<long,0> { static constexpr int id = NPY_LONG; };
template<> struct npy_type<long long,0> { static constexpr int id = NPY_LONGLONG; };
template<> struct npy_type<unsigned char,0> { static constexpr int id = NPY_UBYTE; };
template<> struct npy_type<unsigned int,0> { static constexpr int id = NPY_UINT; };
template<> struct npy_type<unsigned long,0> { static constexpr int id = NPY_ULONG; };
template<> struct npy_type<unsigned long long,0> { static constexpr int id = NPY_ULONGLONG; };
template<> struct npy_type<float,0> { static constexpr int id = NPY_FLOAT; };
template<> struct npy_type<double,0> { static constexpr int id = NPY_DOUBLE; };
template<> struct npy_type<long double,0> { static constexpr int id = NPY_LONGDOUBLE; };
template<> struct npy_type<std::complex<float>,0> { static constexpr int id = NPY_CFLOAT; };
template<> struct npy_type<std::complex<double>,0> { static constexpr int id = NPY_CDOUBLE; };
template<> struct npy_type<std::complex<long double>,0>  { static constexpr int id = NPY_CLONGDOUBLE; };


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

inline void py_array::_check(const char *loc)
{
    if (!PyArray_Check(this->ptr))
	_throw(loc);
}


// FIXME is there a function in the numpy C-API which computes this?
inline int py_array::ncontig() const
{
    int nd = this->ndim();

    if (nd == 0)
	return 0;

    npy_intp *shp = this->shape();
    npy_intp *str = this->strides();
    npy_intp expected_stride = this->itemsize();

    for (int i = nd-1; i >= 0; i--) {
	if (str[i] != expected_stride)
	    return nd-1-i;
	expected_stride *= shp[i];
    }

    return nd;
}


inline py_array py_array::make(int ndim, const npy_intp *shape, int type)
{
    PyObject *p = PyArray_SimpleNew(ndim, const_cast<npy_intp *> (shape), type);
    return py_array::new_reference(p);
}


inline py_array py_array::from_sequence(const py_object &seq, int type, int requirements, int min_ndim, int max_ndim)
{
    // Make sure the returned array is a base-class ndarray.
    requirements |= NPY_ARRAY_ENSUREARRAY;

    // FIXME figure out whether I need to decrement the refcount.
    PyArray_Descr *desc = PyArray_DescrFromType(type);
    if (!desc)
	throw pyerr_occurred("py_array::from_sequence()");

    PyObject *p = PyArray_FromAny(seq.ptr, desc, min_ndim, max_ndim, requirements, NULL);
    return py_array::new_reference(p);
}


// py_array::from_pointer(): static constructor-like member function
inline py_array py_array::from_pointer(int ndim, const npy_intp *shape, const npy_intp *strides, int itemsize, void *data, int npy_type, int flags)
{
    PyObject *p = PyArray_New(&PyArray_Type, ndim, const_cast<npy_intp *> (shape), npy_type,
			      const_cast<npy_intp *> (strides), data, itemsize, flags, NULL);

    return py_array::new_reference(p);
}


// This version of py_array::from_pointer() has a 'base' object.
inline py_array py_array::from_pointer(int ndim, const npy_intp *shape, const npy_intp *strides, int itemsize, void *data, int npy_type, int flags, const py_object &base)
{
    PyObject *p = PyArray_New(&PyArray_Type, ndim, const_cast<npy_intp *> (shape), npy_type,
			      const_cast<npy_intp *> (strides), data, itemsize, flags, NULL);

    py_array ret = py_array::new_reference(p);

    int err = PyArray_SetBaseObject(ret.aptr(), base.ptr);
    if (err < 0)
	throw pyerr_occurred("pyclops::py_array::from_pointer");

    // PyArray_SetBaseObject() steals the 'base' reference, so we need to balance the books.
    Py_INCREF(base.ptr);

    return ret;
}


}  // namespace pyclops

#endif
