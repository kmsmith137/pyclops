// FIXME in hindsight I think it would be better to implement 
//
//     struct np_array<T, ...>
//
// where '...' stands for a list of modifiers, and then define
//
//     template<typename T>
//     using in_array<T> = np_array<const T>;
//
//     template<typename T> 
//     using io_carray<T,N> = np_array<T, arr_rw, arr_contig>
//
// etc.

#ifndef _PYCLOPS_ARRAY_CONVERTERS_HPP
#define _PYCLOPS_ARRAY_CONVERTERS_HPP

#include "core.hpp"
#include "py_array.hpp"
#include "converters.hpp"


namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// in_array, in_carray, in_narray, in_ncarray


template<typename T>
struct in_array : py_array
{
    const T *data;

    in_array(const py_array &arr, const char *where=nullptr);

    // Default flags, passed to PyArray_fromAny() for in_array.
    static constexpr int default_flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_ELEMENTSTRIDES | NPY_ARRAY_FORCECAST;
};


template<typename T>
struct in_carray : in_array<T>
{
    in_carray(const py_array &arr, const char *where=nullptr);
};


template<typename T, int N>
struct in_narray : in_array<T> 
{ 
    static_assert(N >= 0, "pyclops::in_narray is only defined for N >= 0");

    in_narray(const py_array &arr, const char *where=nullptr);
};


template<typename T, int N, int C=N>
struct in_ncarray : in_narray<T,N> 
{ 
    static_assert((C >= 0) && (C <= N), "pyclops::in_ncarray is only defined for 0 <= C <= N");

    in_ncarray(const py_array &arr, const char *where=nullptr);
};


// -------------------------------------------------------------------------------------------------
//
// io_array, io_carray, io_narray, io_ncarray


template<typename T>
struct io_array : py_array
{
    T *data;

    io_array(const py_array &arr, const char *where=nullptr);

    // Default flags, passed to PyArray_fromAny() for io_array.
    static constexpr int default_flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_ELEMENTSTRIDES | NPY_ARRAY_FORCECAST | NPY_ARRAY_WRITEABLE | NPY_ARRAY_UPDATEIFCOPY;
};


template<typename T>
struct io_carray : io_array<T>
{
    io_carray(const py_array &arr, const char *where=nullptr);
};


template<typename T, int N>
struct io_narray : io_array<T> 
{ 
    static_assert(N >= 0, "pyclops::io_narray is only defined for N >= 0");

    io_narray(const py_array &arr, const char *where=nullptr);
};


template<typename T, int N, int C=N>
struct io_ncarray : io_narray<T,N> 
{ 
    static_assert((C >= 0) && (C <= N), "pyclops::io_ncarray is only defined for 0 <= C <= N");

    io_ncarray(const py_array &arr, const char *where=nullptr);
};


// -------------------------------------------------------------------------------------------------
//
// Implementation follows.  Lots of things to improve here!
//
// FIXME for now we just call PyArray_FromAny() with NPY_ARRAY_FORCECAST.
// Some things to think about later:
//   - is this efficient, in the case where 'x' is already an array?
//   - should have boolean flags to control level of casting allowed.
//   - what happens e.g. if the array is 'int' and 'unsigned int' is requested?  does a superfluous copy get made?
//   - can this convert between floating-point types and integer types?
//
// FIXME in_narray<T,N> doesn't work for N=0!
//   - Because PyArray_FromAny() interprets min_nd = max_nd = 0 as "no constraint".
//   - Seems best to fix this in py_array::from_sequence()
//
// FIXME in_ncarray<T,N,C> implementation can call PyArray_FromAny() twice.
//   - Is this efficient?
//   - There's no scenario where we can double-copy, is there?


template<typename T> 
in_array<T>::in_array(const py_array &arr, const char *where) :
    py_array(arr),
    data(reinterpret_cast<const T *> (arr.data()))
{
    if (this->type() != npy_type<T>::id)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array dtype");
}


template<typename T>
in_carray<T>::in_carray(const py_array &arr, const char *where) :
    in_array<T>(arr)
{
    if ((this->flags() & NPY_ARRAY_C_CONTIGUOUS) != NPY_ARRAY_C_CONTIGUOUS)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": array was not contiguous as expected");
}


template<typename T, int N>
in_narray<T,N>::in_narray(const py_array &arr, const char *where) :
    in_array<T>(arr, where)
{
    if (this->ndim() != N)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array rank");
}


template<typename T, int N, int C>
in_ncarray<T,N,C>::in_ncarray(const py_array &arr, const char *where) :
    in_narray<T,N>(arr, where)
{
    if (this->ncontig() < C)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array ncontig");
}


template<typename T>
struct converter<in_array<T>>
{
    static in_array<T> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, in_array<T>::default_flags);
    }

    // No real reason to define a to-python converter, but why not?
    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T>
struct converter<in_carray<T>>
{
    static in_carray<T> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, in_array<T>::default_flags | NPY_ARRAY_C_CONTIGUOUS);
    }

    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T, int N>
struct converter<in_narray<T,N>>
{
    static in_narray<T,N> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, in_array<T>::default_flags, N, N);
    }

    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T, int N, int C>
struct converter<in_ncarray<T,N,C>>
{
    static in_ncarray<T,N,C> from_python(const py_object &x, const char *where=nullptr)
    {
	int flags = in_array<T>::default_flags;
	if (C >= N)
	    flags |= NPY_ARRAY_C_CONTIGUOUS;

	py_array ret = py_array::from_sequence(x, npy_type<T>::id, flags, N, N);
	
	if ((C >= N) || (ret.ncontig() >= C))
	    return ret;

	flags |= NPY_ARRAY_C_CONTIGUOUS;
	return py_array::from_sequence(ret, npy_type<T>::id, flags);
    }

    static py_object to_python(const in_array<T> &x) { return x; }
};


// -------------------------------------------------------------------------------------------------


template<typename T> 
io_array<T>::io_array(const py_array &arr, const char *where) :
    py_array(arr),
    data(reinterpret_cast<T *> (arr.data()))
{
    if (this->type() != npy_type<T>::id)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array dtype");
    if ((this->flags() & NPY_ARRAY_WRITEABLE) != NPY_ARRAY_WRITEABLE)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": io_array is not writeable");
}


template<typename T>
io_carray<T>::io_carray(const py_array &arr, const char *where) :
    io_array<T>(arr)
{
    if ((this->flags() & NPY_ARRAY_C_CONTIGUOUS) != NPY_ARRAY_C_CONTIGUOUS)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": array was not contiguous as expected");
}


template<typename T, int N>
io_narray<T,N>::io_narray(const py_array &arr, const char *where) :
    io_array<T>(arr, where)
{
    if (this->ndim() != N)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array rank");
}


template<typename T, int N, int C>
io_ncarray<T,N,C>::io_ncarray(const py_array &arr, const char *where) :
    io_narray<T,N>(arr, where)
{
    if (this->ncontig() < C)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": unexpected array ncontig");
}


template<typename T>
struct converter<io_array<T>>
{
    static io_array<T> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, io_array<T>::default_flags);
    }

    static py_object to_python(const io_array<T> &x) { return x; }
};


template<typename T>
struct converter<io_carray<T>>
{
    static io_carray<T> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, io_array<T>::default_flags | NPY_ARRAY_C_CONTIGUOUS);
    }

    static py_object to_python(const io_array<T> &x) { return x; }
};


template<typename T, int N>
struct converter<io_narray<T,N>>
{
    static io_narray<T,N> from_python(const py_object &x, const char *where=nullptr) 
    {
	return py_array::from_sequence(x, npy_type<T>::id, io_array<T>::default_flags, N, N);
    }

    static py_object to_python(const io_array<T> &x) { return x; }
};


template<typename T, int N, int C>
struct converter<io_ncarray<T,N,C>>
{
    static io_ncarray<T,N,C> from_python(const py_object &x, const char *where=nullptr)
    {
	int flags = io_array<T>::default_flags;
	if (C >= N)
	    flags |= NPY_ARRAY_C_CONTIGUOUS;

	py_array ret = py_array::from_sequence(x, npy_type<T>::id, flags, N, N);
	
	if ((C >= N) || (ret.ncontig() >= C))
	    return ret;

	flags |= NPY_ARRAY_C_CONTIGUOUS;
	return py_array::from_sequence(ret, npy_type<T>::id, flags);
    }

    static py_object to_python(const io_array<T> &x) { return x; }
};


}  // namespace pyclops

#endif  // _PYCLOPS_ARRAY_CONVERTERS_HPP
