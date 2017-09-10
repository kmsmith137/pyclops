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


template<typename T>
struct in_array : py_array
{
    const T *data;

    in_array(const py_array &arr, const char *where=nullptr);
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
    static_assert(N >= 0, "pyclops::in_ncarray is only defined for N >= 0");
    static_assert(C >= 0, "pyclops::in_ncarray is only defined for C >= 0");
    static_assert(C <= N, "pyclops::in_ncarray is only defined for C <= N");

    in_ncarray(const py_array &arr, const char *where=nullptr);
};


// -------------------------------------------------------------------------------------------------


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
	// FIXME for now we just call PyArray_FromAny() with NPY_ARRAY_FORCECAST.
	// Some things to think about later:
	//   - is this efficient, in the case where 'x' is already an array?
	//   - boolean flags to control level of casting allowed.
	//   - what happens e.g. if the array is 'int' and 'unsigned int' is requested?
	//     does a copy get made?
	//   - what happens if the array is double and 'int' is requested?
	//     does the data get rounded?

	int flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY;
	return py_array::from_sequence(x, npy_type<T>::id, flags);
    }

    // No real reason to define a to-python converter, but why not?
    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T>
struct converter<in_carray<T>>
{
    static in_array<T> from_python(const py_object &x, const char *where=nullptr) 
    {
	int flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_C_CONTIGUOUS;
	return py_array::from_sequence(x, npy_type<T>::id, flags);
    }

    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T, int N>
struct converter<in_narray<T,N>>
{
    static in_narray<T,N> from_python(const py_object &x, const char *where=nullptr) 
    {
	// FIXME doesn't work for N=0!
	// (Because PyArray_FromAny() interprets min_nd = max_nd = 0 as "no constraint".)
	int flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY;
	return py_array::from_sequence(x, npy_type<T>::id, flags, N, N);
    }

    static py_object to_python(const in_array<T> &x) { return x; }
};


template<typename T, int N, int C>
struct converter<in_ncarray<T,N,C>>
{
    static in_ncarray<T,N,C> from_python(const py_object &x, const char *where=nullptr)
    {
	// FIXME is this an efficient implementation?
	// There's no scenario where we can double-copy, is there?

	int flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ENSUREARRAY;
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


}  // namespace pyclops

#endif  // _PYCLOPS_ARRAY_CONVERTERS_HPP
