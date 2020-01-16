// This is coming along!  Some things to improve next:
//
//  - Should pass along 'where' strings so that python caller can get more information if there
//    is an error (e.g. which argument didn't convert from python)
//
//  - If wrap_func() is called with a functional argument which doesn't match either the std::function or
//    function-pointer templates, then should print a verbose error message.
//
//  - Is there a way to template-match a lambda-expression?  I think it might be possible if
//    types for every argument are specified (via kwarg(), if a default_val is specified, or
//    maybe a new modifier pyarg())
//
//  - Scattered minor FIXME's in comments below.

#ifndef _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
#define _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP

#include "core.hpp"
#include "converters.hpp"
#include <unordered_map>
#include <functional>

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// kwarg() is used to specify an (argument_name, default_value) pair,
// when python-wrapping a function.

template<typename T> struct _kwarg;

template<typename T> 
inline _kwarg<T> kwarg(const std::string &arg_name, const T &default_val);


// These functional wrappers are the "bottom line" routines defined in this file.  
// The "args..." at the end are a list of argument specifiers, which can be either strings 
// (representing a named argument with no default value), or kwarg() to specify a default value.


// std::function -> python function
template<typename R, typename... Ts, typename... Us>
inline std::function<py_object(py_tuple,py_dict)> wrap_func(std::function<R(Ts...)> f, const Us & ... args);

// (C++ function pointer) -> python function
template<typename R, typename... Ts, typename... Us>
inline std::function<py_object(py_tuple,py_dict)> wrap_func(R (*f)(Ts...), const Us & ... args);

// (C++ member function pointer) -> python method
template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(R (C::*f)(Ts...), const Us & ... args);

// Need separate version of wrap_method() for const-qualified member functions.
template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(R (C::*f)(Ts...) const, const Us & ... args);

// std::function (with 'self' argument) -> python method 
template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(std::function<R(C*,Ts...)> f, const Us & ... args);

template<class C, typename... Ts, typename... Us>
inline std::function<C* (py_object,py_tuple,py_dict)> wrap_constructor(std::function<C* (Ts...)> f, const Us & ... args);


// -------------------------------------------------------------------------------------------------
//
// Everything after this point is "implementation".
//
// Like most code which uses C++ templates in a complicated way, the code in this source file is
// pretty cryptic, and we rely on extended comments to explain what's going on!
//
// First, external functions defined in functional_wrappers.cpp.


struct argname_hash {
    ssize_t curr_size = 0;
    bool duplicates_detected = false;
    std::unordered_map<std::string,ssize_t> hash;

    void add(const char *s);
    void check(const py_dict &kwds, ssize_t n_min);
};


extern std::runtime_error bad_arg_count(int n_given, int n_min, int n_max);
extern std::runtime_error missing_arg(const char *arg_name);


// -------------------------------------------------------------------------------------------------
//
// _kwarg<T>: two-element struct which holds an (arg_name, default_value) pair.
//
// kwarg(): syntactic-sugar inline function which constructs a _kwarg<T> without explicitly specifying T.


template<typename T>
struct _kwarg {
    const std::string arg_name;
    T default_val;

    _kwarg(const std::string &arg_name_, const T &default_val_) :
	arg_name(arg_name_), 
	default_val(default_val_)
    { }
};

template<typename T> 
inline _kwarg<T> kwarg(const std::string &arg_name, const T &default_val)
{
    return _kwarg<T> (arg_name, default_val);
}


// -------------------------------------------------------------------------------------------------
//
// _call_helper
//
// _call_helper<R>::call_func(f, args...) -> py_object
//     f(args...) -> R -> py_object  [to_python]
//
// 

// Primary template (used for R != void)
template<typename R> 
struct _call_helper
{
    template<typename F, typename... Ts>
    static inline py_object call_func(const F &f, const Ts & ... args)
    {
	return converter<R>::to_python(f(args...));
    }

    template<typename C, typename F, typename... Ts>
    static inline py_object call_method(C *c, const F &f, const Ts & ... args)
    {
	// Apparently this is the C++ syntax for calling a class member function through a function pointer.
	return converter<R>::to_python((c->*f)(args...));
    }
};


// Special case: R=void.
template<>
struct _call_helper<void>
{
    template<typename F, typename... Ts>
    static inline py_object call_func(const F &f, const Ts & ... args)
    {
	f(args...);
	return py_object(); // Py_None
    }

    template<typename C, typename F, typename... Ts>
    static inline py_object call_method(C *c, const F &f, const Ts & ... args)
    {
	(c->*f)(args...);
	return py_object(); // Py_None
    }
};


// -------------------------------------------------------------------------------------------------
//
// An "xarg" struct represents one named argument to a python-wrapped function.
// Three xarg struct types are defined:
//
//   _xarg_invalid
//   _xarg_untyped
//   _xarg_default<T>
//
// Each xarg struct defines the following member function.
//
//   template<typename T>
//   inline T _xarg<X>::from_python(const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i) const
//
// This converts the current argument from python to C++, using converter<T>::from_python().
// The arguments (n,i) are the length of 'args', and the current position in 'args' respectively.
// The "current argument" is either an element of 'args' or 'kwds', depending on whether (i < n) or (i >= n).
//
// FIXME: rename _pyarg?  (I forgot why I named this "xarg".)


struct _xarg_invalid {
    static constexpr bool valid = false;
    static constexpr bool has_default = false;
    static constexpr const char *arg_name = "<invalid>";

    template<typename T>
    _xarg_invalid(T &x) { }
    
    template<typename T>
    inline T from_python(const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i) const
    {
	throw std::runtime_error("should never be called");
    }
};


struct _xarg_untyped {
    static constexpr bool valid = true;
    static constexpr bool has_default = false;
    const char *arg_name = nullptr;

    _xarg_untyped(const std::string &n) : arg_name(strdup(n.c_str())) { }
    
    template<typename T>
    inline T from_python(const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i) const
    {
	PyObject *p;

	if (i < n)
	    p = args._get_item(i);
	else {
	    p = kwds._get_item(arg_name);
	    if (!p)
		throw missing_arg(arg_name);
	}

	return converter<T>::from_python(py_object::borrowed_reference(p));
    }
};


template<typename T>
struct _xarg_default {
    static constexpr bool valid = true;
    static constexpr bool has_default = true;
    const char *arg_name = nullptr;
    const T default_val;

    _xarg_default(const _kwarg<T> &x) : 
	arg_name(strdup(x.arg_name.c_str())),
	default_val(x.default_val)
    { }

    template<typename Tout> 
    inline Tout from_python(const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i) const
    {
	PyObject *p;

	if (i < n)
	    p = args._get_item(i);
	else {
	    p = kwds._get_item(arg_name);
	    if (!p)
		return default_val;
	}

	return converter<Tout>::from_python(py_object::borrowed_reference(p));
    }
};


template<typename T, bool C = std::is_convertible<T,std::string>::value>
struct _xarg_t {
    using type = _xarg_invalid;
};

template<typename T>
struct _xarg_t<T,true> {
    using type = _xarg_untyped;
};

template<typename T>
struct _xarg_t<_kwarg<T>,false> {
    using type = _xarg_default<T>;
};


// -------------------------------------------------------------------------------------------------
//
// An "xargs" struct is a compile-time list of xarg structs.
// Its purpose in life is to aggregate all argument names and default_vals into a single object.
// The following members are defined:
//
//   _xargs<...>::N                        total number of arguments
//   _xargs<...>::Nmin                     number of arguments without default_vals
//   _xargs<...>::add_to_argname_hash()    adds all argument names to specified argname_hash


struct _xargs_empty {
    static constexpr bool type_error = false;
    static constexpr bool ordering_error = false;
    static constexpr bool valid = true;
    static constexpr int Nmin = 0;
    static constexpr int N = 0;

    _xargs_empty() { }
    
    inline void add_to_argname_hash(argname_hash &h) const { }
};


template<typename X, typename Xt>
struct _xargs_composite {
    static constexpr bool type_error = !X::valid || Xt::type_error;
    static constexpr bool ordering_error = !type_error && (Xt::ordering_error || (X::has_default && (Xt::Nmin > 0)));
    static constexpr bool valid = !type_error && !ordering_error;
    static constexpr int Nmin = X::has_default ? 0 : (Xt::Nmin + 1);
    static constexpr int N = Xt::N + 1;
    
    X head;
    Xt tail;

    template<typename T, typename... Ts>
    _xargs_composite(const T &t, const Ts & ... ts) : head(t), tail(ts...) { }

    inline void add_to_argname_hash(argname_hash &h) const
    {
	h.add(head.arg_name);
	tail.add_to_argname_hash(h);
    }
};


template<typename... Ts> struct _xargs_t;

template<>
struct _xargs_t<> {
    using type = _xargs_empty;
};

template<typename T, typename... Ts>
struct _xargs_t<T,Ts...> {
    using X = typename _xarg_t<T>::type;
    using Xt = typename _xargs_t<Ts...>::type;
    using type = _xargs_composite<X, Xt>;
};


// -------------------------------------------------------------------------------------------------
//
// _carg<>, _carg_t<T>::type
//
// FIXME need to implement pointer types.
// FIXME need to think about rvalue references here.
//
// For extension_types, it makes more sense to define converter<T&> than converter<T>.
//
// FIXME now that I think about it, this should really be part of a general from_python<T>() framework,
// for use in e.g. py_tuple::make().


struct _carg_invalid
{
    static constexpr bool valid = false;
};

template<typename T>
struct _carg
{
    static constexpr bool valid = true;
    T arg;

    template<typename X>
    _carg(const X &xarg, const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i=0) :
	arg(xarg.template from_python<T> (args, kwds, n, i))
    { }
};


// Helper templates for _carg_t.
//  _carg_tt<T1,T2,...>: returns _carg<Ti>
//  _carg_tx<C,T1,T2,...>: 

template<typename... Ts> struct _carg_tt;
template<bool C, typename... Ts> struct _carg_tx;

template<> struct _carg_tt<>
{
    using type = _carg_invalid;
};

template<typename T, typename... Ts>
struct _carg_tt<T,Ts...> : _carg_tx<converts_from_python<T>::value,T,Ts...> { };

template<typename T, typename... Ts>
struct _carg_tx<true,T,Ts...>
{
    using type = _carg<T>;
};

template<typename T, typename... Ts>
struct _carg_tx<false,T,Ts...> : _carg_tt<Ts...> { };


// Definition of _carg_t starts here.
// Primary template (if T is not a pointer or reference type)
// In this case, we try converters in the order (const T&), (T&), T.

template<typename T>
struct _carg_t : _carg_tt<const T&, T&, T> { };

// If T is a const-reference type, then we use the same ordering: (const T&), (T&), T.

template<typename T>
struct _carg_t<const T&> : _carg_t<T> { };

// Finally, if T is a non-const reference type, then only converter<T&> is allowed.

template<typename T>
struct _carg_t<T&> : _carg_tt<T&> { };


// -------------------------------------------------------------------------------------------------
//
// _cargs<T>
// _cargs_t<T1,T2,...>


struct _cargs_empty {
    static constexpr bool converter_error = false;
    static constexpr int N = 0;

    _cargs_empty(const _xargs_empty &x, const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i=0)
    { }

    template<typename R, typename F, typename... Ts>
    inline py_object call_func(const F &f, const Ts & ... iargs)
    {
	return _call_helper<R>::call_func(f, iargs...);
    }
    
    template<typename R, typename C, typename F, typename... Ts>
    inline py_object call_method(C *c, const F &f, const Ts & ... iargs)
    {
	return _call_helper<R>::call_method(c, f, iargs...);
    }

    template<typename C, typename F, typename... Ts>
    inline C *call_constructor(const F &f, const Ts & ... iargs)
    {
	return f(iargs...);
    }
};


template<typename S, typename St>
struct _cargs_composite {
    static constexpr bool converter_error = !S::valid || St::converter_error;
    static constexpr int N = St::N + 1;

    S head;
    St tail;
    
    template<typename X, typename Xt>
    _cargs_composite(const _xargs_composite<X,Xt> &xargs, const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i=0) :
	head(xargs.head, args, kwds, n, i),
	tail(xargs.tail, args, kwds, n, i+1)
    { }
    
    template<typename R, typename F, typename... Ts>
    inline py_object call_func(const F &f, const Ts & ... iargs)
    {
	return tail.template call_func<R> (f, iargs..., head.arg);
    }

    template<typename R, typename C, typename F, typename... Ts>
    inline py_object call_method(C *c, const F &f, const Ts & ... iargs)
    {
	return tail.template call_method<R> (c, f, iargs..., head.arg);
    }

    template<typename C, typename F, typename... Ts>
    inline C *call_constructor(const F &f, const Ts & ... iargs)
    {
	return tail.template call_constructor<C> (f, iargs..., head.arg);
    }
};


struct _cargs_dummy {
    template<typename X>
    _cargs_dummy(const X &xargs, const py_tuple &args, const py_dict &kwds, ssize_t n, ssize_t i=0)
    {
	throw std::runtime_error("should never be called");
    }

    template<typename R, typename F>
    inline py_object call_func(const F &f)
    {
	throw std::runtime_error("should never be called");
    }

    template<typename R, typename C, typename F>
    inline py_object call_method(C *c, const F &f)
    {
	throw std::runtime_error("should never be called");
    }

    template<typename C, typename F>
    inline C *call_constructor(const F &f)
    {
	throw std::runtime_error("should never be called");
    }
};


template<typename... Ts> struct _cargs_t;

template<>
struct _cargs_t<> {
    using type = _cargs_empty;
};

template<typename T, typename... Ts>
struct _cargs_t<T,Ts...> {
    using S = typename _carg_t<T>::type;
    using St = typename _cargs_t<Ts...>::type;
    using type = _cargs_composite<S, St>;
};


// -------------------------------------------------------------------------------------------------
//
// _arg_checker
//
// Performs compile-time checks between cargs and xargs (mismatched number of arguments,
// or non-convertible arguments)


// Helper for _arg_checker
template<typename C, typename X>
struct _convert_checker { static constexpr bool error = false; };

template<typename S, typename St, typename X, typename Xt>
struct _convert_checker<_cargs_composite<S,St>, _xargs_composite<X,Xt>>
{
    static constexpr bool error = _convert_checker<St,Xt>::error;
};

template<typename T, typename St, typename U, typename Xt>
struct _convert_checker<_cargs_composite<_carg<T>,St>, _xargs_composite<_xarg_default<U>,Xt>>
{
    static constexpr bool C = std::is_convertible<U,T>::value;
    static constexpr bool E = _convert_checker<St,Xt>::error;
    static constexpr bool error = !C || E;
};


template<typename C, typename X>
struct _arg_checker
{
    static constexpr bool count_error = (C::N != X::N);
    static constexpr bool check_conversion = !C::converter_error && X::valid && !count_error;
    static constexpr bool convert_error = check_conversion && _convert_checker<C,X>::error;
    static constexpr bool valid = check_conversion && !convert_error;
};


// -------------------------------------------------------------------------------------------------
//
// "Bottom-line" functional wrappers.
//
// FIXME reduce cut-and-paste here.


template<typename R, typename... Ts, typename... Us>
inline std::function<py_object(py_tuple,py_dict)> wrap_func(std::function<R(Ts...)> f, const Us & ... args)
{
    using cargs_t = typename _cargs_t<Ts...>::type;

    static_assert(!cargs_t::converter_error, "missing from_python converter for at least one function argument");
	
    using xargs_t = typename _xargs_t<Us...>::type;
            
    static_assert(!xargs_t::type_error, "all python argument specifiers must be strings or kwarg(...)");
    static_assert(!xargs_t::ordering_error, "python arguments with default_vals must go at the end");

    using ac = _arg_checker<cargs_t, xargs_t>;
    
    constexpr bool to_python_error = !std::is_void<R>::value && !converts_to_python<R>::value;
    constexpr bool all_checks_passed = ac::valid && !to_python_error;

    static_assert(!ac::count_error || (xargs_t::N > 0), "python arguments must be specified (either strings or kwarg(...))");
    static_assert(!ac::count_error || (xargs_t::N == 0), "number of python argument specifiers doesn't match number of function arguments");
    static_assert(!ac::convert_error, "type error when converting specified default_val to argument type of C++ function");
    static_assert(!to_python_error, "missing to_python converter for return value from function");
    
    // FIXME memory leaks here (new())    
    xargs_t *x = new xargs_t(args...);
    argname_hash *a = new argname_hash;
    
    if (all_checks_passed)
	x->add_to_argname_hash(*a);

    auto ret = [f,a,x](py_tuple args, py_dict kwds) -> py_object
	{
	    constexpr int Nmin = xargs_t::Nmin;
	    constexpr int Nmax = xargs_t::N;
	    
	    ssize_t nargs = args.size();
	    ssize_t ntot = nargs + kwds.size();

	    // Quick sanity check on argument count.
	    if ((ntot < Nmin) || (ntot > Nmax))
		throw bad_arg_count(ntot, Nmin, Nmax);

	    // Additional checks: invalid keyword args.
	    a->check(kwds, nargs);

	    using cargs_t2 = typename std::conditional<all_checks_passed, cargs_t, _cargs_dummy>::type;
	    
	    // Convert all arguments from python.
	    cargs_t2 cargs(*x, args, kwds, nargs);

	    // Call function and to_python converter.
	    return cargs.template call_func<R> (f);
	};

    return ret;
}


// This version of wrap_func() wraps a function pointer, rather than a std::function.
// FIXME this is actually a little suboptimal, since it's best to have the static_asserts in the top-level wrapper.
template<typename R, typename... Ts, typename... Us>
inline std::function<py_object(py_tuple,py_dict)> wrap_func(R (*f)(Ts...), const Us & ... args)
{
    return wrap_func(std::function<R(Ts...)> (f), args...);
}


template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(R (C::*f)(Ts...), const Us & ... args)
{
    using cargs_t = typename _cargs_t<Ts...>::type;

    static_assert(!cargs_t::converter_error, "missing from_python converter for at least one method argument");
    
    using xargs_t = typename _xargs_t<Us...>::type;
            
    static_assert(!xargs_t::type_error, "all python argument specifiers must be strings or kwarg(...)");
    static_assert(!xargs_t::ordering_error, "python arguments with default_vals must go at the end");

    using ac = _arg_checker<cargs_t, xargs_t>;
    
    constexpr bool to_python_error = !std::is_void<R>::value && !converts_to_python<R>::value;
    constexpr bool all_checks_passed = ac::valid && !to_python_error;

    static_assert(!ac::count_error || (xargs_t::N > 0), "python arguments must be specified (either strings or kwarg(...))");
    static_assert(!ac::count_error || (xargs_t::N == 0), "number of python argument specifiers doesn't match number of function arguments");
    static_assert(!ac::convert_error, "type error when converting specified default_val to argument type of C++ function");
    static_assert(!to_python_error, "missing to_python converter for return value from function");
    
    // FIXME memory leaks here (new())    
    xargs_t *x = new xargs_t(args...);
    argname_hash *a = new argname_hash;
    
    if (all_checks_passed)
	x->add_to_argname_hash(*a);

    auto ret = [f,a,x](C *self, py_tuple args, py_dict kwds) -> py_object
	{
	    constexpr int Nmin = xargs_t::Nmin;
	    constexpr int Nmax = xargs_t::N;
	    
	    ssize_t nargs = args.size();
	    ssize_t ntot = nargs + kwds.size();

	    // Quick sanity check on argument count.
	    if ((ntot < Nmin) || (ntot > Nmax))
		throw bad_arg_count(ntot, Nmin, Nmax);

	    // Additional checks: invalid keyword args.
	    a->check(kwds, nargs);

	    using cargs_t2 = typename std::conditional<all_checks_passed, cargs_t, _cargs_dummy>::type;
	    
	    // Convert all arguments from python.
	    cargs_t2 cargs(*x, args, kwds, nargs);

	    // Call method and to_python converter.
	    return cargs.template call_method<R> (self, f);
	};

    return ret;
}


// Need separate version of wrap_method() for const-qualified member functions.
template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(R (C::*f)(Ts...) const, const Us & ... args)
{
    using nonconst_t = R (C::*)(Ts...);
    return wrap_method(nonconst_t(f), args...);
}


// This version of wrap_method() wraps a std::function, rather than a member function pointer.
// FIXME lots of cut-and-paste here!

template<class C, typename R, typename... Ts, typename... Us>
inline std::function<py_object(C*, py_tuple, py_dict)> wrap_method(std::function<R(C*,Ts...)> f, const Us & ... args)
{
    using cargs_t = typename _cargs_t<Ts...>::type;

    static_assert(!cargs_t::converter_error, "missing from_python converter for at least one method argument");
    
    using xargs_t = typename _xargs_t<Us...>::type;
            
    static_assert(!xargs_t::type_error, "all python argument specifiers must be strings or kwarg(...)");
    static_assert(!xargs_t::ordering_error, "python arguments with default_vals must go at the end");

    using ac = _arg_checker<cargs_t, xargs_t>;
    
    constexpr bool to_python_error = !std::is_void<R>::value && !converts_to_python<R>::value;
    constexpr bool all_checks_passed = ac::valid && !to_python_error;

    static_assert(!ac::count_error || (xargs_t::N > 0), "python arguments must be specified (either strings or kwarg(...))");
    static_assert(!ac::count_error || (xargs_t::N == 0), "number of python argument specifiers doesn't match number of function arguments");
    static_assert(!ac::convert_error, "type error when converting specified default_val to argument type of C++ function");
    static_assert(!to_python_error, "missing to_python converter for return value from function");
    
    // FIXME memory leaks here (new())    
    xargs_t *x = new xargs_t(args...);
    argname_hash *a = new argname_hash;
    
    if (all_checks_passed)
	x->add_to_argname_hash(*a);

    auto ret = [f,a,x](C *self, py_tuple args, py_dict kwds) -> py_object
	{
	    constexpr int Nmin = xargs_t::Nmin;
	    constexpr int Nmax = xargs_t::N;
	    
	    ssize_t nargs = args.size();
	    ssize_t ntot = nargs + kwds.size();

	    // Quick sanity check on argument count.
	    if ((ntot < Nmin) || (ntot > Nmax))
		throw bad_arg_count(ntot, Nmin, Nmax);

	    // Additional checks: invalid keyword args.
	    a->check(kwds, nargs);

	    using cargs_t2 = typename std::conditional<all_checks_passed, cargs_t, _cargs_dummy>::type;
	    
	    // Convert all arguments from python.
	    cargs_t2 cargs(*x, args, kwds, nargs);

	    // Call method and to_python converter.
	    return cargs.template call_func<R> (f, self);
	};

    return ret;
}


// Wrap constructor
// Note: previous version had a 'self' argument, but it doesn't seem to be needed, now that we have py_upcall().
template<class C, typename... Ts, typename... Us>
inline std::function<C* (py_object,py_tuple,py_dict)> wrap_constructor(std::function<C* (Ts...)> f, const Us & ... args)
{
    using cargs_t = typename _cargs_t<Ts...>::type;

    static_assert(!cargs_t::converter_error, "missing from_python converter for at least one constructor argument");
    
    using xargs_t = typename _xargs_t<Us...>::type;
            
    static_assert(!xargs_t::type_error, "all python argument specifiers must be strings or kwarg(...)");
    static_assert(!xargs_t::ordering_error, "python arguments with default_vals must go at the end");

    using ac = _arg_checker<cargs_t, xargs_t>;
    constexpr bool all_checks_passed = ac::valid;

    static_assert(!ac::count_error || (xargs_t::N > 0), "python arguments must be specified (either strings or kwarg(...))");
    static_assert(!ac::count_error || (xargs_t::N == 0), "number of python argument specifiers doesn't match number of function arguments");
    static_assert(!ac::convert_error, "type error when converting specified default_val to argument type of C++ function");
    
    // FIXME memory leaks here (new())    
    xargs_t *x = new xargs_t(args...);
    argname_hash *a = new argname_hash;
    
    if (all_checks_passed)
	x->add_to_argname_hash(*a);

    auto ret = [f,a,x](py_object, py_tuple args, py_dict kwds) -> C*
	{
	    constexpr int Nmin = xargs_t::Nmin;
	    constexpr int Nmax = xargs_t::N;
	    
	    ssize_t nargs = args.size();
	    ssize_t ntot = nargs + kwds.size();

	    // Quick sanity check on argument count.
	    if ((ntot < Nmin) || (ntot > Nmax))
		throw bad_arg_count(ntot, Nmin, Nmax);

	    // Additional checks: invalid keyword args.
	    a->check(kwds, nargs);

	    using cargs_t2 = typename std::conditional<all_checks_passed, cargs_t, _cargs_dummy>::type;
	    
	    // Convert all arguments from python.
	    cargs_t2 cargs(*x, args, kwds, nargs);

	    // Call constructor and return bare pointer.
	    return cargs.template call_constructor<C> (f);
	};

    return ret;
}


}  // namespace pyclops

#endif  // _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
