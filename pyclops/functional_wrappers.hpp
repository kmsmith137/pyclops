// FIXME: the functional_wrappers could use a lot of improvement!

#ifndef _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
#define _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"
#include "converters.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// _ntuple


template<typename... Args> struct _ntuple;

template<> struct _ntuple<> 
{
    _ntuple(const py_tuple &t, ssize_t pos=0) { }
    
    template<typename R, typename F, typename... PArgs>
    inline R _rcall(const F &f, PArgs... pargs)
    {
	return f(pargs...);
    }

    template<typename F, typename... PArgs>
    inline void _vcall(const F &f, PArgs... pargs)
    {
	f(pargs...);
    }
};

template<typename A, typename... Ap>
struct _ntuple<A,Ap...> {
    A head;
    _ntuple<Ap...> tail;

    _ntuple(const py_tuple &t, ssize_t pos=0) :
	head(converter<A>::from_python(t.get_item(pos))),
	tail(t, pos+1)
    { }
    
    // "Returning" call with return type R.
    template<typename R, typename F, typename... PArgs>
    inline R _rcall(const F &f, PArgs... pargs)
    {
	return tail.template _rcall<R> (f, pargs..., head);
    }

    // "Void" call returning void.
    template<typename F, typename... PArgs>
    inline void _vcall(const F &f, PArgs... pargs)
    {
	tail._vcall(f, pargs..., head);
    }    
};


// -------------------------------------------------------------------------------------------------
//
// toy_wrap()


template<typename R, typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(std::function<R(Args...)> f)
{
    return [f](py_tuple args, py_dict kwds) -> py_object
	{
	    // FIXME: improve this error message, and others in this source file!
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("pyclops: wrong number of arguments to wrapped function");
	    _ntuple<Args...> cargs(args);
	    return converter<R>::to_python(cargs.template _rcall<R> (f));
	};
}


template<typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(std::function<void(Args...)> f)
{
    return [f](py_tuple args, py_dict kwds) -> py_object
	{
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("pyclops: wrong number of arguments to wrapped function");
	    _ntuple<Args...> cargs(args);
	    cargs._vcall(f);
	    return py_object();  // returns None
	};
}


template<typename R, typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(R (*f)(Args...))
{
    return toy_wrap(std::function<R(Args...)> (f));
}


}  // namespace pyclops

#endif  // _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
