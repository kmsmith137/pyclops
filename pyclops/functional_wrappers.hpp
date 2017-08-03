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
// _func_context


template<typename R, typename... Args> struct _func_context;


// Void return type, without args
template<> struct _func_context<void>
{
    _func_context(const py_tuple &t, ssize_t pos=0) { }

    template<typename F, typename... PArgs>
    inline py_object _call(const F &f, PArgs... pargs)
    {
	f(pargs...);
	return py_object();  // Py_None
    }
};


// Non-void return type, without args
template<typename R> struct _func_context<R>
{
    _func_context(const py_tuple &t, ssize_t pos=0) { }

    template<typename F, typename... PArgs>
    inline py_object _call(const F &f, PArgs... pargs)
    {
	return converter<R>::to_python(f(pargs...));
    }
};    


// With arguments (return type may be void or non-void)
template<typename R, typename A, typename... Ap> 
struct _func_context<R,A,Ap...> 
{
    A head;
    _func_context<R,Ap...> tail;

    _func_context(const py_tuple &t, ssize_t pos=0) :
	head(converter<A>::from_python(t.get_item(pos))),
	tail(t, pos+1)
    { }

    template<typename F, typename... PArgs>
    inline py_object _call(const F &f, PArgs... pargs)
    {
	return tail._call(f, pargs..., head);
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
	    _func_context<R,Args...> cargs(args);
	    return cargs._call(f);
	};
}


template<typename R, typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(R (*f)(Args...))
{
    return toy_wrap(std::function<R(Args...)> (f));
}


}  // namespace pyclops

#endif  // _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
