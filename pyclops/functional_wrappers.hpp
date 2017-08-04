// FIXME: the functional_wrappers could use improvement!  (especially the constructor wrapper)
//
// Question for the future: is there a way to template-match a lambda-expression?
// Currently we can apply wrappers to a std::function or a C-style function pointer,
// but not a lambda-expression.  (As a workaround, we wrap lambdas in std::function.)

#ifndef _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
#define _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP

#include "core.hpp"

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
    inline py_object _py_call(const F &f, PArgs... pargs)
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
    inline py_object _py_call(const F &f, PArgs... pargs)
    {
	return converter<R>::to_python(f(pargs...));
    }

    template<typename F, typename... PArgs>
    inline R _rcall(const F &f, PArgs... pargs)
    {
	return f(pargs...);
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
    inline py_object _py_call(const F &f, PArgs... pargs)
    {
	return tail._py_call(f, pargs..., head);
    }

    template<typename F, typename... PArgs>
    inline R _rcall(const F &f, PArgs... pargs)
    {
	return tail._rcall(f, pargs..., head);
    }
};


// -------------------------------------------------------------------------------------------------
//
// toy_wrap()


// Wrap std::function.
template<typename R, typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(std::function<R(Args...)> f)
{
    return [f](py_tuple args, py_dict kwds) -> py_object
	{
	    // FIXME: improve this error message
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("pyclops: wrong number of arguments to wrapped function");
	    _func_context<R,Args...> cargs(args);
	    return cargs._py_call(f);
	};
}


// Wrap C-style function pointer.
template<typename R, typename... Args>
inline std::function<py_object(py_tuple,py_dict)> toy_wrap(R (*f)(Args...))
{
    return toy_wrap(std::function<R(Args...)> (f));
}


// Helper class used when wrapping a member function (see next toy_wrap()).
template<typename R, class C, typename... Args>
struct _partial_bind {
    R (C::*f)(Args...);
    C *self;

    _partial_bind(R (C::*f_)(Args...), C *self_) : 
	f(f_), self(self_)
    { }
    
    inline R operator()(Args... args) const 
    { 
	return (self->*f)(args...); 
    }
};


// Wrap class member function.
template<typename R, class C, typename... Args>
inline std::function<py_object(C*, py_tuple, py_dict)> toy_wrap(R (C::*f)(Args...))
{
    return [f](C *self, py_tuple args, py_dict kwds) -> py_object
	{
	    // FIXME: improve this error message
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("pyclops: wrong number of arguments to wrapped method");
	    _func_context<R,Args...> cargs(args);
	    _partial_bind<R,C,Args...> fself(f, self);
	    return cargs._py_call(fself);
	};
}


// Wrap constructor
template<class C, typename... Args>
inline std::function<std::shared_ptr<C>(py_tuple,py_dict)> toy_wrap_constructor(std::function<std::shared_ptr<C>(Args...)> f)
{
    return [f](py_tuple args, py_dict kwds) -> std::shared_ptr<C>
	{
	    // FIXME: improve this error message
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("pyclops: wrong number of arguments to wrapped function");
	    _func_context<std::shared_ptr<C>,Args...> cargs(args);
	    return cargs._rcall(f);
	};
}


}  // namespace pyclops

#endif  // _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
