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
    inline R _pcall(const F &f, PArgs... pargs)
    {
	return f(pargs...);
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

    template<typename R, typename F, typename... PArgs>
    inline R _pcall(const F &f, PArgs... pargs)
    {
	return tail.template _pcall<R> (f, pargs..., head);
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
	    if ((args.size() != sizeof...(Args)) || (kwds.size() != 0))
		throw std::runtime_error("nein!");
	    _ntuple<Args...> cargs(args);
	    return converter<R>::to_python(cargs.template _pcall<R> (f));
	};
}


}  // namespace pyclops

#endif  // _PYCLOPS_FUNCTIONAL_WRAPPERS_HPP
