#ifndef _PYCLOPS_VIRTUAL_FUNCTION_HPP
#define _PYCLOPS_VIRTUAL_FUNCTION_HPP

#include "core.hpp"
#include "extension_type.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// The virtual_function and pure_virtual_function classes are used to define "python-upcalling"
// overrides of C++ virtual functions.  Usage is something like this:
//
//   class X {
//       virtual int f_v(int i) { ... }
//       virtual int f_pv(int i) = 0;
//   };       
//
//   extension_type<X> X_type;   // PyTypeObject
//
//   class py_X {
//       virtual int f_v(int i) override {
//           virtual_function<X,int> v(X_type, this, "f_v");
//           return v.exists ? v.upcall(i) : X::f_v(i);
//       }
//
//       virtual int f_pv(int i) override {
//           pure_virtual_function<X,int> v(X_type, this, "f_pv");
//           return v.upcall(i);
//       }
//   };
//
// Note that converters are applied in virtual_function<X>::upcall().
//
// FIXME too much boilerplate needed to wrap a virtual function!  Needs a second iteration to streamline...
// FIXME explicit specification of template arguments <T,R> is ugly, any way to improve?
// One idea: instead of template arguments <T,R>, let function pointer to class member function be the template argument.


template<typename T, typename R>
struct virtual_function {
    py_object self;
    py_object f_base;
    py_object f_self;
    bool exists;

    virtual_function(const extension_type<T> &type, const T *self, const char *method_name);

    template<typename... Ts>
    inline R upcall(const Ts & ... args);
};


template<typename T, typename R>
struct pure_virtual_function : virtual_function<T,R> {
    // Inherits upcall(), exists, self, func from virtual_function<T,R>.
    pure_virtual_function(const extension_type<T> &type, const T *self, const char *method_name);
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.
//
// FIXME this implementation of virtual_function<> does a double call to PyObject_GetAttrString(),
// which is expensive!  What's the best way of doing it?


// Helper for virtual_function constructor: converts C++ 'self' to python 'self'.
template<typename T>
inline py_object _vf_self(const extension_type<T> &type, const T *self)
{
    PyObject *sp = master_hash_table_query(self);
    
    // FIXME: this error message reminded me that we should have a mechanism for constructing
    // python objects directly from C++ (going through tp_new, etc.)
    
    if (!sp) {
	throw std::runtime_error("pyclops::virtual_function: couldn't find object in master_hash_table"
				 " (either a pyclops bug, or upcalling C++ object was constructed without tp_new()");
    }

    // FIXME not sure if this call to PyObject_IsInstance() is necessary.
    int is_inst = PyObject_IsInstance(sp, (PyObject *)type.tobj);

    if (is_inst < 0)
	throw pyerr_occurred();  // PyObject_IsInstance() failed, and has set a python exception.
    if (is_inst == 0)
	throw std::runtime_error("pyclops::virtual function: 'self' does not have expected type");

    return py_object::borrowed_reference(sp);
}


// Helper for virtual_function constructor: gets method from base class.
template<typename T>
inline py_object _vf_fbase(const extension_type<T> &type, const char *method_name)
{
    PyObject *bp = PyObject_GetAttrString((PyObject *)type.tobj, method_name);
    if (!bp)
	throw std::runtime_error("pyclops::virtual_function: couldn't find '" + std::string(method_name) + "' in base class (probably a bug in the C++ wrapper code)");

    return py_object::new_reference(bp);
}


inline py_object _vf_fself(const py_object &self, const char *method_name)
{
    PyObject *dp = PyObject_GetAttrString((PyObject *) self.ptr->ob_type, method_name);
    if (!dp)
	throw std::runtime_error("pyclops::virtual_function: '" + std::string(method_name) + "' in defined in base class, but not in instance (not sure what's going on)");

    return py_object::new_reference(dp);
}


template<typename T, typename R>
virtual_function<T,R>::virtual_function(const extension_type<T> &type, const T *self_, const char *method_name) :
    self(_vf_self(type, self_)),
    f_base(_vf_fbase(type, method_name)),
    f_self(_vf_fself(self, method_name)),
    exists(f_base.ptr != f_self.ptr)
{ }


template<typename T, typename R>
template<typename... Ts>
inline R virtual_function<T,R>::upcall(const Ts & ... args)
{
    if (!exists)
	throw std::runtime_error("pure_virtual_function::upcall() applied when exists=false (this is a bug in the C++ wrapper code)");

    // Since 'f_self' was obtained from the PyTypeObject (not the PyObject), 
    // it is "unbound" and we need to prepend 'self' to the argument list.

    py_tuple t = py_tuple::make(self, args...);
    py_object ret = f_self.call(t);
    return converter<R>::from_python(ret);
}


// virtual_function::upcall() needs specialization for R=void.
// Currently implemented by specializing the entire virtual_function class to R=void via cut-and-paste.
// FIXME there must be a way of doing this with less cut-and-paste, but I am being lazy!

template<typename T>
struct virtual_function<T,void> {
    py_object self;
    py_object f_base;
    py_object f_self;
    bool exists;

    virtual_function(const extension_type<T> &type, const T *self_, const char *method_name) :
	self(_vf_self(type, self_)),
	f_base(_vf_fbase(type, method_name)),
	f_self(_vf_fself(self, method_name)),
	exists(f_base.ptr != f_self.ptr)
    { }

    template<typename... Ts>
    inline void upcall(const Ts & ... args)
    {
	if (!exists)
	    throw std::runtime_error("pure_virtual_function::upcall() applied when exists=false (this is a bug in the C++ wrapper code)");
	py_tuple t = py_tuple::make(self, args...);
	f_self.call(t);
    }
};


// pure_virtual_function<T,R> is just a thin wrapper around virtual_function<T,R>, which throws an exception if there is no python override.

template<typename T, typename R>
pure_virtual_function<T,R>::pure_virtual_function(const extension_type<T> &type, const T *self_, const char *method_name) :
    virtual_function<T,R> (type, self_, method_name)
{
    // The base class constructor sets the 'exists' flag if the virtual function has been overridden in python.
    // For pure virtuals this is mandatory, so we just throw an exception if exists=false.

    if (!this->exists) {
	std::string tp_name_self = this->self.ptr->ob_type ? this->self.ptr->ob_type->tp_name : "";
	std::string tp_name_base = type.tobj ? type.tobj->tp_name : "";
	throw std::runtime_error(std::string(method_name) + "() must be defined in python subclass " + tp_name_self + ", in order to override pure virtual in C++ base class " + tp_name_base);
    }
}


}  // namespace pyclops

#endif // _PYCLOPS_VIRTUAL_FUNCTION_HPP
