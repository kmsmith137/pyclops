#ifndef _PYCLOPS_EXTENSION_TYPE_HPP
#define _PYCLOPS_EXTENSION_TYPE_HPP

#include "internals.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


template<typename T>
struct class_wrapper {
    PyObject_HEAD
    T *ptr = nullptr;
};


template<typename T>
class extension_type {
public:
    extension_type(const std::string &name, const std::string &docstring="");

    inline void add_constructor(std::function<T* (py_tuple,py_dict)> f);

    inline void add_method(const std::string &name,
			   const std::string &docstring,
			   std::function<py_object(T *,py_tuple,py_dict)> f);

    inline void finalize();

    inline T *from_python(const py_object &obj, const char *where=nullptr);
    inline py_object to_python(const T &x);

protected:
    PyTypeObject *tobj = nullptr;
    bool finalized = false;

    // Note: bare pointer to std::vector is intentional here!
    // Reminder: a PyMethodDef is a (name, cfunc, flags, docstring) quadruple.
    std::vector<PyMethodDef> *methods;

    static inline void tp_dealloc(PyObject *self);
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


template<typename T>
extension_type<T>::extension_type(const std::string &name, const std::string &docstring) :
    methods(new std::vector<PyMethodDef> ())
{ 
    // This is probably silly, but I decided to overallocate the PyTypeObject to avoid
    // a possible segfault if the python interpreter gets recompiled with -DCOUNT_ALLOCS.
    ssize_t nalloc = sizeof(PyTypeObject) + 128;
    tobj = (PyTypeObject *) malloc(nalloc);   // FIXME check for failed allocation
    memset(tobj, 0, nalloc);

    PyObject_INIT_VAR((PyVarObject *) tobj, NULL, 0);

    tobj->tp_name = strdup(name.c_str());
    tobj->tp_doc = strdup(docstring.c_str());
    tobj->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    tobj->tp_basicsize = sizeof(class_wrapper<T>);
    tobj->tp_dealloc = extension_type<T>::tp_dealloc;
}


template<typename T>
inline void extension_type<T>::add_constructor(std::function<T* (py_tuple, py_dict)> f)
{
    if (tobj->tp_new)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::add_constructor()");

    auto g = [f](PyTypeObject *type, py_tuple args, py_dict kwds) -> PyObject* {
	T *tp = f(args, kwds);
	if (!tp)
	    throw std::runtime_error(std::string(type->tp_name) + ": constructor returned null pointer?!");

	PyObject *ret = type->tp_alloc(type, 0);
	if (!ret)
	    throw pyerr_occurred();

	class_wrapper<T> *wret = reinterpret_cast<class_wrapper<T> *> (ret);
	wret->ptr = tp;
	return ret;
    };

    // Convert std::function to cfunction.
    tobj->tp_new = make_kwargs_newfunc(g);
}


template<typename T>
inline T *extension_type<T>::from_python(const py_object &obj, const char *where)
{
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) this->tobj)) {
	throw std::runtime_error(std::string(where ? where : "pyclops") 
				 + ": couldn't convert python object to type " 
				 + tobj->tp_name);
    }

    auto *wobj = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
    return wobj->ptr;
}


template<typename T>
inline py_object extension_type<T>::to_python(const T &t)
{
    T *tp = new T(t);
    if (!tp)
	throw std::runtime_error("pyclops: allocation failed in to_python converter");

    PyObject *obj = this->tobj->tp_alloc(tobj, 0);
    if (!obj)
	throw pyerr_occurred();

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj);
    wp->ptr = tp;

    return py_object::new_reference(obj);
}


template<typename T>
inline void extension_type<T>::tp_dealloc(PyObject *self)
{
    // FIXME it would be nice to check that 'self' is an instance of extension_type<T>,
    // before doing the cast.  This is possible but would require a new cfunction_table I think!

    auto *wp = reinterpret_cast<class_wrapper<T> *> (self);
    delete wp->ptr;
    wp->ptr = nullptr;
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_TYPE_HPP
