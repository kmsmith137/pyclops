#ifndef _PYCLOPS_EXTENSION_TYPE_HPP
#define _PYCLOPS_EXTENSION_TYPE_HPP

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"

#include "cfunction_table.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif



template<typename T>
struct class_wrapper {
    PyObject_HEAD

    // The 'initialized' flag exists in order to detect the case where tp_init()
    // never gets called (say, because the C++ extension class is subclassed in
    // python, and the subclass __init__() forgets to call the base class __init__()).
    //
    // In this case, the fields below should still be zeroed, since the object
    // should be allocated by PyType_GenericAlloc(), which zeroes the allocated
    // memory (in addition to initializing ob_refcount and ob_type).  
    //
    // Therefore, the 'initialized' flag is zero if tp_init() never gets called.
    // In tp_init(), we set it to 1 (see below).  This gives us a mechanism for 
    // detecting whether tp_init() is called, as desired.
    //
    // Note: I wasn't sure whether it was safe to assume that a binary-zeroed
    // (but not explicitly constructed) shared_ptr<> is a well-formed empty pointer.
    // If so, then the 'initialized' flag isn't necessary!

    int initialized = 0;

    // Note: constructed with "placement new" (see add_constructor() and to_python() below), 
    // and destroyed with direct call to shared_ptr<T> destructor (see tp_dealloc() below).
    std::shared_ptr<T> ptr;
};


template<typename T>
struct extension_type {
    extension_type(const std::string &name, const std::string &docstring="");

    inline void add_constructor(std::function<std::shared_ptr<T> (py_tuple,py_dict)> f);

    // FIXME: decide whether the "self" argument of the C++ method should be (T *),
    // as currently assumed, or shared_ptr<T>.
    inline void add_method(const std::string &name,
			   const std::string &docstring,
			   std::function<py_object(T *,py_tuple,py_dict)> f);

    inline void finalize();

    static inline std::shared_ptr<T> from_python(PyTypeObject *tobj, const py_object &obj, const char *where=nullptr);
    static inline py_object to_python(PyTypeObject *tobj, const std::shared_ptr<T> &x);
    static inline void tp_dealloc(PyObject *self);

    PyTypeObject *tobj = nullptr;
    bool finalized = false;

    // Note: bare pointer to std::vector is intentional here!
    // Reminder: a PyMethodDef is a (name, cfunc, flags, docstring) quadruple.
    std::vector<PyMethodDef> *methods;
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
    tobj->tp_new = PyType_GenericNew;
    tobj->tp_dealloc = extension_type<T>::tp_dealloc;
}


template<typename T>
inline void extension_type<T>::add_constructor(std::function<std::shared_ptr<T> (py_tuple, py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_constructor() was called after finalize()");
    if (tobj->tp_init)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::add_constructor()");

    PyTypeObject *type = this->tobj;

    auto g = [f,type](py_object self, py_tuple args, py_dict kwds) -> void {
	if (!PyObject_IsInstance(self.ptr, (PyObject *)type))
	    throw std::runtime_error(std::string(type->tp_name) + ": 'self' argument to __init__() does not have expected type?!");

	class_wrapper<T> *wp = reinterpret_cast<class_wrapper<T> *> (self.ptr);
	if (wp->initialized)
	    throw std::runtime_error(std::string(type->tp_name) + ": double call to __init__()?!");

	std::shared_ptr<T> tp = f(args, kwds);
	if (!tp)
	    throw std::runtime_error(std::string(type->tp_name) + ": constructor function returned null pointer?!");

	// class_wrapper<T>::ptr is constructed here, with "placement new".
	wp->initialized = 1;
	new(&wp->ptr) std::shared_ptr<T> (tp);
    };

    // Convert std::function to C-style function pointer.
    tobj->tp_init = make_kwargs_initproc(g);
}


template<typename T>
inline void extension_type<T>::add_method(const std::string &name, const std::string &docstring, std::function<py_object(T*,py_tuple,py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_method() was called after finalize()");

    char *where = strdup(name.c_str());
    PyTypeObject *tp = this->tobj;

    auto g = [f,where,tp](py_object self, py_tuple args, py_dict kwds) -> py_object {
	std::shared_ptr<T> tself = extension_type<T>::from_python(tp, self, where);
	return f(tself.get(), args, kwds);
    };

    PyMethodDef m;
    m.ml_name = where;
    m.ml_meth = make_kwargs_cmethod(g);
    m.ml_flags = METH_VARARGS | METH_KEYWORDS;
    m.ml_doc = strdup(docstring.c_str());

    this->methods->push_back(m);
}


template<typename T>
inline void extension_type<T>::finalize()
{
    if (!tobj->tp_init)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_constructor() was never called");
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::finalize()");

    int nmethods = methods->size();

    // Note that we include a zeroed sentinel.
    tobj->tp_methods = (PyMethodDef *) malloc((nmethods+1) * sizeof(PyMethodDef));
    memset(tobj->tp_methods, 0, (nmethods+1) * sizeof(PyMethodDef));
    memcpy(tobj->tp_methods, &(*methods)[0], nmethods * sizeof(PyMethodDef));

    this->finalized = true;
}


template<typename T>
inline std::shared_ptr<T> extension_type<T>::from_python(PyTypeObject *tobj, const py_object &obj, const char *where)
{
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) tobj))
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": expected object of type " + tobj->tp_name);

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
    if (!wp->initialized)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": " + tobj->tp_name + ".__init__() was never called?!");

    return wp->ptr;
}


template<typename T>
inline py_object extension_type<T>::to_python(PyTypeObject *tobj, const std::shared_ptr<T> &x)
{
    PyObject *obj = tobj->tp_alloc(tobj, 0);
    if (!obj)
	throw pyerr_occurred();

    // class_wrapper<T>::ptr is constructed here, with "placement new".
    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj);
    new(&wp->ptr) std::shared_ptr<T> (x);
    wp->initialized = 1;

    return py_object::new_reference(obj);
}


template<typename T>
inline void extension_type<T>::tp_dealloc(PyObject *self)
{
    // FIXME it would be nice to check that 'self' is an instance of extension_type<T>,
    // before doing the cast.  This is possible but would require a new cfunction_table I think!

    auto *wp = reinterpret_cast<class_wrapper<T> *> (self);

    if (!wp->initialized)
	return;

    // class_wrapper<T>::ptr is destroyed here, with direct destructor call.
    wp->ptr.~shared_ptr();
    wp->initialized = 0;
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_TYPE_HPP
