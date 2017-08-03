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

    // Note: constructed with "placement new" (in add_constructor() and to_python() below), 
    // and destroyed with direct call to shared_ptr<T> destructor (in tp_dealloc() below).
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
    tobj->tp_dealloc = extension_type<T>::tp_dealloc;
}


template<typename T>
inline void extension_type<T>::add_constructor(std::function<std::shared_ptr<T> (py_tuple, py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_constructor() was called after finalize()");
    if (tobj->tp_new)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::add_constructor()");

    auto g = [f](PyTypeObject *type, py_tuple args, py_dict kwds) -> PyObject* {
	std::shared_ptr<T> tp = f(args, kwds);
	if (!tp)
	    throw std::runtime_error(std::string(type->tp_name) + ": constructor returned null pointer?!");

	PyObject *ret = type->tp_alloc(type, 0);
	if (!ret)
	    throw pyerr_occurred();

	std::cerr << "tp_new(1): tp=" << tp.get() << ", refcount=" << tp.use_count() << "\n";

	// class_wrapper<T>::ptr is constructed here, with "placement new".
	class_wrapper<T> *wp = reinterpret_cast<class_wrapper<T> *> (ret);	
	new(&wp->ptr) std::shared_ptr<T> (tp);

	std::cerr << "tp_new(2): tp=" << tp.get() << ", refcount=" << tp.use_count() << "\n";
	std::cerr << "tp_new(3): wp->ptr=" << wp->ptr.get() << ", refcount=" << wp->ptr.use_count() << "\n";

	return ret;
    };

    // Convert std::function to C-style function pointer.
    tobj->tp_new = make_kwargs_newfunc(g);
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
    if (!tobj->tp_new)
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
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) tobj)) {
	throw std::runtime_error(std::string(where ? where : "pyclops") 
				 + ": couldn't convert python object to type " 
				 + tobj->tp_name);
    }

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
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

    return py_object::new_reference(obj);
}


template<typename T>
inline void extension_type<T>::tp_dealloc(PyObject *self)
{
    std::cerr << "tp_dealloc\n";

    // FIXME it would be nice to check that 'self' is an instance of extension_type<T>,
    // before doing the cast.  This is possible but would require a new cfunction_table I think!

    auto *wp = reinterpret_cast<class_wrapper<T> *> (self);

    // Should be unnecessary, but I'm paranoid.
    wp->ptr.reset();

    // class_wrapper<T>::ptr is destroyed here, with direct destructor call.
    // I wonder if this is also unnecessary, now that the shared_ptr has been reset!
    wp->ptr.~shared_ptr();
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_TYPE_HPP
