#ifndef _PYCLOPS_EXTENSION_TYPE_HPP
#define _PYCLOPS_EXTENSION_TYPE_HPP

#include "core.hpp"
#include "cfunction_table.hpp"
#include "functional_wrappers.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// Externally visible extension_type.


template<typename T>
struct extension_type {
    extension_type(const std::string &name, const std::string &docstring="");

    inline void add_constructor(std::function<T* (py_object,py_tuple,py_dict)> f);

    // FIXME: decide whether the "self" argument of the C++ method should be (T *), as currently assumed, or shared_ptr<T>.
    // FIXME: currently "fragile", in the sense that if add_method() is called when add_pure_virtual() is needed, the result will be an infinite loop!
    inline void add_method(const std::string &name, const std::string &docstring, std::function<py_object(T *,py_tuple,py_dict)> f, bool pure_virtual=false);
    inline void add_pure_virtual(const std::string &name, const std::string &docstring, std::function<py_object(T *,py_tuple,py_dict)> f);

    // General property API.
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get);
    // inline void add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get, const std::function<py_object(py_object,py_object) f_set);

    // This more specific API suffices for simple properties.
    //    struct X { int x; ... };
    //    std::function<int(const X*)> f_ro = [](const X *self) { return self->x; };
    //    std::function<int& (X*)> f_rw = [](X *self) -> int& { return self->x; };

    template<typename R>
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<R(const T *)> &f);

    //template<class C, typename R>
    //inline void add_property(const std::string &name, const std::string &docstring, const std::function<R& (C *)> &f);

    inline void finalize();

    // These guys are intended to be wrapped by converters.
    // FIXME wouldn't it be better to make them member functions?
    static inline py_object to_python(PyTypeObject *tobj, const std::shared_ptr<T> &x);
    static inline T *bare_pointer_from_python(PyTypeObject *tobj, const py_object &obj, const char *where=nullptr);
    static inline std::shared_ptr<T> shared_ptr_from_python(PyTypeObject *tobj, const py_object &obj, const char *where=nullptr);

    static inline void tp_dealloc(PyObject *self);

    PyTypeObject *tobj = nullptr;
    bool finalized = false;

    // Note: bare pointer to std::vector is intentional here!
    std::vector<PyMethodDef> *methods = nullptr;     // (name, cfunc, flags, docstring)
    std::vector<PyGetSetDef> *getsetters = nullptr;  // (name, getter, setter, doc, closure)
};


// -------------------------------------------------------------------------------------------------
//
// Implementation.


// The class_wrapper<T> type is used "under the hood" to embed the shared_ptr<T> in a PyObject.
// Members of class_wrapper<T> are only accessed by extension_type<T> (in this source file).

template<typename T>
struct class_wrapper {
    PyObject_HEAD

    // The 'p' pointer can be NULL, in the corner case where tp_init() never gets called 
    // (say, because the C++ extension class is subclassed in python, and the subclass
    // __init__() forgets to call the base class __init__()).
    //
    // Code which uses 'p' (e.g. from-python converter) should always check
    // that it is non-NULL!
    //
    // Note: it is safe to assume that if tp_init() never gets called (or tp_new()), 
    // then 'p' is a null pointer (rather than uninitialized memory).  This is because
    // PyType_GenericAlloc() binary-zeroes its allocated memory.
    //
    // An important invariant of class_wrapper<T> which must be preserved: a
    // master_hash_table entry exists for the (T *, PyObject *) pair if and only 
    // if (p != nullptr).

    T *p = nullptr;

    // The precise semantics of the 'ref' field are nontrivial to explain!
    //
    // An object can either be "C++ managed" if it was originally constructed in C++,
    // or "python-managed" if originally constructed in python.
    //
    // If an object is C++ managed, then 'ref' will be a nonempty shared_ptr<> which
    // is constructed in tp_init() and destroyed in tp_dealloc().  When the PyObject
    // is converted to a shared_ptr<T> by its from_python converter, we simply return
    // a copy of 'ref'.  In this setup, the PyObject holds one reference, but more
    // references can exist in C++ data structures elsewhere, and the PyObject's
    // lifetime can be shorter than the C++ object's lifetime.
    // 
    // If an object is python-managed, then 'ref' will be an empty pointer, and the
    // pointer 'p' is allocated with new(), and destroyed with delete().  When a
    // shared_ptr is requested by C++ code (e.g. via the from_python_converter), we
    // increment the python refcount, and return a C++ shared_ptr whose deleter is
    // responsible for decrementing the refcount.  In this scenario, the PyObject's
    // lifetime and the C++ object's lifetime are always the same.
    //
    // Note that 'ref' is constructed with "placement new" and destroyed with a direct
    // destructor call.  I wanted to avoid making the assumption that a binary-zeroed
    // shared_ptr<> is a valid (empty) pointer.  Therefore, there is an invariant that
    // 'ref' is a valid shared_ptr<> if and only if (p != nullptr).  If p is NULL,
    // then 'ref' should be treated as unintialized memory.
    //
    // Summarizing this comment and the preceding one:
    //
    //     if p == NULL:
    //         no master_hash_table entry exists
    //        'ref' is uninitialized memory
    //     else:
    //         master_tash_table entry exists
    //         'ref' is a valid shared_ptr
    //         if ref is an empty pointer:
    //             object is python-managed
    //         else:
    //             object is C++ managed
    //             'ref' points to 'p'.

    std::shared_ptr<T> ref;
};


template<typename T>
extension_type<T>::extension_type(const std::string &name, const std::string &docstring)
{ 
    this->methods = new std::vector<PyMethodDef> ();
    this->getsetters = new std::vector<PyGetSetDef> ();

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
inline void extension_type<T>::add_constructor(std::function<T* (py_object, py_tuple, py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_constructor() was called after finalize()");
    if (tobj->tp_init)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::add_constructor()");

    PyTypeObject *type = this->tobj;

    auto tp_init = [f,type](py_object self, py_tuple args, py_dict kwds) -> void {
	if (!PyObject_IsInstance(self.ptr, (PyObject *)type))
	    throw std::runtime_error(std::string(type->tp_name) + ": 'self' argument to __init__() does not have expected type?!");

	class_wrapper<T> *wp = reinterpret_cast<class_wrapper<T> *> (self.ptr);
	if (wp->p)
	    throw std::runtime_error(std::string(type->tp_name) + ": double call to __init__()?!");

	T *tp = f(self, args, kwds);
	if (!tp)
	    throw std::runtime_error(std::string(type->tp_name) + ": constructor function returned null pointer?!");

	// Initialize new python-managed object.
	new(&wp->ref) std::shared_ptr<T> ();   // "placement new"
	master_hash_table_add(tp, self.ptr);
	wp->p = tp;
    };

    // Convert std::function to C-style function pointer.
    tobj->tp_init = make_kwargs_initproc(tp_init);
}


template<typename T>
inline void extension_type<T>::add_method(const std::string &name, const std::string &docstring, std::function<py_object(T*,py_tuple,py_dict)> f, bool pure_virtual)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_method() was called after finalize()");

    char *fname = strdup(name.c_str());
    PyTypeObject *tp = this->tobj;

    auto py_method = [f,tp,fname,pure_virtual](py_object self, py_tuple args, py_dict kwds) -> py_object {
	if (!PyObject_IsInstance(self.ptr, (PyObject *) tp))
	    throw std::runtime_error(std::string(tp->tp_name) + "." + fname + ": expected 'self' of type " + tp->tp_name);

	auto *wp = reinterpret_cast<class_wrapper<T> *> (self.ptr);
	if (!wp->p)
	    throw std::runtime_error(std::string(tp->tp_name) + ".__init__() was never called (probably missing call in subclass constructor");

	bool is_python_managed = !wp->ref;

	// For pure virtual functions, this simple mechanism suffices to prevent an infinite loop.
	// FIXME: for non-pure virtuals, something different is needed!
	// FIXME: currently "fragile", in the sense that if add_method() is called when add_pure_virtual() is needed, the result will be an infinite loop!

	if (pure_virtual && is_python_managed) {
	    PyTypeObject *sp = self.ptr->ob_type;
	    const char *sp_name = sp ? sp->tp_name : "";
	    throw std::runtime_error(std::string(fname) + "() must be defined in python subclass " + sp_name + ", in order to override pure virtual in C++ base class " + tp->tp_name);
	}

	return f(wp->p, args, kwds);
    };

    PyMethodDef m;
    m.ml_name = fname;
    m.ml_meth = make_kwargs_cmethod(py_method);
    m.ml_flags = METH_VARARGS | METH_KEYWORDS;
    m.ml_doc = strdup(docstring.c_str());

    this->methods->push_back(m);
}


template<typename T>
inline void extension_type<T>::add_pure_virtual(const std::string &name, const std::string &docstring, std::function<py_object(T*,py_tuple,py_dict)> f)
{
    this->add_method(name, docstring, f, true);
}


template<typename T>
inline void extension_type<T>::add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_property() was called after finalize()");

    // FIXME memory leaks!
    property_closure *p = new property_closure;
    p->f_get = f_get;
    
    PyGetSetDef gs;
    gs.name = strdup(name.c_str());
    gs.get = pyclops_getter;
    gs.set = NULL;
    gs.doc = strdup(docstring.c_str());
    gs.closure = p;
    
    getsetters->push_back(gs);
}


template<typename T> template<typename R>
inline void extension_type<T>::add_property(const std::string &name, const std::string &docstring, const std::function<R(const T *)> &f)
{
    // FIXME memory leak
    PyTypeObject *tp = this->tobj;
    std::string propname = std::string(tp->tp_name) + "." + name;
    const char *cpropname = strdup(propname.c_str());

    std::function<py_object(py_object)> f_get = [f,tp,cpropname](py_object self) -> py_object
	{
	    T *cself = bare_pointer_from_python(tp, self, cpropname);
	    R cret = f(cself);
	    return converter<R>::to_python(cret);
	};

    this->add_property(name, docstring, f_get);
}


template<typename T>
inline void extension_type<T>::finalize()
{
    if (!tobj->tp_init)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_constructor() was never called");
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": double call to extension_type::finalize()");

    // Note that we include zeroed sentinels.

    int nmethods = methods->size();
    tobj->tp_methods = (PyMethodDef *) malloc((nmethods+1) * sizeof(PyMethodDef));
    memset(tobj->tp_methods, 0, (nmethods+1) * sizeof(PyMethodDef));
    memcpy(tobj->tp_methods, &(*methods)[0], nmethods * sizeof(PyMethodDef));

    int ngetsetters = getsetters->size();
    tobj->tp_getset = (PyGetSetDef *) malloc((ngetsetters+1) * sizeof(PyGetSetDef));
    memset(tobj->tp_getset, 0, (ngetsetters+1) * sizeof(PyGetSetDef));
    memcpy(tobj->tp_getset, &(*getsetters)[0], ngetsetters * sizeof(PyGetSetDef));

    this->finalized = true;
}


template<typename T>
inline T *extension_type<T>::bare_pointer_from_python(PyTypeObject *tobj, const py_object &obj, const char *where)
{
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) tobj))
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": expected object of type " + tobj->tp_name);

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
    if (!wp->p)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": " + tobj->tp_name + ".__init__() was never called?!");

    return wp->p;
}


template<typename T>
inline std::shared_ptr<T> extension_type<T>::shared_ptr_from_python(PyTypeObject *tobj, const py_object &obj, const char *where)
{
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) tobj))
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": expected object of type " + tobj->tp_name);

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
    if (!wp->p)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": " + tobj->tp_name + ".__init__() was never called?!");

    if (wp->ref)
	return wp->ref;  // object is C++ managed

    // Object is python-managed.  We increment the refcount, and the shared_ptr is 
    // responsible for decrementing it later, via master_hash_table_deleter().
    Py_INCREF(obj.ptr);
    return std::shared_ptr<T> (wp->p, master_hash_table_deleter);
}


template<typename T>
inline py_object extension_type<T>::to_python(PyTypeObject *tobj, const std::shared_ptr<T> &x)
{
    // FIXME: another option is to return None here, but I need to think about what's best!
    if (!x)
	throw std::runtime_error("pyclops: empty pointer in to_python converter");

    // Check master_hash_table, to see whether python object already exists.
    PyObject *obj = master_hash_table_query(x.get());
    if (obj) 
	return py_object::borrowed_reference(obj);

    // Make new object.
    // FIXME I suspect this should be tp_new(), rather than tp_alloc().
    obj = tobj->tp_alloc(tobj, 0);
    if (!obj)
	throw pyerr_occurred();

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj);
    new(&wp->ref) std::shared_ptr<T> ();   // "placement new"

    master_hash_table_add(x.get(), obj);
    wp->p = x.get();
    wp->ref = x;

    return py_object::new_reference(obj);
}


template<typename T>
inline void extension_type<T>::tp_dealloc(PyObject *self)
{
    // FIXME it would be nice to check that 'self' is an instance of extension_type<T>,
    // before doing the cast.  This is possible but would require a new cfunction_table I think!

    auto *wp = reinterpret_cast<class_wrapper<T> *> (self);

    if (wp->p) {
	void *p = wp->p;
	wp->p = nullptr;
	master_hash_table_remove(p, self);
	wp->ref.reset();
	wp->ref.~shared_ptr();  // direct destructor call (counterpart of "placement new")
    }
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_TYPE_HPP
