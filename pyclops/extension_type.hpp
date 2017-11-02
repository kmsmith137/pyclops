#ifndef _PYCLOPS_EXTENSION_TYPE_HPP
#define _PYCLOPS_EXTENSION_TYPE_HPP

#include <memory>
#include "core.hpp"
#include "converters.hpp"
#include "cfunction_table.hpp"
#include "functional_wrappers.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// Externally visible extension_type.


template<typename B>
struct _extension_subtype {
    // Returns NULL if dynamic_pointer_cast fails (throws exception on miscellaneous failure).
    // Caller must check that 'x' is a nonempty pointer, and x.get() is not in the master_hash_table.
    virtual PyObject *_to_python(const std::shared_ptr<B> &x) = 0;
};


template<typename T, typename B=T>
struct extension_type : _extension_subtype<B>
{
    static_assert(std::is_base_of<B,T>::value, "extension_type<T,B>: B must be a base class of T");
    
    using wrapped_type = T;
    using wrapped_base = B;

    // Use this constructor if there is no python-wrapped base class (i.e. B == T).
    inline extension_type(const std::string &name, const std::string &docstring);

    // Use this constructor if there is a python-wrapped base class (i.e. B != T).
    // The constructor will check (via static_assert) that E::wrapped_type == B.
    template<typename E>
    inline extension_type(const std::string &name, const std::string &docstring, E &base);
    
    inline void add_constructor(std::function<T* (py_object,py_tuple,py_dict)> f);

    // The 'f' argument will usually be obtained from wrap_method(), in pyclops/functional_wrappers.hpp.
    inline void add_method(const std::string &name, const std::string &docstring, std::function<py_object(T *,py_tuple,py_dict)> f);

    // The 'f' argument will usually be obtained from wrap_func(), in pyclops/functional_wrappers.hpp.
    inline void add_staticmethod(const std::string &name, const std::string &docstring, std::function<py_object(py_tuple,py_dict)> f);

    // General property API.
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get);
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get, const std::function<void(py_object,py_object)> &f_set);

    // This more specific API suffices for simple properties.
    //    struct X { int x; ... };
    //    std::function<int(const X*)> f_ro = [](const X *self) { return self->x; };
    //    std::function<int& (X*)> f_rw = [](X *self) -> int& { return self->x; };

    template<typename R>
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<R(const T *)> &f);

    template<typename R>
    inline void add_property(const std::string &name, const std::string &docstring, const std::function<R& (T *)> &f);

    // Sets the 'finalize' flag.
    // Note: this is called automatically in extension_module::add_type().
    inline void finalize();

    // These guys are intended to be wrapped by converters.
    // FIXME wouldn't it be better to make them member functions?
    static inline T *bare_pointer_from_python(PyTypeObject *tobj, const py_object &obj, const char *where=nullptr);
    static inline std::shared_ptr<T> shared_ptr_from_python(PyTypeObject *tobj, const py_object &obj, const char *where=nullptr);

    // This version of to_python() is "the" to_python converter for the wrapped type T.
    // It checks for an empty pointer, checks the master_hash_table, and checks all of the derived_types.
    inline py_object to_python(const std::shared_ptr<T> &x);

    static inline void tp_dealloc(PyObject *self);

    // This version of _to_python() is called recursively via the base class.
    // (See _extension_subtype above.)
    inline PyObject *_to_python(const std::shared_ptr<B> &x) override;

    // Helper function called by constructors.
    inline void _construct(const std::string &name, const std::string &docstring);

    // Helper function called by to_python() and _to_python().
    // Returns new reference; never returns NULL.
    inline PyObject *_make(const std::shared_ptr<T> &p);

    // Allocated and initialized at construction.
    PyTypeObject *tobj = nullptr;

    // Note: bare pointer to std::vector is intentional here!
    std::vector<PyMethodDef> *methods = nullptr;     // (name, cfunc, flags, docstring)
    std::vector<PyGetSetDef> *getsetters = nullptr;  // (name, getter, setter, doc, closure)
    bool finalized = false;                          // if true, no methods or getsetters may be added.

    // Note: base_types have pointers to their derived_types, but not vice versa!
    // Note: it's OK to add derived_types after the 'finalized' flag is set.
    // This is because new derived types may be defined at any time (e.g. when another module is imported).
    std::vector<_extension_subtype<T> *> derived_types;
};


// FIXME superseded by 'class virtual function', should it be removed?
inline py_object _py_upcall(void *self, const char *meth_name, const py_tuple &args);



// -------------------------------------------------------------------------------------------------
//
// xconverter: used to define converters for extension_types.
//
// FIXME should this code be moved elsewhere?
// FIXME 'xconverter' needs better name!
//
// FIXME has_xconverter could be improved (just checks that static member function 'type' exists and is a pointer)


// expects 'static constexpr extension_type<T,B> *type = ...'
template<typename T> struct xconverter;


template<typename T, typename = void>
struct has_xconverter : std::false_type { };

template<typename T>
struct has_xconverter<T, typename std::enable_if<std::is_pointer<decltype(xconverter<T>::type)>::value>::type> : std::true_type { };


template<typename T>
struct predicated_converter<T&, typename std::enable_if<has_xconverter<T>::value>::type>
{
    static inline T& from_python(const py_object &x, const char *where=nullptr)
    {
	auto p = extension_type<T>::shared_ptr_from_python(xconverter<T>::type->tobj, x, where);
	return *p;
    }
};


template<typename T>
struct predicated_converter<std::shared_ptr<T>, typename std::enable_if<has_xconverter<T>::value>::type>
{
    static std::shared_ptr<T> from_python(const py_object &obj, const char *where=nullptr)
    {
	return extension_type<T>::shared_ptr_from_python(xconverter<T>::type->tobj, obj, where);
    }
	
    static py_object to_python(const std::shared_ptr<T> &x)
    {
	return xconverter<T>::type->to_python(x);
    }
};


// FIXME invokes copy constructor - should diasable this template if no copy constructor defined.
// Even if copy constructor is defined, it may be a hidden source of overhead - should there be a boolean
// flag somewhere to enable this template?

template<typename T>
struct predicated_converter<T, typename std::enable_if<has_xconverter<T>::value>::type>
{
    static inline py_object to_python(const T &x)
    {
	auto p = std::make_shared<T> (x);
	return xconverter<T>::type->to_python(p);
    }
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


// This constructor is called if there is no python-wrapped base class.
template<typename T, typename B>
extension_type<T,B>::extension_type(const std::string &name, const std::string &docstring)
{ 
    static_assert(std::is_same<T,B>::value, "extension_type<T,B> was constructed with B != T, but no base_type object was specified");
    _construct(name, docstring);
}


// This constructor is called if there is a python-wrapped base class B.
template<typename T, typename B> template<typename E>
extension_type<T,B>::extension_type(const std::string &name, const std::string &docstring, E &base)
{
    using Et = typename E::wrapped_type;

    static_assert(!std::is_same<T,B>::value, "extension_type<T,B> was constructed with B == T, but base_type object was specified");
    static_assert(std::is_same<B,Et>::value, "extension_type<T,B>: base class B does not match base_type::wrapped_type");

    _construct(name, docstring);
    
    tobj->tp_base = base.tobj;
    base.derived_types.push_back(this);
}


// Helper function called by constructors.
template<typename T, typename B>
inline void extension_type<T,B>::_construct(const std::string &name, const std::string &docstring)
{
    this->methods = new std::vector<PyMethodDef> ();
    this->getsetters = new std::vector<PyGetSetDef> ();

    // This is probably silly, but I decided to overallocate the PyTypeObject to avoid
    // a possible segfault if the python interpreter gets recompiled with -DCOUNT_ALLOCS.
    ssize_t nalloc = sizeof(PyTypeObject) + 128;
    tobj = (PyTypeObject *) malloc(nalloc);   // FIXME check for failed allocation
    memset(tobj, 0, nalloc);

    // Idiomatic initialization produces superfluous warnings with gcc5
    // PyObject_INIT((PyVarObject *) tobj, NULL);

    // This initialization is equivalent (see Include/objimpl.h in python interpreter source code)
    _Py_NewReference(tobj);

    tobj->tp_name = strdup(name.c_str());
    tobj->tp_doc = strdup(docstring.c_str());
    tobj->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    tobj->tp_basicsize = sizeof(class_wrapper<T>);
    tobj->tp_new = PyType_GenericNew;
    tobj->tp_dealloc = extension_type<T>::tp_dealloc;
}


template<typename T, typename B>
inline void extension_type<T,B>::add_constructor(std::function<T* (py_object, py_tuple, py_dict)> f)
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
	master_hash_table_add(tp, self.ptr);
	new(&wp->ref) std::shared_ptr<T> ();   // "placement new"
	wp->p = tp;
    };

    // Convert std::function to C-style function pointer.
    tobj->tp_init = make_kwargs_initproc(tp_init);
}


template<typename T, typename B>
inline void extension_type<T,B>::add_method(const std::string &name, const std::string &docstring, std::function<py_object(T*,py_tuple,py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_method() was called after finalize()");

    char *fname = strdup(name.c_str());
    PyTypeObject *tp = this->tobj;

    auto py_method = [f,tp,fname](py_object self, py_tuple args, py_dict kwds) -> py_object {
	if (!PyObject_IsInstance(self.ptr, (PyObject *) tp))
	    throw std::runtime_error(std::string(tp->tp_name) + "." + fname + ": expected 'self' of type " + tp->tp_name);

	auto *wp = reinterpret_cast<class_wrapper<T> *> (self.ptr);
	if (!wp->p)
	    throw std::runtime_error(std::string(tp->tp_name) + ".__init__() was never called (probably missing call in subclass constructor");

	return f(wp->p, args, kwds);
    };

    PyMethodDef m;
    m.ml_name = fname;
    m.ml_meth = make_kwargs_cmethod(py_method);
    m.ml_flags = METH_VARARGS | METH_KEYWORDS;
    m.ml_doc = strdup(docstring.c_str());

    this->methods->push_back(m);
}


template<typename T, typename B>
inline void extension_type<T,B>::add_staticmethod(const std::string &name, const std::string &docstring, std::function<py_object(py_tuple,py_dict)> f)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_staticmethod() was called after finalize()");

    PyMethodDef m;
    m.ml_name = strdup(name.c_str());
    m.ml_meth = make_kwargs_cfunction(f);
    m.ml_flags = METH_STATIC | METH_VARARGS | METH_KEYWORDS;
    m.ml_doc = strdup(docstring.c_str());

    this->methods->push_back(m);
}


template<typename T, typename B>
inline void extension_type<T,B>::add_property(const std::string &name, const std::string &docstring, const std::function<py_object(py_object)> &f_get)
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


template<typename T, typename B>
inline void extension_type<T,B>::add_property(const std::string &name, const std::string &docstring, 
					      const std::function<py_object(py_object)> &f_get, 
					      const std::function<void(py_object,py_object)> &f_set)
{
    if (finalized)
	throw std::runtime_error(std::string(tobj->tp_name) + ": extension_type::add_property() was called after finalize()");

    // FIXME memory leaks!
    property_closure *p = new property_closure;
    p->f_get = f_get;
    p->f_set = f_set;
    
    PyGetSetDef gs;
    gs.name = strdup(name.c_str());
    gs.get = pyclops_getter;
    gs.set = pyclops_setter;
    gs.doc = strdup(docstring.c_str());
    gs.closure = p;
    
    getsetters->push_back(gs);
}


template<typename T, typename B> template<typename R>
inline void extension_type<T,B>::add_property(const std::string &name, const std::string &docstring, const std::function<R(const T *)> &f)
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


template<typename T, typename B> template<typename R>
inline void extension_type<T,B>::add_property(const std::string &name, const std::string &docstring, const std::function<R& (T *)> &f)
{
    // FIXME memory leak
    PyTypeObject *tp = this->tobj;
    std::string propname = std::string(tp->tp_name) + "." + name;
    const char *cpropname = strdup(propname.c_str());

    std::function<py_object(py_object)> f_get = [f,tp,cpropname](py_object self) -> py_object
	{
	    T *cself = bare_pointer_from_python(tp, self, cpropname);
	    R &cret = f(cself);
	    return converter<R>::to_python(cret);
	};

    std::function<void(py_object,py_object)> f_set = [f,tp,cpropname](py_object self, py_object value) -> void
	{
	    T *cself = bare_pointer_from_python(tp, self, cpropname);
	    R &cret = f(cself);
	    cret = converter<R>::from_python(value);
	};

    this->add_property(name, docstring, f_get, f_set);
}


template<typename T, typename B>
inline void extension_type<T,B>::finalize()
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


template<typename T, typename B>
inline T *extension_type<T,B>::bare_pointer_from_python(PyTypeObject *tobj, const py_object &obj, const char *where)
{
    if (!PyObject_IsInstance(obj.ptr, (PyObject *) tobj))
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": expected object of type " + tobj->tp_name);

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj.ptr);
    if (!wp->p)
	throw std::runtime_error(std::string(where ? where : "pyclops") + ": " + tobj->tp_name + ".__init__() was never called?!");

    return wp->p;
}


template<typename T, typename B>
inline std::shared_ptr<T> extension_type<T,B>::shared_ptr_from_python(PyTypeObject *tobj, const py_object &obj, const char *where)
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


template<typename T, typename B>
inline py_object extension_type<T,B>::to_python(const std::shared_ptr<T> &x)
{
    // FIXME: another option is to return None here, but I need to think about what's best!
    if (!x)
	throw std::runtime_error("pyclops: empty pointer in to_python converter");

    // Check master_hash_table, to see whether python object already exists.
    PyObject *obj = master_hash_table_query(x.get());
    if (obj) 
	return py_object::borrowed_reference(obj);

    // If we get here, a new python object will be created (or an exception will be thrown).
    // First we check the derived_types.

    for (const auto &d: derived_types) {
	PyObject *p = d->_to_python(x);
	if (p != NULL)  // dynamic_pointer_cast succeeded
	    return py_object::new_reference(p);
    }

    return py_object::new_reference(this->_make(x));
}


// Returns NULL if dynamic_pointer_cast fails (throws exception on miscellaneous failure).
// Caller has checked that 'x' is a nonempty pointer, and x.get() is not in the master_hash_table.
template<typename T, typename B>
inline PyObject *extension_type<T,B>::_to_python(const std::shared_ptr<B> &x)
{
    std::shared_ptr<T> y = std::dynamic_pointer_cast<T> (x);

    if (!y)
	return NULL;

    for (const auto &d: derived_types) {
	PyObject *p = d->_to_python(y);
	if (p != NULL)  // dynamic_pointer_cast succeeded
	    return p;
    }

    return this->_make(y);
}


template<typename T, typename B>
inline PyObject *extension_type<T,B>::_make(const std::shared_ptr<T> &x)
{
    // Make new object.
    // FIXME I suspect this should be tp_new(), rather than tp_alloc().
    PyObject *obj = tobj->tp_alloc(tobj, 0);
    if (!obj)
	throw pyerr_occurred();

    auto *wp = reinterpret_cast<class_wrapper<T> *> (obj);
    new(&wp->ref) std::shared_ptr<T> ();   // "placement new"

    master_hash_table_add(x.get(), obj);
    wp->p = x.get();
    wp->ref = x;

    return obj;
}


template<typename T, typename B>
inline void extension_type<T,B>::tp_dealloc(PyObject *self)
{
    // FIXME it would be nice to check that 'self' is an instance of extension_type<T>,
    // before doing the cast.  This is possible but would require a new cfunction_table I think!

    auto *wp = reinterpret_cast<class_wrapper<T> *> (self);

    if (!wp->p)
	return;

    T *p = wp->p;
    wp->p = nullptr;
    master_hash_table_remove((void *)p, self);

    if (!wp->ref) {
	// Object is python-managed, i.e. allocated with new() when python object
	// is constructed, and deallocated with delete() when python object is destroyed.
	delete p;
	return;
    }

    // Object is C++-managed, i.e. python object holds a reference via shared_ptr<>.
    wp->ref.reset();
    wp->ref.~shared_ptr();  // direct destructor call (counterpart of "placement new")
}


// -------------------------------------------------------------------------------------------------


inline py_object _py_upcall(void *self, const char *meth_name, const py_tuple &args)
{
    PyObject *s = master_hash_table_query(self);

    // FIXME how to improve this error message?
    if (!s)
	throw std::runtime_error("pyclops internal error: couldn't find object in master_hash_table during upcall");

    // This should never fail, since the method should be defined by the base class.
    // FIXME how to improve this error message?
    PyObject *fp = PyObject_GetAttrString(s, meth_name);
    if (!fp)
	throw std::runtime_error("pyclops internal error: couldn't find attribute during upcall");

    py_object func = py_object::new_reference(fp);
    return func.call(args);
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_TYPE_HPP
