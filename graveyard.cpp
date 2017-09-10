
// -------------------------------------------------------------------------------------------------
//
// mcpp_arrays


template<typename T> 
struct converter<mcpp_arrays::rs_array<T>> {
    static mcpp_arrays::rs_array<T> from_python(const py_object &x, const char *where=nullptr)
    {
	py_array a(x);

	int ndim = a.ndim();
	npy_intp *np_shape = a.shape();
	npy_intp *np_strides = a.strides();
	npy_intp np_itemsize = a.itemsize();

	// Both numpy and mcpp_arrays define 'shape' and 'strides' arrays, but there are
	// two minor differences.  First, numpy uses npy_intp whereas mcpp_arrays uses ssize_t,
	// and in principle these types can be different.  Second, the definitions of the
	// strides differ by a factor of 'itemsize'.

	std::vector<ssize_t> tmp(2*ndim);
	ssize_t *m_shape = &tmp[0];
	ssize_t *m_strides = &tmp[ndim];
	
	for (int i = 0; i < ndim; i++) {
	    // FIXME does numpy allow the stride to not be divisible by the itemsize?
	    // If so, this is an annoying corner case we should eventually handle!
	    if (np_strides[i] % np_itemsize != 0)
		throw std::runtime_error("pyclops internal error: can't divide stride by itemsize");
	    m_shape[i] = np_shape[i];
	    m_strides[i] = np_strides[i] / np_itemsize;
	}
	
	auto dtype = mcpp_typeid_from_npy_type(a.type(), where);
	auto ref = make_mcpp_ref_from_pybase(a._base());

	mcpp_arrays::rs_array<T> ret(a.data(), ndim, m_shape, m_strides, dtype, ref, where);

	// This check should never fail, but seemed like a good idea.
	if (ret.itemsize != np_itemsize)
	    throw std::runtime_error("pyclops internal error: itemsize mismatch in numpy array from_python converter");
	
	return ret;
    }

    // FIXME: here is one case that could be handled better by our rs_array converters.
    // Suppose we python-wrap a function which accepts an rs_array, and returns it:
    //
    //   rs_array<T> f(rs_array<T> x) { return x; }
    //
    // The from_python converter will be called, followed by the to_python converter.
    // This has the effect of returning a new numpy array which points to the same data
    // as the input array.  It would be better to return a reference to the input array!

    static py_object to_python(const mcpp_arrays::rs_array<T> &x)
    {
	py_object pybase = make_pybase_from_mcpp_ref(x.ref);
	int npy_type = npy_type_from_mcpp_typeid(x.dtype, "rs_array to-python converter");

	std::vector<npy_intp> v(2 * x.ndim);
	npy_intp *shape = &v[0];
	npy_intp *strides = &v[x.ndim];

	for (int i = 0; i < x.ndim; i++) {
	    shape[i] = x.shape[i];
	    strides[i] = x.strides[i] * x.itemsize;    // Note factor of 'itemsize' here
	}

	// FIXME: figure out with 100% certainty which flags we should set here.
	int flags = NPY_ARRAY_NOTSWAPPED;

	return py_array::from_pointer(x.ndim, shape, strides, x.itemsize, x.data, npy_type, flags, pybase);
    }
};


mcpp_typeid mcpp_typeid_from_npy_type(int npy_type, const char *where)
{
    switch (npy_type) {
	case NPY_BYTE: return mcpp_arrays::mcpp_type<char>::id;
	case NPY_UBYTE: return mcpp_arrays::mcpp_type<unsigned char>::id;
	case NPY_SHORT: return mcpp_arrays::mcpp_type<short>::id;
	case NPY_USHORT: return mcpp_arrays::mcpp_type<unsigned short>::id;
	case NPY_INT: return mcpp_arrays::mcpp_type<int>::id;
	case NPY_UINT: return mcpp_arrays::mcpp_type<unsigned int>::id;
	case NPY_LONG: return mcpp_arrays::mcpp_type<long>::id;
	case NPY_ULONG: return mcpp_arrays::mcpp_type<unsigned long>::id;
	case NPY_FLOAT: return mcpp_arrays::mcpp_type<float>::id;
	case NPY_DOUBLE: return mcpp_arrays::mcpp_type<double>::id;
	case NPY_CFLOAT: return mcpp_arrays::mcpp_type<std::complex<float> >::id;
	case NPY_CDOUBLE: return mcpp_arrays::mcpp_type<std::complex<double> >::id;
    }

    stringstream ss;
    if (where)
	ss << where << ": ";
    ss << "numpy type " << npy_type << "(" << npy_typestr(npy_type) << ") is not supported by mcpp_arrays";
    throw runtime_error(ss.str());
}


int npy_type_from_mcpp_typeid(mcpp_typeid mcpp_type, const char *where)
{
    switch (mcpp_type) {
	case MCPP_INT8: return NPY_INT8;
	case MCPP_INT16: return NPY_INT16;
	case MCPP_INT32: return NPY_INT32;
	case MCPP_INT64: return NPY_INT64;
	case MCPP_UINT8: return NPY_UINT8;
	case MCPP_UINT16: return NPY_UINT16;
	case MCPP_UINT32: return NPY_UINT32;
	case MCPP_UINT64: return NPY_UINT64;
	case MCPP_FLOAT32: return NPY_FLOAT32;
	case MCPP_FLOAT64: return NPY_FLOAT64;
	case MCPP_COMPLEX64: return NPY_COMPLEX64;
	case MCPP_COMPLEX128: return NPY_COMPLEX128;
	case MCPP_INVALID: break;  // compiler pacifier
    }
    
    stringstream ss;
    if (where)
	ss << where << ": ";
    ss << "invalid mcpp_typeid: " << mcpp_type;
    throw runtime_error(ss.str());
}


// -------------------------------------------------------------------------------------------------
//
// Helpers for garbage collection.
//
// Reminder: mcpp_array garbage collection is done via a shared_ptr<void>,
// whereas numpy garbage collection is done via a 'pybase' object.


// FIXME should use general machinery here!
struct mcpp_pybase {
    PyObject_HEAD

    int is_initialized = 0;
    shared_ptr<void> ref;
    
    static PyObject *tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
	PyObject *self_ = type->tp_alloc(type, 0);
	if (!self_)
	    return NULL;

	mcpp_pybase *self = (mcpp_pybase *) self_;
	
	// "Placement new"!
	new(&self->ref) std::shared_ptr<void> (); 
	self->is_initialized = 1;

	return self_;
    }

    static void tp_dealloc(PyObject *self_)
    {
	mcpp_pybase *self = (mcpp_pybase *) self_;

	if (self->is_initialized) {
	    // Direct destructor call (counterpart of "placement new")
	    self->ref.reset();
	    self->ref.~shared_ptr();
	    self->is_initialized = 0;
	}

	Py_TYPE(self)->tp_free(self_);
    }                

    static constexpr const char *docstring = 
	"Helper class for sharing reference counts between C++ and Python arrays";
};


static PyTypeObject mcpp_pybase_type {
    PyVarObject_HEAD_INIT(NULL, 0)
    "pyclops.mcpp_pybase",      /* tp_name */
    sizeof(mcpp_pybase),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    mcpp_pybase::tp_dealloc,   /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    mcpp_pybase::docstring,    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    mcpp_pybase::tp_new,       /* tp_new */
};


static bool _mcpp_pybase_added = false;

void _add_mcpp_pybase(PyObject *module)
{
    // FIXME is this best?
    if (_mcpp_pybase_added)
	return;

    Py_INCREF(&mcpp_pybase_type);
    PyModule_AddObject(module, "mcpp_pybase", (PyObject *)&mcpp_pybase_type);
    _mcpp_pybase_added = true;
}

bool _mcpp_pybase_ready()
{
    return (PyType_Ready(&mcpp_pybase_type) >= 0);
}


// -------------------------------------------------------------------------------------------------


// Helper for make_mcpp_ref_from_pybase()
static void decref_callback(void *p)
{
    PyObject *op = (PyObject *) p;
    Py_XDECREF(op);
}

// Externally-visible function to make a C++ shared_ptr<void> from PyArray::base.
shared_ptr<void> make_mcpp_ref_from_pybase(const py_object &x)
{
    // FIXME improve!
    if (!_mcpp_pybase_added)
	throw runtime_error("pyclops: currently you need to 'import pyclops' by hand");

    PyObject *p = (PyObject *) x.ptr;

    // General case: return new shared_ptr which holds reference to object.
    if (!PyObject_IsInstance(p, (PyObject *) &mcpp_pybase_type)) {
	Py_INCREF(p);
	return shared_ptr<void> (p, decref_callback);
    }

    // Special case: 'x' is a python object which wraps a C++ shared_ptr.
    // In this case, we return a copy of the existing shared_ptr, rather than making a new one.

    mcpp_pybase *mp = (mcpp_pybase *) p;

    if (!mp->is_initialized)
	throw runtime_error("pyclops internal error: unintialized object in make_mcpp_ref_from_pybase()");
    if (!mp->ref)
	throw runtime_error("pyclops internal error: unexpected empty pointer in make_mcpp_ref_from_pybase()");

    return mp->ref;
}


// Externally-visible function to make a PyArray::base from a C++ shared_ptr<void>.
py_object make_pybase_from_mcpp_ref(const shared_ptr<void> &ref)
{
    // FIXME improve!
    if (!_mcpp_pybase_added)
	throw runtime_error("pyclops: currently you need to 'import pyclops' by hand");

    if (!ref)
	throw runtime_error("pyclops internal error: empty pointer passed to make_pybase_from_mcpp_ref()");

    auto d = std::get_deleter<void (*)(void *)> (ref);

    if (d && (*d == decref_callback)) {
	// Special case: 'ref' is holding a reference to a python object (via a decref_callback).
	// In this case, we return a new reference to the existing object, rather than making a new object.
	PyObject *p = (PyObject *) ref.get();
	return py_object::borrowed_reference(p);
    }

    // General case: construct new python object (of type mcpp_pybase) which wraps the shared_ptr<void>.

    PyObject *p = mcpp_pybase::tp_new(&mcpp_pybase_type, NULL, NULL);
    py_object ret = py_object::new_reference(p);
    mcpp_pybase *mp = (mcpp_pybase *) (p);

    if (!mp->is_initialized)
	throw runtime_error("pyclops internal error: uninitialized mcpp_pybase object in make_pybase_from_mcpp_ref()");

    mp->ref = ref;
    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// pyclops.cpp
//
// This source file is compiled into pyclops.so (the python extension module).
// It is not compiled into libpyclops.so (the C++ library).

#include "pyclops/internals.hpp"

using namespace std;
using namespace pyclops;


static PyMethodDef module_methods[] = {
    { NULL, NULL, 0, NULL }
};


PyMODINIT_FUNC initpyclops(void)
{
    import_array();

    if (!_mcpp_pybase_ready())
	return;

    PyObject *m = Py_InitModule3("pyclops", module_methods, "pyclops: a C++ library for writing python extension modules");
    if (!m)
        return;

    _add_mcpp_pybase(m);
}
