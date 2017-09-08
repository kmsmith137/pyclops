#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;
using namespace mcpp_arrays;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


const char *npy_typestr(int npy_type)
{
    // All numpy types listed in numpy/ndarraytypes.h

    switch (npy_type) {
	case NPY_BOOL: return "NPY_BOOL";
	case NPY_BYTE: return "NPY_BYTE";
	case NPY_UBYTE: return "NPY_UBYTE";
	case NPY_SHORT: return "NPY_SHORT";
	case NPY_USHORT: return "NPY_USHORT";
	case NPY_INT: return "NPY_INT";
	case NPY_UINT: return "NPY_UINT";
	case NPY_LONG: return "NPY_LONG";
	case NPY_ULONG: return "NPY_ULONG";
	case NPY_LONGLONG: return "NPY_LONGLONG";
	case NPY_ULONGLONG: return "NPY_ULONGLONG";
	case NPY_FLOAT: return "NPY_FLOAT";
	case NPY_DOUBLE: return "NPY_DOUBLE";
	case NPY_LONGDOUBLE: return "NPY_LONGDOUBLE";
	case NPY_CFLOAT: return "NPY_CFLOAT";
	case NPY_CDOUBLE: return "NPY_CDOUBLE";
	case NPY_CLONGDOUBLE: return "NPY_CLONGDOUBLE";
	case NPY_OBJECT: return "NPY_OBJECT";
	case NPY_STRING: return "NPY_STRING";
	case NPY_UNICODE: return "NPY_UNICODE";
	case NPY_VOID: return "NPY_VOID";
	case NPY_DATETIME: return "NPY_DATETIME";
	case NPY_TIMEDELTA: return "NPY_TIMEDELTA";
	case NPY_HALF: return "NPY_HALF";
	case NPY_NTYPES: return "NPY_NTYPES";
	case NPY_NOTYPE: return "NPY_NOTYPE";
	case NPY_CHAR: return "NPY_CHAR";
	case NPY_USERDEF: return "NPY_USERDEF";
    }
    
    return "unrecognized numpy type";
}


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


}  // namespace pyclops
