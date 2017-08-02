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
// Reminder: mcpp_array garbage collection is done via a 'reaper' class,
// whereas numpy garbage collection is done via a 'pybase' object.
//
// We define two new classes:
//
//   - np_reaper: an mccp_arrays::array_reaper which wraps a python object
//   - mcpp_pybase: a new python type whose instances wrap an mccp_arrays::array_reaper.


struct np_reaper : public mcpp_arrays::mcpp_reaper {
    py_object x;
    np_reaper(const py_object &x_) : x(x_) { }
    virtual ~np_reaper() { }
};


struct mcpp_pybase {
    PyObject_HEAD

    shared_ptr<mcpp_arrays::mcpp_reaper> *reaper;
    
    static PyObject *tp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
	PyObject *self_ = type->tp_alloc(type, 0);
	if (!self_)
	    return NULL;

	mcpp_pybase *self = (mcpp_pybase *) self_;
	self->reaper = nullptr;
	return self_;
    }

    static void tp_dealloc(PyObject *self_)
    {
	mcpp_pybase *self = (mcpp_pybase *) self_;

	delete self->reaper;
	self->reaper = nullptr;
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


static bool _reaper_type_added = false;

void _add_reaper_type(PyObject *module)
{
    // FIXME: what action should be taken if _add_reaper_type() is called twice?
    Py_INCREF(&mcpp_pybase_type);
    PyModule_AddObject(module, "mcpp_pybase", (PyObject *)&mcpp_pybase_type);
    _reaper_type_added = true;
}

bool _reaper_type_ready()
{
    return (PyType_Ready(&mcpp_pybase_type) >= 0);
}


// -------------------------------------------------------------------------------------------------
//
// Externally-visible helpers for garbage collection.


// Make a C++ reaper which wraps python object 'x'.
shared_ptr<mcpp_arrays::mcpp_reaper> make_mcpp_reaper_from_pybase(const py_object &x)
{
    // FIXME improve!
    if (!_reaper_type_added)
	throw runtime_error("pyclops: currently you need to 'import pyclops' by hand");

    // Typical case: construct new reaper.
    if (!PyObject_IsInstance(x.ptr, (PyObject *) &mcpp_pybase_type))
	return make_shared<np_reaper> (x);

    // Special case: 'x' is a python object which wraps a C++ reaper.
    // In this case, rather than creating a new reaper, we return a pointer to the old one.
    mcpp_pybase *mp = (mcpp_pybase *) (x.ptr);
    
    if (!mp->reaper)
	throw runtime_error("pyclops internal error: unexpected null pointer in make_mcpp_reaper_from_pybase()");

    shared_ptr<mcpp_arrays::mcpp_reaper> ret = *(mp->reaper);

    if (!ret)
	throw runtime_error("pyclops internal error: unexpected empty pointer in make_mcpp_reaper_from_pybase()");

    return ret;
}


py_object make_pybase_from_mcpp_reaper(const shared_ptr<mcpp_arrays::mcpp_reaper> &reaper)
{
    if (!reaper)
	throw runtime_error("pyclops internal error: empty 'reaper' pointer passed to make_pybase_from_mcpp_reaper()");

    // FIXME improve!
    if (!_reaper_type_added)
	throw runtime_error("pyclops: currently you need to 'import pyclops' by hand");

    // Special case: 'reaper' is a C++ reaper which wraps a pybase.
    // In this case, rather than creating a new pybase, we return the old one.
    np_reaper *npp = dynamic_cast<np_reaper *> (reaper.get());
    
    if (npp)
	return npp->x;

    // Typical case: construct new pybase object.

    PyObject *p = mcpp_pybase::tp_new(&mcpp_pybase_type, NULL, NULL);
    py_object ret = py_object::new_reference(p);

    mcpp_pybase *mp = (mcpp_pybase *) (p);
    mp->reaper = new shared_ptr<mcpp_arrays::mcpp_reaper> (reaper);
    
    return ret;
}


}  // namespace pyclops
