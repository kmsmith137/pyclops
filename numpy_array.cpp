#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;
using namespace mccp_arrays;

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


int mcpp_typeid_from_npy_type(int npy_type)
{
    switch (npy_type) {
	case NPY_INT8: return MCPP_INT8;
	case NPY_INT16: return MCPP_INT16;
	case NPY_INT32: return MCPP_INT32;
	case NPY_INT64: return MCPP_INT64;
	case NPY_UINT8: return MCPP_UINT8;
	case NPY_UINT16: return MCPP_UINT16;
	case NPY_UINT32: return MCPP_UINT32;
	case NPY_UINT64: return MCPP_UINT64;
	case NPY_FLOAT32: return MCPP_FLOAT32;
	case NPY_FLOAT64: return MCPP_FLOAT64;
	case NPY_COMPLEX64: return MCPP_COMPLEX64;
	case NPY_COMPLEX128: return MCPP_COMPLEX128;
    }

    stringstream ss;
    ss << "numpy type " << npy_type << "(" << npy_typestr(npy_type) << ") is not supported by mcpp_arrays";
    throw runtime_error(ss.str());
}


int npy_type_from_mcpp_typeid(mcpp_array::typeid mcpp_type)
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
    }
    
    throw runtime_error("invalid mcpp_array::typeid: " + to_string(mcpp_type));
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


struct np_reaper : public mcpp_arrays::mcpp_array_reaper {
    py_object x;
    np_reaper(const py_object &x_) : x(x_) { }
    virtual ~np_reaper() { }
};


struct mcpp_pybase {
    PyObject_HEAD

    shared_ptr<mcpp_arrays::mcpp_array_reaper> *reaper;
    
    // These tp_new() and tp_dealloc() are 
    
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
    mcpp_pybase_docstring,  /* tp_doc */
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


// -------------------------------------------------------------------------------------------------
//
// Externally-visible helpers for garbage collection.


shared_ptr<mcpp_arrays::array_reaper> mcpp_reaper_from_pybase(const py_object &x)
{
    if (!PyObject_IsInstance(x.ptr(), (PyObject *) &mcpp_pybase_type))
	return make_shared<np_reaper> (x);

    mcpp_pybase *mp = (mcpp_pybase *) (x.ptr());
    
    if (!mp->reaper)
	throw runtime_error(xx);

    shared_ptr<mcpp_arrays::array_reaper> ret = *(mp->reaper);

    if (!ret)
	throw runtime_error(xx);

    return ret;
}


py_object make_mcpp_pybase(const shared_ptr<mcpp_arrays::array_reaper> &reaper)
{
    if (!reaper)
	throw runtime_error("pyclops internal error: empty 'reaper' pointer passed to make_mcpp_pybase()");

    np_reaper *npp = dynamic_cast<np_reaper *> (reaper.get());
    
    if (npp)
	return npp->x;

    PyObject *p = mcpp_pybase::tp_new(&mcpp_pybase, NULL, NULL);
    py_object ret = py_object::new_reference(p);

    mcpp_pybase *mp = (mcpp_pybase *) (p);
    p->reaper = new shared_ptr<mcpp_arrays::array_reaper> (reaper);
    
    return ret;
}


}  // namespace pyclops
