#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


static constexpr int max_kwargs_cfunctions = 50;
static constexpr int max_kwargs_cmethods = 50;
static constexpr int max_kwargs_initprocs = 20;


// -------------------------------------------------------------------------------------------------
//
// kwargs_cfunction
//
// Input:    std::function<py_object(py_tuple,py_dict)>
// Output:   PyObject (*)(PyObject *, PyObject *, PyObject *)


struct kwargs_cfunction {
    std::function<py_object(py_tuple,py_dict)> cpp_func;
    PyObject * (*c_func)(PyObject *, PyObject *, PyObject *);
};

static vector<kwargs_cfunction> kwargs_cfunctions(max_kwargs_cfunctions);
static int num_kwargs_cfunctions = 0;


// non-inline
PyObject *_kwargs_cfunction_body(PyObject *self, PyObject *args, PyObject *kwds, int N)
{
    try {
	py_tuple a = py_tuple::borrowed_reference(args);
	py_dict k = kwds ? py_dict::borrowed_reference(kwds) : py_dict();
	py_object r = kwargs_cfunctions[N].cpp_func(a, k);

	PyObject *ret = r.ptr;
	r.ptr = NULL;  // steal reference
	return ret;
    }
    catch (std::exception &e) {
	set_python_error(e);
	return NULL;
    } catch (...) {
	PyErr_SetString(PyExc_RuntimeError, "C++ exception was thrown, but not a subclass of std::exception");
	return NULL;
    }
}

template<int N>
static PyObject *kwargs_cfunction_body(PyObject *self, PyObject *args, PyObject *kwds)
{
    return _kwargs_cfunction_body(self, args, kwds, N);
}


template<int N, typename std::enable_if<(N==0),int>::type = 0>
inline void initialize_kwargs_cfunctions() { }

template<int N, typename std::enable_if<(N>0),int>::type = 0>
inline void initialize_kwargs_cfunctions()
{
    initialize_kwargs_cfunctions<N-1>();    
    kwargs_cfunctions[N-1].c_func = kwargs_cfunction_body<N-1>;
}


PyCFunction make_kwargs_cfunction(std::function<py_object(py_tuple,py_dict)> f)
{
    if (num_kwargs_cfunctions >= max_kwargs_cfunctions)
	throw runtime_error("pyclops: cfunction_table is full!");

    kwargs_cfunctions[num_kwargs_cfunctions].cpp_func = f;
    return (PyCFunction) kwargs_cfunctions[num_kwargs_cfunctions++].c_func;
}


// -------------------------------------------------------------------------------------------------
//
// kwargs_cmethod
//
// Input:    std::function<py_object(py_object,py_tuple,py_dict)>
// Output:   PyObject (*)(PyObject *, PyObject *, PyObject *)


struct kwargs_cmethod {
    std::function<py_object(py_object,py_tuple,py_dict)> cpp_func;
    PyObject * (*c_func)(PyObject *, PyObject *, PyObject *);
};

static vector<kwargs_cmethod> kwargs_cmethods(max_kwargs_cmethods);
static int num_kwargs_cmethods = 0;


// non-inline
PyObject *_kwargs_cmethod_body(PyObject *self, PyObject *args, PyObject *kwds, int N)
{
    try {
	py_object s = py_tuple::borrowed_reference(self);
	py_tuple a = py_tuple::borrowed_reference(args);
	py_dict k = kwds ? py_dict::borrowed_reference(kwds) : py_dict();
	py_object r = kwargs_cmethods[N].cpp_func(s, a, k);

	PyObject *ret = r.ptr;
	r.ptr = NULL;  // steal reference
	return ret;
    }
    catch (std::exception &e) {
	set_python_error(e);
	return NULL;
    } catch (...) {
	PyErr_SetString(PyExc_RuntimeError, "C++ exception was thrown, but not a subclass of std::exception");
	return NULL;
    }
}

template<int N>
static PyObject *kwargs_cmethod_body(PyObject *self, PyObject *args, PyObject *kwds)
{
    return _kwargs_cmethod_body(self, args, kwds, N);
}


template<int N, typename std::enable_if<(N==0),int>::type = 0>
inline void initialize_kwargs_cmethods() { }

template<int N, typename std::enable_if<(N>0),int>::type = 0>
inline void initialize_kwargs_cmethods()
{
    initialize_kwargs_cmethods<N-1>();    
    kwargs_cmethods[N-1].c_func = kwargs_cmethod_body<N-1>;
}


PyCFunction make_kwargs_cmethod(std::function<py_object(py_object,py_tuple,py_dict)> f)
{
    if (num_kwargs_cmethods >= max_kwargs_cmethods)
	throw runtime_error("pyclops: cmethod_table is full!");

    kwargs_cmethods[num_kwargs_cmethods].cpp_func = f;
    return (PyCFunction) kwargs_cmethods[num_kwargs_cmethods++].c_func;
}


// -------------------------------------------------------------------------------------------------
//
// kwargs_initproc
//
// Input:    std::function<void (py_object, py_tuple, py_dict)>
// Output:   int (*)(PyObject *, PyObject *, PyObject *)
//
// Note: the "outer" C-style wrapper function returns -1 if the "inner" C++ std::function
// threw an exception, and 0 if it returned successfully.

struct kwargs_initproc {
    std::function<void(py_object, py_tuple, py_dict)> cpp_func;
    int (*c_func)(PyObject *, PyObject *, PyObject *);
};

static vector<kwargs_initproc> kwargs_initprocs(max_kwargs_initprocs);
static int num_kwargs_initprocs = 0;


// non-inline
int _kwargs_initproc_body(PyObject *self, PyObject *args, PyObject *kwds, int N)
{
    try {
	py_object s = py_object::borrowed_reference(self);
	py_tuple a = py_tuple::borrowed_reference(args);
	py_dict k = kwds ? py_dict::borrowed_reference(kwds) : py_dict();
	kwargs_initprocs[N].cpp_func(s, a, k);
	return 0;
    }
    catch (std::exception &e) {
	set_python_error(e);
	return -1;
    } catch (...) {
	PyErr_SetString(PyExc_RuntimeError, "C++ exception was thrown, but not a subclass of std::exception");
	return -1;
    }
}


template<int N>
static int kwargs_initproc_body(PyObject *type, PyObject *args, PyObject *kwds)
{
    return _kwargs_initproc_body(type, args, kwds, N);
}


template<int N, typename std::enable_if<(N==0),int>::type = 0>
inline void initialize_kwargs_initprocs() { }

template<int N, typename std::enable_if<(N>0),int>::type = 0>
inline void initialize_kwargs_initprocs()
{
    initialize_kwargs_initprocs<N-1>();    
    kwargs_initprocs[N-1].c_func = kwargs_initproc_body<N-1>;
}


initproc make_kwargs_initproc(std::function<void (py_object, py_tuple, py_dict)> f)
{
    if (num_kwargs_initprocs >= max_kwargs_initprocs)
	throw runtime_error("pyclops: initproc_table is full!");

    kwargs_initprocs[num_kwargs_initprocs].cpp_func = f;
    return kwargs_initprocs[num_kwargs_initprocs++].c_func;
}


// -------------------------------------------------------------------------------------------------
//
// Getters/setters.


PyObject *pyclops_getter(PyObject *self, void *closure)
{
    property_closure *pc = reinterpret_cast<property_closure *> (closure);

    try {
	py_object s = py_object::borrowed_reference(self);
	py_object r = pc->f_get(s);

	PyObject *ret = r.ptr;
	r.ptr = NULL;  // steal reference
	return ret;
    }
    catch (std::exception &e) {
	set_python_error(e);
	return NULL;
    } catch (...) {
	PyErr_SetString(PyExc_RuntimeError, "C++ exception was thrown, but not a subclass of std::exception");
	return NULL;
    }
}


int pyclops_setter(PyObject *self, PyObject *value, void *closure)
{
    property_closure *pc = reinterpret_cast<property_closure *> (closure);

    try {
	py_object s = py_object::borrowed_reference(self);
	py_object v = py_object::borrowed_reference(value);
	pc->f_set(s,v);
	return 0;
    }
    catch (std::exception &e) {
	set_python_error(e);
	return -1;
    } catch (...) {
	PyErr_SetString(PyExc_RuntimeError, "C++ exception was thrown, but not a subclass of std::exception");
	return -1;
    }
}


// -------------------------------------------------------------------------------------------------


namespace {
    struct _initializer {
	_initializer() 
	{
	    initialize_kwargs_cfunctions<max_kwargs_cfunctions> ();
	    initialize_kwargs_cmethods<max_kwargs_cmethods> ();
	    initialize_kwargs_initprocs<max_kwargs_initprocs> ();
	}
    } _ini;
}


}  // namespace pyclops
