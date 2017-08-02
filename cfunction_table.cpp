#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


static constexpr int table_capacity = 50;


struct kwargs_entry {
    PyCFunction c_func;
    std::function<py_object(py_tuple,py_dict)> cpp_func;
};

static vector<kwargs_entry> kwargs_table(table_capacity);
static int kwargs_table_size = 0;


// non-inline
PyObject *_f_kwargs(PyObject *self, PyObject *args, PyObject *kwds, int N)
{
    try {
	py_tuple a = py_tuple::borrowed_reference(args);
	py_dict k = kwds ? py_dict::borrowed_reference(kwds) : py_dict();
	py_object r = kwargs_table[N].cpp_func(a, k);

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
static PyObject *f_kwargs(PyObject *self, PyObject *args, PyObject *kwds)
{
    return _f_kwargs(self, args, kwds, N);
}


template<int N, typename std::enable_if<(N==0),int>::type = 0>
void populate_kwargs_table() { }

template<int N, typename std::enable_if<(N>0),int>::type = 0>
void populate_kwargs_table()
{
    populate_kwargs_table<N-1>();
    
    PyObject * (*f)(PyObject *, PyObject *, PyObject *) = f_kwargs<N-1>;
    kwargs_table[N-1].c_func = (PyCFunction) f;
}


PyCFunction make_kwargs_cfunction(std::function<py_object(py_tuple,py_dict)> f)
{
    if (kwargs_table_size >= table_capacity)
	throw runtime_error("pyclops: cfunction_table is full!");

    kwargs_table[kwargs_table_size].cpp_func = f;
    return kwargs_table[kwargs_table_size++].c_func;
}


namespace {
    struct _initializer {
	_initializer() 
	{
	    populate_kwargs_table<table_capacity> ();
	}
    } _ini;
}


}  // namespace pyclops
