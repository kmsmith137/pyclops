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
