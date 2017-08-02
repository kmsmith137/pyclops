#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


cmodule::cmodule(const string &name, const string &docstring) :
    module_name(name),
    module_docstring(docstring)
{
    if (name.size() == 0)
	throw runtime_error("pyclops: cmodule name must be a nonempty string");
}


void cmodule::add_function(const string &func_name, const string &docstring, std::function<py_object(py_tuple,py_dict)> func)
{
    if (finalized)
	throw runtime_error("pyclops: cmodule::add_function() called after cmodule::finalize()");

    PyMethodDef m;
    m.ml_name = strdup(func_name.c_str());
    m.ml_meth = make_kwargs_cfunction(func);
    m.ml_flags = METH_VARARGS | METH_KEYWORDS;
    m.ml_doc = strdup(docstring.c_str());

    this->module_methods.push_back(m);
}


void cmodule::add_function(const string &func_name, std::function<py_object(py_tuple,py_dict)> func)
{
    add_function(func_name, "", func);
}


void cmodule::finalize()
{
    if (finalized)
	throw runtime_error("pyclops: double call to cmodule::finalize()");

    this->module_methods.push_back({ NULL, NULL, 0, NULL });
    
    PyObject *m = Py_InitModule3(strdup(module_name.c_str()),
				 &module_methods[0],
				 strdup(module_docstring.c_str()));

    if (!m)
	throw runtime_error("pyclops: Py_InitModule3() failed");

    this->finalized = true;
}


}  // namespace pyclops
