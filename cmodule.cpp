#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


extension_module::extension_module(const string &name, const string &docstring) :
    module_name(name),
    module_docstring(docstring)
{
    if (name.size() == 0)
	throw runtime_error("pyclops: extension_module name must be a nonempty string");
}


void extension_module::add_function(const string &func_name, const string &func_docstring, std::function<py_object(py_tuple,py_dict)> func)
{
    if (finalized)
	throw runtime_error("pyclops: extension_module::add_function() called after extension_module::finalize()");

    this->module_methods.push_back({ func_name, func_docstring, func });
}


void extension_module::add_function(const string &func_name, std::function<py_object(py_tuple,py_dict)> func)
{
    add_function(func_name, "", func);
}


void extension_module::finalize()
{
    if (finalized)
	throw runtime_error("pyclops: double call to extension_module::finalize()");

    // FIXME(?): small memory leaks below.

    int n = module_methods.size();

    PyMethodDef *mdefs = (PyMethodDef *) malloc((n+1) * sizeof(PyMethodDef));
    memset(mdefs, 0, (n+1) * sizeof(PyMethodDef));

    for (int i = 0; i < n; i++) {
	mdefs[i].ml_name = strdup(module_methods[i].func_name.c_str());
	mdefs[i].ml_meth = make_kwargs_cfunction(module_methods[i].func);
	mdefs[i].ml_flags = METH_VARARGS | METH_KEYWORDS;
	mdefs[i].ml_doc = strdup(module_methods[i].func_docstring.c_str());
    }
    
    PyObject *m = Py_InitModule3(strdup(module_name.c_str()),
				 mdefs,
				 strdup(module_docstring.c_str()));

    if (!m)
	throw runtime_error("pyclops: Py_InitModule3() failed");

    this->finalized = true;
}


}  // namespace pyclops
