#ifndef _PYCLOPS_EXTENSION_MODULE_HPP
#define _PYCLOPS_EXTENSION_MODULE_HPP

#include <string>
#include <vector>
#include <functional>

#include "core.hpp"
#include "extension_type.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


struct extension_module {
public:
    extension_module(const std::string &name, const std::string &docstring="");

    void add_function(const std::string &func_name, const std::string &func_docstring, std::function<py_object(py_tuple,py_dict)> func);
    void add_function(const std::string &func_name, std::function<py_object(py_tuple,py_dict)> func);   // empty docstring

    template<typename T, typename B>
    inline void add_type(extension_type<T,B> &type);

    // Registers module with the python interpreter (by calling Py_InitModule3())
    void finalize();

protected:
    const std::string module_name;
    const std::string module_docstring;

    // Reminder: a PyMethodDef is a (name, func, flags, docstring) quadruple.
    std::vector<PyMethodDef> module_methods;

    std::vector<PyTypeObject *> module_types;

    bool finalized = false;
};


template<typename T, typename B>
inline void extension_module::add_type(extension_type<T,B> &type)
{
    if (finalized)
	throw std::runtime_error("pyclops: extension_module::add_type() called after extension_module::finalize()");

    type.finalize();
    module_types.push_back(type.tobj);
}


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_MODULE_HPP
