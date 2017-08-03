#ifndef _PYCLOPS_EXTENSION_MODULE_HPP
#define _PYCLOPS_EXTENSION_MODULE_HPP

#include <string>
#include <vector>
#include <functional>

#include "py_object.hpp"
#include "py_tuple.hpp"
#include "py_dict.hpp"
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

    // Registers module with the python interpreter (by calling Py_InitModule3())
    void finalize();

protected:
    const std::string module_name;
    const std::string module_docstring;

    // Reminder: a PyMethodDef is a (name, func, flags, docstring) quadruple.
    std::vector<PyMethodDef> module_methods;
    bool finalized = false;
};


}  // namespace pyclops

#endif  // _PYCLOPS_EXTENSION_MODULE_HPP
