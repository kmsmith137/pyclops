#ifndef _PYCLOPS_INTERNALS_HPP
#define _PYCLOPS_INTERNALS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with -std=c++11"
#endif

#include "pyclops/core.hpp"
#include "pyclops/py_array.hpp"
#include "pyclops/py_list.hpp"
#include "pyclops/py_type.hpp"
#include "pyclops/py_weakref.hpp"

#include <sstream>
#include <iostream>

#include "pyclops/converters.hpp"
#include "pyclops/cfunction_table.hpp"
#include "pyclops/extension_type.hpp"
#include "pyclops/extension_module.hpp"
#include "pyclops/functional_wrappers.hpp"

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// This is called whenever we want to "swallow" a C++ exception, but propagate it into the python error indicator.
extern void set_python_error(const std::exception &e) noexcept;


}  // namespace pyclops

#endif  // _PYCLOPS_INTERNALS_HPP
