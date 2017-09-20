#ifndef _PYCLOPS_HPP
#define _PYCLOPS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with -std=c++11"
#endif

#include "pyclops/core.hpp"
#include "pyclops/py_array.hpp"
#include "pyclops/py_list.hpp"
#include "pyclops/py_type.hpp"
#include "pyclops/py_weakref.hpp"

#include "pyclops/converters.hpp"
#include "pyclops/array_converters.hpp"
#include "pyclops/extension_type.hpp"
#include "pyclops/extension_module.hpp"
#include "pyclops/functional_wrappers.hpp"
#include "pyclops/virtual_function.hpp"

#endif  // _PYCLOPS_HPP
