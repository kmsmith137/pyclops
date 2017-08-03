#ifndef _PYCLOPS_HPP
#define _PYCLOPS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with -std=c++11"
#endif

#include <mcpp_arrays.hpp>

#include "pyclops/py_object.hpp"
#include "pyclops/py_tuple.hpp"
#include "pyclops/py_dict.hpp"
#include "pyclops/py_array.hpp"
#include "pyclops/py_type.hpp"

#include "pyclops/converters.hpp"
#include "pyclops/extension_type.hpp"
#include "pyclops/extension_module.hpp"
#include "pyclops/functional_wrappers.hpp"

#endif  // _PYCLOPS_HPP
