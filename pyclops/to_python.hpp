#ifndef _PYCLOPS_TO_PYTHON_HPP
#define _PYCLOPS_TO_PYTHON_HPP

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// -------------------------------------------------------------------------------------------------
//
// to_python()


template<typename T> py_object to_python(T x);


// -------------------------------------------------------------------------------------------------
//
// Implementations


template<> inline py_object to_python(ssize_t x)
{
    return py_object::new_reference(PyInt_FromSsize_t(x));
}

template<> inline py_object to_python(double x)
{
    return py_object::new_reference(PyFloat_FromDouble(x));
}

// FIXME: how to make (const string &) work?
template<> inline py_object to_python(std::string x)
{
    return py_object::new_reference(PyString_FromString(x.c_str()));
}


}  // namespace pyclops

#endif  // _PYCLOPS_TO_PYTHON_HPP
