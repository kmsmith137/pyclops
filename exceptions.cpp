#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


void py_tuple::_throw(const char *where)
{
    if (!where)
	where = "pyclops: internal error";
    throw runtime_error(string(where) + ": object was not a tuple as expected");
}

void py_dict::_throw(const char *where)
{
    if (!where)
	where = "pyclops: internal error";
    throw runtime_error(string(where) + ": object was not a dictionary as expected");
}

void py_array::_throw(const char *where)
{
    if (!where)
	where = "pyclops: internal error";
    throw runtime_error(string(where) + ": object was not an array as expected");
}

void py_type::_throw(const char *where)
{
    if (!where)
	where = "pyclops: internal error";
    throw runtime_error(string(where) + ": object was not a type object as expected");
}

void py_weakref::_throw(const char *where)
{
    if (!where)
	where = "pyclops: internal error";
    throw runtime_error(string(where) + ": object was not a weakref object as expected");
}


// -------------------------------------------------------------------------------------------------


pyerr_occurred::pyerr_occurred(const char *where_)
{
    this->where = where_ ? where_ : "pyclops internal error";

    if (!PyErr_Occurred())
	throw std::runtime_error(std::string(where) + ": pyerr_occurred constructor called, but PyErr_Occurred() returned false!");
}

// virtual
char const *pyerr_occurred::what() const noexcept
{
    if (!PyErr_Occurred())
	throw std::runtime_error(std::string(where) + ": pyerr_occurred::what() called, but PyErr_Occurred() returned false!");

    // FIXME: get string corresponding to python exception.
    throw std::runtime_error("An exception was thrown in the python interpreter, and its exception text must remain shrouded in mystery for now");
}


// This is called whenever we want to "swallow" a C++ exception, but propagate it into the python error indicator.
void set_python_error(const std::exception &e) noexcept
{
    if (PyErr_Occurred())
	return;

    const pyerr_occurred *pyerr = dynamic_cast<const pyerr_occurred *> (&e);

    if (pyerr) {
	PyErr_SetString(PyExc_RuntimeError, "pyclops: internal error: pyerr_occurred was thrown, but PyErr_Occurred() returned false!");
	return;
    }

    // TODO: currently we use PyExc_RuntimeError here, but it would be better
    // to use an exception type which depends on the C++ exception type.

    PyErr_SetString(PyExc_RuntimeError, e.what());
}


}  // namespace pyclops
