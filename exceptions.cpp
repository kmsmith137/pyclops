#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"
#include <frameobject.h>

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


void py_tuple::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not a tuple as expected");
}

void py_dict::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not a dictionary as expected");
}

void py_list::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not a list as expected");
}

void py_array::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not an array as expected");
}

void py_type::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not a type object as expected");
}

void py_weakref::_throw(const char *where)
{
    if (!where)
	where = "pyclops";
    throw runtime_error(string(where) + ": object was not a weakref object as expected");
}


// -------------------------------------------------------------------------------------------------
//
// Here is something that seems like it should be part of the Python C-API but isn't!  
// A standalone function
//
//    shared_ptr<const char> get_exception_text(const char *where)
//
// which returns a C-style string containing a textual description of the current exception.
//
// We use C strings instead of std::string, so that we are guaranteed never to throw a C++ exception
// if the process is out of memory.  (In this case we would return "Out of Memory" for the exception
// text.)
//
// We use shared_ptr<const char> instead of (const char *) so that we can either return a malloc()-ed
// string, or a static string (in the latter case, the shared_ptr "deleter" would no-op).  Another
// advantage of shared_ptr<const char> is that we never need to worry about copy/move subtleties.
//
//   PyErr_Display()           [ pythonrun.c ]
//      PyTraceBack_Print()    [ traceback.c ]
//      parse_syntax_error()   [ pythonrun.c ]


// get_traceback_text(): helper for get_exception_text()
//
// Reference: PyTraceBack_Print() in Python/traceback.c
//
// FIXME room for improvement here
//   - It gets confused by callbacks, but works well enough to give the filename and line number.
//   - There is some signal handling stuff in PyTraceBack_Print() which I didn't try to emulate but probably should.
//   - I didn't dig up the actual line of text in the source file (by emalating _Py_DisplaySourceLine() in traceback.c), 
//     I just took whatever information was available in the traceback object.


static void dummy_deleter(const void *p) { }
static void free_deleter(const void *p) { free(const_cast<void *> (p)); }

static shared_ptr<const char> static_string(const char *s) { return shared_ptr<const char> (s, dummy_deleter); }
static shared_ptr<const char> malloced_string(const char *s) { return shared_ptr<const char> (s, free_deleter); }


static shared_ptr<const char> get_traceback_text(PyObject *ptraceback)
{
    if (!ptraceback)
	return static_string("No traceback available [traceback object was NULL, this is probably some sort of internal error]\n");
    if (!PyTraceBack_Check(ptraceback))
	return static_string("No traceback available [PyTraceBack_Check() returned false, this is probably some sort of internal error]\n");

    static const int tb_maxcount = 10;
    static const int tb_maxline = 1000;
    static const int tb_nheader = 100;
    static const int nalloc = tb_nheader + (tb_maxcount * tb_maxline) + 1;
    
    char *ret = (char *)malloc(nalloc);
    if (!ret)
	return static_string("No traceback available [out of memory]\n");
	
    memset(ret, 0, nalloc);
    int istr = snprintf(ret, tb_nheader, "Traceback (most recent call last):\n");
    int itb = 0;
    
    for (PyTracebackObject *tb = (PyTracebackObject *)ptraceback; tb; tb = tb->tb_next) {
	if (itb >= tb_maxcount)
	    break;

	const char *filename = PyString_AsString(tb->tb_frame->f_code->co_filename);
	const char *code = PyString_AsString(tb->tb_frame->f_code->co_name);
	int lineno = tb->tb_lineno;

	if (!filename)
	    filename = "[filename unavailable]";
	if (!code)
	    filename = "[code unavailable]";

	istr += snprintf(ret+istr, tb_maxline, "  File %s, line %d\n    %s\n", filename, lineno, code);
	itb++;
    }

    return malloced_string(ret);
}


// get_exception_type_text(): helper for get_exception_text()
//
// Reference: PyErr_Display() in Python/pythonrun.c
//
// FIXME room for improvement here
//   - In PyErr_Display(), there is a magic print_file_and_line() method which I didn't look into
//   - In PyErr_Display(), there is module lookup logic which I didn't look into

static shared_ptr<const char> get_exception_type_text(PyObject *ptype)
{
    if (!ptype)
	return static_string("[unknown exception type]");
    if (!PyExceptionClass_Check(ptype))
	return static_string("[unknown exception type]");

    char *s = PyExceptionClass_Name(ptype);
    if (!s)
	return static_string("[unknown exception type]");

    char *dot = strrchr(s, '.');
    if (dot != NULL)
	s = dot+1;
    
    const char *ret = strdup(s);
    if (!ret)
	return static_string("MemoryError");

    return malloced_string(ret);
}


// get_exception_type_text(): helper for get_exception_text()
//
// Reference: PyErr_Display() in Python/pythonrun.c
//
// FIXME room for improvement here
//   - In PyErr_Display(), there is a PyFile_WriteObject(..., Py_PRINT_RAW) path

static shared_ptr<const char> get_exception_value_text(PyObject *pvalue)
{
    if (!pvalue)
	return static_string("[exception value unavailable]");

    PyObject *s = PyObject_Str(pvalue);
    if (!s)
	return static_string("[exception value unavailable]");

    const char *s2 = PyString_AsString(s);
    if (!s2) {
	Py_XDECREF(s);
	return static_string("[unavailable exception value]");
    }
	
    char *ret = strdup(s2);
    if (!ret) {
	Py_XDECREF(s);
	return static_string("[out of memory]");
    }

    return malloced_string(ret);
}


// Using the three helper functions above, here is get_exception_text()
static shared_ptr<const char> get_exception_text()
{
    PyObject *ptype = PyErr_Occurred();  // returns borrowed reference
    
    if (!ptype)
	return static_string("Unknown exception occurred [maybe internal error? get_exception_text() was called, but PyErr_Occurred() returned NULL]");
    if (PyErr_GivenExceptionMatches(ptype, PyExc_MemoryError))
	return static_string("Out of memory");

    ptype = NULL;
    PyObject *pvalue = NULL;
    PyObject *ptraceback = NULL;

    // Caller of PyErr_Fetch() owns all 3 references.  Calling PyErr_Fetch() clears the interpreter error state!
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    shared_ptr<const char> ptype_text = get_exception_type_text(ptype);
    shared_ptr<const char> pvalue_text = get_exception_value_text(pvalue);
    shared_ptr<const char> ptraceback_text = get_traceback_text(ptraceback);

    // Undoes PyErr_Fetch(), by setting the interpreter error state and returning the 3 references.
    PyErr_Restore(ptype, pvalue, ptraceback);

    int nalloc = strlen(ptype_text.get()) + strlen(pvalue_text.get()) + strlen(ptraceback_text.get()) + 10;
    char *ret = (char *) malloc(nalloc);

    if (!ret)
	return static_string("Out of memory");

    snprintf(ret, nalloc, "%s%s: %s", ptraceback_text.get(), ptype_text.get(), pvalue_text.get());
    return malloced_string(ret);
}


// -------------------------------------------------------------------------------------------------


pyerr_occurred::pyerr_occurred(const char *where_)
{
    // FIXME 'where' argument currently ignored.
    this->msg = get_exception_text();
}

// virtual
char const *pyerr_occurred::what() const noexcept
{
    // FIXME do I need to worry about the lifetime of the pointer here?
    return msg ? msg.get() : "[unknown error]";
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
