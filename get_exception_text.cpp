
// -------------------------------------------------------------------------------------------------
//
// Here is something that seems like it should be part of the Python C-API but isn't!  
// A standalone function
//
//    const char *get_exception_text(int &free_flag)
//
// which returns a C-style string containing a textual description of the current exception.  If the 
// returned string was malloc()-ed, then 'free_flag' will be set, to tell the caller that free() is needed.
//
// We use C strings instead of std::string, so that we are guaranteed never to throw a C++ exception
// if the process is out of memory.  (In this case we return "Out of Memory" for the exception
// text.)  
//
//   PyErr_Display()           [ pythonrun.c ]
//      PyTraceBack_Print()    [ traceback.c ]
//      parse_syntax_error()   [ pythonrun.c ]


//
// get_traceback_text(): helper for get_exception_text()
//
// Reference: PyTraceBack_Print() in Python/traceback.c
//
// FIXME room for improvement here
//   - It gets confused by callbacks, but works well enough to give the filename and line number
//   - There is some signal handling stuff in PyTraceBack_Print() which I didn't try to emulate but probably should
//   - I didn't dig up the actual line of text in the source file (by emalating _Py_DisplaySourceLine() in traceback.c), 
//     I just took whatever information was available in the traceback object.
//
static const char *get_traceback_text(PyObject *ptraceback, int &free_flag)
{
    free_flag = 0;

    if (!ptraceback)
	return "No traceback available [traceback object was NULL, this is probably some sort of internal error]\n";
    if (!PyTraceBack_Check(ptraceback))
	return "No traceback available [PyTraceBack_Check() returned false, this is probably some sort of internal error]\n";

    static const int tb_maxcount = 10;
    static const int tb_maxline = 1000;
    static const int tb_nheader = 100;
    static const int nalloc = tb_nheader + (tb_maxcount * tb_maxline) + 1;
    
    char *ret = (char *)malloc(nalloc);
    if (!ret)
	return "No traceback available [out of memory]\n";
	
    free_flag = 1;
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

    return ret;
}


//
// get_exception_type_text(): helper for get_exception_text()
//
// Reference: PyErr_Display() in Python/pythonrun.c
//
// FIXME room for improvement here
//   - In PyErr_Display(), there is a magic print_file_and_line() method which I didn't look into
//   - In PyErr_Display(), there is module lookup logic which I didn't look into
//
static const char *get_exception_type_text(PyObject *ptype, int &free_flag)
{
    free_flag = 0;

    if (!ptype)
       return "[unknown exception type]";
    if (!PyExceptionClass_Check(ptype))
       return "[unknown exception type]";	

    char *s = PyExceptionClass_Name(ptype);
    if (!s)
       return "[unknown exception type]";

    char *dot = strrchr(s, '.');
    if (dot != NULL)
	s = dot+1;
    
    const char *ret = strdup(s);
    if (!ret)
	return "MemoryError";

    free_flag = 1;
    return ret;
}


//
// get_exception_type_text(): helper for get_exception_text()
//
// Reference: PyErr_Display() in Python/pythonrun.c
//
// FIXME room for improvement here
//   - In PyErr_Display(), there is a PyFile_WriteObject(..., Py_PRINT_RAW) path
//
static const char *get_exception_value_text(PyObject *pvalue, int &free_flag)
{
    free_flag = 0;

    if (!pvalue)
	return "[exception value unavailable]";

    PyObject *s = PyObject_Str(pvalue);
    if (!s)
	return "[exception value unavailable]";

    const char *s2 = PyString_AsString(s);
    if (!s2) {
	Py_XDECREF(s);
	return "[unavailable exception value]";
    }
	
    char *ret = strdup(s2);
    if (!ret) {
	Py_XDECREF(s);
	return "[out of memory]";
    }

    free_flag = 1;
    return ret;
}


// Using the three helper functions above, here is get_exception_text()
const char *get_exception_text(int &free_flag)
{
    free_flag = 0;

    PyObject *ptype = PyErr_Occurred();  // returns borrowed reference
    
    if (!ptype)
	return "Unknown exception occurred [maybe internal error? get_exception_text() was called, but PyErr_Occurred() returned NULL]";
    if (PyErr_GivenExceptionMatches(ptype, PyExc_MemoryError))
	return "Out of memory";  // special handling of out-of-memory case, to avoid any malloc()-ing

    ptype = NULL;
    PyObject *pvalue = NULL;
    PyObject *ptraceback = NULL;

    // Caller of PyErr_Fetch() owns all 3 references.  Calling PyErr_Fetch() clears the interpreter error state!
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    int ptype_free_flag = 0;
    int pvalue_free_flag = 0;
    int ptraceback_free_flag = 0;

    const char *ptype_text = get_exception_type_text(ptype, ptype_free_flag);
    const char *pvalue_text = get_exception_value_text(pvalue, pvalue_free_flag);
    const char *ptraceback_text = get_traceback_text(ptraceback, ptraceback_free_flag);

    // Undoes PyErr_Fetch(), by setting the interpreter error state and returning the 3 references.
    PyErr_Restore(ptype, pvalue, ptraceback);

    int nalloc = strlen(ptype_text) + strlen(pvalue_text) + strlen(ptraceback_text) + 10;
    char *ret = (char *) malloc(nalloc);

    if (ret) {
	snprintf(ret, nalloc, "%s%s: %s", ptraceback_text, ptype_text, pvalue_text);
	free_flag = 1;
    }
    else
	ret = (char *) "Out of memory";

    if (ptype_free_flag)
	free((void *) ptype_text);
    if (pvalue_free_flag)
	free((void *) pvalue_text);
    if (ptraceback_free_flag)
	free((void *) ptraceback_text);

    return ret;
}
