#define NO_IMPORT_ARRAY
#include "pyclops/functional_wrappers.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// n_given = number of arguments (both keyword and non-keyword) specified by python caller
// n_min = minimum allowed number of arguments (= number of C++ arguments which do not have default_vals)
// n_max = maximum allowed number of arguments (= total C++ arguments, with or without default_vals)
//
// This function gets called if (n_given < nmin) or (n_given > n_max).
// I got a little carried away here, making the error messages exactly the same as the python interpreter!

std::runtime_error bad_arg_count(int n_given, int n_min, int n_max)
{
    stringstream ss;

    ss << "function takes ";

    if (n_min == n_max) {
	if (n_min == 0)
	    ss << "no arguments";
	else if (n_min == 1)
	    ss << "exactly 1 argument";
	else
	    ss << "exactly " << n_min << " arguments";
    }
    else if (n_given < n_min)
	ss << "at least " << n_min << " argument" << ((n_min > 1) ? "s" : "");
    else
	ss << "at most " << n_max << " argument" << ((n_max > 1) ? "s" : "");

    ss << " (" << n_given << " given)";

    // FIXME ends up as RuntimeError in python (should be TypeError).
    return runtime_error(ss.str());
}


std::runtime_error missing_arg(const char *arg_name)
{
    return runtime_error("function argument '" + string(arg_name) + "' must be specified");
}


void argname_hash::add(const char *sp)
{
    // This implementation is a little inefficient, but this won't matter in practice,
    // since add() is only called when a new python module is imported.
    
    string s(sp);

    auto p = hash.find(s);

    if (p == hash.end()) {
	hash[s] = curr_size++;
	return;  // success
    }
    
    // FIXME I think it makes more sense to throw an exception here (meaning that the module import will fail),
    // but pyclops currently can't handle an exception which is thrown during the module initialization process.
    // This will be fixed eventually, but it's nontrivial!

    cout << "pyclops: duplicate argname detected!  This is a problem in the C++ source code, and you'll need to recompile." << endl;
    duplicates_detected = true;
}


// The function vgetargskeywords() in the python interpreter (Python/getargs.c) is a good reference here.
void argname_hash::check(const py_dict &kwds, ssize_t nargs)
{
    // FIXME if duplicate argnames are specified, we're currently not throwing an exception during module
    // import (see comment above), so we need to detect this condition and throw an exception here.

    if (duplicates_detected)
	throw runtime_error("pyclops: duplicate argname detected!  This is a problem in the C++ source code, and you'll need to recompile.");

    PyObject *key = NULL;
    PyObject *val = NULL;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwds.ptr, &pos, &key, &val)) {
	if (!PyString_Check(key))
	    throw runtime_error("keywords must be strings");

	char *kp = PyString_AsString(key);
	
	// FIXME there is an unnecessary malloc/copy/free here (in the string constructor).
	// I'd like to remove this, but I think the hash table will need to be re-implemented by hand,
	// instead of using unordered_map, so it's nontrivial.

	string k(kp);

	auto p = hash.find(k);

	if (p == hash.end())
	    throw runtime_error("'" + k + "' is an invalid keyword argument for this function");

	if (p->second < nargs)
	    throw runtime_error("Argument '" + k  + "' given by name and position");
    }
}


}  // namespace pyclops
