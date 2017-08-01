#include <sstream>
#include <iostream>
#include "pyclops.hpp"

using namespace std;
using namespace pyclops;


static ssize_t add(ssize_t x, ssize_t y) { return x+y; }


static string describe_array(py_array a)
{
    stringstream ss;

    int ndim = a.ndim();
    npy_intp *shape = a.shape();
    npy_intp *strides = a.strides();
    
    ss << "array(npy_type=" << a.npy_type()
       << " (" << npy_typestr(a.npy_type())
       << "), shape=(";

    for (int i = 0; i < ndim; i++) {
	if (i > 0) ss << ",";
	ss << shape[i];
    }

    ss << "), strides=(";

    for (int i = 0; i < ndim; i++) {
	if (i > 0) ss << ",";
	ss << strides[i];
    }

    ss << "), itemsize=" << a.itemsize()
       << ")";

    return ss.str();
}


PyMODINIT_FUNC initthe_greatest_module(void)
{
    import_array();

    cmodule m("the_greatest_module", "The greatest!");

    m.add_function("add", 
		   "Addition, baby",
		   toy_wrap(std::function<ssize_t(ssize_t,ssize_t)> (add)));

    m.add_function("describe_array",
		   toy_wrap(std::function<string(py_array)> (describe_array)));

    m.finalize();
}
