#include <sstream>
#include <iostream>
#include <mcpp_arrays.hpp>
#include "pyclops.hpp"

using namespace std;
using namespace mcpp_arrays;
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
       << ")\n";

    return ss.str();
}


static double _sum(int ndim, const ssize_t *shape, const ssize_t *strides, const double *data)
{
    if (ndim == 0)
	return data[0];

    double ret = 0.0;
    for (int i = 0; i < shape[0]; i++)
	ret += _sum(ndim-1, shape+1, strides+1, data + i*strides[0]);

    return ret;
}


static double sum_array(rs_array<double> a)
{
    return _sum(a.ndim, a.shape, a.strides, a.data);
}


// Currently has to be called from python as make_array((2,3,4)).
static py_object make_array(py_tuple dims)
{
    int ndims = dims.size();
    vector<ssize_t> shape(ndims);

    for (int i = 0; i < ndims; i++)
	shape[i] = converter<ssize_t>::from_python(dims.get_item(i));

    rs_array<int> a(ndims, &shape[0]);

    if (a.ncontig != ndims)
	throw runtime_error("make_array: rs_array was not fully contiguous as expected");

    for (int i = 0; i < a.size; i++)
	a.data[i] = 100 * i;

    py_array ret = converter<rs_array<int>>::to_python(a);
    cout << "make_array returning: " << describe_array(ret) << endl;
    return ret;
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

    m.add_function("sum_array",
		   toy_wrap(std::function<double(rs_array<double>)> (sum_array)));

    m.add_function("make_array",
		   toy_wrap(std::function<py_object(py_tuple)> (make_array)));

    m.finalize();
}
