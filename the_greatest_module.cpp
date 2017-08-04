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


static void print_float(double x)
{
    cout << "print_float: " << x << endl;
}


// -------------------------------------------------------------------------------------------------


struct X {
    ssize_t x;    
    X(ssize_t x_) : x(x_) { cout << "    X::X(" << x << ") " << this << endl; }
    X(const X &x_) : x(x_.x) { cout << "    X::X(" << x << ") " << this << endl; }
    ~X() { cout << "    X::~X(" << x << ") " << this << endl; }
    ssize_t get() { return x; }
};


// Declare X type object.
static extension_type<X> X_type("X", "The awesome X class");


// Converters for working with X objects (not X& references).
// Using these will lead to many unnecessary copy constructors!
template<>
struct converter<X> {
    static X from_python(const py_object &obj, const char *where=nullptr)
    {
	auto p = extension_type<X>::from_python(X_type.tobj, obj, where);
	return *p;
    }

    static py_object to_python(const X &x)
    {
	auto p = make_shared<X> (x);
	return extension_type<X>::to_python(X_type.tobj, p);
    }
};


// Converters for working with shared_ptr<X> objects.
// Using these will lead to many shared_ptr<X>() copy constructors, but no X() copy constructors.
template<>
struct converter<shared_ptr<X>> {
    static shared_ptr<X> from_python(const py_object &obj, const char *where=nullptr)
    {
	return extension_type<X>::from_python(X_type.tobj, obj, where);
    }

    static py_object to_python(const shared_ptr<X> &x)
    {
	return extension_type<X>::to_python(X_type.tobj, x);
    }
};


// -------------------------------------------------------------------------------------------------



struct Base {
    const string name;
    Base(const string &name_) : name(name_) { }    

    string get_name() { return name; }
    virtual ssize_t f(ssize_t n) = 0;
};

// Helper function for Derived constructor.
static string _der_name(ssize_t m)
{
    stringstream ss;
    ss << "Derived(" << m << ")";
    return ss.str();
}

struct Derived : public Base {
    const ssize_t m;
    Derived(ssize_t m_) : Base(_der_name(m_)), m(m_) { }
    virtual ssize_t f(ssize_t n) override { return m+n; }
};


static shared_ptr<Base> make_derived(ssize_t m)
{
    return make_shared<Derived> (m);
}


// Declare Base type object
static extension_type<Base> Base_type("Base", "This base class has a pure virtual function.");


template<>
struct converter<shared_ptr<Base>> {
    static shared_ptr<Base> from_python(const py_object &obj, const char *where=nullptr)
    {
	return extension_type<Base>::from_python(Base_type.tobj, obj, where);
    }

    static py_object to_python(const shared_ptr<Base> &x)
    {
	return extension_type<Base>::to_python(Base_type.tobj, x);
    }
};


// -------------------------------------------------------------------------------------------------


PyMODINIT_FUNC initthe_greatest_module(void)
{
    import_array();

    extension_module m("the_greatest_module", "The greatest!");

    // ----------------------------------------------------------------------

    m.add_function("add", toy_wrap(add));
    m.add_function("describe_array", toy_wrap(describe_array));
    m.add_function("sum_array", toy_wrap(sum_array));
    m.add_function("make_array", toy_wrap(make_array));
    m.add_function("print_float", toy_wrap(print_float));

    auto get_basicsize = [](py_type t) -> ssize_t { return t.get_basicsize(); };
    auto make_tuple = []() -> py_tuple { return py_tuple::make(ssize_t(2), 3.5, string("hi")); };

    m.add_function("get_basicsize",
		   toy_wrap(std::function<ssize_t(py_type)> (get_basicsize)));

    m.add_function("make_tuple",
		   toy_wrap(std::function<py_tuple()> (make_tuple)));


    // ----------------------------------------------------------------------

    auto X_constructor1 = [](ssize_t i) { return make_shared<X> (i); };
    auto X_constructor2 = std::function<shared_ptr<X>(ssize_t)> (X_constructor1);

    X_type.add_constructor(toy_wrap_constructor(X_constructor2));
    X_type.add_method("get", "get!", toy_wrap(&X::get));
    m.add_type(X_type);

    auto make_X = [](ssize_t i) -> X { return X(i); };
    auto get_X = [](X x) -> ssize_t { return x.get(); };
    auto make_Xp = [](ssize_t i) -> shared_ptr<X> { return make_shared<X> (i); };
    auto get_Xp = [](shared_ptr<X> x) -> ssize_t { return x->get(); };
    auto clone_Xp = [](shared_ptr<X> x) -> shared_ptr<X> { return x; };

    m.add_function("make_X",
		   toy_wrap(std::function<X(ssize_t)> (make_X)));

    m.add_function("get_X",
		   toy_wrap(std::function<ssize_t(X)> (get_X)));

    m.add_function("make_Xp",
		   toy_wrap(std::function<shared_ptr<X>(ssize_t)> (make_Xp)));

    m.add_function("get_Xp",
		   toy_wrap(std::function<ssize_t(shared_ptr<X>)> (get_Xp)));

    m.add_function("clone_Xp",
		   toy_wrap(std::function<shared_ptr<X>(shared_ptr<X>)> (clone_Xp)));

    // ----------------------------------------------------------------------

    Base_type.add_method("get_name", "get the name!", toy_wrap(&Base::get_name));
    Base_type.add_method("f", "a pure virtual function", toy_wrap(&Base::f));

    // This doesn't really make sense.
    auto dummy_cons1 = []() -> shared_ptr<Base> { return make_derived(100); };
    auto dummy_cons2 = std::function<shared_ptr<Base>()> (dummy_cons1);
    Base_type.add_constructor(toy_wrap_constructor(dummy_cons2));

    m.add_function("make_derived", toy_wrap(make_derived));

    m.add_type(Base_type);

    m.finalize();
}
