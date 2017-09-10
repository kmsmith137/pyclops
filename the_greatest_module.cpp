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
    
    ss << "array(npy_type=" << a.type()
       << " (" << npy_typestr(a.type())
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


// For simplicity (but not efficiency), we force the array to be contiguous and convert to double.
static double sum_array(in_carray<double> a)
{
    npy_intp size = a.size();
    
    double ret = 0.0;
    for (npy_intp i = 0; i < size; i++)
	ret += a.data[i];

    return ret;
}


// Currently has to be called from python as make_array((2,3,4)).
static py_object make_array(py_tuple dims)
{
    int ndims = dims.size();
    vector<npy_intp> shape(ndims);

    for (int i = 0; i < ndims; i++)
	shape[i] = converter<ssize_t>::from_python(dims.get_item(i));

    py_array ret = py_array::make(ndims, &shape[0], npy_type<int>::id);

    if (ret.ncontig() != ndims)
	throw runtime_error("make_array: array was not fully contiguous as expected");

    int size = ret.size();
    int *data = (int *) ret.data();

    for (int i = 0; i < size; i++)
	data[i] = 100 * i;

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
	auto p = extension_type<X>::shared_ptr_from_python(X_type.tobj, obj, where);
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
	return extension_type<X>::shared_ptr_from_python(X_type.tobj, obj, where);
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

    virtual ssize_t f(ssize_t n) = 0;

    string get_name() { return name; }
    ssize_t f_cpp(ssize_t n) { return f(n); }   // Forces call to f() to go through C++ code
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


// Represents a Base which has been subclassed from python.
struct PyBase : public Base {
    PyBase(const py_object &self, const string &name) : 
	Base(name)
    { }

    virtual ssize_t f(ssize_t n) override
    {
	py_tuple args = py_tuple::make(n);
	py_object ret = _py_upcall(this, "f", args);
	return converter<ssize_t>::from_python(ret, "Base.f");
    }
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
	return extension_type<Base>::shared_ptr_from_python(Base_type.tobj, obj, where);
    }

    static py_object to_python(const shared_ptr<Base> &x)
    {
	return extension_type<Base>::to_python(Base_type.tobj, x);
    }
};


static shared_ptr<Base> g_Base;

static void set_global_Base(shared_ptr<Base> b) { g_Base = b; }
static void clear_global_Base() { g_Base.reset(); }
static ssize_t f_global_Base(ssize_t n) { return g_Base ? g_Base->f(n) : 0; }


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

    auto X_constructor1 = [](py_object self, ssize_t i) { return new X(i); };
    auto X_constructor2 = std::function<X* (py_object,ssize_t)> (X_constructor1);

    X_type.add_constructor(toy_wrap_constructor(X_constructor2));
    X_type.add_method("get", "get!", toy_wrap(&X::get));

    std::function<ssize_t(const X *x)> X_xget = [](const X *x) { return x->x; };
    X_type.add_property("xget", "get x!", X_xget);

    std::function<ssize_t& (X *x)> X_xset = [](X *x) -> ssize_t & { return x->x; };
    X_type.add_property("xset", "set x!", X_xset);

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
    Base_type.add_method("f_cpp", "forces call to f() to go through C++", toy_wrap(&Base::f_cpp));
    Base_type.add_pure_virtual("f", "a pure virtual function", toy_wrap(&Base::f));

    // This python constructor allows a python subclass to override the pure virtual function f().
    auto Base_constructor1 = [](py_object self, string name) { return new PyBase(self, name); };
    auto Base_constructor2 = std::function<Base* (py_object,string)> (Base_constructor1);
    Base_type.add_constructor(toy_wrap_constructor(Base_constructor2));

    m.add_type(Base_type);

    m.add_function("make_derived", toy_wrap(make_derived));
    m.add_function("set_global_Base", toy_wrap(set_global_Base));
    m.add_function("clear_global_Base", toy_wrap(clear_global_Base));
    m.add_function("f_global_Base", toy_wrap(f_global_Base));

    m.finalize();
}
