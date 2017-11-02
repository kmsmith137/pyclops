// Suggest #including pyclops first, to avoid gcc warning "_POSIX_C_SOURCE redefined"
#include "pyclops.hpp"

#include <sstream>
#include <iostream>

using namespace std;
using namespace pyclops;


static int add(int x, int y) { return x+y; }

static bool boolean_not(bool x) { return !x; }

static bool is_string(const py_object &x) { return x.is_string(); }
static string echo_string(const string &s) { return s + "!"; }

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


static void add_21(io_ncarray<float,2,1> a, double t)
{
    npy_intp *shape = a.shape();
    npy_intp *strides = a.strides();

    npy_intp m = shape[0];
    npy_intp n = shape[1];
    npy_intp s = strides[0] / sizeof(float);
    
    for (npy_intp i = 0; i < m; i++)
	for (npy_intp j = 0; j < n; j++)
	    a.data[i*s+j] += t;
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
    ssize_t get() const { return x; }
    static double sm(double x, double y) { return x+y; }   // example staticmethod
};


// Declare X type object.
static extension_type<X> X_type("X", "The awesome X class");

namespace pyclops {
    template<> struct xconverter<X> { static constexpr extension_type<X> *type = &X_type; };
}

// -------------------------------------------------------------------------------------------------


struct Base {
    const string name;
    Base(const string &name_) : name(name_) { }    
    virtual ~Base() { }

    string get_name() { return name; }

    // f() is pure virtual
    virtual ssize_t f(ssize_t n) = 0;
    ssize_t f_cpp(ssize_t n) { return f(n); }   // Forces call to f() to go through C++ code.

    // g() is virtual, but not pure virtual
    virtual ssize_t g(ssize_t n) { return 10*n; }
    ssize_t g_cpp(ssize_t n) { return g(n); }   // Forces call to g() to go through C++ code.
    
    // virtual function returning void
    virtual void h(const string &s) { cout << s << endl; }
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
    virtual ssize_t g(ssize_t n) override { return 20*n; }
};


// -------------------------------------------------------------------------------------------------

// Declare Base type object
static extension_type<Base> Base_type("Base", "This base class has a pure virtual function f() and a non-pure virtual g().");
static extension_type<Derived,Base> Derived_type("Derived", "This derived class overrides the pure virtual Base::f(), but does not override Base::g().", Base_type);

namespace pyclops {
    template<> struct xconverter<Base> { static constexpr extension_type<Base> *type = &Base_type; };
    template<> struct xconverter<Derived> { static constexpr extension_type<Derived,Base> *type = &Derived_type; };
}

// "Upcalling" base class.
struct PyBase : public Base {
    PyBase(const string &name) : 
	Base(name)
    { }
    
    // Pure virtual
    virtual ssize_t f(ssize_t n) override
    {
	pure_virtual_function<ssize_t> v(Base_type, this, "f");
	return v.upcall(n);
    }

    // Non pure virtual
    virtual ssize_t g(ssize_t n) override
    {
	virtual_function<ssize_t> v(Base_type, this, "g");
	if (v.exists)
	    return v.upcall(n);
	return Base::g(n);
    }

    // Non pure virtual
    virtual void h(const string &s) override
    {
	virtual_function<void> v(Base_type, this, "h");
	if (v.exists) v.upcall(s);
	else Base::h(s);
    }
}; 

// "Upcalling" derived class.
// FIXME: currently definitions of "upcalling" f(), g(), h() must be repeated between PyBase and PyDerived.
// Fixing this is nontrivial, since PyDerived is not a subclass of PyBase.  I made a note on my todo list 
// to think about it more!  It helps a little to define some helper functions (see example in rf_pipelines),
// but I'd like to find a better solution.

struct PyDerived : public Derived {
    PyDerived(ssize_t m_) : Derived(m_) { }

    // In class Derived, f() has been overridden, so it is virtual (not pure virtual).
    virtual ssize_t f(ssize_t n) override
    {
	virtual_function<ssize_t> v(Derived_type, this, "f");
	if (v.exists) return v.upcall(n);
	return Derived::f(n);
    }

    virtual ssize_t g(ssize_t n) override
    {
	virtual_function<ssize_t> v(Derived_type, this, "g");
	if (v.exists)
	    return v.upcall(n);
	return Derived::g(n);	
    }

    // Non pure virtual
    virtual void h(const string &s) override
    {
	virtual_function<void> v(Derived_type, this, "h");
	if (v.exists) v.upcall(s);
	else Derived::h(s);
    }    
};

static shared_ptr<Base> make_derived(ssize_t m)
{
    return make_shared<Derived> (m);
}

static shared_ptr<Derived> make_derived2(ssize_t m)
{
    return make_shared<Derived> (m);
}

static shared_ptr<Base> g_Base;
static void set_global_Base(shared_ptr<Base> b) { g_Base = b; }
static void clear_global_Base() { g_Base.reset(); }
static ssize_t f_global_Base(ssize_t n) { return g_Base ? g_Base->f(n) : 0; }



// -------------------------------------------------------------------------------------------------


static string f_kwargs(int a, int b, int c=2, int d=3)
{
    stringstream ss;
    ss << "a=" << a << ", b=" << b << ", c=" << c << ", d=" << d;
    return ss.str();
}

// -------------------------------------------------------------------------------------------------



PyMODINIT_FUNC initexample_module(void)
{
    import_array();

    extension_module m("example_module", "Example module");

    // ----------------------------------------------------------------------

    m.add_function("add", wrap_func(add, "x", "y"));
    m.add_function("boolean_not", wrap_func(boolean_not, "x"));
    m.add_function("is_string", wrap_func(is_string, "x"));
    m.add_function("echo_string", wrap_func(echo_string, "x"));
    m.add_function("describe_array", wrap_func(describe_array, "a"));
    m.add_function("sum_array", wrap_func(sum_array, "a"));
    m.add_function("make_array", wrap_func(make_array, "dims"));
    m.add_function("add_21", wrap_func(add_21, "a", "t"));
    m.add_function("print_float", wrap_func(print_float, "x"));

    std::function<ssize_t(py_type)> get_basicsize = [](py_type t) { return t.get_basicsize(); };
    std::function<py_tuple()> make_tuple = []() { return py_tuple::make(ssize_t(2), 3.5, string("hi")); };

    m.add_function("get_basicsize", wrap_func(get_basicsize, "t"));
    m.add_function("make_tuple", wrap_func(make_tuple));

    // ----------------------------------------------------------------------

    auto X_constructor1 = [](ssize_t i) { return new X(i); };
    auto X_constructor2 = std::function<X* (ssize_t)> (X_constructor1);

    X_type.add_constructor(wrap_constructor(X_constructor2, "i"));
    X_type.add_method("get", "get!", wrap_method(&X::get));

    std::function<ssize_t(X*)> X_get2 = [](X *x) { return x->get(); };
    X_type.add_method("get2", "get(), wrapped via std::function", wrap_method(X_get2));

    std::function<ssize_t(const X *x)> X_xget = [](const X *x) { return x->x; };
    X_type.add_property("xget", "get x!", X_xget);

    std::function<ssize_t& (X *x)> X_xset = [](X *x) -> ssize_t & { return x->x; };
    X_type.add_property("xset", "set x!", X_xset);

    // Example staticmethod.  Note wrap_func() here, not wrap_method().
    X_type.add_staticmethod("sm", "example staticmethod", wrap_func(&X::sm, "x", "y"));

    m.add_type(X_type);

    std::function<X(ssize_t)> make_X = [](ssize_t i) { return X(i); };
    std::function<ssize_t(X)> get_X = [](X x) { return x.get(); };
    std::function<shared_ptr<X>(ssize_t)> make_Xp = [](ssize_t i) { return make_shared<X> (i); };
    std::function<ssize_t(shared_ptr<X>)> get_Xp = [](shared_ptr<X> x) { return x->get(); };
    std::function<shared_ptr<X>(shared_ptr<X>)> clone_Xp = [](shared_ptr<X> x) { return x; };

    m.add_function("make_X", wrap_func(make_X, "i"));
    m.add_function("get_X", wrap_func(get_X, "x"));
    m.add_function("make_Xp", wrap_func(make_Xp, "i"));
    m.add_function("get_Xp", wrap_func(get_Xp, "x"));
    m.add_function("clone_Xp", wrap_func(clone_Xp, "x"));

    // ----------------------------------------------------------------------

    // Base, Derived.
    // For Derived, we don't wrap f(), g(), h(), etc. but we do wrap a constructor.

    std::function<Base* (const string &)> 
	Base_init = [](const string &name) { return new PyBase(name); };

    std::function<Derived* (ssize_t)>
	Derived_init = [](ssize_t m) { return new PyDerived(m); };

    Base_type.add_constructor(wrap_constructor(Base_init, "name"));
    Derived_type.add_constructor(wrap_constructor(Derived_init, "m"));

    Base_type.add_method("get_name", "get the name!", wrap_method(&Base::get_name));
    Base_type.add_method("f", "a pure virtual function", wrap_method(&Base::f, "n"));
    Base_type.add_method("f_cpp", "forces call to f() to go through C++", wrap_method(&Base::f_cpp, "n"));
    Base_type.add_method("g", "a virtual function, but not pure virtual", wrap_method(&Base::g, "n"));
    Base_type.add_method("g_cpp", "forces call to g() to go through C++", wrap_method(&Base::g_cpp, "n"));
    Base_type.add_method("h", "a virtual function returning void", wrap_method(&Base::h, "s"));

    m.add_type(Base_type);
    m.add_type(Derived_type);

    // These functions all operate on Base objects (not Derived).
    // Exception: make_derived2().
    m.add_function("make_derived", wrap_func(make_derived, "m"));
    m.add_function("make_derived2", wrap_func(make_derived2, "m"));
    m.add_function("set_global_Base", wrap_func(set_global_Base, "b"));
    m.add_function("clear_global_Base", wrap_func(clear_global_Base));
    m.add_function("f_global_Base", wrap_func(f_global_Base, "n"));

    // ----------------------------------------------------------------------

    m.add_function("f_kwargs", wrap_func(f_kwargs, "a", "b", kwarg("c",2), kwarg("d",3)));

    m.finalize();
}
