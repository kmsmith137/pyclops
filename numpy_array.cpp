#define NO_IMPORT_ARRAY
#include "pyclops/internals.hpp"

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


const char *npy_typestr(int npy_type)
{
    // All numpy types listed in numpy/ndarraytypes.h

    switch (npy_type) {
	case NPY_BOOL: return "NPY_BOOL";
	case NPY_BYTE: return "NPY_BYTE";
	case NPY_UBYTE: return "NPY_UBYTE";
	case NPY_SHORT: return "NPY_SHORT";
	case NPY_USHORT: return "NPY_USHORT";
	case NPY_INT: return "NPY_INT";
	case NPY_UINT: return "NPY_UINT";
	case NPY_LONG: return "NPY_LONG";
	case NPY_ULONG: return "NPY_ULONG";
	case NPY_LONGLONG: return "NPY_LONGLONG";
	case NPY_ULONGLONG: return "NPY_ULONGLONG";
	case NPY_FLOAT: return "NPY_FLOAT";
	case NPY_DOUBLE: return "NPY_DOUBLE";
	case NPY_LONGDOUBLE: return "NPY_LONGDOUBLE";
	case NPY_CFLOAT: return "NPY_CFLOAT";
	case NPY_CDOUBLE: return "NPY_CDOUBLE";
	case NPY_CLONGDOUBLE: return "NPY_CLONGDOUBLE";
	case NPY_OBJECT: return "NPY_OBJECT";
	case NPY_STRING: return "NPY_STRING";
	case NPY_UNICODE: return "NPY_UNICODE";
	case NPY_VOID: return "NPY_VOID";
	case NPY_DATETIME: return "NPY_DATETIME";
	case NPY_TIMEDELTA: return "NPY_TIMEDELTA";
	case NPY_HALF: return "NPY_HALF";
	case NPY_NTYPES: return "NPY_NTYPES";
	case NPY_NOTYPE: return "NPY_NOTYPE";
	case NPY_CHAR: return "NPY_CHAR";
	case NPY_USERDEF: return "NPY_USERDEF";
    }
    
    return "unrecognized numpy type";
}


}  // namespace pyclops
