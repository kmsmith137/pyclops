#define NO_IMPORT_ARRAY
#include "pyclops/core.hpp"

#include <vector>
#include <algorithm>
#include <unordered_map>

// To enable some verbose debugging output, turn this on!
#define PYCLOPS_MASTER_HASH_TABLE_DEBUG 0

using namespace std;

namespace pyclops {
#if 0
}  // emacs pacifier
#endif


// Global hash table is declared here.
// Note: the hash table does not hold a reference!
static unordered_map<const void*, PyObject*> master_hash_table;


void master_hash_table_add(const void *cptr, PyObject *pptr)
{
#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_add() called: " << cptr << " -> " << pptr << endl;
#endif

    if (!cptr || !pptr)
	throw runtime_error("pyclops: internal error: null pointer in master_hash_table_add()");

    auto p = master_hash_table.find(cptr);
    if (p != master_hash_table.end())
	throw runtime_error("pyclops: internal error in master_hash_table_add(): entry already exists");

    master_hash_table[cptr] = pptr;

#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_add(): success.  Current hash table follows." << endl;
    master_hash_table_print();
#endif
}


void master_hash_table_remove(const void *cptr, PyObject *pptr)
{
#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_remove() called: " << cptr << " -> " << pptr << endl;
#endif

    if (!cptr || !pptr)
	throw runtime_error("pyclops: internal error: null pointer in master_hash_table_remove()");

    auto p = master_hash_table.find(cptr);
    
    if (p == master_hash_table.end())
	throw runtime_error("pyclops: internal error in master_hash_table_remove(): entry not found");
    if (p->second != pptr)
	throw runtime_error("pyclops: internal error in master_hash_table_remove(): PyObject mismatch");

    master_hash_table.erase(p);

#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_remove(): success.  Current hash table follows." << endl;
    master_hash_table_print();
#endif
}


PyObject *master_hash_table_query(const void *cptr)
{
#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_query() called: " << cptr << endl;
#endif

    if (!cptr)
	throw runtime_error("pyclops: internal error: null pointer in master_hash_table_query()");

    auto p = master_hash_table.find(cptr);
    PyObject *ret = (p != master_hash_table.end()) ? p->second : NULL;

#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_query() returning " << ret << endl;
#endif

    return ret;
}


void master_hash_table_deleter(const void *cptr)
{
#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_deleter() called: " << cptr << endl;
#endif

    if (!cptr)
	throw runtime_error("pyclops: internal error: null pointer in master_hash_table_deleter()");

    PyObject *op = NULL;

    // This extra block ensures that p's destructor is called before Py_DECREF.
    // This is because Py_DECREF() can remove the entry from the hash table, invalidating p.
    // I doubt is actually necessary, but just being paranoid... !

    do {
	auto p = master_hash_table.find(cptr);
	if (p == master_hash_table.end())
	    throw runtime_error("pyclops: internal error in master_hash_table_remove(): couldn't find entry");
	op = p->second;
    } while (0);

#if PYCLOPS_MASTER_HASH_TABLE_DEBUG
    cout << "master_hash_table_deleter() found " << op << " , calling PyDECREF()" << endl;
#endif

    Py_XDECREF(op);
}


// Suboptimal implementation, intended for debugging.
void master_hash_table_print()
{
    vector<pair<const void*,PyObject*>> v;

    for (const auto &p: master_hash_table)
	v.push_back({p.first, p.second});

    sort(v.begin(), v.end());
    
    for (const auto &p: v)
	cout << "    " << p.first << " -> " << p.second << "\n";
}


}  // namespace pyclops
