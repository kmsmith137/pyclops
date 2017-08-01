- segfault from interpreter??

- how does mcpp_pybase get initialized?  does there need to be a pyclops.so python module?

- py_array member functions for things like FromAny(), Writeback()?

- type_error exception (and think more generally about exceptions)

- use general-purpose type-wrapping logic (once it exists) for mcpp_pybase

  this should have the side effect of cleaning up some code ugliness (e.g. shared_ptr<> *)  
  
- "pretty" compile-time errors

    - if from_python<mcpp_arrays::rs_array<T>> is called with an invalid type T
      (note: need a type_trait in mcpp_arrays!)

- 'where'
    - toy_wrap
    - do some simple tests
    - pyerr_occurred()

- low priority: overloads
