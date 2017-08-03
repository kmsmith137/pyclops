#!/usr/bin/env python

import numpy as np
import numpy.random
import pyclops
import the_greatest_module

print 'Should be 15:', the_greatest_module.add(5,10)
print the_greatest_module.describe_array(np.zeros((3,4,5), dtype=np.float32))

a = numpy.random.uniform(size=(3,4,5))
s1 = the_greatest_module.sum_array(a)
s2 = a.sum()
print 'The following should be equal:', s1, s2

a = the_greatest_module.make_array((2,3,4))
print 'Output of make_array() follows'
print a

print
print 'Calling X()'
x = the_greatest_module.X(23)
print 'Calling X.get()'
i = x.get()
print 'Should be 23:', i
print 'Deleting'
del x

print 'Calling make_X()'
x = the_greatest_module.make_X(24)
print 'Calling X.get()'
i = x.get()
print 'Should be 24:', i
print 'Calling get_X()'
i = the_greatest_module.get_X(x)
print 'Should be 24:', i
print 'Deleting'
del x

print 'Calling make_Xp()'
x = the_greatest_module.make_Xp(25)
print 'Calling clone_Xp()'
y = the_greatest_module.clone_Xp(x)
print 'Calling X.get()'
i = x.get()
print 'Should be 25:', i
print 'Calling get_Xp()'
i = the_greatest_module.get_Xp(y)
print 'Should be 25:', i
print 'Deleting x'
del x
print 'Deleting y'
del y

print 'All done!'
