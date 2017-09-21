#!/usr/bin/env python

import numpy as np
import numpy.random
import the_greatest_module as tgm

print 'Should be 15:', tgm.add(5,10)
print tgm.describe_array(np.zeros((3,4,5), dtype=np.float32))

a = numpy.random.uniform(size=(3,4,5))
s1 = tgm.sum_array(a)
s2 = a.sum()
print 'The following should be equal:', s1, s2

a = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
print 'Should equal 21:', tgm.sum_array(a)

a = tgm.make_array((2,3,4))
print 'Output of make_array() follows'
print a

t = tgm.make_tuple()
print 'Output of make_tuple() follows'
print t

print
print 'Calling X()'
x = tgm.X(23)
print 'Calling X.get()'
i = x.get()
print 'Should be 23:', i
print 'Deleting'
del x

print 'Calling make_X()'
x = tgm.make_X(24)
print 'Calling X.get()'
i = x.get()
print 'Should be 24:', i
print 'Calling get_X()'
i = tgm.get_X(x)
print 'Should be 24:', i
print 'Deleting'
del x

print 'Calling make_Xp()'
x = tgm.make_Xp(25)
print 'Calling clone_Xp()'
y = tgm.clone_Xp(x)
print 'Calling X.get()'
i = x.get()
print 'Should be 25:', i
print 'Calling get_Xp()'
i = tgm.get_Xp(y)
print 'Should be 25:', i
print 'Deleting x'
del x
print 'Deleting y'
del y

class E(tgm.Derived):
    def f(self, m):
        return m+1

    def g(self, m):
        return m+2

    def h(self, s):
        print 'E.%s' % s

e = E(10)
print e
print e.f(100)
print e.g(100)
print e.h("hi")

print 'All done!'
