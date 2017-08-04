#!/usr/bin/env python

import the_greatest_module as tgm

class Derived2(tgm.Base):
    def __init__(self, m):
        tgm.Base.__init__(self, "Derived2(%d)" % m)
        self.m = m

    def f(self, n):
        print '    Derived2.f() called'
        return self.m * n

x = Derived2(5)
print x.get_name()
print x.f_cpp(10)
print x.f(20)

tgm.set_global_Base(x)
print tgm.f_global_Base(30)
del x
print tgm.f_global_Base(40)
