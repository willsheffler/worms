from worms.util import PING

def foo():
   return bar()

def bar():
   return baz()

def baz():
   return PING('hello from baz', printit=False)

if __name__ == '__main__':
   print(foo())
