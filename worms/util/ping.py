import os, traceback

def ping(message='', debug=True, printit=True, bounds=(1, -1)):
   message = str(message)
   if debug:

      stack = traceback.extract_stack()
      framestrs = list()
      for iframe, frame in enumerate(stack[bounds[0]:bounds[1]]):
         # print(iframe)
         # print(frame[0])
         # print(type(frame[0]))
         path = frame[0].split('/')[-1].replace('.py', '')
         line = path + '.' + frame[2] + ':' + str(frame[1])
         framestrs.append(line)
         # for ix, x in enumerate(frame):
         # print('   ', ix, x)
   msg = 'PING (' + message + ')' + '/'.join(framestrs)
   if printit:
      print(msg)
   return msg

def foo():
   bar()

def bar():
   baz()

def baz():
   ping('something about baz')

if __name__ == '__main__':
   print('-' * 80)
   foo()
   print('-' * 80)
