import os, traceback

def PING(
   message='',
   printit=True,
   ntraceback=-1,
   skip_process=True,
   flush=False,
):
   message = str(message)

   stack = traceback.extract_stack()
   framestrs = list()
   for iframe, frame in enumerate(stack[1:ntraceback]):  # 0 is ping func, skip
      path = frame[0].split('/')[-1].replace('.py', '')
      line = path + '.' + frame[2] + ':' + str(frame[1])
      if skip_process and line.startswith(('process.', 'popen_fork.', 'context.')):
         continue
      framestrs.append(line)
      # for ix, x in enumerate(frame):
      # print('   ', ix, x)
   msg = 'PING ' + message + ' FROM ' + '/'.join(framestrs)
   if printit:
      print(msg, flush=True)
   return msg

def foo():
   bar()

def bar():
   baz()

def baz():
   PING('hello from baz')

if __name__ == '__main__':
   print('-' * 80)
   foo()
   print('-' * 80)
