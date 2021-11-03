import os, traceback, sys

def PING(
   message='',
   printit=True,
   ntraceback=-1,
   skip_process=True,
   flush=True,
   exit=False,
   emphasis=0,
   **kw,
):
   return
   message = str(message) if message else '<no msg>'
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
   for i in range(emphasis):
      msg = '!' * 80 + '\n' + msg + '\n' + '!' * 80
   if printit:
      print(msg, flush=flush)
   if exit:
      sys.exit()
   return msg
