from deferred_import import deferred_import

main = deferred_import('worms.app.main')
stackapp = deferred_import('worms.app.stackapp')
plug_from_oligomer = deferred_import('worms.app.plug_from_oligomer')
from worms.app.simple import run_simple, output_simple
