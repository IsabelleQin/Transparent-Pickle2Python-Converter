
import dill
import sys
import textwrap
import importlib.util

from core.configs import *
from core.engine import Convert
from core.engine import find_class as find_class_pkl
from io import BytesIO as StringIO

import __main__ as _main_module
_main = _main_module

class Unpickler(Convert):
    """python's Unpickler extended to interpreter sessions and more types"""
    _session = False

    def __init__(self, *args, **kwds):
        Convert.__init__(self, *args, **kwds)
        self._main = _main_module

    def load(self): #NOTE: if settings change, need to update attributes
        result = Convert.load(self)
        source_code = "from core.engine import _instantiate, build, get_extension\n"
        source_code += "from core.dill import find_class\n"
        source_code += "def main():\n"
        source_code += textwrap.indent(result.getvalue(), indent)
        return source_code
    load.__doc__ = Convert.load.__doc__
    pass

def generate(path: str, code_path: str, **kwds):
    with open(path, "rb") as f:
        source_code = Unpickler(f, **kwds).load()
    with open(code_path, "w") as f:
        f.write(source_code)
    return source_code

def find_class(module, name, proto, fix_imports):
    if (module, name) == ('__builtin__', '__main__'):
        return _main.__dict__ #XXX: above set w/save_module_dict
    elif (module, name) == ('__builtin__', 'NoneType'):
        return type(None) #XXX: special case: NoneType missing
    if module == 'dill.dill': module = 'dill._dill'
    return find_class_pkl(module, name, proto, fix_imports)

def execute(code_path: str, ignore=None):
    spec = importlib.util.spec_from_file_location("pypickle", code_path)
    pypickle = importlib.util.module_from_spec(spec)
    sys.modules["pypickle"] = pypickle
    spec.loader.exec_module(pypickle)
    res = pypickle.main()

    settings = dill.Pickler.settings
    # Ignore is used here 
    _ignore = settings['ignore'] if ignore is None else _ignore
    if type(res).__module__ == getattr(_main_module, '__name__', '__main__'):
        if not _ignore:
            # point obj class to main
            try: res.__class__ = getattr(_main, type(res).__name__)
            except (AttributeError,TypeError): pass # defined in a file
    #_main_module.__dict__.update(obj.__dict__) #XXX: should update globals ?
    return res