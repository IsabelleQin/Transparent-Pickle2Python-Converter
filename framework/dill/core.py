
import dill
import sys
import importlib.util

from core import convert, Assemble
from core import find_class as find_class_pkl

import __main__ as _main_module
_main = _main_module

def generate(
          path: str, 
          code_path: str, 
          indent: int = 4,
          fix_imports: bool = True, 
          encoding: str = "ASCII", 
          errors: str = "strict", 
          buffers = None
          ):
    # The generation process is the same as Python Pickle
    with open(path, "rb") as f:
        raw_code = convert(f, indent=indent, fix_imports=fix_imports, buffers=buffers,
                     encoding=encoding, errors=errors)
    Assemble("dill", indent).output(raw_code, code_path)

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