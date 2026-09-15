import sys
from core import convert, Assemble
import importlib.util

def generate(
          path: str, 
          code_path: str, 
          indent: int = 4,
          fix_imports: bool = True, 
          encoding: str = "ASCII", 
          errors: str = "strict", 
          buffers = None
          ):
          
    with open(path, "rb") as f:
        raw_code = convert(f, indent=indent, fix_imports=fix_imports, buffers=buffers,
                     encoding=encoding, errors=errors)
    Assemble("pickle", indent).output(raw_code, code_path)

def execute(code_path: str):
    spec = importlib.util.spec_from_file_location("pypickle", code_path)
    pypickle = importlib.util.module_from_spec(spec)
    sys.modules["pypickle"] = pypickle
    spec.loader.exec_module(pypickle)
    return pypickle.main()