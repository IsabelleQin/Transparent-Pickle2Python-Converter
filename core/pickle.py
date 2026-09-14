import sys
import textwrap

from core.engine import convert
from core.configs import *
import importlib.util

def generate(
          path: str, 
          code_path: str, 
          fix_imports: bool = True, 
          encoding: str = "ASCII", 
          errors: str = "strict", 
          buffers = None
          ):
    with open(path, "rb") as f:
        result = convert(f, fix_imports=fix_imports, encoding=encoding, 
                    errors=errors, buffers=buffers)
    source_code = "from core.engine import _instantiate, build, find_class, get_extension\n"
    source_code += "def main():\n"
    source_code += textwrap.indent(result.getvalue(), indent)
    with open(code_path, "w") as f:
        f.write(source_code)
    return source_code

def execute(code_path: str):
    spec = importlib.util.spec_from_file_location("pypickle", code_path)
    pypickle = importlib.util.module_from_spec(spec)
    sys.modules["pypickle"] = pypickle
    spec.loader.exec_module(pypickle)
    return pypickle.main()