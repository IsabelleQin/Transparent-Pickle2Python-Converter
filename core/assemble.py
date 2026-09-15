from typing import Literal, Union, List
from io import StringIO
import textwrap
import logging
import os

# Assemble the code pieces
class Assemble:
    def __init__(self, 
                 framework: Literal["pickle", "dill", "torch"], 
                 indent: int = 4
                 ):
        self.framework = framework
        self.indent = indent*" "

    def _get_imports(self):
        """
            Get import supports for the corresponding framework.
        """
        if self.framework == "pickle":
            return "from framework.pickle import _instantiate, build, find_class, get_extension\n"
        elif self.framework == "dill":
            header = "from core import _instantiate, build, get_extension\n"
            header += "from framework.dill import find_class\n"
            return header
        elif self.framework == "torch":
            # NeMo models depend on the PyTorch framework
            return "from framework.torch import _instantiate, build, find_class, get_extension, persistent_load\n"
        else:
            raise NotImplementedError # Should never reach this line

    def _structure_code(self, raw_code: Union[StringIO, List]):
        """
            Structure code blocks
        """
        if isinstance(raw_code, StringIO):
            code = textwrap.indent(raw_code.getvalue(), self.indent)
            return f"def main():\n{code}"
        elif isinstance(raw_code, List):
            # Always verify
            assert self.framework == "torch"
            funcs = ["magic_number", "protocol_version", "_sys_info", "main", "deserialized_storage_keys"]
            res = ""
            try:
                for i in range(5):
                    # Allow malformed PyTorch files, raise a warning though
                    code = textwrap.indent(raw_code[i].getvalue(), self.indent)
                    res += f"def {funcs[i]}():\n{code}\n"
                # The last line: object offset
                res += f"OBJ_OFFSET = {raw_code[5]}"
            except Exception as e:
                print(e)
            finally: return res
        else:
            raise TypeError(f"Generated code type invalid! Expecting StringIO or List but got {type(raw_code)}")

    def assemble(self, raw_code: Union[StringIO, List]):
        """
            Assemble the code only, no output
        """
        header = self._get_imports()
        code = self._structure_code(raw_code)
        return header + code

    def output(self, raw_code: Union[StringIO, List], code_path: str):
        code = self.assemble(raw_code)
        with open(code_path, "w") as f:
            f.write(code)
            logging.info(f"Code saved to {os.path.abspath(code_path)}")