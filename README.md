# Transparent-Pickle2Python-Converter
Pickle deserialization decoupled. Converts Pickle binaries to executable Python code. 

## Usage

### Getting Started
Generate Python code for a Pickle test case:

```
python main.py generate pickle -c test/obfuscation.pkl -o source_code.py
```

The generated code looks like this:

```
# source_code.py
from core.engine import _instantiate, build, find_class, get_extension
def main():
    proto = 0
    fix_imports = True
    glob0 = find_class('_codecs', 'encode', proto, fix_imports)
    var1 = glob0(*('bf', 'rot_13'))
    glob2 = find_class('_codecs', 'encode', proto, fix_imports)
    var3 = glob2(*('flfgrz', 'rot_13'))
    glob4 = find_class(var1, var3, proto, fix_imports)
    var5 = glob4(*('echo Yippie!',))
    return var5
```

Execute the generated code with: 

```
python main.py execute pickle -o source_code.py
```

### Arguments
| Arg | Values | Decription |
| -------- | -------- | -------- |
| ```mode``` | ```["generate", "execute"]``` | Generate code from a Pickle-based checkpoint/Execute a generated Python script. |
| ```framework``` | ```["pickle", "dill", "torch", "nemo"]``` | Target generation/execution framework. |
| ```-c / --checkpoint``` | - | Path to the original Pickle-based checkpoint. |
| ```-o / --output_path``` | - | Path to the code output. |
| ```-mm / --model_module``` | - | Model module for NeMo models. |
| ```-mc / --model_class``` | - | Model class for NeMo models. |

### Corresponding Framework Versions
- dill=0.4.1
- torch=2.13.0
- nemo-toolkit=3.0.0

## Additional Comments
### What is Pickle? 
See (pickle — Python object serialization)[https://docs.python.org/3/library/pickle.html].

### Why Pickle?
It is a popular serialization format, but it is not safe. 

### What is the benefit of converting Pickle to Python code?
Pickle-based files are essentially binary blobs that are hard to comprehend directly. 
Detecting vulnerabilities can be hard! 
However, Python code is (usually) easier for programmers to comprehend. 
Converting Pickle binaries to Python code enables transparent auditing, such as static analysis. 

### What is the difference between this tool and other converters?
Our converter decouples the parse and reconstruct stages in a Pickle deserialization process and generates **executable** code! 
This means you can reconstruct a model directly with the generated code. 
Or, you may also customize and enhance the generated code. 

### Current limitations/TODO
1. PyTorch legacy TAR format is not supported. This format has been deprecated for a long time, and PyTorch no longer provides serialization support for it.
2. Configured deserialization (e.g., dill-based PyTorch model) is not supported at this time. 
3. Small batches of tests have been conducted on the PickleBall dataset. More tests on correctness and efficiency are needed.
