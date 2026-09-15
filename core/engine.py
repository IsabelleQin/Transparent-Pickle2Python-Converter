import _compat_pickle
import io
import pickle
import sys

from copyreg import _extension_cache, _inverted_registry
from core.configs import indent
from pickle import HIGHEST_PROTOCOL
from pickle import PROTO, NEWOBJ, STACK_GLOBAL, NEWOBJ_EX, REDUCE, APPEND, APPENDS, \
    SETITEM, SETITEMS, ADDITEMS, BUILD, STOP
from pickle import UnpicklingError

# Represent objects
class ObjRep:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return self.name

# Unpickling machinery
class Convert(pickle._Unpickler):
    def __init__(self, file, *args, **kwargs):
        # Unique ID for globals
        super().__init__(file, *args, **kwargs)
        self.uid = 0
        self.source_code = io.StringIO()
        # Default settings
        self.source_code.write(f"proto = {self.proto}\n")
        self.source_code.write(f"fix_imports = {self.fix_imports}\n")
        
    def log(self, name, stat):
        self.source_code.write(f"{name} = {stat}\n")

    def load_proto(self):
        proto = self.read(1)[0]
        if not 0 <= proto <= HIGHEST_PROTOCOL:
            raise ValueError("unsupported pickle protocol: %d" % proto)
        self.proto = proto
        self.log("proto", proto)
    pickle._Unpickler.dispatch[PROTO[0]] = load_proto

    # INST and OBJ differ only in how they get a class object.  It's not
    # only sensible to do the rest in a common routine, the two routines
    # previously diverged and grew different bugs.
    # klass is the class to instantiate, and k points to the topmost mark
    # object, following which are the arguments for klass.__init__.    
    def _instantiate(self, klass, args):
        name = f"obj{self.uid}"
        self.uid += 1
        self.log(name, f"_instantiate({klass}, {args})")
        self.stack.append(ObjRep(name))

    def load_newobj(self):
        args = self.stack.pop()
        cls = self.stack.pop()
        name = f"obj{self.uid}"
        self.uid += 1
        self.log(name, f"{cls}.__new__({cls}, *{args})")
        self.stack.append(ObjRep(name))
    pickle._Unpickler.dispatch[NEWOBJ[0]] = load_newobj

    def load_stack_global(self):
        name = self.stack.pop()
        module = self.stack.pop()
        if type(name) not in (str, ObjRep) or type(module) not in (str, ObjRep):
            raise UnpicklingError("STACK_GLOBAL requires str")
        self.append(self.find_class(module, name))
    pickle._Unpickler.dispatch[STACK_GLOBAL[0]] = load_stack_global

    def load_newobj_ex(self):
        kwargs = self.stack.pop()
        args = self.stack.pop()
        cls = self.stack.pop()
        name = f"obj{self.uid}"
        self.uid += 1
        self.log(name, f"{cls}.__new__({cls}, *{args}, **{kwargs})")
        self.stack.append(ObjRep(name))
    pickle._Unpickler.dispatch[NEWOBJ_EX[0]] = load_newobj_ex

    def get_extension(self, code):
        # Solve this during runtime
        glob = f"glob{self.uid}"
        self.uid += 1
        self.log(glob, f"get_extension({code}, proto, fix_imports)")
        self.stack.append(ObjRep(glob))

    def find_class(self, module, name):
        # Subclasses may override this.
        glob = f"glob{self.uid}"
        self.uid += 1
        self.log(glob, f"find_class({module!r}, {name!r}, proto, fix_imports)")
        return ObjRep(glob)

    def load_reduce(self):
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        name = f"var{self.uid}"
        self.uid += 1
        self.log(name, f"{func}(*{args})")
        stack[-1] = ObjRep(name)
    pickle._Unpickler.dispatch[REDUCE[0]] = load_reduce

    def load_append(self):
        stack = self.stack
        value = stack.pop()
        list_obj = stack[-1]
        if isinstance(list_obj, ObjRep):
            self.source_code.write(f"{list_obj}.append({value})\n")
        else:
            list_obj.append(value)
    pickle._Unpickler.dispatch[APPEND[0]] = load_append

    def load_appends(self):
        items = self.pop_mark()
        list_obj = self.stack[-1]
        if isinstance(list_obj, ObjRep):
            self.log("items", items)
            self.source_code.write(f"try: {list_obj}.extend(items)\n")
            self.source_code.write(f"except:\n{indent}for item in items: \n{indent}{list_obj}.append(item)\n")
            return
        try:
            extend = list_obj.extend
        except AttributeError:
            pass
        else:
            extend(items)
            return
        # Even if the PEP 307 requires extend() and append() methods,
        # fall back on append() if the object has no extend() method
        # for backward compatibility.
        append = list_obj.append
        for item in items:
            append(item)
    pickle._Unpickler.dispatch[APPENDS[0]] = load_appends

    def load_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        dict_obj = stack[-1]
        if isinstance(dict_obj, ObjRep):
            self.log(f"{dict_obj}[{key!r}]", f"{value}")
        else:
            dict_obj[key] = value
    pickle._Unpickler.dispatch[SETITEM[0]] = load_setitem

    def load_setitems(self):
        items = self.pop_mark()
        dict_obj = self.stack[-1]
        if isinstance(dict_obj, ObjRep):
            self.log("items", items)
            self.source_code.write(f"for i in range(0, len(items), 2): \n{indent}{dict_obj}[items[i]] = items[i+1]\n")
        else:
            for i in range(0, len(items), 2):
                dict_obj[items[i]] = items[i + 1]
    pickle._Unpickler.dispatch[SETITEMS[0]] = load_setitems

    def load_additems(self):
        items = self.pop_mark()
        set_obj = self.stack[-1]
        if isinstance(set_obj, ObjRep):
            self.log("items", items)
            self.source_code.write(f"if isinstance({set_obj}, set): {set_obj}.update(items)\n")
            self.source_code.write(f"else:\n{indent}for item in items: {set_obj}.add(item)\n")
            return
        if isinstance(set_obj, set):
            set_obj.update(items)
        else:
            add = set_obj.add
            for item in items:
                add(item)
    pickle._Unpickler.dispatch[ADDITEMS[0]] = load_additems

    def load_build(self):
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        assert isinstance(inst, ObjRep)
        # Do this at runtime
        self.source_code.write(f"build({inst}, {state})\n")
    pickle._Unpickler.dispatch[BUILD[0]] = load_build

    def load_stop(self):
        value = self.stack[-1]
        self.source_code.write(f"return {value}\n")
        raise pickle._Stop(self.source_code)
    pickle._Unpickler.dispatch[STOP[0]] = load_stop

def convert(file, *, fix_imports=True, encoding="ASCII", errors="strict",
          buffers=None):
    return Convert(file, fix_imports=fix_imports, buffers=buffers,
                     encoding=encoding, errors=errors).load()

# Functions for execution
def _instantiate(klass, args):
    if (args or not isinstance(klass, type) or
        hasattr(klass, "__getinitargs__")):
        try:
            value = klass(*args)
        except TypeError as err:
            raise TypeError("in constructor for %s: %s" %
                            (klass.__name__, str(err)), sys.exc_info()[2])
    else:
        value = klass.__new__(klass)
    return value

def build(inst, state):
    setstate = getattr(inst, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    slotstate = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slotstate = state
    if state:
        inst_dict = inst.__dict__
        intern = sys.intern
        for k, v in state.items():
            if type(k) is str:
                inst_dict[intern(k)] = v
            else:
                inst_dict[k] = v
    if slotstate:
        for k, v in slotstate.items():
            setattr(inst, k, v)

def find_class(module, name, proto, fix_imports):
    # Subclasses may override this.
    if proto < 3 and fix_imports:
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        elif module in _compat_pickle.IMPORT_MAPPING:
            module = _compat_pickle.IMPORT_MAPPING[module]
    __import__(module, level=0)
    if proto >= 4:
        return pickle._getattribute(sys.modules[module], name)[0]
    else:
        return getattr(sys.modules[module], name)

def get_extension(code, proto, fix_imports):
    nil = []
    obj = _extension_cache.get(code, nil)
    if obj is not nil:
        return obj
    key = _inverted_registry.get(code)
    if not key:
        if code <= 0: # note that 0 is forbidden
            # Corrupt or hostile pickle.
            raise UnpicklingError("EXT specifies code <= 0")
        raise ValueError("unregistered extension code %d" % code)
    obj = find_class(*key, proto, fix_imports)
    _extension_cache[code] = obj
    return obj
