import sys
import _compat_pickle
import pickle
from pickle import UnpicklingError
from copyreg import _extension_cache, _inverted_registry

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