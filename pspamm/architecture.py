from importlib import import_module

def init():
    global arch
    global generator
    global operands
    arch = "none"
    generator = None
    operands = None

    

#https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname

def get_class( kls ):
    return import_module(kls)
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


