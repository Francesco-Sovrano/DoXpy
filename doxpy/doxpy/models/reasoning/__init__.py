import re
wh_elements = ['why','how','what','where','when','who','which','whose','whom']
wh_elements_regexp = re.compile('('+'|'.join(map(re.escape, wh_elements))+')', re.IGNORECASE)
is_not_wh_word = lambda x: re.match(wh_elements_regexp, x) is None # use match instead of search

#############

__all__ = []

import pkgutil
import inspect

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        __all__.append(name)