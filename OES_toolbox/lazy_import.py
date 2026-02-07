# import importlib
import importlib.util
import sys
import time

from OES_toolbox.logger import Logger

logger = Logger(instance=None,context={"class":"lazy_import"})

def lazy_import(name):
    tstart = time.perf_counter()
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    logger.debug(f"Lazy importing {name} in {(time.perf_counter()-tstart)*1000:.3g} ms")
    return module