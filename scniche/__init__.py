import sys
from importlib.metadata import version

from . import datasets
from . import plot as pl
from . import preprocess as pp
from . import trainer as tr
from . import analysis as al

# has to be done at the end, after everything has been imported
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tr", "pp", "pl", "al"]})

__version__ = version("scniche")

__all__ = ["__version__", "datasets", "tr", "pp", "pl", "al"]

