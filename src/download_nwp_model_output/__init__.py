"""Download specified NWP model data.

A quick conversion of my images-on-demand python/django code to just
save the downloaded data.  It appears GFS, NAM, and ECMWF work.  I
added code for NAVGEM but it appears their update cycle is a bit
slower.

At present, this is designed to be run in a for loop as the files become
available, with cron determining the proper run times.  The script
will try to find the most recent model run likely to have data and
download the variables I used to make plots in my server.

"""
try:
    from importlib.metadata import version  # type: ignore
except ImportError:
    from importlib_metadata import version  # type: ignore

from .nwp_models import NWP_MODELS

__version__ = version("weather-data-downloader")

__all__ = ["NWP_MODELS", "__version__"]
