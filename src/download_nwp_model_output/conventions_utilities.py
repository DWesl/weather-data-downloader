# ~*~ coding: utf8 ~*~
"""Utility functions for convention compliance

Built around the Climate and Forecast (CF) and Attribute Conventions
for Data Discovert (ACDD) conventions.  Implementation taken from
"""
import datetime
import functools
import logging
import os
import socket
import subprocess
import sys
from pwd import getpwuid
from typing import Any, Dict, Hashable

try:
    from shlex import quote
except ImportError:
    from pipes import quote

import cmdline_provenance
import dateutil.tz
import xarray

try:
    FILE_DIR = os.path.abspath(os.path.dirname(__file__))
except Exception:
    FILE_DIR = os.path.abspath(".")
REPO_ROOT = os.path.abspath(FILE_DIR)

_LOGGER = logging.getLogger(__name__)
UTC = dateutil.tz.tzutc()

UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)

HOST = socket.gethostbyaddr(socket.gethostbyname(os.uname()[1]))[0]
MAIN_HOST = ".".join(HOST.split(".")[-3:])
COMMAND_LINE = " ".join(quote(arg) for arg in sys.argv)
INPUT_LOGS = {}


############################################################
# Save CF/ACDD compatible netCDF output
def save_nonsparse_netcdf(
    ds_to_save: xarray.Dataset,
    save_name: str,
    data_created: bool = True,
    data_modified: bool = True,
) -> None:
    """Save the dataset at the name.

    Parameters
    ----------
    ds_to_save: xarray.Dataset
    save_name: str
    data_created: bool
        Was the dataset created here?  Is this a new variable
        calculated from other variables?
    data_modified: bool
        Was the dataset modified?  Was there averaging, reformatting,
        or data QC/QA?
    """
    cf_attrs = global_attributes_dict()
    if not data_created:
        del cf_attrs["date_created"]
    else:
        if not data_modified:
            del cf_attrs["date_modified"]
    ds_to_save.attrs.update(
        {
            key: value
            for key, value in cf_attrs.items()
            if key not in ds_to_save.attrs or "date" in key
        }
    )
    if "history" not in ds_to_save.attrs:
        ds_to_save.attrs["history"] = get_output_log()
    # I may need to rethink the fill value with integer datasets
    encoding = {
        name: {"zlib": True, "_FillValue": -9.99e9} for name in ds_to_save.data_vars
    }  # type: Dict[Hashable, Dict[str, Any]]
    encoding.update(
        {name: {"zlib": True, "_FillValue": None} for name in ds_to_save.coords}
    )
    if "projection" in ds_to_save.attrs:
        del ds_to_save.attrs["projection"]
    for name in ds_to_save.data_vars:
        if "projection" in ds_to_save[name].attrs:
            del ds_to_save[name].attrs["projection"]
    for name in ds_to_save.coords:
        if "projection" in ds_to_save.coords[name].attrs:
            del ds_to_save.coords[name].attrs["projection"]
    _LOGGER.debug("Saving file %s from dataset:\n%s", save_name, ds_to_save)
    ds_to_save.to_netcdf(
        save_name, mode="w", format="NETCDF4", encoding=encoding, engine="netcdf4"
    )


@functools.lru_cache()
def global_attributes_dict():
    # type: () -> Dict[str, str]
    """Set global attributes required by conventions.

    Currently CF-1.6 and ACDD-1.3.

    Returns
    -------
    global_atts: dict
        Still needs title, summary, source, creator_institution,
        product_version, references, cdm_data_type, institution,
        geospatial_vertical_{min,max,positive,units}, ...

    References
    ----------
    CF Conventions document:
    http://cfconventions.org
    ACDD document:
    http://wiki.esipfed.org/index.php/Category:Attribute_Conventions_Dataset_Discovery
    NCEI Templates:
    https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/
    """
    username = getpwuid(os.getuid())[0]
    global_atts = dict(
        Conventions="CF-1.6 ACDD-1.3",
        standard_name_vocabulary="CF Standard Name Table v32",
        history=(
            "{now:{date_fmt:s}}: Created by {progname:s} "
            "with command line: {cmd_line:s}"
        ).format(
            now=RUN_DATE,
            date_fmt=UDUNITS_DATE,
            progname=sys.argv[0],
            cmd_line=COMMAND_LINE,
        ),
        source=("Created by {progname:s} " "with command line: {cmd_line:s}").format(
            progname=sys.argv[0],
            cmd_line=COMMAND_LINE,
        ),
        date_created="{now:{date_fmt:s}}".format(now=RUN_DATE, date_fmt=ACDD_DATE),
        date_modified="{now:{date_fmt:s}}".format(now=RUN_DATE, date_fmt=ACDD_DATE),
        date_metadata_modified="{now:{date_fmt:s}}".format(
            now=RUN_DATE, date_fmt=ACDD_DATE
        ),
        creator_name=username,
        creator_email="{username:s}@{host:s}".format(
            username=username,
            host=MAIN_HOST,
        ),
        creator_institution=MAIN_HOST,
    )

    try:
        global_atts["conda_packages"] = subprocess.check_output(
            # Full urls including package, version, build, and MD5
            ["conda", "list", "--explicit", "--md5"],
            universal_newlines=True,
        )
    except OSError:
        pass

    try:
        global_atts["pip_packages"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            universal_newlines=True,
        )
    except OSError:
        pass

    return global_atts


############################################################
# Track file history attribute
def add_input_log(file_name: str, history: str) -> None:
    """Add the history of the named file to the inputs.

    Parameters
    ----------
    file_name: str
        The file name associated with the history
    history: str
        The history of that file
    """
    INPUT_LOGS[file_name] = history


def get_output_log() -> str:
    """Get the history string for an output file.

    Returns
    -------
    history: str
        A history attribute to use in an output file
    """
    result: str = cmdline_provenance.new_log(
        infile_history=INPUT_LOGS, git_repo=REPO_ROOT
    )
    return result
