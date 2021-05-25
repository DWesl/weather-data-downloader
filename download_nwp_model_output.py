#!/usr/bin/env python3
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

import argparse
import collections
import dataclasses
import datetime
import functools
import getpass
import logging
import os.path
import pathlib
import socket
import subprocess
import sys
import typing
import urllib
from contextlib import closing
from pwd import getpwuid
from typing import Any, Dict, Hashable

try:
    from shlex import quote
except ImportError:
    from pipes import quote

import cmdline_provenance
import dateutil.parser
import dateutil.tz
import eccodes
import numpy as np
import xarray
from metpy.calc import vorticity, wind_speed
from metpy.constants import earth_avg_radius as EARTH_RADIUS
from metpy.units import units
from siphon.catalog import TDSCatalog

try:
    FILE_DIR = os.path.abspath(os.path.dirname(__file__))
except Exception:
    FILE_DIR = os.path.abspath(".")
REPO_ROOT = os.path.abspath(FILE_DIR)
RUN_DIR = os.path.abspath(".")

_LOGGER = logging.getLogger(__name__)
ONE_HOUR = datetime.timedelta(hours=1)
UTC = dateutil.tz.tzutc()
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

UDUNITS_DATE = "%Y-%m-%d %H:%M:%S%z"
ACDD_DATE = "%Y-%m-%dT%H:%M:%S%z"
CALENDAR = "standard"
RUN_DATE = datetime.datetime.now(tz=UTC)

HOST = socket.gethostbyaddr(socket.gethostbyname(os.uname()[1]))[0]
MAIN_HOST = ".".join(HOST.split(".")[-3:])
COMMAND_LINE = " ".join(quote(arg) for arg in sys.argv)
INPUT_LOGS = {}

BboxWesn = collections.namedtuple("BboxWesn", ["west", "east", "south", "north"])
N_AMER_BBOX = BboxWesn(-180, 10, 0, 90)
METPY_DEPENDENT_VARIABLES = {
    "wind_speed": ["x_wind", "y_wind"],
    "vorticity": ["x_wind", "y_wind"],
}


@dataclasses.dataclass
class NwpModel:
    abbrev: str
    short_desc: str
    cycling_interval: int
    cycling_offset: int
    pressure_level_units: str
    modeling_center: str
    data_access_protocall: str
    variable_mapping: typing.Dict[str, str]
    tds_catalog_url: str = ""
    tds_catalog_pattern: str = ""
    ftp_domain: str = ""
    ftp_login_name: str = ""
    ftp_data_path_pattern: str = ""

    def convert_pressure_to_reported_units(self, pressure: int) -> int:
        return int(units(f"{pressure:d} hPa").to(self.pressure_level_units).magnitude)

    def is_model_start(self, init_time: datetime.datetime) -> bool:
        return (init_time.hour - self.cycling_offset) % self.cycling_interval == 0

    def get_last_model_start(self) -> datetime.datetime:
        last_hour = datetime.datetime.utcnow().replace(
            minute=0, second=0, microsecond=0
        )
        while not self.is_model_start(last_hour):
            last_hour -= ONE_HOUR
        assert self.is_model_start(last_hour)
        return last_hour

    def get_previous_model_start(
        self, init_time: datetime.datetime
    ) -> datetime.datetime:
        return init_time - datetime.timedelta(hours=self.cycling_interval)

    def model_start_has_data(self, init_time: datetime.datetime) -> bool:
        assert self.is_model_start(init_time)
        now = datetime.datetime.utcnow()
        return (now - init_time) > datetime.timedelta(hours=self.cycling_interval)

    def get_variable_name(self, variable: str) -> str:
        return self.variable_mapping[variable]

    def get_model_data(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.List[str],
        pressure_mb: int,
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        assert self.is_model_start(init_time)
        assert init_time <= valid_time
        level = int(self.convert_pressure_to_reported_units(pressure_mb))
        variable_mapping = self.variable_mapping

        def get_variable_name(var_name: str) -> str:
            try:
                result = [variable_mapping[var_name]]
            except KeyError:
                result = [
                    variable_mapping[name]
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        if self.data_access_protocall == "tds":
            try:
                main_catalog = TDSCatalog(self.tds_catalog_url)
                this_catalog = main_catalog.catalog_refs[
                    self.tds_catalog_pattern.format(init_time=init_time)
                ].follow()
                tds_ds = this_catalog.datasets[0]
            except KeyError:
                raise KeyError("Data for specified initial time not available")
            ncss = tds_ds.subset()
            query = ncss.query()
            query.lonlat_box(
                west=bbox_wesn[0],
                east=bbox_wesn[1],
                south=bbox_wesn[2],
                north=bbox_wesn[3],
            )
            query.time(valid_time)
            query.accept("netcdf4")

            query.variables(
                *[
                    var_name
                    for name in variables
                    for var_name in get_variable_name(name)
                ]
            )

            query.vertical_level(level)
            data = ncss.get_data(query)
            dataset = xarray.open_dataset(xarray.backends.NetCDF4DataStore(data))
            # The dataset isn't fully CF, since Unidata doesn't set standard
            # names, but this at least gets me the projection.
            dataset_cf = dataset.metpy.parse_cf()
        elif self.data_access_protocall == "ftp":
            dataset = xarray.Dataset()
            lead_time = valid_time - init_time
            lead_time_hours = (
                lead_time.days * HOURS_PER_DAY + lead_time.seconds // SECONDS_PER_HOUR
            )
            for var in variables:
                var_names = get_variable_name(var)
                for var_name in var_names:
                    with closing(
                        urllib.request.urlopen(
                            "ftp://{user:s}:{password:s}@{domain:s}/{path:s}".format(
                                user=self.ftp_login_name,
                                password=getpass.getpass(),
                                domain=self.ftp_domain,
                                path=self.ftp_data_path_pattern.format(
                                    init_time=init_time,
                                    variable_upper=var_name.upper(),
                                    variable_lower=(
                                        var_name.lower() if var_name != "H" else "gh"
                                    ),
                                    lead_time_hours=lead_time_hours,
                                    level_kpa=pressure_mb // 10,
                                    level_hpa=pressure_mb,
                                ),
                            )
                        )
                    ) as ftp_data:
                        grib_data = ftp_data.read()
                    dataset[var_name] = xarray_from_grib_data(grib_data)
            dataset_cf = dataset.metpy.parse_cf()
        if "wind_speed" in variables:
            dataset_cf["wind_speed"] = wind_speed(
                dataset_cf[get_variable_name("x_wind")[0]].metpy.quantify(),
                dataset_cf[get_variable_name("y_wind")[0]].metpy.quantify(),
            )
        if "vorticity" in variables:
            dataset_cf["vorticity"] = vorticity(
                dataset_cf[get_variable_name("x_wind")[0]].metpy.quantify(),
                dataset_cf[get_variable_name("y_wind")[0]].metpy.quantify(),
            )
        return dataset_cf


def xarray_from_grib_data(grib_data):
    msg_id = eccodes.codes_new_from_message(grib_data)
    try:
        keys_iterator = eccodes.codes_keys_iterator_new(msg_id)
        grib_attributes = {}
        while eccodes.codes_keys_iterator_next(keys_iterator):
            key_name = eccodes.codes_keys_iterator_get_name(keys_iterator)
            if eccodes.codes_get_size(msg_id, key_name) == 1:
                if eccodes.codes_is_missing(msg_id, key_name):
                    grib_attributes[key_name] = np.nan
                else:
                    grib_attributes[key_name] = eccodes.codes_get(msg_id, key_name)
            else:
                grib_attributes[key_name] = eccodes.codes_get_array(msg_id, key_name)
    finally:
        eccodes.codes_release(msg_id)
    Nj = grib_attributes.pop("Nj")
    Ni = grib_attributes.pop("Ni")
    field_values = grib_attributes.pop("values").reshape(Nj, Ni)
    latitudes = grib_attributes.pop("latitudes").reshape(Nj, Ni)
    longitudes = grib_attributes.pop("longitudes").reshape(Nj, Ni)
    # latitude = grib_attributes.pop("distinctLatitudes")
    # longitude = grib_attributes.pop("distinctLongtitudes")
    del grib_attributes["latLonValues"]
    forecast_reference_time = datetime.datetime(
        grib_attributes.pop("year"),
        grib_attributes.pop("month"),
        grib_attributes.pop("day"),
        grib_attributes.pop("hour"),
        grib_attributes.pop("minute"),
        grib_attributes.pop("second"),
    )
    forecast_period = datetime.timedelta(hours=grib_attributes.pop("forecastTime"))
    level_type = grib_attributes.pop("typeOfLevel")
    if level_type.startswith("isobaricIn"):
        level = xarray.DataArray(
            grib_attributes.pop("level"),
            name="pressure",
            attrs={
                "standard_name": "air_pressure",
                "units": level_type[10:],
            },
        )
    else:
        level = xarray.DataArray(
            grib_attributes.pop("level"),
            name="level",
            attrs={"level_type": level_type},
        )
    standard_name = grib_attributes.pop("cfName")
    units = grib_attributes.pop("units")
    field_min = grib_attributes.pop("minimum")
    field_max = grib_attributes.pop("maximum")
    missing_value = grib_attributes.pop("missingValue")
    # earth_radius = grib_attributes.pop("earthRadius")

    result = (
        xarray.DataArray(
            field_values,
            {
                # "latitude": (
                #     ("latitude",),
                #     latitude,
                #     {
                #         "standard_name": "latitude",
                #         "units": "degrees_north",
                #     },
                # ),
                # "longitude": (
                #     ("longitude",),
                #     longitude,
                #     {
                #         "standard_name": "longitude",
                #         "units": "degrees_east",
                #     },
                # ),
                "latitudes": (
                    ("latitude", "longitude"),
                    latitudes,
                    {
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    },
                ),
                "longitudes": (
                    ("latitude", "longitude"),
                    longitudes,
                    {
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    },
                ),
                "forecast_reference_time": (
                    (),
                    forecast_reference_time,
                    {"standard_name": "forecast_reference_time"},
                ),
                "forecast_period": (
                    (),
                    forecast_period,
                    {"standard_name": "forecast_period"},
                ),
                "valid_time": (
                    (),
                    forecast_reference_time + forecast_period,
                    {"standard_name": "time"},
                ),
                "level": level,
            },
            ("latitude", "longitude"),
            standard_name,
            {
                "standard_name": standard_name,
                "units": units,
                "actual_range": (field_min, field_max),
            },
            {"_FillValue": missing_value, "grid_mapping_name": "latlon_crs"},
        )
        .metpy.assign_crs(
            grid_mapping_name="latitude_longitude",
            # earth_radius=earth_radius,
            earth_radius=EARTH_RADIUS.magnitude,
        )
        .load()
    )
    result.coords["latlon_crs"] = ((), -1, result.metpy.pyproj_crs.to_cf())
    result.attrs.update(
        {
            f"GRIB_{name:s}": value
            for name, value in grib_attributes.items()
            if not isinstance(value, type(None))
        }
    )
    return result


NCEP_VARIABLE_MAP = {
    "air_temperature": "Temperature_isobaric",
    "geopotential_height": "Geopotential_height_isobaric",
    "x_wind": "u-component_of_wind_isobaric",
    "y_wind": "v-component_of_wind_isobaric",
}
ECMWF_VARIABLE_MAP = {
    "air_temperature": "T",
    "geopotential_height": "H",  # might be z instead
    "x_wind": "U",
    "y_wind": "V",
}
FNMOC_VARIABLE_MAP = {
    "air_temperature": "air_temp_isobaric",
    "geopotential_height": "geop_ht_isobaric",
    "x_wind": "wnd_ucmp_isobaric",
    "y_wind": "wnd_vcmp_isobaric",
}


NWP_MODELS = dict(
    GFS=NwpModel(
        "GFS",
        "NCEP Global Forecast System",
        6,
        0,
        "Pa",
        "NCEP",
        "tds",
        NCEP_VARIABLE_MAP,
        "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/catalog.xml",
        "GFS_Global_0p5deg_{init_time:%Y%m%d_%H%M}.grib2",
    ),
    NAM=NwpModel(
        "NAM",
        "North American Mesoscale",
        6,
        0,
        "Pa",
        "NCEP",
        "tds",
        NCEP_VARIABLE_MAP,
        "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NAM/CONUS_40km/conduit/catalog.xml",
        "NAM_CONUS_40km_conduit_{init_time:%Y%m%d_%H%M}.grib2",
    ),
    RAP=NwpModel(
        "RAP",
        "Rapid Refresh",
        3,
        0,
        "Pa",
        "NCEP",
        "tds",
        NCEP_VARIABLE_MAP,
        "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_40km/catalog.xml",
        "RR_CONUS_40km_{init_time:%Y%m%d_%H%M}.grib2",
    ),
    ECMWF=NwpModel(
        "ECMWF",
        "Integrated Forecast System",
        12,
        0,
        "hPa",
        "ECMWF",
        "ftp",
        ECMWF_VARIABLE_MAP,
        ftp_domain="dissemination.ecmwf.int",
        ftp_login_name="wmo",
        ftp_data_path_pattern="{init_time:%Y%m%d%H%M%S}/A_H{variable_upper:s}XE{level_kpa:d}ECMF{init_time:%d%H%M}_C_ECMF_{init_time:%Y%m%d%H%M%S}_{lead_time_hours:d}h_{variable_lower:s}_{level_hpa:d}hPa_global_0p5deg_grib2.bin",
    ),
    NAVGEM=NwpModel(
        "NAVGEM",
        "NAVGEM",
        6,
        0,
        "hPa",
        "FNMOC",
        "tds",
        FNMOC_VARIABLE_MAP,
        "https://thredds.ucar.edu/thredds/catalog/grib/FNMOC/NAVGEM/Global_0p5deg/catalog.xml",
        "FNMOC NAVGEM Global 0.5 Degree_{init_time:%Y%m%d_%H%M}.grib1",
    ),
)


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
    return cmdline_provenance.new_log(infile_history=INPUT_LOGS, git_repo=REPO_ROOT)


############################################################
# Program logic
PARSER = argparse.ArgumentParser(__doc__)
PARSER.add_argument("model_abbrev", choices=NWP_MODELS.keys())
PARSER.add_argument("forecast_hour", type=int)
PARSER.add_argument("--init-time", type=dateutil.parser.parse, default=None)

if __name__ == "__main__":
    args = PARSER.parse_args(sys.argv[1:])
    model = NWP_MODELS[args.model_abbrev]
    if args.init_time == None:
        last_start = model.get_last_model_start()
        if not model.model_start_has_data(last_start):
            last_start = model.get_previous_model_start(last_start)
    else:
        last_start = args.init_time
        assert last_start < RUN_DATE
    valid_time = last_start + datetime.timedelta(hours=args.forecast_hour)
    save_dir = pathlib.Path(
        os.path.join(
            RUN_DIR, model.abbrev, last_start.isoformat(), valid_time.isoformat()
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if model.abbrev != "ECMWF":
        for pressure_mb in (1000, 925, 850, 700, 500, 300, 200):
            dataset = model.get_model_data(
                last_start,
                valid_time,
                NCEP_VARIABLE_MAP.keys(),
                pressure_mb,
                N_AMER_BBOX,
            )
            del dataset.coords["metpy_crs"]
            save_nonsparse_netcdf(
                dataset,
                os.path.join(
                    save_dir,
                    f"{model.abbrev}_{last_start:%Y%m%dT%H}_f{args.forecast_hour:02d}_{pressure_mb:04d}mb_data.nc4",
                ),
            )
    else:
        # ECMWF
        for pressure_mb, variables in (
            (850, ["x_wind", "y_wind", "air_temperature"]),
            (500, ["geopotential_height"]),
        ):
            dataset = model.get_model_data(
                last_start, valid_time, variables, pressure_mb, N_AMER_BBOX
            )
            del dataset.coords["metpy_crs"]
            save_nonsparse_netcdf(
                dataset,
                os.path.join(
                    save_dir,
                    f"{model.abbrev}_{last_start:%Y%m%dT%H}_f{args.forecast_hour:02d}_{pressure_mb:04d}mb_data.nc4",
                ),
            )
