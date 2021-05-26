# -*- coding: utf-8 -*-
"""Collect the data and algorithms needed to get model data.

One general-ish class (general for models available through TDS, less
so for those only available through FTP, I think, plus a few instances
for different models.

"""
import collections
import dataclasses
import datetime
import getpass
import typing
import urllib.request
from contextlib import closing

import numpy as np
import xarray
from metpy.constants import earth_avg_radius as EARTH_RADIUS  # noqa: N812
from metpy.units import units
from siphon.catalog import TDSCatalog

ONE_HOUR = datetime.timedelta(hours=1)
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600

BboxWesn = collections.namedtuple("BboxWesn", ["west", "east", "south", "north"])
N_AMER_BBOX = BboxWesn(-180, 10, 0, 90)
METPY_DEPENDENT_VARIABLES = {
    "wind_speed": ["x_wind", "y_wind"],
    "vorticity": ["x_wind", "y_wind"],
}


@dataclasses.dataclass
class NwpModel:
    """Encapsulate the data needed to get data from an NWP model.

    Some of this is more useful for generating graphics, but I may
    dump it into metadata.  :pypi:`cc-plugin-ncei` would install a
    good checker.
    """

    # pylint: disable=too-many-instance-attributes

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
        """Convert pressure from hPa to model-reported units.

        Parameters
        ----------
        pressure: int

        Returns
        -------
        int
        """
        return int(units(f"{pressure:d} hPa").to(self.pressure_level_units).magnitude)

    def is_model_start(self, init_time: datetime.datetime) -> bool:
        """Check if there is a model run starting at the given time.

        Parameters
        ----------
        init_time: datetime.datetime

        Returns
        -------
        bool
        """
        return (init_time.hour - self.cycling_offset) % self.cycling_interval == 0

    def get_last_model_start(self) -> datetime.datetime:
        """Get the start of the most recent model run.

        Returns
        -------
        datetime.datetime
        """
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
        """Get the model start time before the given one.

        Parameters
        ----------
        init_time: datetime.datetime

        Returns
        -------
        datetime.datetime
        """
        return init_time - datetime.timedelta(hours=self.cycling_interval)

    def model_start_has_data(self, init_time: datetime.datetime) -> bool:
        """Report whether the model start is likely to have data.

        Parameters
        ----------
        init_time: datetime.datetime

        Returns
        -------
        bool
        """
        assert self.is_model_start(init_time)
        now = datetime.datetime.utcnow()
        return (now - init_time) > datetime.timedelta(hours=self.cycling_interval)

    def get_variable_name(self, variable: str) -> str:
        """Get the name used by the model for the variable.

        Parameters
        ----------
        variable: str

        Returns
        -------
        str
        """
        return self.variable_mapping[variable]

    def get_model_data(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        pressure_mb: int,
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        pressure_mb: int
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """
        assert self.is_model_start(init_time)
        assert init_time <= valid_time
        level = int(self.convert_pressure_to_reported_units(pressure_mb))
        variable_mapping = self.variable_mapping

        def get_variable_name(var_name: str) -> typing.List[str]:
            """Get the name(s) used by the model for the variable.

            Will give variables used to derive that variable if needed.

            Parameters
            ----------
            var_name: str

            Returns
            -------
            list of str
            """
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
                raise KeyError(
                    "Data for specified initial time not available"
                ) from None
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
            dataset: xarray.Dataset = xarray.open_dataset(  # type: ignore
                xarray.backends.NetCDF4DataStore(data)  # type: ignore
            )
            # The dataset isn't fully CF, since Unidata doesn't set standard
            # names, but this at least gets me the projection.
            dataset_cf: xarray.Dataset = dataset.metpy.parse_cf()
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
        return dataset_cf


def xarray_from_grib_data(  # pylint: disable=too-many-locals
    grib_data: bytes,
) -> xarray.DataArray:
    """Produce an XArray dataset from binary grib data.

    Parameters
    ----------
    grib_data: bytes

    Returns
    -------
    xarray.DataArray
    """
    import eccodes

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
    num_rows = grib_attributes.pop("Nj")
    num_cols = grib_attributes.pop("Ni")
    field_values = grib_attributes.pop("values").reshape(num_rows, num_cols)
    latitudes = grib_attributes.pop("latitudes").reshape(num_rows, num_cols)
    longitudes = grib_attributes.pop("longitudes").reshape(num_rows, num_cols)
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
    grib_units = grib_attributes.pop("units")
    field_min = grib_attributes.pop("minimum")
    field_max = grib_attributes.pop("maximum")
    missing_value = grib_attributes.pop("missingValue")
    # earth_radius = grib_attributes.pop("earthRadius")

    result: xarray.DataArray = (
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
                "units": grib_units,
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
            if value is not None
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
        (
            "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/"
            "Global_0p5deg/catalog.xml"
        ),
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
        (
            "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NAM/CONUS_40km/"
            "conduit/catalog.xml"
        ),
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
        ftp_data_path_pattern=(
            "{init_time:%Y%m%d%H%M%S}/"
            "A_H{variable_upper:s}XE{level_kpa:d}ECMF{init_time:%d%H%M}_C_ECMF_"
            "{init_time:%Y%m%d%H%M%S}_{lead_time_hours:d}h_{variable_lower:s}_"
            "{level_hpa:d}hPa_global_0p5deg_grib2.bin"
        ),
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
        (
            "https://thredds.ucar.edu/thredds/catalog/grib/FNMOC/NAVGEM/"
            "Global_0p5deg/catalog.xml"
        ),
        "FNMOC NAVGEM Global 0.5 Degree_{init_time:%Y%m%d_%H%M}.grib1",
    ),
)
