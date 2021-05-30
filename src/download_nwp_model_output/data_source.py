# -*- coding: utf-8 -*-
"""Functions to get data from NWP models.

So far just TDS data and the public ECMWF FTP site.
"""
import abc
import dataclasses
import datetime
import getpass
import os
import typing
import urllib.request
from contextlib import closing

import numpy as np
import xarray
from metpy.constants import earth_avg_radius as EARTH_RADIUS  # noqa: N812
from metpy.units import units
from siphon.catalog import TDSCatalog

HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600

METPY_DEPENDENT_VARIABLES = {
    "wind_speed": ["x_wind", "y_wind"],
    "vorticity": ["x_wind", "y_wind"],
    "dew_point_temperature": ["air_temperature", "relative_humidity"],
}

PRESSURE_VARIABLES = frozenset(
    ["air_temperature", "geopotential_height", "x_wind", "y_wind", "relative_humidity"]
)
HEIGHT_2M_VARIABLES = frozenset(["air_temperature", "dew_point_temperature"])
HEIGHT_10M_VARIABLES = frozenset(["x_wind", "y_wind"])
SINGLE_LEVEL_VARIABLES = frozenset(
    [
        "air_pressure_at_mean_sea_level",
        "low_type_cloud_area_fraction",
        "medium_type_cloud_area_fraction",
        "high_type_cloud_area_fraction",
        "cloud_area_fraction",
        "atmosphere_mass_content_of_water_vapor",
    ]
)

VariableMap = typing.Dict[str, typing.Union[str, typing.Dict[str, str]]]


@dataclasses.dataclass  # type: ignore
class NwpDataSource(abc.ABC):
    """Interface for data sources."""

    pressure_level_units: str
    variable_mapping: VariableMap

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

    @abc.abstractmethod
    def get_model_data_pressure(
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

    @abc.abstractmethod
    def get_model_data_height(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        height_m: int,
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        height_m: int
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """

    @abc.abstractmethod
    def get_model_data_single_level(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """


@dataclasses.dataclass
class TDSDataSource(NwpDataSource):
    """Class for data from TDS server."""

    tds_catalog_url: str
    tds_catalog_pattern: str

    def get_model_data_pressure(
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
        assert init_time <= valid_time
        level = int(self.convert_pressure_to_reported_units(pressure_mb))
        variable_mapping = self.variable_mapping

        cast = typing.cast

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
                result = [
                    cast("Dict[str, str]", variable_mapping[var_name])["pressure"]
                ]
            except KeyError:
                result = [
                    cast("Dict[str, str]", variable_mapping[name])["pressure"]
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        try:
            main_catalog = TDSCatalog(self.tds_catalog_url)
            this_catalog = main_catalog.catalog_refs[
                self.tds_catalog_pattern.format(init_time=init_time)
            ].follow()
            tds_ds = this_catalog.datasets[0]
        except KeyError:
            raise KeyError("Data for specified initial time not available") from None
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
            *[var_name for name in variables for var_name in get_variable_name(name)]
        )

        query.vertical_level(level)
        data = ncss.get_data(query)
        dataset: xarray.Dataset = xarray.open_dataset(  # type: ignore
            xarray.backends.NetCDF4DataStore(data)  # type: ignore
        )
        # The dataset isn't fully CF, since Unidata doesn't set standard
        # names, but this at least gets me the projection.
        return dataset

    def get_model_data_height(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        height_m: int,
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        height_m: int
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """
        assert init_time <= valid_time
        level = height_m
        variable_mapping = self.variable_mapping
        cast = typing.cast

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
                result = [cast("Dict[str, str]", variable_mapping[var_name])["height"]]
            except KeyError:
                result = [
                    cast("Dict[str, str]", variable_mapping[name])["height"]
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        try:
            main_catalog = TDSCatalog(self.tds_catalog_url)
            this_catalog = main_catalog.catalog_refs[
                self.tds_catalog_pattern.format(init_time=init_time)
            ].follow()
            tds_ds = this_catalog.datasets[0]
        except KeyError:
            raise KeyError("Data for specified initial time not available") from None
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
            *[var_name for name in variables for var_name in get_variable_name(name)]
        )

        query.vertical_level(level)
        data = ncss.get_data(query)
        dataset: xarray.Dataset = xarray.open_dataset(  # type: ignore
            xarray.backends.NetCDF4DataStore(data)  # type: ignore
        )
        # The dataset isn't fully CF, since Unidata doesn't set standard
        # names, but this at least gets me the projection.
        return dataset

    def get_model_data_single_level(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """
        assert init_time <= valid_time
        variable_mapping = self.variable_mapping
        cast = typing.cast

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
                result = [cast("str", variable_mapping[var_name])]
            except KeyError:
                result = [
                    cast("str", variable_mapping[name])
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        try:
            main_catalog = TDSCatalog(self.tds_catalog_url)
            this_catalog = main_catalog.catalog_refs[
                self.tds_catalog_pattern.format(init_time=init_time)
            ].follow()
            tds_ds = this_catalog.datasets[0]
        except KeyError:
            raise KeyError("Data for specified initial time not available") from None
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
            *[var_name for name in variables for var_name in get_variable_name(name)]
        )

        data = ncss.get_data(query)
        dataset: xarray.Dataset = xarray.open_dataset(  # type: ignore
            xarray.backends.NetCDF4DataStore(data)  # type: ignore
        )
        # The dataset isn't fully CF, since Unidata doesn't set standard
        # names, but this at least gets me the projection.
        return dataset


@dataclasses.dataclass
class EcmwfFtpDataSource(NwpDataSource):
    """Class for data from public ECMWF FTP server."""

    ftp_domain: str
    ftp_login_name: str
    ftp_pressure_data_path_pattern: str
    ftp_single_level_data_path_pattern: str

    def get_model_data_pressure(
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
        assert init_time <= valid_time
        variable_mapping = self.variable_mapping
        cast = typing.cast

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
                result = [cast("str", variable_mapping[var_name])]
            except KeyError:
                result = [
                    cast("str", variable_mapping[name])
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        dataset = xarray.Dataset()
        lead_time = valid_time - init_time
        lead_time_hours = (
            lead_time.days * HOURS_PER_DAY + lead_time.seconds // SECONDS_PER_HOUR
        )
        time_chars = {96: "M", 72: "K", 48: "I", 24: "E", 120: "O", 144: "Q"}
        time_chr = time_chars[lead_time_hours]
        password = os.environ.get("ECMWF_PUBLIC_FTP_PASSWORD", "")
        if password == "":
            password = getpass.getpass("ECMWF public FTP server password:")
        for var in variables:
            var_names = get_variable_name(var)
            for var_name in var_names:
                with closing(
                    urllib.request.urlopen(
                        "ftp://{user:s}:{password:s}@{domain:s}/{path:s}".format(
                            user=self.ftp_login_name,
                            password=password,
                            domain=self.ftp_domain,
                            path=self.ftp_pressure_data_path_pattern.format(
                                init_time=init_time,
                                variable_upper=var_name.upper(),
                                variable_lower=(
                                    var_name.lower() if var_name != "H" else "gh"
                                ),
                                lead_time_hours=lead_time_hours,
                                level_kpa=pressure_mb // 10,
                                level_hpa=pressure_mb,
                                time_chr=time_chr,
                            ),
                        )
                    )
                ) as ftp_data:
                    grib_data = ftp_data.read()
                dataset[var_name] = xarray_from_grib_data(grib_data)
        return dataset

    def get_model_data_height(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        height_m: int,
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        height_m: int
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """
        raise NotImplementedError("ECMWF doesn't provide data on height levels")

    def get_model_data_single_level(
        self,
        init_time: datetime.datetime,
        valid_time: datetime.datetime,
        variables: typing.Iterable[str],
        bbox_wesn: typing.Tuple[float, float, float, float],
    ) -> xarray.Dataset:
        """Get the given data from the model.

        Parameters
        ----------
        init_time: datetime.datetime
        valid_time: datetime.datetime
        variables: list of str
        bbox_wesn: tuple of float

        Returns
        -------
        xarray.Dataset
        """
        assert init_time <= valid_time
        variable_mapping = self.variable_mapping
        cast = typing.cast

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
                result = [cast("str", variable_mapping[var_name])]
            except KeyError:
                result = [
                    cast("str", variable_mapping[name])
                    for name in METPY_DEPENDENT_VARIABLES[var_name]
                ]
            return result

        dataset = xarray.Dataset()
        lead_time = valid_time - init_time
        lead_time_hours = (
            lead_time.days * HOURS_PER_DAY + lead_time.seconds // SECONDS_PER_HOUR
        )
        time_chars = {96: "M", 72: "K", 48: "I", 24: "E", 120: "O", 144: "Q"}
        time_chr = time_chars[lead_time_hours]
        password = os.environ.get("ECMWF_PUBLIC_FTP_PASSWORD", "")
        if password == "":
            password = getpass.getpass("ECMWF public FTP server password:")
        for var in variables:
            var_names = get_variable_name(var)
            for var_name in var_names:
                with closing(
                    urllib.request.urlopen(
                        "ftp://{user:s}:{password:s}@{domain:s}/{path:s}".format(
                            user=self.ftp_login_name,
                            password=password,
                            domain=self.ftp_domain,
                            path=self.ftp_single_level_data_path_pattern.format(
                                init_time=init_time,
                                variable_upper=var_name.upper(),
                                variable_lower=(
                                    var_name.lower() if var_name != "H" else "gh"
                                ),
                                lead_time_hours=lead_time_hours,
                                time_chr=time_chr,
                            ),
                        )
                    )
                ) as ftp_data:
                    grib_data = ftp_data.read()
                dataset[var_name] = xarray_from_grib_data(grib_data)
        return dataset


def xarray_from_grib_data(  # pylint: disable=too-many-locals,too-many-branches
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
    # Not sure what to do with these.
    del grib_attributes["codedValues"]
    latitudes = grib_attributes.pop("latitudes").reshape(num_rows, num_cols)
    longitudes = grib_attributes.pop("longitudes").reshape(num_rows, num_cols)
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

    result: xarray.DataArray = xarray.DataArray(
        field_values,
        {
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
    ).load()
    if "earthRadius" in grib_attributes and np.isfinite(grib_attributes["earthRadius"]):
        earth_radius = grib_attributes.pop("earthRadius")
    else:
        earth_radius = EARTH_RADIUS.magnitude
    result = result.metpy.assign_crs(
        grid_mapping_name="latitude_longitude", earth_radius=earth_radius
    )
    if "name" in grib_attributes:
        result.attrs["long_name"] = grib_attributes.pop("name")
    result.coords["latlon_crs"] = ((), -1, result.metpy.pyproj_crs.to_cf())
    try:
        latitude = grib_attributes.pop("distinctLatitudes")
        longitude = grib_attributes.pop("distinctLongitudes")
        if all(np.isfinite(latitude)) and all(np.isfinite(longitude)):
            result.coords.update(
                {
                    "latitude": (
                        ("latitude",),
                        latitude,
                        {
                            "standard_name": "latitude",
                            "units": "degrees_north",
                        },
                    ),
                    "longitude": (
                        ("longitude",),
                        longitude,
                        {
                            "standard_name": "longitude",
                            "units": "degrees_east",
                        },
                    ),
                }
            )
    except KeyError:
        pass
    result.attrs.update(
        {
            f"GRIB_{name:s}": value
            for name, value in grib_attributes.items()
            if value is not None
        }
    )
    return result
