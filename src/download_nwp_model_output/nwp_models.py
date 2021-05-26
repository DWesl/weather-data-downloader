# -*- coding: utf-8 -*-
"""Collect the data needed to get model data.

Mostly plugs data into classes from :mod:`.data_source`.
"""
import collections
import dataclasses
import datetime
import math
import typing

import xarray

from .data_source import EcmwfFtpDataSource, NwpDataSource, TDSDataSource, VariableMap

ONE_HOUR = datetime.timedelta(hours=1)

BboxWesn = collections.namedtuple("BboxWesn", ["west", "east", "south", "north"])


@dataclasses.dataclass
class NwpModel:
    """Encapsulate the data needed to get data from an NWP model.

    Some of this is more useful for generating graphics, but I may
    dump it into metadata.  :pypi:`cc-plugin-ncei` would install a
    good checker.
    """

    abbrev: str
    short_desc: str
    cycling_interval: int
    cycling_offset: int
    modeling_center: str
    data_access: NwpDataSource
    availability_delay: int = -1

    def __post_init__(self) -> None:
        """Set attributes if they weren't specified.

        Currently just availability_delay.
        """
        if self.availability_delay < 0:
            self.availability_delay = int(math.ceil(self.cycling_interval / 2))

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
        return (now - init_time) > datetime.timedelta(hours=self.availability_delay)

    def get_model_start_with_data(self) -> datetime.datetime:
        """Get most recent model start with data.

        Returns
        -------
        datetime.datetime
        """
        init_time = self.get_last_model_start()
        while not self.model_start_has_data(init_time):
            init_time = self.get_previous_model_start(init_time)
        return init_time

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
        result: xarray.Dataset = self.data_access.get_model_data_pressure(
            init_time, valid_time, variables, pressure_mb, bbox_wesn
        )
        return result

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
        result: xarray.Dataset = self.data_access.get_model_data_height(
            init_time, valid_time, variables, height_m, bbox_wesn
        )
        return result

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
        result: xarray.Dataset = self.data_access.get_model_data_single_level(
            init_time, valid_time, variables, bbox_wesn
        )
        return result


NCEP_VARIABLE_MAP: VariableMap = {
    "air_temperature": {
        "pressure": "Temperature_isobaric",
        "height": "Temperature_height_above_ground",
    },
    "geopotential_height": {
        "pressure": "Geopotential_height_isobaric",
    },
    "x_wind": {
        "pressure": "u-component_of_wind_isobaric",
        "height": "u-component_of_wind_height_above_ground",
    },
    "y_wind": {
        "pressure": "v-component_of_wind_isobaric",
        "height": "v-component_of_wind_height_above_ground",
    },
    "relative_humidity": {
        "pressure": "Relative_humidity_isobaric",
        "height": "Relative_humidity_height_above_ground",
    },
    "atmosphere_mass_content_of_water_vapor": (
        "Precipitable_water_entire_atmosphere_single_layer"
    ),
    "air_pressure_at_mean_sea_level": "Pressure_reduced_to_MSL_msl",
    "high_type_cloud_area_fraction": "High_cloud_cover_high_cloud",
    "medium_type_cloud_area_fraction": "Medium_cloud_cover_middle_cloud",
    "low_type_cloud_area_fraction": "Low_cloud_cover_low_cloud",
    "cloud_area_fraction": "Total_cloud_cover_entire_atmosphere",
}
ECMWF_VARIABLE_MAP: VariableMap = {
    "air_temperature": "T",
    "geopotential_height": "H",
    "x_wind": "U",
    "y_wind": "V",
    "air_pressure_at_mean_sea_level": "P",
}
FNMOC_VARIABLE_MAP: VariableMap = {
    "air_temperature": {
        "pressure": "air_temp_isobaric",
        "height": "air_temp_height_above_ground",
    },
    "geopotential_height": {
        "pressure": "geop_ht_isobaric",
    },
    "x_wind": {
        "pressure": "wnd_ucmp_isobaric",
        "height": "wnd_ucmp_height_above_ground",
    },
    "y_wind": {
        "pressure": "wnd_vcmp_isobaric",
        "height": "wnd_vcmp_height_above_ground",
    },
    "relative_humidity": {
        "pressure": "rltv_hum_isobaric",
        "height": "rltv_hum_height_above_ground",
    },
    "air_pressure_at_mean_sea_level": "pres_reduced_msl",
    "low_type_cloud_area_fraction": "ttl_cld_cvr_low_cld",
    "medium_type_cloud_area_fraction": "ttl_cld_cvr_mid_cld",
    "high_type_cloud_area_fraction": "ttl_cld_cvr_hi_cld",
    "cloud_area_fraction": "ttl_cld_cvr_sky_cvr",
}

NWP_MODELS = dict(
    GFS=NwpModel(
        "GFS",
        "NCEP Global Forecast System",
        6,
        0,
        "NCEP",
        TDSDataSource(
            "Pa",
            NCEP_VARIABLE_MAP,
            (
                "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/"
                "Global_0p5deg/catalog.xml"
            ),
            "GFS_Global_0p5deg_{init_time:%Y%m%d_%H%M}.grib2",
        ),
        availability_delay=6,
    ),
    NAM=NwpModel(
        "NAM",
        "North American Mesoscale",
        6,
        0,
        "NCEP",
        TDSDataSource(
            "Pa",
            NCEP_VARIABLE_MAP,
            (
                "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/NAM/CONUS_40km/"
                "conduit/catalog.xml"
            ),
            "NAM_CONUS_40km_conduit_{init_time:%Y%m%d_%H%M}.grib2",
        ),
    ),
    RAP=NwpModel(
        "RAP",
        "Rapid Refresh",
        3,
        0,
        "NCEP",
        TDSDataSource(
            "Pa",
            NCEP_VARIABLE_MAP,
            (
                "https://thredds.ucar.edu/thredds/catalog/grib/"
                "NCEP/RAP/CONUS_40km/catalog.xml"
            ),
            "RR_CONUS_40km_{init_time:%Y%m%d_%H%M}.grib2",
        ),
    ),
    ECMWF=NwpModel(
        "ECMWF",
        "Integrated Forecast System",
        12,
        0,
        "ECMWF",
        EcmwfFtpDataSource(
            "hPa",
            ECMWF_VARIABLE_MAP,
            ftp_domain="dissemination.ecmwf.int",
            ftp_login_name="wmo",
            ftp_pressure_data_path_pattern=(
                "{init_time:%Y%m%d%H%M%S}/"
                "A_H{variable_upper:s}X{time_chr:s}{level_kpa:d}ECMF"
                "{init_time:%d%H%M}_C_ECMF_{init_time:%Y%m%d%H%M%S}_"
                "{lead_time_hours:d}h_{variable_lower:s}_"
                "{level_hpa:d}hPa_global_0p5deg_grib2.bin"
            ),
            ftp_single_level_data_path_pattern=(
                "{init_time:%Y%m%d%H%M%S}/"
                "A_H{variable_upper:s}X{time_chr:s}89ECMF{init_time:%d%H%M}_C_ECMF_"
                "{init_time:%Y%m%d%H%M%S}_{lead_time_hours:d}h_msl_global_"
                "0p5deg_grib2.bin"
            ),
        ),
    ),
    NAVGEM=NwpModel(
        "NAVGEM",
        "NAVGEM",
        6,
        0,
        "FNMOC",
        TDSDataSource(
            "hPa",
            FNMOC_VARIABLE_MAP,
            (
                "https://thredds.ucar.edu/thredds/catalog/grib/FNMOC/NAVGEM/"
                "Global_0p5deg/catalog.xml"
            ),
            "FNMOC NAVGEM Global 0.5 Degree_{init_time:%Y%m%d_%H%M}.grib1",
        ),
        availability_delay=18,
    ),
)
