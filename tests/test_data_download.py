# -*- coding: utf-8 -*-
"""Test that the functions will download data."""
import datetime

import pytest

from download_nwp_model_output.data_source import (
    HEIGHT_2M_VARIABLES,
    HEIGHT_10M_VARIABLES,
    PRESSURE_VARIABLES,
    SINGLE_LEVEL_VARIABLES,
)
from download_nwp_model_output.nwp_models import NWP_MODELS, BboxWesn

PA_BBOX = BboxWesn(-81, -74, 39, 43)


@pytest.mark.parametrize("model_abbrev", ["GFS", "NAM", "RAP", "NAVGEM"])
@pytest.mark.parametrize("forecast_hour", [3, 12])
@pytest.mark.parametrize("pressure_mb", [1000, 500, 250])
def test_model_pressure(
    model_abbrev: str, forecast_hour: int, pressure_mb: int
) -> None:
    """Test whether models can get variables on pressure surfaces.

    Parameters
    ----------
    model_abbrev: str
    forecast_hour: int
    pressure_mb: int
    """
    model = NWP_MODELS[model_abbrev]
    init_time = model.get_model_start_with_data()
    valid_time = init_time + datetime.timedelta(hours=forecast_hour)
    variables = PRESSURE_VARIABLES & model.data_access.variable_mapping.keys()
    data = model.get_model_data_pressure(
        init_time, valid_time, variables, pressure_mb, PA_BBOX
    )
    for data_var in data.data_vars.values():
        assert data_var.count() > 0


@pytest.mark.parametrize("model_abbrev", ["NAVGEM", "GFS", "NAM", "RAP"])
@pytest.mark.parametrize("forecast_hour", [3, 12])
def test_model_surface(model_abbrev: str, forecast_hour: int) -> None:
    """Test whether models can get surface fields.

    Parameters
    ----------
    model_abbrev: str
    forecast_hour: int
    """
    model = NWP_MODELS[model_abbrev]
    init_time = model.get_model_start_with_data()
    valid_time = init_time + datetime.timedelta(hours=forecast_hour)
    variables_2m = HEIGHT_2M_VARIABLES & model.data_access.variable_mapping.keys()
    data_2m = model.get_model_data_height(
        init_time,
        valid_time,
        variables_2m,
        2,
        PA_BBOX,
    )
    for data_var in data_2m.data_vars.values():
        assert data_var.count() > 0
    variables_10m = HEIGHT_10M_VARIABLES & model.data_access.variable_mapping.keys()
    data_10m = model.get_model_data_height(
        init_time,
        valid_time,
        variables_10m,
        10,
        PA_BBOX,
    )
    for data_var in data_10m.data_vars.values():
        assert data_var.count() > 0
    variables_one_level = (
        SINGLE_LEVEL_VARIABLES & model.data_access.variable_mapping.keys()
    )
    data_one_level = model.get_model_data_single_level(
        init_time,
        valid_time,
        variables_one_level,
        PA_BBOX,
    )
    for data_var in data_one_level.data_vars.values():
        assert data_var.count() > 0


@pytest.mark.parametrize("forecast_hour", range(24, 72, 24))
def test_ecmwf_pressure(forecast_hour: int) -> None:
    """Test whether code can download ECMWF pressure fields.

    Parameters
    ----------
    forecast_hour: int
    """
    pytest.importorskip("eccodes")
    model = NWP_MODELS["ECMWF"]
    init_time = model.get_model_start_with_data()
    valid_time = init_time + datetime.timedelta(hours=forecast_hour)
    variables_850 = ["x_wind", "y_wind", "air_temperature"]
    data_850 = model.get_model_data_pressure(
        init_time, valid_time, variables_850, 850, PA_BBOX
    )
    for data_var in data_850.data_vars.values():
        assert data_var.count() > 0
    variables_500 = ["geopotential_height"]
    data_500 = model.get_model_data_pressure(
        init_time, valid_time, variables_500, 500, PA_BBOX
    )
    for data_var in data_500.data_vars.values():
        assert data_var.count() > 0


@pytest.mark.parametrize("forecast_hour", range(24, 72, 24))
def test_ecmwf_surface(forecast_hour: int) -> None:
    """Test whether code can download ECMWF pressure fields.

    Parameters
    ----------
    forecast_hour: int
    """
    pytest.importorskip("eccodes")
    model = NWP_MODELS["ECMWF"]
    init_time = model.get_model_start_with_data()
    valid_time = init_time + datetime.timedelta(hours=forecast_hour)
    variables_one_level = ["air_pressure_at_mean_sea_level"]
    data_one_level = model.get_model_data_single_level(
        init_time, valid_time, variables_one_level, PA_BBOX
    )
    for data_var in data_one_level.data_vars.values():
        assert data_var.count() > 0
