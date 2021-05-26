#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to download and save data for given model and time.

Each level currently saved in a different file to avoid a half-empty
ECMWF file.

"""
import argparse
import datetime
import os.path
import pathlib
import sys
import typing

import dateutil.parser

from .conventions_utilities import RUN_DATE, save_nonsparse_netcdf
from .nwp_models import N_AMER_BBOX, NCEP_VARIABLE_MAP, NWP_MODELS

RUN_DIR = os.path.abspath(".")

############################################################
# Program logic
PARSER = argparse.ArgumentParser(__doc__)
PARSER.add_argument("model_abbrev", choices=NWP_MODELS.keys())
PARSER.add_argument("forecast_hour", type=int)
PARSER.add_argument("--init-time", type=dateutil.parser.parse, default=None)


def main_argv(argv: typing.List[str]) -> int:
    """Call script using given arguments.

    Parameters
    ----------
    argv: list of str

    Returns
    -------
    int
    """
    args = PARSER.parse_args(argv)
    model = NWP_MODELS[args.model_abbrev]
    if args.init_time is None:
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
                    (
                        f"{model.abbrev}_{last_start:%Y%m%dT%H}_"
                        f"f{args.forecast_hour:02d}_{pressure_mb:04d}mb_data.nc4"
                    ),
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
                    (
                        f"{model.abbrev}_{last_start:%Y%m%dT%H}_"
                        f"f{args.forecast_hour:02d}_{pressure_mb:04d}mb_data.nc4"
                    ),
                ),
            )
    return 0


def main() -> int:
    """Call script using command-line arguments.

    Returns
    -------
    int
    """
    return main_argv(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
