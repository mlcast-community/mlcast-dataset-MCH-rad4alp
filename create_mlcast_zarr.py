#!/usr/bin/env python3

import argparse
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import zarr
from numcodecs import Zstd
from pyproj import CRS, Transformer

logger = logging.getLogger("mylogger")

ARCHIVE_ROOT = Path("/store_new/mch/msrad/radar/swiss/data/hdf5")


@dataclass(frozen=True)
class ProductConfig:
    name: str
    archive_prefix: str
    member_regex: str
    variable_name: str
    standard_name: str
    long_name: str
    units: str
    timestep_seconds: int
    time_resolution: str


PRODUCTS = {
    "rzc": ProductConfig(
        name="RZC_2.5MIN",
        archive_prefix="RZC",
        member_regex=r"^RZC\d{9}.*\.h5$",
        variable_name="rain_rate",
        standard_name="rain_rate",
        long_name="MeteoSwiss radar precipitation rate",
        units="mm h-1",
        timestep_seconds=150,
        time_resolution="PT2M30S",
    ),
    "cpc5": ProductConfig(
        name="CPC_5MIN",
        archive_prefix="CPCHhdf5",
        member_regex=r"^CPC\d{10}_00005\..*\.h5$",
        variable_name="precipitation_amount",
        standard_name="precipitation_amount",
        long_name="MeteoSwiss CombiPrecip 5-minute precipitation accumulation",
        units="kg m-2",
        timestep_seconds=300,
        time_resolution="PT5M",
    ),
    "cpc60": ProductConfig(
        name="CPC_60MIN",
        archive_prefix="CPCHhdf5",
        member_regex=r"^CPC\d{10}_00060\..*\.h5$",
        variable_name="precipitation_amount",
        standard_name="precipitation_amount",
        long_name="MeteoSwiss CombiPrecip 60-minute precipitation accumulation",
        units="kg m-2",
        timestep_seconds=3600,
        time_resolution="PT1H",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--product",
        choices=PRODUCTS,
        required=True,
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start time, e.g. 2016-01-01 or 2016-01-01T12:00",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End time, inclusive",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ARCHIVE_ROOT,
    )

    parser.add_argument(
        "--creator",
        default="Daniel Wolfensberger <daniel.wolfensberger@meteoswiss.ch>",
        help="Must be in 'Name <email>' format",
    )

    parser.add_argument(
        "--created-with",
        default=("https://github.com/mlcast-community/mlcast-dataset-MCH@0.1.0"),
    )

    parser.add_argument(
        "--dataset-version",
        default="0.1.0",
    )

    parser.add_argument(
        "--license",
        default="CC-BY-4.0",
    )

    parser.add_argument(
        "--standard-name",
        default=None,
        help="Override the MLCast standard_name for the data variable",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
    )

    return parser.parse_args()


def parse_datetime(value):
    return datetime.fromisoformat(value)


def iter_days(start, end):
    current = start.date()

    while current <= end.date():
        yield current
        current += timedelta(days=1)


def day_to_julian(day):
    return day.strftime("%y%j")


def archive_path(root, day, product):
    yyddd = day_to_julian(day)

    return root / f"{day.year:04d}" / yyddd / f"{product.archive_prefix}{yyddd}.zip"


def timestamp_from_filename(filename):
    """
    Examples:

        RZC162760000VL.801.h5
        CPC1627600009_00005.801.h5

    Date/time section is YYJJJHHMM.
    """
    basename = Path(filename).name

    match = re.match(
        r"^[A-Z]{3}"
        r"(?P<year>\d{2})"
        r"(?P<julian>\d{3})"
        r"(?P<hour>\d{2})"
        r"(?P<minute>\d{2})",
        basename,
    )

    if match is None:
        raise ValueError(f"Could not extract timestamp from {basename}")

    year = 2000 + int(match["year"])
    julian = int(match["julian"])

    return datetime.strptime(f"{year}-{julian:03d}", "%Y-%j").replace(
        hour=int(match["hour"]),
        minute=int(match["minute"]),
        tzinfo=timezone.utc,
    )


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode()

    return value


def find_data_group(h5):
    for dataset_name in sorted(h5):
        if not dataset_name.startswith("dataset"):
            continue

        dataset = h5[dataset_name]

        for data_name in sorted(dataset):
            if not data_name.startswith("data"):
                continue

            group = dataset[data_name]

            if "data" in group:
                return group

    raise ValueError("Could not locate ODIM data array")


def read_odim_file(fileobj):
    """
    Read one MeteoSwiss ODIM-HDF5 field.

    Returns
    -------
    data : ndarray
        Shape (y, x), float32.

    x, y : ndarray
        Projected coordinate arrays.

    crs : pyproj.CRS
    """
    with h5py.File(fileobj, "r") as h5:
        data_group = find_data_group(h5)

        raw = data_group["data"][...]

        what = data_group.get("what")

        gain = 1.0
        offset = 0.0
        nodata = None
        undetect = None

        if what is not None:
            gain = float(what.attrs.get("gain", 1.0))
            offset = float(what.attrs.get("offset", 0.0))
            nodata = what.attrs.get("nodata")
            undetect = what.attrs.get("undetect")

        data = raw.astype(np.float32) * gain + offset

        invalid = np.zeros(
            raw.shape,
            dtype=bool,
        )

        if nodata is not None:
            invalid |= raw == nodata

        if undetect is not None:
            invalid |= raw == undetect

        data[invalid] = np.nan

        where = h5["where"]

        projdef = decode_attr(
            where.attrs.get(
                "projdef",
                "EPSG:2056",
            )
        )

        try:
            crs = CRS.from_user_input(projdef)
        except Exception:  # noqa: BLE001
            crs = CRS.from_epsg(2056)

        xsize = int(
            where.attrs.get(
                "xsize",
                raw.shape[1],
            )
        )

        ysize = int(
            where.attrs.get(
                "ysize",
                raw.shape[0],
            )
        )

        xscale = float(
            where.attrs.get(
                "xscale",
                1000.0,
            )
        )

        yscale = float(
            where.attrs.get(
                "yscale",
                1000.0,
            )
        )

        ll_lon = float(where.attrs["LL_lon"])
        ll_lat = float(where.attrs["LL_lat"])

        transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            crs,
            always_xy=True,
        )

        x0, y0 = transformer.transform(
            ll_lon,
            ll_lat,
        )

        x = x0 + (np.arange(xsize) + 0.5) * xscale

        y = y0 + (np.arange(ysize) + 0.5) * yscale

        #
        # Ensure y increases.
        #
        if y[0] > y[-1]:
            y = y[::-1]
            data = data[::-1]

        return data, x, y, crs


def make_latlon(x, y, crs):
    xx, yy = np.meshgrid(x, y)

    transformer = Transformer.from_crs(
        crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    lon, lat = transformer.transform(
        xx,
        yy,
    )

    return (
        lat.astype(np.float32),
        lon.astype(np.float32),
    )


def build_dataset(
    data,
    times,
    x,
    y,
    crs,
    product,
    standard_name,
    creator,
    created_with,
    dataset_version,
    license_name,
    consistent_timestep_start,
):
    lat, lon = make_latlon(
        x,
        y,
        crs,
    )

    crs_wkt = crs.to_wkt()

    times = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    ds = xr.Dataset(
        data_vars={
            product.variable_name: (
                ("time", "y", "x"),
                data.astype(np.float32),
            ),
            "crs": xr.DataArray(
                np.int32(0),
                attrs={
                    "spatial_ref": crs_wkt,
                    "crs_wkt": crs_wkt,
                    **crs.to_cf(),
                },
            ),
        },
        coords={
            "time": (
                "time",
                times,
                {
                    "standard_name": "time",
                    "axis": "T",
                },
            ),
            "x": (
                "x",
                x,
                {
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate of projection",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                "y",
                y,
                {
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate of projection",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "lat": (
                ("y", "x"),
                lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "lon": (
                ("y", "x"),
                lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
        },
    )

    ds[product.variable_name].attrs.update(
        {
            "standard_name": standard_name,
            "long_name": product.long_name,
            "units": product.units,
            "grid_mapping": "crs",
        }
    )

    ds.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": product.long_name,
            "institution": ("Federal Office of Meteorology and Climatology MeteoSwiss"),
            "source": ("MeteoSwiss operational radar precipitation product"),
            "license": license_name,
            "consistent_timestep_start": consistent_timestep_start,
            "mlcast_created_on": datetime.now(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="seconds"),
            "mlcast_created_by": creator,
            "mlcast_created_with": created_with,
            "mlcast_dataset_version": dataset_version,
            "mlcast_dataset_identifier": ("CH-MeteoSwiss-precipitation"),
        }
    )

    return ds


def open_existing(path):
    try:
        return xr.open_zarr(
            path,
            consolidated=True,
        )
    except (ValueError, KeyError):
        logger.warning(
            "Consolidated metadata not found for %s; opening unconsolidated",
            path,
        )

        return xr.open_zarr(
            path,
            consolidated=False,
        )


def get_existing_last_time(path):
    if not path.exists():
        return None

    ds = open_existing(path)

    try:
        if ds.sizes.get("time", 0) == 0:
            return None

        return np.datetime64(
            ds.time.values[-1],
            "ns",
        )

    finally:
        ds.close()


def get_existing_first_time(path):
    if not path.exists():
        return None

    ds = open_existing(path)

    try:
        if ds.sizes.get("time", 0) == 0:
            return None

        return np.datetime64(
            ds.time.values[0],
            "ns",
        )

    finally:
        ds.close()


def check_existing_geometry(
    ds,
    output,
):
    existing = open_existing(output)

    try:
        if existing.sizes["x"] != ds.sizes["x"]:
            raise ValueError("Existing Zarr has different x dimension")

        if existing.sizes["y"] != ds.sizes["y"]:
            raise ValueError("Existing Zarr has different y dimension")

        if not np.allclose(
            existing.x.values,
            ds.x.values,
        ):
            raise ValueError("Existing Zarr has different x coordinates")

        if not np.allclose(
            existing.y.values,
            ds.y.values,
        ):
            raise ValueError("Existing Zarr has different y coordinates")

    finally:
        existing.close()


def create_encoding(
    ds,
    variable_name,
):
    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    compressor = Zstd(level=5)

    return {
        variable_name: {
            "chunks": (1, ny, nx),
            "compressor": compressor,
            "dtype": "float32",
        },
        "time": {
            "chunks": (4096,),
        },
        "x": {
            "chunks": (nx,),
        },
        "y": {
            "chunks": (ny,),
        },
        "lat": {
            "chunks": (ny, nx),
            "compressor": compressor,
        },
        "lon": {
            "chunks": (ny, nx),
            "compressor": compressor,
        },
    }


def write_dataset(ds, output, variable_name):
    output = Path(output)

    if output.exists():
        ds.to_zarr(
            output,
            mode="a",
            append_dim="time",
            consolidated=False,
        )

    else:
        ds.to_zarr(
            output,
            mode="w",
            encoding=create_encoding(
                ds,
                variable_name,
            ),
            consolidated=False,
            zarr_format=2,
        )


def read_one_day(
    archive,
    start,
    end,
    product,
    minimum_time=None,
):
    arrays = []
    times = []

    reference_x = None
    reference_y = None
    reference_crs = None

    regex = re.compile(product.member_regex)

    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if regex.match(Path(name).name)]

        members.sort(key=timestamp_from_filename)

        for member in members:
            timestamp = timestamp_from_filename(member)

            if timestamp < start or timestamp > end:
                continue

            t64 = np.datetime64(
                timestamp,
                "ns",
            )

            if minimum_time is not None and t64 <= minimum_time:
                continue

            logger.debug(
                "Reading %s [%s]",
                member,
                timestamp,
            )

            raw_bytes = zf.read(member)

            #
            # h5py can read an in-memory file image.
            #
            with h5py.File(
                io.BytesIO(raw_bytes),
                "r",
            ):
                #
                # read_odim_file expects something h5py can open,
                # so duplicate the small decode logic here by
                # passing the BytesIO object instead.
                #
                pass

            buffer = io.BytesIO(raw_bytes)

            data, x, y, crs = read_odim_file(buffer)

            if reference_x is None:
                reference_x = x
                reference_y = y
                reference_crs = crs

            else:
                if not np.allclose(
                    reference_x,
                    x,
                ):
                    raise ValueError(f"x grid changed in {member}")

                if not np.allclose(
                    reference_y,
                    y,
                ):
                    raise ValueError(f"y grid changed in {member}")

            arrays.append(data)
            times.append(timestamp)

    if not arrays:
        return None

    times64 = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    order = np.argsort(times64)

    data = np.stack(arrays)[order]
    times64 = times64[order]

    if len(np.unique(times64)) != len(times64):
        raise ValueError(f"Duplicate timestamps in {archive}")

    return (
        data,
        times64,
        reference_x,
        reference_y,
        reference_crs,
    )


def compute_missing_times(
    times,
    timestep_seconds,
):
    """
    Compute all expected timestamps between first and last
    actual observation which are absent from the dataset.
    """
    times = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    if len(times) < 2:
        return np.array(
            [],
            dtype="datetime64[ns]",
        )

    step = np.timedelta64(
        timestep_seconds,
        "s",
    )

    expected = np.arange(
        times[0],
        times[-1] + step,
        step,
        dtype="datetime64[ns]",
    )

    return np.setdiff1d(
        expected,
        times,
    )


def update_time_metadata(
    output,
    product,
):
    """
    Recompute missing_times from the complete Zarr time axis.

    - Preserves all existing global attributes.
    - Does not create missing_times when no timestamps are missing.
    - Stores missing_times as a proper coordinate when needed.
    """
    output = Path(output)

    ds = open_existing(output)

    try:
        times = np.asarray(
            ds.time.values,
            dtype="datetime64[ns]",
        )

        if len(times) == 0:
            logger.warning("Dataset contains no timesteps")
            return

        dt = np.diff(times)

        if np.any(dt <= np.timedelta64(0, "ns")):
            raise ValueError("Time coordinate is not strictly increasing")

        missing = compute_missing_times(
            times,
            product.timestep_seconds,
        )

        first_time = np.datetime_as_string(
            times[0],
            unit="s",
        )

    finally:
        ds.close()

    logger.info(
        "Dataset starts at %s",
        first_time,
    )

    logger.info(
        "Found %d missing expected timesteps",
        len(missing),
    )

    #
    # Open root group and preserve attributes.
    #
    group = zarr.open_group(
        str(output),
        mode="a",
    )

    original_attrs = dict(group.attrs)

    #
    # ---------------------------------------------------------
    # No missing timestamps:
    #
    # Do NOT create an empty missing_times variable.
    #
    # Also delete an old one left by a previous run.
    # ---------------------------------------------------------
    #
    if len(missing) == 0:
        if "missing_times" in group:
            logger.info("Removing existing empty 'missing_times' array")

            del group["missing_times"]

    else:
        #
        # Remove an existing version first.
        #
        if "missing_times" in group:
            del group["missing_times"]

        #
        # IMPORTANT:
        #
        # Use the SAME name for the coordinate and dimension.
        # This makes missing_times an xarray index coordinate
        # when the store is reopened.
        #
        missing_ds = xr.Dataset(
            coords={
                "missing_times": (
                    "missing_times",
                    missing.astype("datetime64[ns]"),
                ),
            }
        )

        missing_ds["missing_times"].attrs.update(
            {
                "standard_name": "time",
                "long_name": ("Expected regular timesteps missing from the dataset"),
            }
        )

        missing_ds.to_zarr(
            output,
            mode="a",
            consolidated=False,
        )

    #
    # Restore all original root attributes.
    #
    group = zarr.open_group(
        str(output),
        mode="a",
    )

    group.attrs.clear()
    group.attrs.update(original_attrs)

    #
    # Add/update MLCast temporal metadata.
    #
    group.attrs["consistent_timestep_start"] = first_time

    #
    # Consolidate once.
    #
    zarr.consolidate_metadata(str(output))


def main():
    args = parse_args()

    logger.basicConfig(
        level=getattr(
            logging,
            args.log_level.upper(),
        ),
        format=("%(asctime)s %(levelname)s %(message)s"),
    )

    product = PRODUCTS[args.product]

    standard_name = args.standard_name or product.standard_name

    start = parse_datetime(args.start)

    end = parse_datetime(args.end)

    if end < start:
        raise ValueError("--end must be >= --start")

    last_time = get_existing_last_time(args.output)

    if last_time is not None:
        logger.info(
            "Existing dataset ends at %s",
            last_time,
        )

    first_written = False
    geometry_checked = False

    first_time = get_existing_first_time(args.output)

    last_time = get_existing_last_time(args.output)

    for day in iter_days(
        start,
        end,
    ):
        archive = archive_path(
            args.archive_root,
            day,
            product,
        )

        if not archive.exists():
            logger.warning(
                "Archive does not exist: %s",
                archive,
            )
            continue

        logger.info(
            "Processing %s",
            archive,
        )

        result = read_one_day(
            archive=archive,
            start=start,
            end=end,
            product=product,
            minimum_time=last_time,
        )

        if result is None:
            logger.info(
                "No new data in %s",
                archive,
            )
            continue

        (
            data,
            times,
            x,
            y,
            crs,
        ) = result

        #
        # Existing dataset: use its original start time.
        #
        if first_time is not None:
            consistent_start = np.datetime_as_string(
                first_time,
                unit="s",
            )
        else:
            consistent_start = np.datetime_as_string(
                times[0],
                unit="s",
            )
            first_time = times[0]

        ds = build_dataset(
            data=data,
            times=times,
            x=x,
            y=y,
            crs=crs,
            product=product,
            standard_name=standard_name,
            creator=args.creator,
            created_with=args.created_with,
            dataset_version=(args.dataset_version),
            license_name=args.license,
            consistent_timestep_start=(consistent_start),
        )

        logger.info(
            "Writing %d timesteps: %s -> %s",
            len(times),
            times[0],
            times[-1],
        )

        if args.output.exists() and not geometry_checked:
            check_existing_geometry(
                ds,
                args.output,
            )
            geometry_checked = True

        write_dataset(
            ds=ds,
            output=args.output,
            variable_name=(product.variable_name),
        )

        last_time = times[-1]
        first_written = True

    if not args.output.exists():
        logger.warning("No output was created")
        return

    #
    # Recompute MLCast temporal metadata across the complete
    # output after all daily appends.
    #
    update_time_metadata(
        args.output,
        product,
    )

    if first_written:
        logger.info(
            "Finished writing %s",
            args.output,
        )
    else:
        logger.info("Nothing new was written; temporal metadata was refreshed")


if __name__ == "__main__":
    main()
