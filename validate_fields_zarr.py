#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


logger = logging.getLogger("fast_validation")


PRODUCTS = {
    "rzc": {
        "filename": "RZC.zarr",
        "variable": "prate",
        "label": "RZC",
        "units": "kg m-2 h-1",
    },
    "cpc5": {
        "filename": "CPC5.zarr",
        "variable": "precipitation_amount",
        "label": "CPC5",
        "units": "kg m-2",
    },
    "cpc60": {
        "filename": "CPC60.zarr",
        "variable": "precipitation_amount",
        "label": "CPC60",
        "units": "kg m-2",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_images"),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of wet cases to save",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum number of random RZC timesteps to inspect",
    )

    parser.add_argument(
        "--min-max",
        type=float,
        default=5.0,
        help=(
            "RZC field is considered wet if its maximum "
            "is at least this value"
        ),
    )

    parser.add_argument(
        "--min-wet-fraction",
        type=float,
        default=0.002,
        help=(
            "Minimum fraction of pixels above --wet-threshold"
        ),
    )

    parser.add_argument(
        "--wet-threshold",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
    )

    return parser.parse_args()


def open_store(path):
    try:
        return xr.open_zarr(
            path,
            consolidated=True,
        )
    except (ValueError, KeyError):
        return xr.open_zarr(
            path,
            consolidated=False,
        )


def nearest_time_index(times, target):
    """
    Return index of timestamp nearest to target.
    """
    times = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    target = np.datetime64(
        target,
        "ns",
    )

    return int(
        np.argmin(
            np.abs(times - target)
        )
    )


def is_wet_field(
    data,
    min_max,
    min_wet_fraction,
    wet_threshold,
):
    finite = np.isfinite(
        data
    )

    if not np.any(finite):
        return False

    max_value = np.nanmax(
        data
    )

    wet_fraction = np.mean(
        finite
        & (data >= wet_threshold)
    )

    return (
        max_value >= min_max
        and wet_fraction >= min_wet_fraction
    )


def save_image(
    ds,
    variable,
    index,
    label,
    units,
    output_file,
):
    field = ds[variable].isel(
        time=index
    ).values

    timestamp = np.datetime64(
        ds.time.values[index],
        "s",
    )

    x = ds.x.values
    y = ds.y.values

    positive = field[
        np.isfinite(field)
        & (field > 0)
    ]

    if positive.size:
        vmax = float(
            np.percentile(
                positive,
                99.5,
            )
        )
    else:
        vmax = None

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        field,
        origin="upper",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max(),
        ],
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )
    
    image = ax.pcolormesh(x,y,field)
    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        units
    )

    ax.set_title(
        f"{label} — "
        f"{np.datetime_as_string(timestamp, unit='m')}"
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level.upper(),
        ),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Open stores once.
    #
    stores = {}

    for name, config in PRODUCTS.items():
        path = (
            args.input_dir
            / config["filename"]
        )

        logger.info(
            "Opening %s",
            path,
        )

        stores[name] = open_store(
            path
        )

    rzc = stores["rzc"]

    ntime = rzc.sizes["time"]

    rng = np.random.default_rng(
        args.seed
    )

    #
    # Randomly probe RZC only.
    #
    candidates = rng.permutation(
        ntime
    )

    selected = []

    for index in candidates[
        : min(args.max_attempts, ntime)
    ]:
        index = int(
            index
        )

        #
        # Load ONE radar image only.
        #
        field = rzc[
            PRODUCTS["rzc"]["variable"]
        ].isel(
            time=index
        ).values

        if not is_wet_field(
            field,
            min_max=args.min_max,
            min_wet_fraction=args.min_wet_fraction,
            wet_threshold=args.wet_threshold,
        ):
            continue

        timestamp = rzc.time.values[
            index
        ]

        logger.info(
            "Selected wet case %s",
            timestamp,
        )

        selected.append(
            (
                index,
                timestamp,
            )
        )

        if len(selected) >= args.n:
            break

    if not selected:
        logger.warning(
            "No wet cases found"
        )

        for ds in stores.values():
            ds.close()

        return

    logger.info(
        "Found %d wet cases",
        len(selected),
    )

    #
    # Produce images.
    #
    for case_number, (
        rzc_index,
        timestamp,
    ) in enumerate(
        selected,
        start=1,
    ):
        timestamp_s = np.datetime64(
            timestamp,
            "s",
        )

        tag = (
            np.datetime_as_string(
                timestamp_s,
                unit="m",
            )
            .replace("-", "")
            .replace(":", "")
            .replace("T", "_")
        )

        case_dir = (
            args.output_dir
            / f"{case_number:02d}_{tag}"
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        #
        # RZC exact selected timestep.
        #
        save_image(
            ds=stores["rzc"],
            variable=PRODUCTS["rzc"]["variable"],
            index=rzc_index,
            label=PRODUCTS["rzc"]["label"],
            units=PRODUCTS["rzc"]["units"],
            output_file=case_dir / "RZC.png",
        )

        #
        # CPC5 nearest timestamp.
        #
        cpc5_index = nearest_time_index(
            stores["cpc5"].time.values,
            timestamp,
        )

        save_image(
            ds=stores["cpc5"],
            variable=PRODUCTS["cpc5"]["variable"],
            index=cpc5_index,
            label=PRODUCTS["cpc5"]["label"],
            units=PRODUCTS["cpc5"]["units"],
            output_file=case_dir / "CPC5.png",
        )

        #
        # CPC60 nearest timestamp.
        #
        cpc60_index = nearest_time_index(
            stores["cpc60"].time.values,
            timestamp,
        )

        save_image(
            ds=stores["cpc60"],
            variable=PRODUCTS["cpc60"]["variable"],
            index=cpc60_index,
            label=PRODUCTS["cpc60"]["label"],
            units=PRODUCTS["cpc60"]["units"],
            output_file=case_dir / "CPC60.png",
        )

        logger.info(
            "Saved case %d: %s",
            case_number,
            case_dir,
        )

    for ds in stores.values():
        ds.close()


if __name__ == "__main__":
    main()
