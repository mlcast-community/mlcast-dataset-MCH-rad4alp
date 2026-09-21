# MeteoSwiss → MLCast Zarr Converter

This repository contains tools to convert historical MeteoSwiss radar precipitation products into MLCast-compatible Zarr datasets.

The converter currently supports three products:

| Product    | CLI name | Temporal resolution | Output variable        | Output store |
| ---------- | -------- | ------------------: | ---------------------- | ------------ |
| RZC        | `rzc`    |             2.5 min | `rain_rate`            | `RZC.zarr`   |
| CPC 5 min  | `cpc5`   |               5 min | `precipitation_amount` | `CPC5.zarr`  |
| CPC 60 min | `cpc60`  |              60 min | `precipitation_amount` | `CPC60.zarr` |

The three products are written to separate Zarr stores so that each dataset has a consistent temporal resolution.

---

## Input data

The converter reads the historical MeteoSwiss HDF5 radar archive located under:

```text
/store_new/mch/msrad/radar/swiss/data/hdf5/
```

The archive is organized by year and Julian day.

For example:

```text
/store_new/mch/msrad/radar/swiss/data/hdf5/2016/16276/
```

### RZC

RZC data are stored in one ZIP archive per day:

```text
RZC16276.zip
```

with HDF5 files such as:

```text
RZC162760000VL.801.h5
RZC162760002VL.801.h5
RZC162760005VL.801.h5
RZC162760007VL.801.h5
...
```

The nominal temporal resolution is 2.5 minutes.

### CPC

CPC data are also stored in daily ZIP archives:

```text
CPCHhdf516276.zip
```

The same archive contains both 5-minute and 60-minute products.

Examples:

```text
CPC1627600009_00005.801.h5
CPC1627600009_00060.801.h5
CPC1627600059_00005.801.h5
...
```

The suffix identifies the accumulation period:

```text
_00005   → 5-minute CPC
_00060   → 60-minute CPC
```

The converter automatically filters the required files according to `--product`.

---

# Converter

The main program is:

```text
create_mlcast_zarr.py
```

It reads the source ZIP archives, extracts the HDF5 fields, converts them to an xarray dataset, and writes an MLCast-compatible Zarr v2 store.

## Main features

The converter:

* supports RZC, CPC 5-minute and CPC 60-minute products;
* reads directly from the daily ZIP files;
* converts ODIM-style HDF5 data to physical values;
* converts missing/undetect values to `NaN`;
* writes data with dimensions:

```text
(time, y, x)
```

* writes projected `x` and `y` coordinates;
* generates 2-D latitude and longitude coordinates;
* stores CRS information using CF-compatible metadata;
* writes `float32` precipitation data;
* uses ZSTD compression;
* uses one complete radar image per Zarr chunk:

```text
(1, ny, nx)
```

* supports appending to an existing Zarr store;
* skips timestamps that are already present;
* checks that appended data use the same spatial grid;
* checks that time remains strictly increasing;
* detects missing timestamps;
* adds MLCast metadata;
* consolidates Zarr metadata after processing.

---

# Usage

## RZC

```bash
python create_mlcast_zarr.py \
    --product rzc \
    --start 2016-01-01 \
    --end 2016-01-10 \
    --output /store_new/mch/msrad/radar/MLCast/RZC.zarr
```

## CPC 5-minute

```bash
python create_mlcast_zarr.py \
    --product cpc5 \
    --start 2016-01-01 \
    --end 2016-01-10 \
    --output /store_new/mch/msrad/radar/MLCast/CPC5.zarr
```

## CPC 60-minute

```bash
python create_mlcast_zarr.py \
    --product cpc60 \
    --start 2016-01-01 \
    --end 2016-01-10 \
    --output /store_new/mch/msrad/radar/MLCast/CPC60.zarr
```

Dates can also contain a time:

```bash
--start 2016-01-01T12:00
--end   2016-01-02T18:00
```

---

# Appending data

If the output Zarr store already exists, the converter appends new timesteps rather than rewriting the complete dataset.

For example:

```bash
python create_mlcast_zarr.py \
    --product rzc \
    --start 2016-01-01 \
    --end 2017-12-31 \
    --output RZC.zarr
```

followed later by:

```bash
python create_mlcast_zarr.py \
    --product rzc \
    --start 2016-01-01 \
    --end 2020-12-31 \
    --output RZC.zarr
```

will skip timesteps already present and append only data newer than the last timestamp in the existing store.

The intended workflow is therefore chronological:

```text
2016
 ↓
2017
 ↓
2018
 ↓
...
```

The converter is not intended to insert new timesteps into the middle of an already-written Zarr dataset.

---

# MLCast metadata

The output contains the global metadata required by the MLCast validator, including:

```text
license
mlcast_created_on
mlcast_created_by
mlcast_created_with
mlcast_dataset_version
mlcast_dataset_identifier
mlcast_dataset_identifier_format
consistent_timestep_start
```

The dataset identifiers distinguish the individual MeteoSwiss products.

For example:

```text
CH-MeteoSwiss-precipitation-RZC
CH-MeteoSwiss-precipitation-CPC5
CH-MeteoSwiss-precipitation-CPC60
```

with an identifier format such as:

```text
{country_code}-{entity}-{physical_variable}-{common_name}
```

The exact identifier and repository metadata should be kept synchronized with the corresponding MLCast contribution repository.

---

# Missing timestamps

The converter compares the actual time coordinate with the expected regular time axis.

Expected intervals are:

```text
RZC     150 s
CPC5    300 s
CPC60  3600 s
```

If missing timestamps are detected, they are recorded in the Zarr metadata according to the MLCast convention.

If no timestamps are missing, an empty `missing_times` array is not written.

---

# Running with SLURM

The wrapper script runs the three products in parallel using a SLURM job array.

Example wrapper:

```bash
#!/bin/bash
#SBATCH --job-name=mlcast
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/mlcast_%A_%a.out
#SBATCH --error=logs/mlcast_%A_%a.err

set -euo pipefail

mkdir -p logs

source /scratch/mch/wolfensb/pyrad_venv/miniforge3/bin/activate
source /scratch/mch/wolfensb/.mlcast_venv/bin/activate

#################################
# Modify here
#################################
START_TIME="2016-01-01"
END_TIME="2016-01-10"
OUTPUT_DIR="/store_new/mch/msrad/radar/MLCast/"
#################################

PRODUCTS=(rzc cpc5 cpc60)
OUTPUTS=(RZC.zarr CPC5.zarr CPC60.zarr)

PRODUCT="${PRODUCTS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"

echo "Processing product: ${PRODUCT}"
echo "Period: ${START_TIME} -> ${END_TIME}"
echo "Output: ${OUTPUT}"

python create_mlcast_zarr.py \
    --product "${PRODUCT}" \
    --start "${START_TIME}" \
    --end "${END_TIME}" \
    --output "${OUTPUT_DIR}/${OUTPUT}"
```

Submit the job with:

```bash
sbatch create_mlcast.slurm
```

The job array launches three independent tasks:

```text
SLURM_ARRAY_TASK_ID=0 → rzc   → RZC.zarr
SLURM_ARRAY_TASK_ID=1 → cpc5  → CPC5.zarr
SLURM_ARRAY_TASK_ID=2 → cpc60 → CPC60.zarr
```

Because each task writes to a different Zarr store, the three jobs can run concurrently without competing for the same output.

---

# SLURM logs

Standard output and errors are written to:

```text
logs/mlcast_<JOB_ID>_<ARRAY_ID>.out
logs/mlcast_<JOB_ID>_<ARRAY_ID>.err
```

For example:

```text
logs/mlcast_123456_0.out
logs/mlcast_123456_1.out
logs/mlcast_123456_2.out
```

To inspect the status of the jobs:

```bash
squeue -u "$USER"
```

To inspect completed jobs:

```bash
sacct -j <JOB_ID>
```

---

# Processing a larger archive

For a complete historical conversion, only the date range in the wrapper needs to be changed:

```bash
START_TIME="2016-01-01"
END_TIME="2025-12-31"
```

The converter processes the daily archives sequentially within each product, while SLURM parallelizes the three products.

Parallelizing several jobs that write to the **same** Zarr store is intentionally avoided because simultaneous appends to one Zarr store can cause metadata and chunk-writing conflicts.

---

# Validating the output

Install the MLCast dataset validator in a suitable Python environment:

```bash
pip install mlcast-dataset-validator
```

Then validate each Zarr store:

```bash
mlcast.validate_dataset \
    source_data \
    radar_precipitation \
    /store_new/mch/msrad/radar/MLCast/RZC.zarr
```

For CPC5:

```bash
mlcast.validate_dataset \
    source_data \
    radar_precipitation \
    /store_new/mch/msrad/radar/MLCast/CPC5.zarr
```

For CPC60:

```bash
mlcast.validate_dataset \
    source_data \
    radar_precipitation \
    /store_new/mch/msrad/radar/MLCast/CPC60.zarr
```

A short test dataset will fail the MLCast minimum temporal-coverage requirement because MLCast requires at least three years of data. This is expected during testing.

Structural checks such as the following should nevertheless pass:

```text
time coordinate
latitude/longitude coordinates
projected coordinates
spatial resolution
spatial dimensions
chunking
compression
dimension order
data type
CF variable naming
CRS metadata
license metadata
MLCast metadata
Zarr v2 compatibility
consolidated metadata
```

---

# Inspecting the Zarr manually

The resulting dataset can be opened using xarray:

```python
import xarray as xr

ds = xr.open_zarr(
    "/store_new/mch/msrad/radar/MLCast/CPC5.zarr",
    consolidated=True,
)

print(ds)
print(ds.attrs)
```

The precipitation field should have dimensions:

```python
print(ds["precipitation_amount"].dims)
```

```text
('time', 'y', 'x')
```

For RZC:

```python
print(ds["rain_rate"].dims)
```

```text
('time', 'y', 'x')
```

The chunk structure can be inspected with:

```python
print(ds["precipitation_amount"].chunks)
```

The time chunks should contain one timestep each.

---

# Checking temporal resolution

The actual time intervals can be checked with:

```python
import numpy as np
import xarray as xr

ds = xr.open_zarr(
    "CPC5.zarr",
    consolidated=True,
)

dt = np.diff(ds.time.values).astype("timedelta64[s]").astype(int)

values, counts = np.unique(
    dt,
    return_counts=True,
)

for value, count in zip(values, counts):
    print(f"{value:5d} s : {count}")
```

Expected dominant values are:

```text
RZC     →  150 s
CPC5    →  300 s
CPC60   → 3600 s
```

Larger intervals indicate missing source timesteps.

---

# Output structure

A typical CPC5 dataset looks conceptually like:

```text
<xarray.Dataset>

Dimensions:
    time
    y: 640
    x: 710

Coordinates:
    time
    x
    y
    lat (y, x)
    lon (y, x)

Data variables:
    precipitation_amount (time, y, x)
    crs

Attributes:
    Conventions
    title
    institution
    source
    license
    consistent_timestep_start
    mlcast_created_on
    mlcast_created_by
    mlcast_created_with
    mlcast_dataset_version
    mlcast_dataset_identifier_format
    mlcast_dataset_identifier
```

RZC has the same structure but uses `rain_rate` as the meteorological data variable.

---

# Notes

* Zarr v2 is currently used.
* Consolidated metadata are created at the end of the conversion.
* The source radar grid has a spatial resolution of 1 km.
* The current grid size is approximately `640 × 710` pixels.
* CPC precipitation amounts are stored using CF-compatible metadata.
* Source `nodata` and `undetect` values are converted to `NaN`.
* The converter should normally be run chronologically when extending an existing dataset.
* RZC, CPC5, and CPC60 should remain separate datasets because they have different temporal characteristics.
