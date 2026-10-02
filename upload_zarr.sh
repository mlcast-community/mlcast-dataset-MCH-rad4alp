#!/bin/bash

set -euo pipefail

SOURCE_PATH="/store_new/mch/msrad/radar/MLCast"
REMOTE_BASE="ewc:mlcast-source-datasets/radar_precipitation/CH-rad4alp/v1.0.0"

ZARRS=(
    "RZC.zarr"
    "CPC5.zarr"
    "CPC60.zarr"
)

for ZARR in "${ZARRS[@]}"; do

    echo
    echo "============================================================"
    echo "Uploading ${ZARR}"
    echo "============================================================"

    rclone sync \
        "${SOURCE_PATH}/${ZARR}/" \
        "${REMOTE_BASE}/${ZARR}/" \
        -P \
        --stats=2s \
        --stats-one-line

done

echo
echo "All Zarr stores uploaded successfully."
