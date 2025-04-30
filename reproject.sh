#!/usr/bin/env bash

#COILED n-tasks 3111
#COILED max-workers 100
#COILED region us-west-2
#COILED memory 8 GiB
#COILED container ghcr.io/osgeo/gdal
#COILED forward-aws-credentials True

# Install aws CLI
if [ ! "$(which aws)" ]; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -qq awscliv2.zip
    ./aws/install
fi

# Download file to be processed
filename=$(aws s3 ls --no-sign-request --recursive  s3://sentinel-cogs/sentinel-s2-l2a-cogs/54/E/XR/ | \
           grep ".tif" | \
           awk '{print $4}' | \
           awk "NR==$(($COILED_BATCH_TASK_ID + 1))")
aws s3 cp --no-sign-request s3://sentinel-cogs/$filename in.tif

# Reproject GeoTIFF
gdalwarp -t_srs EPSG:4326 in.tif out.tif

# Move result to processed bucket
aws s3 mv out.tif s3://oss-scratch-space/sentinel-reprojected/$filename