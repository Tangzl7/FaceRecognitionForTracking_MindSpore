#!/bin/bash

if [ $# -ne 2 ]
then
  echo "Wrong parameter format."
  echo "Usage:"
  echo "         bash $0 [INPUT_AIR_PATH] [OUTPUT_OM_PATH_NAME]"
  echo "Example: "
  echo "         bash convert_om.sh  xxx.air xx"

  exit 1
fi

input_air_path=$1
output_om_path=$2

atc --input_format=NCHW \
    --framework=1 \
    --model="${input_air_path}" \
    --input_shape="x:1,3,96,64"  \
    --output="${output_om_path}" \
    --output_type=FP32 \
    --op_select_implmode=high_precision \
    --soc_version=Ascend310
