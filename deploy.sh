#!/bin/bash

set -e

if [ "$#" -eq 5 ]; then
	BOARD=$1
	MODEL_NAME=$2
	BATCH_SIZE=$3
	DATASET=$4
	NR_IMAGES=$5
else
	echo "Error: please provide all required arguments."
	echo "Example: ./deploy.sh KV260 path/to/float_model.h5 batchsize path/to/calibration_dataset nr_images"
	exit 1
fi

sudo mkdir -p /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/
sudo cp arch.json /opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/

python quantize.py --float_model ${MODEL_NAME} --batchsize ${BATCH_SIZE} --dataset ${DATASET} --nr_images ${NR_IMAGES}

compile() {
      vai_c_tensorflow2 \
            --model           build/quant_model/q_model.h5 \
            --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/${BOARD}/arch.json \
		        --output_dir ./deployment_directory \
		        --net_name NERONE4FPGA${MODEL}
}

mkdir -p deployment_directory/
cp NeroneRidingPynq-Classification.ipynb ./deployment_directory/
cp NeroneRidingPynq-Segmentation.ipynb deployment_directory/
compile 2>&1 | tee build/logs/compile_$TARGET.log



echo "-----------------------------------------"
echo "MODEL DEPLOYED"
echo "-----------------------------------------"