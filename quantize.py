'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

DIVIDER = '-----------------------------------------'



def quant_model(float_model,quant_model,batchsize,image_directory,nr_images):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model, compile=False)

    # get input dimensions of the floating-point model
    height = float_model.input_shape[1]
    width = float_model.input_shape[2]
    colormode = "rgb"

    # build calibration dataset
    total_images = len([name for name in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, name))])
    split = nr_images/total_images
    split = min(1.0, split)
    
    data_gen = ImageDataGenerator(validation_split=split)
    
    
    calibration_dataset = data_gen.flow_from_directory(directory=os.path.dirname(image_directory),
                                                       target_size=(height, width),
                                                       color_mode=colormode,
                                                       batch_size=batchsize,
                                                       subset='validation')
    sys.stdout.write("\033[F")
    # calibration_dataset = DataGen(batchsize, (height, width), image_directory, nr_images, colormode)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=calibration_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to ',quant_model)



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='build/float_model/f_model.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5.')
    ap.add_argument('-q', '--quant_model',  type=str, default='build/quant_model/q_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5.')
    ap.add_argument('-b', '--batchsize',    type=int, default=32, help='Batchsize for quantization. Default is 32.')
    ap.add_argument('-d', '--dataset',      type=str, default='build/calibration_dataset', help='Full path to folder containing TFRecord files. Default is build/tfrecords.')
    ap.add_argument('-n', '--nr_images',    type=int, default=500, help='Number of images for the calibration dataset, default is 500.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --dataset      : ', args.dataset)
    print (' --nr_images    : ', args.nr_images)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.dataset, args.nr_images)


if __name__ ==  "__main__":
    main()