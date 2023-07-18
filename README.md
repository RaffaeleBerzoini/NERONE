# NERONE : the Fast Way to Efficiently Execute Your Deep Learning Algorithm at the Edge


## Setup

* Open a command prompt and execute:
    ```console
    git clone https://github.com/Xilinx/Vitis-AI.git
    cd Vitis-AI
    git checkout 1.4.1
    ```
* Follow the Vitis-AI installation process [here](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Host) 
  * Once the installation is completed open a terminal in the Vitis-AI directory and execute:  
  ```console
  git clone https://github.com/necst/NERONE
  ./docker_run.sh xilinx/vitis-ai-cpu:1.4.1.978
  ```


* Put your image into the working directory. You should have something like this:

```text
NERONE   # your WRK_DIR
.
├── build
  ├── calibration_dataset
├── arch.json
├── deploy.sh
└── ...
```

## Quantization and compilation

* In the command prompt execute:
  ```console
    Vitis-AI /workspace > conda activate vitis-ai-tensorflow2
    (vitis-ai-tensorflow2) Vitis-AI /workspace > cd NERONE
    (vitis-ai-tensorflow2) chmod +x deploy.sh
    (vitis-ai-tensorflow2) ./deploy.sh ZCU104 build/float_model/f_model.h5 32 build/calibration_dataset 500
    ```
* Change the arguments to suit your needs in the last command

## Deployment on the evaluation board

Set up the evaluation board as stated [here](https://github.com/Xilinx/DPU-PYNQ).


Copy the `deployment_directory` directory to your board with `scp -r deployment_directory/ root@192.168.1.227:~/.` assuming that the target board IP address is 192.168.1.227 - adjust this as appropriate for your system.

You could also directly copy the folder to the board SD card 

On the board open `NeroneRidingPynq-Classification.ipynb` or `NeroneRidingPynq-Segmentation.ipynb` for classification and segmentation, respectively, and execute it. Your quantized and compile model is now executing on the FPGA!

## Asjustment Options

Please keep in mind that both in the `quantize.py` file and in the `.ipynb` files images are loaded and preprocessed based on our tests. You might want to change that. In all files you can proceed exactly as you did for training and inference on GPU/CPU without worrying about being working on an FPGA.

## Our models

In the folder `results` you can find float, quantized and compiled models (for the ZCU104) obtained and described in the associated publication.

Refer also to the [following repository](https://github.com/necst/SENECA) for a detailed description of one of the test cases.

<a id="paper_ref"></a>
# Associated Publication

If you find this repository useful, please use the following citation:

```
@ARTICLE{10185039,
  author={Berzoini, Raffaele and D'Arnese, Eleonora and Conficconi, Davide and Santambrogio, Marco D.},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={NERONE: The Fast Way to Efficiently Execute Your Deep Learning Algorithm At the Edge}, 
  year={2023},
  pages={1-9},
  doi={10.1109/JBHI.2023.3296142}}