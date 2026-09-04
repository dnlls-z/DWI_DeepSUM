# DeepSUM-DWI
Image registration and super-resolution network for diffusion-weighted imaging (DWI) MRI, based on the DeepSUM model (https://github.com/diegovalsesia/deepsum).

DeepSUM is a Multi Image Super-Resolution (MISR) deep neural network that exploits both spatial and temporal correlations to recover a single high resolution image from multiple unregistered low resolution images. 

DeepSUM-DWI is a TensorFlow 2 implementation of DeepSUM for repeated diffusion-weighted imaging (DWI) MRI. It jointly registers multiple low-resolution DWI repetitions and reconstructs a single high-resolution image. The model uses TensorFlow's `compat.v1` graph workflow while running on TensorFlow 2.

This project was funded by UNAM-PAPIIT grant: TA101224.

## Setup
DeepSUM-DWI uses Tensorflow 2.20. It is recommended to follow the official TensorFlow installation instructions when GPU support or platform-specific setup is needed: https://www.tensorflow.org/install/pip 

## Workflow
1. Open `master_scripts/I_config_custom.ipynb`. Set the project paths, input MATLAB file, run nickname, split parameters, and patch size, then run all cells. This creates or updates a versioned configuration in `runs/<nickname>/config_files/`.

2. Open `master_scripts/II_master_script.ipynb`. Set the same nickname and desired number of epochs, then run all cells to create the dataset, train `UDN.py`, and super-resolve the test set.

The input MATLAB file must contain dwi_lr, dwi_hr, mask_lr, mask_hr, and norm_value arrays. Run configuration files record the input path, output dataset/test-set directories, number of repetitions (T_in), scale (R), image size, normalization statistics, and train/test split.

Dataset creation and inference can also be run from the project root:

```python CD.py --config config_rd_12sj_9r_s3_v1.json```

```python SR.py --config-path config_rd_12sj_9r_s3_v1.json --project-root global_files```

`SR.py` restores the matching checkpoint from `global_files/checkpoints/`, writes MATLAB outputs, previews, and image-quality metrics. Use `--out-dir`, `--image-index`, `--no-png`, or `--no-mat` to adjust inference output.

### Project Layout
 - `global_files/`: dataset creation (`CD.py`), model (`UDN.py`), inference (`SR.py`), base configuration, checkpoints, and outputs.
 - `master_scripts/`: configuration and end-to-end training notebooks.
 - `runs/`: versioned experiment configurations plus generated training and test datasets.

## Authors & Contacts
The original DeepSUM code is based on work by team SuperPip from the Image Processing and Learning group of Politecnico di Torino: Andrea Bordone Molini (andrea.bordone AT polito.it), Diego Valsesia (diego.valsesia AT polito.it), Giulia Fracastoro (giulia.fracastoro AT polito.it), Enrico Magli (enrico.magli AT polito.it). The original team were not involved in this DWI implementation.

The DWI extension was implemented at the National MRI Laboratory (LANIREM) of the Universidad Nacional Autónoma de México, Mexico, by Carolina Daniells Zaldivar, María Guadalupe García-Garcia, Paola Ocampo Luna, Jesús E Solís-Pérez, Luis Concha and Merlin J Fair (merlin.fair AT unam.mx).

## Acknowledgements
This project was funded by UNAM-PAPIIT grant: TA101224.
