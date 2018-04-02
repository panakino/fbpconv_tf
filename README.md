# FBPConvNet 
http://ieeexplore.ieee.org/document/7949028/

This is tensorflow implementation for ``Deep Convolutional Neural Network for Inverse Problems in Imaging``, TIP (2017).
2D sparse-view CT reconstruction and reconstruction of accelerated MRI. 

Whole codes are forked and modified from https://github.com/jakeret/tf_unet.

## Training configuration
* Tensorflow 1.1.0
* 1 or 2 GPUs (TITAN X pascal arch.)
* MacOS X 10.12.6
* Python 2.7.12

## Data (ellipsoids x 20)
* train : https://drive.google.com/open?id=1FTOgM2vOQaGSokEDtOaPNdBTto6h5yFi
* test : https://drive.google.com/open?id=1w_kPao6L2UwhTKIgcr_3o62A6vYYtX_r

### illustration
![alt tag](https://github.com/panakino/3dbpconv/blob/master/structure.png)

## Commands
To start training a model for FBPConvNet:
```bash
python main.py --lr=1e-4 --output_path='logs/' --data_path='data_path/*.h5' --test_path='test_path/*.h5' --features_root=32 --layers=5 --is_training=True
```

To deploy trained model:
```bash
python main.py --lr=1e-4 --output_path='logs/' --data_path='data_path/*.h5' --test_path='test_path/*.h5' --features_root=32 --layers=5 --is_training=False
```

You may find more details in main.py.


## Contact
kyonghwan.jin@gmail.com
