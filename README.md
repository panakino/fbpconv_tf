# 3D BPConvNet

3D parallel MRI reconstruction for accelerated MRI. In our paper (to be appeared), we reconstructed 320x320x256x8 volume using 3D BPConvNet. It takes sub 10 seconds. We used Knee dataset from http://mridata.org. We also used wavelet transform from built-in of Matlab 2016, however, in this repo, we do not provide wavelet transform. The codes only cover the part network.

Whole codes are forked from https://github.com/jakeret/tf_unet.

## Training configuration
* Tensorflow 1.1.0
* 2 GPUs (TITAN X pascal arch.)
* MacOS X 10.12.6
* Python 2.7.12

### illustration
![alt tag](https://github.com/panakino/3dbpconv/blob/master/structure.png)

## Commands
To start training a model for 3D BPConvNet:
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
