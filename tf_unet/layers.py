# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import h5py
import hashlib
import glob
import scipy.io as sio
import tensorflow as tf
import os
import numpy as np
from skimage.transform import radon, rescale,iradon
from dump_tools import loadh5,saveh5

def rsnr_tf(rec,oracle):
    "regressed SNR"
    sumP    =        tf.reduce_sum(oracle)
    sumI    =        tf.reduce_sum(rec)
    sumIP   =        tf.reduce_sum( oracle * rec )
    sumI2   =        tf.reduce_sum(rec**2)

    Aup     =       tf.stack([sumI2,sumI],axis=0)
    Adw     =       tf.stack([sumI,tf.cast(tf.size(oracle),tf.float32)],axis=0)
    A       =       tf.concat([tf.expand_dims(Aup,0),tf.expand_dims(Adw,0)],0)
    b       =       tf.expand_dims(tf.stack([sumIP,sumP],axis=0),1)

    c       =        tf.matmul(tf.matrix_inverse(A),b)
    rec     =        tf.gather(c,0)*rec+tf.gather(c,1)
    err     =        tf.reduce_sum((oracle-rec)**2)
    SNR     =        10.0*log10_tf(tf.reduce_sum(oracle**2)/err)

    return SNR

def rsnr(rec,oracle):
    "regressed SNR"
    sumP    =        sum(oracle.reshape(-1))
    sumI    =        sum(rec.reshape(-1))
    sumIP   =        sum( oracle.reshape(-1) * rec.reshape(-1) )
    sumI2   =        sum(rec.reshape(-1)**2)
    A       =        np.matrix([[sumI2, sumI],[sumI, oracle.size]])
    b       =        np.matrix([[sumIP],[sumP]])
    c       =        np.linalg.inv(A)*b #(A)\b
    rec     =        c[0,0]*rec+c[1,0]
    err     =        sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR     =        10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR=0.0;
    return SNR

def flipping(img,gt):
    if np.random.rand(1)>0.5:
        out=np.fliplr(img)
        out_gt=np.fliplr(gt)
    else:
        out=img
        out_gt=gt
    if np.random.rand(1)>0.5:
        out=np.flipud(out)
        out_gt=np.flipud(out_gt)
    return out, out_gt

def psnr(refined_img, gt_img):
    psnr = (
          20. * np.log10(np.amax(
          abs(gt_img)
          )) -
          10. * np.log10(np.mean(
          np.square(abs(gt_img - refined_img)
          ))))
    return psnr

def log10_tf(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def psnr_tf(refined_img, gt_img):
    psnr = (
        20. * log10_tf(tf.reduce_max(
        abs(gt_img), axis=[1, 2, 3]
        )) -
        10. * log10_tf(tf.reduce_mean(
        tf.square(abs(gt_img - refined_img)), axis=[1, 2, 3]
        ))
        )
    return psnr

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.random_uniform(shape=shape,minval=0.0,maxval=0.1)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_, padding,stride=1):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride, padding):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')
def avg_pool(x,n):
    return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))

## bring from tf_unet.layers
def _find_data_files(search_path):
    all_files=glob.glob(search_path)
    return all_files

def _load_file(path,opt):
    h5f=h5py.File(path,'r')
    x=np.array(h5f[opt])
    return x

## new function
def _load_whole_data(current_version, file, dump_folder):
    hashstr = hashlib.sha256((
        "".join(file)+current_version
        ).encode()).hexdigest()
    dump_file=os.path.join(dump_folder,hashstr+".h5")
    print(dump_file)
    rebuild_data=True
    if os.path.exists(dump_file):
        print("dump file existed")
        with h5py.File(dump_file,"r") as h5file:
           if "version" in list(h5file.keys()):
               if h5file["version"].value==current_version:
                       rebuild_data=False

    print("rebuild_data",rebuild_data)
    if rebuild_data:
        data=[]
        mat_contents=sio.loadmat(file)
        gt=np.squeeze(mat_contents['data_gt'])
        sparse=np.zeros_like(gt)
        full=np.zeros_like(gt)

        for ind in range(gt.shape[2]):
            img = np.squeeze(gt[:,:,ind])
            theta = np.linspace(0., 180., 1e3, endpoint=False)
            sinogram = radon(img, theta=theta, circle=False)
            theta_down = theta[0:1000:20]
            sparse[:,:,ind] =iradon(sinogram[:,0:1000:20],theta=theta_down,circle=False)
            full[:,:,ind] =iradon(sinogram,theta=theta,circle=False)
            print("iteration : " , ind, "/", gt.shape[2])

        norm_val=np.amax(sparse)
        print("norm_val", norm_val)
        print("finished rebuild")
        saveh5({"label":full/norm_val*255.,"sparse":sparse/norm_val*255.,"version":current_version},dump_file)

    f_handle=h5py.File(dump_file,"r")
    label=np.array(f_handle["label"])
    sparse=np.array(f_handle["sparse"])
    print("size of label, " , label.shape)
    print("size of sparse, " , sparse.shape)
