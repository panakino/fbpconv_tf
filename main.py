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
Created on Jul 28, 2016

author: jakeret
modified at Feb 2018
modified by : Kyong Jin (kyonghwan.jin@gmail.com)

running command example
python main.py --lr=1e-4 --output_path='logs/' --features_root=32 --layers=5 --restore=False
python main.py --lr=1e-4 --output_path='logs/' --features_root=32 --layers=5 --restore=False
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
from tf_unet import image_gen, image_util
from tf_unet import unet
from tf_unet import util,layers
from tf_unet.layers import _load_whole_data
import os

flags=tf.app.flags
flags.DEFINE_boolean('is_training',True,'training phase/ deploying phase')
flags.DEFINE_boolean('is_flipping',True,'flipping augmentation')
flags.DEFINE_float('lr',1e-4,'learning rate')
flags.DEFINE_integer('features_root',64,'learning rate')
flags.DEFINE_integer('layers',5,'number of depth')
flags.DEFINE_integer('Ngpu',1,'number of GPUs[1/2]')
flags.DEFINE_string('optimizer','adam','optimizing algorithm : adam / momentum')
#flags.DEFINE_string('dump_train','dump_train/','hash file path for train dataset')
#flags.DEFINE_string('dump_test','dump_test/','hash file path for test dataset')
flags.DEFINE_string('train_path','train_data/*.mat','file path for train dataset')
flags.DEFINE_string('test_path','test_data/*.mat','file path for test dataset')
flags.DEFINE_boolean('restore',True,'resotre model')
flags.DEFINE_string('output_path','logs/','log folder')
flags.DEFINE_boolean('maxpool',True,'true : maxpool, false:avgpool')
conf=flags.FLAGS

if __name__ == '__main__':
    nx = 512
    ny = 512

    # parameters
    training_iters = 475
    epochs = 100
    display_step = 10
    restore =conf.restore
    current_version='fbpconv'
    current_version_test='fbpconv_test'
    net = unet.Unet(channels=1,
                    n_class=1,
                    layers=conf.layers,
                    features_root=conf.features_root,
                    Ngpu=conf.Ngpu,
                    maxpool=conf.maxpool,
                    summaries=True,
                    cost="euclidean") # cost="dice_coefficient"

    if conf.is_training:
        #file_train='/Volumes/Disk_addition/[2016_07]deconv_ct/fulldata/train_set_biomed.mat'
        #file_test='/Volumes/Disk_addition/[2016_07]deconv_ct/fulldata/test_set_biomed.mat'
        #dump_train=conf.dump_train
        #dump_test=conf.dump_test
        ## hash dump for full or sparse view fbp data
        #_load_whole_data(current_version,file_train,dump_train)
        #_load_whole_data(current_version_test,file_test,dump_test)
        #data_loader=image_util.ImageDataProvider_hdf5(dump_train+'*.h5',is_flipping=conf.is_flipping)
        #data_loader_test=image_util.ImageDataProvider_hdf5(dump_test+'*.h5',shuffle_data=False,is_flipping=False)

        data_loader=image_util.ImageDataProvider_mat(conf.train_path,is_flipping=conf.is_flipping)
        data_loader_test=image_util.ImageDataProvider_mat(conf.test_path,shuffle_data=False,is_flipping=False)


        if conf.optimizer=='momentum':
            trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2,learning_rate=conf.lr))
        else:
            trainer = unet.Trainer(net, optimizer="adam", opt_kwargs=dict(learning_rate=conf.lr))

        path = trainer.train(data_loader,data_loader_test,
                conf.output_path+'train/',conf.output_path+'test/',
                             training_iters=training_iters,
                             epochs=epochs,
                             display_step=display_step,
                             restore=restore)
    else:
        save_path = os.path.join(conf.output_path, "model.cpkt")
        x_test, y_test = data_loader_test(1)
        prediction = net.predict(save_path, x_test)

        print("Testing avg RSNR: {:.2f}".format(layers.rsnr(prediction, y_test)))

