# dump_utils.py ---
#
# Filename: dump_utils.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Mon Jan 18 16:52:58 2016 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Thu Sep 15 20:30:33 2016 (+0200)
#           By: kwang
#     Update #: 40
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

import gzip
import os
import pickle

import h5py
import numpy as np


def savepklz(data_to_dump, dump_file_full_name, force_run=False):
    ''' Saves a pickle object and gzip it '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name, force_run=False):
    ''' Loads a gziped pickle object '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data


def saveh5(dict_to_dump, dump_file_full_name):
    ''' Saves a dictionary as h5 file '''

    with h5py.File(dump_file_full_name, 'w') as h5file:
        writeh5(dict_to_dump, h5file)


def writeh5(dict_to_dump, h5node):
    ''' Recursive function to write dictionary to h5 nodes '''

    for _key in dict_to_dump.keys():
        if isinstance(dict_to_dump[_key], dict):
            h5node.create_group(_key)
            cur_grp = h5node[_key]
            writeh5(dict_to_dump[_key], cur_grp)
        else:
            h5node[_key] = dict_to_dump[_key]


def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    with h5py.File(dump_file_full_name, 'r') as h5file:
        dict_from_file = readh5(h5file)

    return dict_from_file


def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file


def loadMonitor(dump_file_full_name_no_ext):
    ''' Saves the monitor file '''

    # Check h5 file, load that if it exists, else use the old one
    if os.path.exists(dump_file_full_name_no_ext + '.h5'):
        with h5py.File(dump_file_full_name_no_ext + '.h5', 'r') as h5file:
            monitor = {}
            for _key in h5file.keys():
                monitor[_key] = h5file[_key].value
                # turn into lists if it was a list
                if _key.endswith('_list'):
                    monitor[_key] = list(monitor[_key])

    elif os.path.exists(dump_file_full_name_no_ext + '.pklz'):
        monitor = loadpklz(dump_file_full_name_no_ext + '.pklz')
    else:
        raise RuntimeError('Dump file for {} does not exist!'.format(
            dump_file_full_name_no_ext))

    return monitor


def saveMonitor(monitor, dump_file_full_name_no_ext):
    ''' Loads the monitor file '''

    with h5py.File(dump_file_full_name_no_ext + '.h5', 'w') as h5file:
        for _key in monitor.keys():
            val = monitor[_key]
            # If it's a list, turn it into np array

            #  NOTE: the lists in the monitor are just to have
            #  appending operation not allocate everytime an element
            #  is added. In fact, they can be easily transformed into
            #  an ndarray
            if _key.endswith('_list'):
                val = np.asarray(val)

            h5file[_key] = val

#
# dump_utils.py ends here
