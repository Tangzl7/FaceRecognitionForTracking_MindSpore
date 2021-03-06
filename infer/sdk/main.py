#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re
import os
import sys
import json
import base64
import argparse
import numpy as np
from tqdm import tqdm

from StreamManagerApi import *


def parse_args():
    parser = argparse.ArgumentParser(description="SDK infer")
    parser.add_argument("-d", "--dataset", type=str, 
                        default="../data/input/valid_data/",
                        help="Specify the directory of dataset")
    parser.add_argument("-p", "--pipeline", type=str,
                        default="../data/face_recognition_for_tracking.pipeline",
                        help="Specify the path of pipeline file")
    parser.add_argument("-o", "--output", type=str,
                        default="../data/output.txt", 
                        help="Specify the infer output file")
    return parser.parse_args()

def get_dataset(dataset_dir):
    file_list = os.listdir(dataset_dir)
    paths, names = [], []
    for sub_path in file_list:
        for im_path in os.listdir(os.path.join(dataset_dir, sub_path)):
            names.append(im_path.split('.')[0])
            paths.append(os.path.join(dataset_dir, sub_path, im_path))

    return names, paths

def get_stream_manager(pipeline_path):
    stream_mgr_api = StreamManagerApi()
    ret = stream_mgr_api.InitManager()
    if ret != 0:
        print(f"Failed to init Stream manager, ret={ret}")
        exit(1)

    with open(pipeline_path, 'rb') as f:
        pipeline_content = f.read()

    ret = stream_mgr_api.CreateMultipleStreams(pipeline_content)
    if ret != 0:
        print(f"Failed to create stream, ret={ret}")
        exit(1)
    return stream_mgr_api

def infer_image(stream_mgr_api, image_path):
    stream_name = b'im_facerecognitionfortracking'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    unique_id = stream_mgr_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit(1)

    infer_result = stream_mgr_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error. errorCode={infer_result.errorCode},"
              f"errorMsg={infer_result.data.decode()}")
        exit(1)

    infer_result_data = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_data['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    infer_array = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    infer_array = infer_array.reshape(tensor_shape)
    return np.squeeze(infer_array, axis=0)

def inclass_likehood(ims_info, type='cos'):
    obj_feas = {}
    likehoods = []
    for name, _, fea in ims_info:
        if re.split('_\d\d\d\d', name)[0] not in obj_feas:
            obj_feas[re.split('_\d\d\d\d', name)[0]] = []
        obj_feas[re.split('_\d\d\d\d', name)[0]].append(fea)
    for _, feas in tqdm(obj_feas.items()):
        feas = np.array(feas)
        if type == 'cos':
            likehood_mat = np.dot(feas, np.transpose(feas)).tolist()
            for row in likehood_mat:
                likehoods += row
        else:
            for fea in feas.tolist():
                likehoods += np.sum(-(fea - feas) ** 2, axis=1).tolist()

    likehoods = np.array(likehoods)
    return likehoods


def btclass_likehood(ims_info, type='cos'):
    likehoods = []
    count = 0
    for name1, _, fea1 in tqdm(ims_info):
        count += 1
        frame_id1, obj_id1 = re.split('_\d\d\d\d', name1)[0], name1.split('_')[-1]
        fea1 = np.array(fea1)
        for name2, _, fea2 in ims_info:
            frame_id2, obj_id2 = re.split('_\d\d\d\d', name2)[0], name2.split('_')[-1]
            if frame_id1 == frame_id2:
                continue
            fea2 = np.array(fea2)
            if type == 'cos':
                likehoods.append(np.sum(fea1 * fea2))
            else:
                likehoods.append(np.sum(-(fea1 - fea2) ** 2))

    likehoods = np.array(likehoods)
    return likehoods

def tar_at_far(inlikehoods, btlikehoods):
    test_point = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    tar_far = []
    for point in test_point:
        thre = btlikehoods[int(btlikehoods.size * point)]
        n_ta = np.sum(inlikehoods > thre)
        tar_far.append((point, float(n_ta) / inlikehoods.size, thre))

    return tar_far


if __name__ == '__main__':
    parser = parse_args()
    img_names, img_paths = get_dataset(parser.dataset)
    stream_mgr_api = get_stream_manager(parser.pipeline)
    features = []
    for img_path in img_paths:
        feature = infer_image(stream_mgr_api, img_path)
        features.append(feature)
    ims_info = list(zip(img_names, img_paths, features))

    stream_mgr_api.DestroyAllStreams()

    inlikehoods = inclass_likehood(ims_info)
    inlikehoods[::-1].sort()
    btlikehoods = btclass_likehood(ims_info)
    btlikehoods[::-1].sort()
    tar_far = tar_at_far(inlikehoods, btlikehoods)

    for far, tar, thre in tar_far:
        print('---{}: {}@{}'.format(far, tar, thre))

    if os.path.exists(parser.output):
        os.remove(parser.output)

    with open(parser.output, 'a+') as result_fw:
        for far, tar, thre in tar_far:
            result_fw.write('{}: {}@{} \n'.format(far, tar, thre))

