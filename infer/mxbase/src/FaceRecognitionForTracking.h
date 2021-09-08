/*
 * Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FACE_RECOGNITION_FOR_TRACKING_H
#define FACE_RECOGNITION_FOR_TRACKING_H

#include <opencv2/opencv.hpp>
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string modelPath;
};

class FaceRecognitionForTracking {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imgMat);
    APP_ERROR Resize(const cv::Mat &srcMat, cv::Mat &dstMat);
    APP_ERROR CvMatToTensorBase(const cv::Mat &imgMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs, std::vector<std::string> &names, cv::Mat &output);
    APP_ERROR Process(const std::string &imgPath);
    double GetInferCostTimeMs() const {
        return inferCostTimeMs;
    }

private:
    std::vector<std::string> GetFileList(const std::string &dirPath);
    void InclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, std::vector<float> &inclassLikehood);
    void BtclassLikehood(cv::Mat &featureMat, std::vector<std::string> &names, std::vector<float> &btclassLikehood);
    void TarAtFar(std::vector<float> &inclassLikehood, std::vector<float> &btclassLikehood, std::vector<std::vector<float>> &tarFars);

private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
    double inferCostTimeMs = 0.0;
};

#endif