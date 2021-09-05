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

#include "FaceRecognitionForTracking.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"

APP_ERROR FaceRecognitionForTracking::Init(const InitParam &initParam) {
    deviceId_ = initParam.deviceId;
    // 设备初始化
    APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
    if (ret != APP_ERR_OK) {
        logError << "Init devices failed, ret=" << ret << ".";
        return ret;
    }

    // 上下文初始化
    ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Set context failed, ret=" << ret << ".";
        return ret;
    }

    // dvpp组件初始化
    dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
    ret = dvppWrapper_->Init();
    if (ret != APP_ERR_OK) {
        LogError << "DvppWrapper init failed, ret=" << ret << ".";
        return ret;
    }

    // 加载模型
    model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
    ret = model_->Init(initParam.modelPath, modelDesc_);
    if (ret != APP_ERR_OK) {
        LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
        return ret;
    }

    // 模型后处理配置信息
    MxBase::ConfigData configData;
    const std::string softmax = initParam.softmax ? "true" : "false";
    const std::string checkTensor = initParam.checkTensor ? "true" : "false";

    configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
    configData.SetJsonValue("TOP_K", std::to_string(initParam.topk));
    configData.SetJsonValue("SOFTMAX", softmax);
    configData.SetJsonValue("CHECK_MODEL", checkTensor);

    auto jsonStr = configData.GetCfgJson().serialize();
    std::map<std::string, std::shared_ptr<void>> config;
    config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
    config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);

    // 模型输出后处理初始化
    post_ = std::make_shared<MxBase::Resnet50PostProcess>();
    ret = post_->Init(config);
    if (ret != APP_ERR_OK) {
        LogError << "Resnet50PostProcess init failed, ret=" << ret << ".";
        return ret;
    }

    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::DeInit() {
    dvppWrapper_->DeInit();
    model_->DeInit();
    post_->DeInit();
    MxBase::DeviceManager::GetInstance()->DestroyDevices();
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::ReadImage(const std::string &imgPath, cv::Mat &imgMat) {
    imgMat = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (imgMat.empty()) {
        LogError << "imread failed. img: " << imgPath;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::Resize(const cv::Mat &srcMat, cv::Mat &dstMat) {
    static constexpr uint32_t resizeWidth = 64
    static constexpr uint32_t resizeHeight = 96
    cv::resize(srcMat, dstMat, cv::Size(resizeWidth, resizeHeight), 0, 0, cv::INTER_LINEAR);
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::CvMatToTensorBase(const cv::Mat &imgMat, MxBase::TensorBase &tensorBase) {
    
}