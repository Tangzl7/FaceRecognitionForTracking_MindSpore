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
    const uint32_t dataSize = imgMat.cols * imgMat.rows * YUV444_RGB_WIDTH_NU;
    MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
    MemoryData memoryDataSrc(imgMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);

    APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Memory malloc failed.";
        return ret;
    }

    std::vector<uint32_t> shape = {imgMat.rows * YUV444_RGB_WIDTH_NU, static_cast<uint32_t>(imgMat.cols)};
    tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT16);
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs) {
    auto dtypes = model_->GetOutputDataType();
    for (size_t i = 0; i < modelDesc_.outputTensors.size(); i++) {
        std::vector<uint32_t> shape = {};
        for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); j++) {
            shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
        }
        TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
        if (ret != APP_ERR_OK) {
            LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
            return ret;
        }
        outputs.push_back(tensor);
    }
    DynamicInfo dynamicInfo = {};
    dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
    auto startTime = std::chrono::high_resolution_clock::now();
    APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
    auto endTime = std::chrono::high_resolution_clock::now();
    if (ret != APP_ERR_OK) {
        LogError << "ModelInference failed, ret=" << ret << ".";
        return ret;
    }
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    inferCostTimeMs += costMs;
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::PostProcess(const std::vector<MxBase::TensorBase> &inputs, std::vector<std::vector<MxBase::ClassInfo>> &clsInfos) {
    APP_ERROR ret = post_->Process(inputs, clsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "Process failed, ret=" << ret << ".";
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR FaceRecognitionForTracking::Process(const std::string &imgPath) {
    cv::Mat image;
    APP_ERROR ret = ReadImage(imgPath, image);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImage failed, ret=" << ret << ".";
        return ret;
    }
    cv::Mat resizeImage;
    ret = Resize(image, resizeImage);
    if (ret != APP_ERR_OK) {
        LogError << "Resize failed, ret=" << ret << ".";
        return ret;
    }

    TensorBase imageTensor;
    ret = CvMatToTensorBase(resizeImage, imageTensor);
    if (ret != APP_ERR_OK) {
        LogError << "CvMatToTensorBase failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<MxBase::TensorBase> inputs = {};
    std::vector<MxBase::TensorBase> outputs = {};
    inputs.push_back(imageTensor);
    ret = Inference(inputs, outputs);
    if (ret != APP_ERR_OK) {
        LogError << "Inference failed, ret=" << ret << ".";
        return ret;
    }
    std::vector<std::vector<MxBase::ClassInfo>> batchClsInfos = {};
    ret = PostProcess(outputs, batchClsInfos);
    if (ret != APP_ERR_OK) {
        LogError << "PostProcess failed, ret=" << ret << ".";
        return ret;
    }

    for (auto &clsInfos : batchClsInfos) {
        std::string resDataStr;
        uint32_t topkIndex = 1;
        for (auto &clsInfo : clsInfos) {
            resDataStr += std::to_string(clsInfos.classId) + " ";
            std::cout << "batchIndex:" << batchIndex << " top" << topkIndex << " className:" << clsInfo.className
                << " confidence:" << clsInfo.confidence << " classIndex:" << clsInfo.classId;
            topkIndex++;
        }
        batchIndex++;
    }
}