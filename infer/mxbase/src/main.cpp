#include "FaceRecognitionForTracking.h"
#include "MxBase/Log/Log.h"
#include <algorithm>
#include <dirent.h>

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './ghostnet /path/to/jpeg_image_dir'.";
        return APP_ERR_OK;
    }

    std::string imgPath = argv[1];
    InitParam initParam;
    initParam.deviceId = 0;
    initParam.modelPath = "../../../model/face_recognition_for_tracking.om";
    auto face_recognition_for_tracking = std::make_shared<FaceRecognitionForTracking>();
    APP_ERROR ret = face_recognition_for_tracking->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "FaceRecognitionForTracking init failed, ret=" << ret << ".";
        return ret;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    ret = face_recognition_for_tracking->Process(imgPath);
    if (ret != APP_ERR_OK) {
    	LogError << "FaceRecognitionForTracking process failed, ret=" << ret << ".";
    	face_recognition_for_tracking->DeInit();
    	return ret;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    face_recognition_for_tracking->DeInit();
    double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogInfo << "[Total process delay] cost: " << costMs << " ms";
    return APP_ERR_OK;
}