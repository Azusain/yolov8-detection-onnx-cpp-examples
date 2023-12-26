#ifndef __DETECTION_MODEL_Y8_
#define __DETECTION_MODEL_Y8_

#include <string_view>
#include <vector>
#include <unordered_map>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "onnx_model_wrapper.h"

class DetectionModelYolov8 : private ONNXModel {
public:
  struct Result {
    long id;
    float conf;
    float left;
    float top;
    float height;
    float width;
  };

  DetectionModelYolov8(std::string_view model_path, Ort::SessionOptions opt={});

  std::unordered_map<int, std::vector<Result>> Predict(
    const cv::Mat& img, float conf_threshhold=0.4, float iou_threshhold=0.4);

private:
  cv::Mat PreProcessing(const cv::Mat& img);

  std::vector<Ort::Value> Inference(const cv::Mat& blob);

  std::unordered_map<int, std::vector<Result>> ProcessInferenceOutput(
    std::vector<Ort::Value>& node_outputs, float conf_threshhold, float iou_threshhold);
}; 

#endif