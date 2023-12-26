#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "detection_model_y8.h"

int main(int argc, char* argv[]) {
  cv::Mat img = cv::imread("../data/images/bus.jpg");
  auto height = img.rows;
  auto width = img.cols;

  DetectionModelYolov8 m{"../data/models/yolov8m.onnx"};
  auto results = m.Predict(img);
  fmt::println("read image ({}, {})", height, width);
  for(auto& [k, v] : results) {
    fmt::println("+-------- id: {} --------+", k);
    for(auto& res : v) {
      fmt::println(
        "xywh: {}, {}, {}, {}", 
        res.left * width, 
        res.top * height, 
        res.width * width, 
        res.height * height
      );
    }
  }
  return 0;
}

