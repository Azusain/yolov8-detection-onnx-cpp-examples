#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "detection_model_y8.h"

int main(int argc, char* argv[]) {
  cv::Mat img = cv::imread("../data/images/bus.jpg");
  
  DetectionModelYolov8 m{"../data/models/yolov8m.onnx"};
  auto results = m.Predict(img);
  
  // draw rectangles
  auto height = img.rows;
  auto width = img.cols;
  for(auto& [k, v] : results) {
    for(auto& res : v) {
      cv::Rect rect(
        res.left * width, 
        res.top * height, 
        res.width * width, 
        res.height * height
      );
      cv::rectangle(img, rect, cv::Scalar(0, 240, 0), 2);
    }
  }

  cv::imwrite("output.jpg", img);
  return 0;
}

