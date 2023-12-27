#include <unordered_map>
#include <vector>
#include <ctime>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "detection_model_y8.h"

using ResultMap = std::unordered_map<int, std::vector<DetectionModelYolov8::Result>>;

void DrawResultsAndSave(
  const cv::Mat& img, 
  const ResultMap& results,
  const char* path
){
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
  cv::imwrite(path, img);
}

int main(int argc, char* argv[]) {
  cv::Mat img = cv::imread("../data/images/bus.jpg");
  bool enable_gpu = true;
  DetectionModelYolov8 m{"../data/models/yolov8m.onnx", enable_gpu};

  clock_t t0 = clock(); 
  m.Predict(img);     // takes more time for preheating with gpu enabled 
  clock_t t1 = clock();
  auto res = m.Predict(img);   
  clock_t t2 = clock();

  fmt::print("first inference time: {}ms\n", (t1 - t0) / 1000);
  fmt::print("second inference time: {}ms\n", (t2 - t1) / 1000);

  DrawResultsAndSave(img, res, "output.jpg");
  return 0;
}

