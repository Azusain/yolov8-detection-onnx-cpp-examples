#include "detection_model_y8.h"

DetectionModelYolov8::DetectionModelYolov8(
  std::string_view model_path, 
  bool enable_gpu
):
  ONNXModel()
{
  Ort::SessionOptions opts;
  if(enable_gpu) {
    OrtCUDAProviderOptions cuda_opts;
    opts.AppendExecutionProvider_CUDA(cuda_opts);
  }
  this->Load(model_path, std::move(opts));
}

std::unordered_map<int, std::vector<DetectionModelYolov8::Result>> DetectionModelYolov8::Predict(
  const cv::Mat& img, 
  float conf_threshhold, 
  float iou_threshhold
) {
  auto blob = this->PreProcessing(img);
  auto node_outputs = this->Inference(blob);
  return this->ProcessInferenceOutput(
    node_outputs, conf_threshhold, iou_threshhold
  );
}

cv::Mat DetectionModelYolov8::PreProcessing(const cv::Mat& img) {
  cv::Mat resized_img;
  cv::resize(
    img, 
    resized_img, 
    cv::Size(input_shapes_[0][3], input_shapes_[0][2]), 
    0, 
    0, 
    cv::INTER_LINEAR
  ); 
  return cv::dnn::blobFromImage(
    resized_img, 
    1.0f / 255, 
    resized_img.size(), 
    cv::Scalar(0, 0, 0), 
    true, 
    false, 
    CV_32F
  );
}

std::vector<Ort::Value> DetectionModelYolov8::Inference(const cv::Mat& blob) {
  std::vector<std::vector<float>> tensors_data;
  tensors_data.emplace_back(
    std::vector<float>{blob.begin<float>(), blob.end<float>()}
  );
  return this->Run(tensors_data);
}

std::unordered_map<int, std::vector<DetectionModelYolov8::Result>> DetectionModelYolov8::ProcessInferenceOutput(
  std::vector<Ort::Value>& node_outputs,
  float conf_threshhold,
  float iou_threshhold
) 
{
  float* raw_results = node_outputs[0].GetTensorMutableData<float>();
  // transposed to [1, 8400, 84]
  cv::Mat results_transposed = cv::Mat( 
    cv::Size(output_shapes_[0][2], output_shapes_[0][1]), 
    CV_32F, 
    raw_results
  ).t();
  auto rows = results_transposed.rows;
  auto cols = results_transposed.cols;
  auto res_t_ptr = (float*)results_transposed.data;

  std::vector<cv::Rect> boxes;
  std::vector<float> confs;
  std::vector<long> ids;

  // extract xywh
  for(size_t i = 0; i < rows; ++i) {
    std::vector<float> row_data{ // row_data[0 ~ 3] -> cen_x, cen_y, w, h
      res_t_ptr + i * cols, res_t_ptr + (i + 1) * cols
    };
    auto max_conf_iter = std::max_element(   
      row_data.begin() + 4, row_data.end()    
    );
    auto max_conf_idx = max_conf_iter - row_data.begin() - 4;

    if(*max_conf_iter > conf_threshhold) {
      auto width = row_data[2];
      auto height = row_data[3];
      auto left = row_data[0] - width / 2;
      auto top = row_data[1] - height / 2;
      ids.emplace_back(max_conf_idx);
      boxes.emplace_back(left, top, width, height);
      confs.emplace_back(*max_conf_iter);
    }
  }

  // nms
  std::vector<int> left_boxes_idx;
  cv::dnn::NMSBoxes(boxes, confs, conf_threshhold, iou_threshhold, left_boxes_idx);

  // format output
  std::unordered_map<int, std::vector<Result>> results_map;
  
  for(auto& idx : left_boxes_idx) {
    auto box{boxes[idx]};
    auto res = Result{
      .id = ids[idx],
      .conf = confs[idx],
      .left = (float)box.x / input_shapes_[0][3],
      .top = (float)box.y / input_shapes_[0][2],
      .height = (float)box.height / input_shapes_[0][2],
      .width = (float)box.width / input_shapes_[0][3]
    };
    
    auto it = results_map.find(res.id);
    if(it != results_map.end()){
      it->second.emplace_back(res);
    } else {
      results_map.emplace(res.id, std::vector<Result>{res});
    }
  }
  return results_map;
}

