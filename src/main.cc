// std libraries
#include <string>
#include <string_view>
#include <numeric>
#include <vector>
#include <unordered_map>

// 3rd parties
#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

class ONNXModel {
protected:
  ONNXModel(
    std::string_view model_path,
    Ort::SessionOptions options={}
  ):
    env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, ""),
    session_(env_, model_path.data(), options),
    mem_info_(
      Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeDefault
      )
    )
  {
    // inputs | outputs' shapes & names
    auto input_counts = session_.GetInputCount();
    auto output_counts = session_.GetOutputCount();

    for(size_t i = 0; i < input_counts; ++i) {
      auto input_info = session_.GetInputTypeInfo(i);
      input_shapes_.push_back(
        input_info.GetTensorTypeAndShapeInfo().GetShape()
      );
      input_size_.push_back(
        input_info.GetTensorTypeAndShapeInfo().GetElementCount()
      );
      auto input_name_ptr = session_.GetInputNameAllocated(i, allocator_);  
      input_names_.push_back(input_name_ptr.get());
      input_names_ptrs_.push_back(
        std::move(input_name_ptr)
      );
    }
    
    for(size_t i = 0; i < output_counts; ++i) {
      auto output_info = session_.GetOutputTypeInfo(i);
      output_shapes_.push_back(
        output_info.GetTensorTypeAndShapeInfo().GetShape()
      );
      auto output_name_ptr = session_.GetOutputNameAllocated(i, allocator_);
      output_names_.push_back(output_name_ptr.get());
      output_names_ptrs_.push_back(
        std::move(output_name_ptr)
      );
    }
  }

  std::vector<Ort::Value> Run(
    std::vector<std::vector<float>> input_tensors_data,  
    Ort::RunOptions opts={}
  ) 
  {
    for(size_t i = 0; i < input_tensors_data.size(); ++i) {
      input_tensors_.push_back(
        Ort::Value::CreateTensor<float>(
          mem_info_, 
          input_tensors_data[i].data(), 
          input_size_[i], 
          input_shapes_[i].data(), 
          input_shapes_[i].size()
        )
      );
    }
 
    std::vector<Ort::Value> output_tensors = session_.Run(
      opts,
      input_names_.data(),
      input_tensors_.data(),
      input_names_.size(),
      output_names_.data(),
      output_names_.size() 
    );
    input_tensors_.clear();
    return output_tensors;
  }

  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo mem_info_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::vector<Ort::Value> input_tensors_;
  std::vector<size_t> input_size_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<Ort::AllocatedStringPtr> input_names_ptrs_;
  std::vector<Ort::AllocatedStringPtr> output_names_ptrs_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

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

  DetectionModelYolov8(
    std::string_view model_path, 
    Ort::SessionOptions opts={}
  ):
    ONNXModel(model_path, std::move(opts))
  {}

  std::unordered_map<int, std::vector<Result>> Predict(
    const cv::Mat& img, 
    float conf_threshhold=0.4, 
    float iou_threshhold=0.4
  ) {
    auto blob = this->PreProcessing(img);
    auto node_outputs = this->Inference(blob);
    return this->ProcessInferenceOutput(
      node_outputs, conf_threshhold, iou_threshhold
    );
  }

private:
  cv::Mat PreProcessing(const cv::Mat& img) {
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

  std::vector<Ort::Value> Inference(const cv::Mat& blob) {
    std::vector<std::vector<float>> tensors_data;
    tensors_data.emplace_back(
      std::vector<float>{blob.begin<float>(), blob.end<float>()}
    );
    return this->Run(tensors_data);
  }

  std::unordered_map<int, std::vector<Result>> ProcessInferenceOutput(
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
}; 


int main(int argc, char* argv[]) {

  cv::Mat img = cv::imread("../data/images/bus.jpg");
  DetectionModelYolov8 m{"../data/models/yolov8m.onnx"};
  auto results = m.Predict(img);
  for(auto& [k, v] : results) {
    fmt::println("id: {}, nums: {}", k, v.size());
  }
  return 0;
}

