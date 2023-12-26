#ifndef __ONNX_MODEL_H
#define __ONNX_MODEL_H
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <string_view>

class ONNXModel {
protected:
  ONNXModel(std::string_view model_path, Ort::SessionOptions options={});

  std::vector<Ort::Value> Run(std::vector<std::vector<float>> input_tensors_data, 
    Ort::RunOptions opts={}); 

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

#endif