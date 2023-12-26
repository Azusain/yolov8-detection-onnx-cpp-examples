#include <onnxruntime_training_cxx_api.h>
#include <string_view>

#include "onnx_model_wrapper.h"

ONNXModel::ONNXModel(
  std::string_view model_path,
  Ort::SessionOptions options
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

std::vector<Ort::Value> ONNXModel::Run(
  std::vector<std::vector<float>> input_tensors_data,  
  Ort::RunOptions opts
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

