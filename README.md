# yolov8 detection tasks with onnxruntime 
This repository contains work on performing inference with the ONNX Runtime APIs.  

### Progress
- basic yolov8 detection tasks on cpu ✔️
- option for default cuda settings ✔️
- studying on enabling io-bindings... ❓
- utilizing more hardware-accelerated APIs(TensorRT or CUDA) before the next century ❓

### NOTES!
you will have to download the .onnx format model personally cuz the file is too large :D


### Build
- gcc
- cmake
- vcpkg (optional)

### Reference
thanks for the examples from these guys:
- onnxruntime https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx

- ultralytics https://github.com/ultralytics/ultralytics/tree/main/examples

- stackoverflow https://stackoverflow.com/questions/65379070/how-to-use-onnx-model-in-c-code-on-linux

- Li-99's repo https://github.com/Li-99/yolov8_onnxruntime.git

 - azusaings@gmail.com
