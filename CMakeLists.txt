cmake_minimum_required(VERSION 3.25)
project(Template VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)


find_package(fmt CONFIG REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(Main src/main.cc)

target_link_libraries(Main
    fmt::fmt
    ${OpenCV_LIBRARIES}
    onnxruntime
)