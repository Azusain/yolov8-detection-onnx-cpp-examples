cmake_minimum_required(VERSION 3.15)
project(Template VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(fmt CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES_CODES "src/*.cc")
file(GLOB HEARDERS_DIR "include")

add_executable(Main ${SOURCES_CODES})

target_include_directories(Main PRIVATE ${HEARDERS_DIR})

target_link_libraries(Main
    fmt::fmt
    ${OpenCV_LIBRARIES}
    onnxruntime
)