cmake_minimum_required(VERSION 3.25)
project(Template VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt CONFIG REQUIRED)

add_executable(Main main.cc)

target_link_libraries(Main
    fmt::fmt
)