cmake_minimum_required(VERSION 3.10)
project(yolo_obb_test)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenVINO REQUIRED)

add_executable(yolov8n_obb_test yolov8n_obb_test.cpp)

target_link_libraries(yolov8n_obb_test PRIVATE ${OpenCV_LIBS} openvino::runtime)