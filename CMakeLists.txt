cmake_minimum_required(VERSION 2.8.3)
project(stereo_visual_slam)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

## Specify additional locations of header files
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cc)

## Your package locations should be listed before other locations
include_directories(
  ./include
  /usr/include/eigen3
)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_module)

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})


set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBS} 
                      ${Sophus_LIBRARIES} 
                      ${CSPARSE_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension)
target_link_libraries(${PROJECT_NAME} fmt)
