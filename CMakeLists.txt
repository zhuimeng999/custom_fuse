cmake_minimum_required(VERSION 3.9)
project(custom_fuse)

set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA CXX)
include(CheckLanguage)
check_language(CUDA)

include_directories(inc)

add_executable(${PROJECT_NAME} src/main.cc inc/endian.hpp inc/helper_cuda.h
  src/Options.cc inc/Options.hpp
  src/ply.cc inc/ply.hpp
  src/depth_filter_kernel.cu inc/depth_filter_kernel.cuh
  src/depth_filter.cc inc/depth_filter.hpp
  src/load_inputs.cc inc/load_inputs.hpp)

find_package(Eigen3 REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS filesystem program_options log)
target_link_libraries(${PROJECT_NAME} Boost::filesystem Boost::program_options Boost::log)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DBOOST_LOG_DYN_LINK)

find_package( OpenCV REQUIRED )
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS} )
#target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} opencv_core opencv_imgcodecs opencv_highgui)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 52 75)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
