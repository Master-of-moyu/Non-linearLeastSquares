cmake_minimum_required(VERSION 3.13)
project(LeastSquare)

set(CMAKE_CXX_STANDARD 17)

# 3rdparty
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# add examples
file(GLOB ALL_EXAMPLE_CPPS "*.cpp")
foreach(ONE_EXAMPLE_CPP ${ALL_EXAMPLE_CPPS})
    get_filename_component(EXAMPLE_NAME ${ONE_EXAMPLE_CPP} NAME_WE)
    message(STATUS "  generate example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${ONE_EXAMPLE_CPP})
    target_link_libraries(${EXAMPLE_NAME} ${CERES_LIBRARIES} ${OpenCV_LIBS})
endforeach()
