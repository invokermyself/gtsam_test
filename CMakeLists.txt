
cmake_minimum_required(VERSION 3.5)
project(gtsam_test)

SET(CMAKE_BUILD_TYPE Debug)

find_package(Boost COMPONENTS thread filesystem date_time system REQUIRED)
FIND_PACKAGE(GTSAM REQUIRED)
set(CMAKE_CXX_STANDARD 11)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GTSAM_INCLUDE_DIR})
INCLUDE_DIRECTORIES("/usr/include/eigen3")
INCLUDE_DIRECTORIES("/opt/ros/melodic/include/")

add_executable(gtsam_test src/main.cpp)
target_link_libraries(gtsam_test ${Boost_LIBRARIES} -lgtsam -ltbb)
