cmake_minimum_required(VERSION 3.28)

project(LexoCraft)

set(CMAKE_CXX_STANDARD 20)

include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)

add_executable(LexoCraft ${PROJECT_SOURCE_DIR}/src/main.cpp)
