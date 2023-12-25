cmake_minimum_required(VERSION 3.28)

project(LexoCraft)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/lexocraft)
# add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
