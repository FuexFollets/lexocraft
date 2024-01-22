cmake_minimum_required(VERSION 3.28)

project(LexoCraft)

include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/lexocraft)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/rapidfuzz-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/robin-map)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/benchmark)

enable_testing()
