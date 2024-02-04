cmake_minimum_required(VERSION 3.10)

project(LexoCraft)

option(COMPILE_TESTS "Compile tests" OFF)
option(USE_GDB "Use GDB" OFF)

include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/src/lexocraft)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/rapidfuzz-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/robin-map)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib/annoy)

enable_testing()
