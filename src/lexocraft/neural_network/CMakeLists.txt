include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_library(neural_network neural_network.cpp neural_network_diff.cpp)
target_compile_options(neural_network PRIVATE -std=c++20 -g -O0 -Wall -Wextra -Wpedantic)
