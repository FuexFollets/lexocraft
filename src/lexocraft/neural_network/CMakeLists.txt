include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)

add_library(neural_network STATIC nn.cpp)
