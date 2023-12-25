include_directories(${CMAKE_CURRENT_SOURCE_DIR}/neural_network/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/neural_network)

add_executable(LexoCraft main.cpp)
target_link_libraries(LexoCraft NeuralNetwork)
