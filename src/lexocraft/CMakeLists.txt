add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/neural_network)

include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_executable(LexoCraft main.cpp)
target_link_libraries(LexoCraft PUBLIC neural_network)
