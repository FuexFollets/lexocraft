include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_library(lexocraft_llm lexer.cpp vector_database.cpp text_completion.cpp text_completion_nn.cpp text_completion_interface.cpp)

target_link_libraries(lexocraft_llm PUBLIC lexocraft_neural_network)
