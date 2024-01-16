include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

add_library(llm lexer.cpp vector_database.cpp text_completion.cpp text_completion_nn.cpp)

target_link_libraries(llm PUBLIC neural_network)
