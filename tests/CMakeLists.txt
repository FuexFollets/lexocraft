include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

function(test_executable name)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/tests)
    add_executable(${name} ${PROJECT_SOURCE_DIR}/tests/${name}.cpp)
    target_link_libraries(${name} ${LEXOCRAFT_LIBS})
    target_compile_options(${name} PRIVATE -std=c++20 -g -O0 -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

test_executable(test1)
test_executable(eigen_serialize)
test_executable(neural_network_serialization)
test_executable(nn_ostream)
test_executable(neural_network_compute)
test_executable(vector_database)
test_executable(word_vector_comparison)
test_executable(vector_database_search)
