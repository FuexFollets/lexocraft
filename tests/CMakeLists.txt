include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

function(test_executable name)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/tests)
    add_executable(${name} ${PROJECT_SOURCE_DIR}/tests/${name}.cpp)
    target_link_libraries(${name} ${LEXICRAFT_LIBS})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

test_executable(test1)
test_executable(eigen_serialize)
test_executable(neural_network_serialization)
