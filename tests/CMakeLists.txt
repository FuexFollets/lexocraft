function(test_executable name)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/tests)
    add_executable(${name} ${PROJECT_SOURCE_DIR}/tests/${name}.cpp)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

test_executable(test1)
