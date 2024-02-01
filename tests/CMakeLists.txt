set(TEST_FILES
    test1
    eigen_serialize
    neural_network_serialization
    nn_ostream
    neural_network_compute
    vector_database
    word_vector_comparison
    vector_database_search
    tsl_robin_map
    passage_tokenization
    text_completion
    vector_subdatabases
    text_prediction
    vector_database_synonyms
    vector_database_serialization
)

if(COMPILE_TESTS)
    include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
    include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)

    function(test_executable name)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/tests)
        add_executable(${name} ${PROJECT_SOURCE_DIR}/tests/${name}.cpp)
        target_link_libraries(${name} ${LEXOCRAFT_LIBS})
        target_compile_options(${name} PRIVATE ${CMAKE_CSS_TEST_FLAGS})
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    enable_testing()

    foreach(test_file ${TEST_FILES})
        test_executable(${test_file})
    endforeach()
endif()
