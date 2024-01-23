include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)
include_directories(${PROJECT_SOURCE_DIR}/lib/cereal/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/eternal/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/rapidfuzz-cpp)
include_directories(${PROJECT_SOURCE_DIR}/lib/robin-map/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/annoy/src)
include_directories(${PROJECT_SOURCE_DIR}/src)

if(UseGoogleBenchmark)
    include_directories(${PROJECT_SOURCE_DIR}/lib/benchmark/include)
endif()

set(LEXOCRAFT_LIBS lexocraft_neural_network
    lexocraft_llm
    rapidfuzz::rapidfuzz
    tsl::robin_map
    $<IF:$<BOOL:${USE_GOOGLE_BENCHMARK}>,benchmark::benchmark,>)

# include(${PROJECT_SOURCE_DIR}/cmake/libs.cmake)
# include(${PROJECT_SOURCE_DIR}/cmake/opts.cmake)
