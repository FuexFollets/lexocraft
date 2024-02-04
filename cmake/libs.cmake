include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)
include_directories(${PROJECT_SOURCE_DIR}/lib/cereal/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/eternal/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/rapidfuzz-cpp)
include_directories(${PROJECT_SOURCE_DIR}/lib/robin-map/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/annoy/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/nanobench/src/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/icecream-cpp)
include_directories(${PROJECT_SOURCE_DIR}/src)

option(USE_NANOBENCH "Use nanobench" OFF)

set(LEXOCRAFT_LIBS lexocraft_neural_network
    lexocraft_llm
    rapidfuzz::rapidfuzz
    tsl::robin_map
    $<IF:$<BOOL:${USE_NANOBENCH}>,nanobench,>
    )
