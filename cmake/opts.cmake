set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

if (USE_GDB)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if ($<NOT:$<EQUAL:${CMAKE_CXX_COMPILER_ID},MSVC>>)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")
endif()

set(CMAKE_CXX_TEST_FLAGS "${CMAKE_CXX_FLAGS} -g")

if ($<NOT:$<EQUAL:${CMAKE_CXX_COMPILER_ID},MSVC>>)
    set(CMAKE_CXX_TEST_FLAGS "${CMAKE_CXX_TEST_FLAGS} \
    -Wall -Wextra -Wpedantic \
    -fsanitize=leak -fsanitize=undefined -fsanitize=address ")
endif()
