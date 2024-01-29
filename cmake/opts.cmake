set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(base_flags "-Wall -Wextra -Wpedantic -g")

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Remove -Wextra for MSVC
    string(REPLACE "-Wextra" "" flags "${base_flags}")
    string(REPLACE "-Wpedantic" "" flags "${base_flags}")
else()
    # Use base flags for other compilers
    set(flags "${base_flags}")
endif()

# Apply the final flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flags}")
