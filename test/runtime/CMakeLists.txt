function(_set_target_options NAME)
    target_include_directories(
        ${NAME}
        PRIVATE
        "${Halide_SOURCE_DIR}/src"
        "${Halide_SOURCE_DIR}/src/runtime"
    )
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Halide runtime lib has declarations for memcmp etc that conflict with GNU stdlib
        target_compile_options(${NAME} PRIVATE -Wno-builtin-declaration-mismatch)
    endif ()
    math(EXPR bits "8 * ${CMAKE_SIZEOF_VOID_P}")
    target_compile_definitions(
        ${NAME}
        PRIVATE
        HALIDE_VERSION_MAJOR=${Halide_VERSION_MAJOR}
        HALIDE_VERSION_MINOR=${Halide_VERSION_MINOR}
        HALIDE_VERSION_PATCH=${Halide_VERSION_PATCH}
        BITS_${bits}
        COMPILING_HALIDE_RUNTIME
        COMPILING_HALIDE_RUNTIME_TESTS
    )
    set_halide_compiler_warnings(${NAME})
endfunction()

function(halide_define_runtime_internal_test NAME)
    add_executable(runtime_internal_${NAME} ${NAME}.cpp $<TARGET_OBJECTS:runtime_internal_common>)
    _set_target_options(runtime_internal_${NAME})
    target_link_libraries(runtime_internal_${NAME} PRIVATE Halide::Test)
    add_halide_test(runtime_internal_${NAME} GROUPS runtime_internal)
endfunction()

# NOTE: These tests directly include runtime_internal.h which isn't compatible with MSVC
if (NOT MSVC)
    # Weak linkages are easier to get right with OBJECT libraries
    add_library(
        runtime_internal_common OBJECT
        common.cpp
        "${Halide_SOURCE_DIR}/src/runtime/msan_stubs.cpp"
        "${Halide_SOURCE_DIR}/src/runtime/to_string.cpp"
    )
    _set_target_options(runtime_internal_common)

    # block_allocator.cpp
    halide_define_runtime_internal_test(block_allocator)

    # block_storage.cpp
    halide_define_runtime_internal_test(block_storage)

    # device_buffer_utils.cpp
    halide_define_runtime_internal_test(device_buffer_utils)

    # linked_list.cpp
    halide_define_runtime_internal_test(linked_list)

    # memory_arena.cpp
    halide_define_runtime_internal_test(memory_arena)

    # string_storage.cpp
    halide_define_runtime_internal_test(string_storage)

    # string_table.cpp
    halide_define_runtime_internal_test(string_table)

    # The mock Vulkan driver exercises the complete device interface, so link
    # the host runtime for its synchronization and buffer management functions.
    find_package(VulkanHeaders 1.3.296 REQUIRED HINTS "${Halide_SOURCE_DIR}/dependencies/vulkan")
    add_halide_runtime(vulkan_transfers_runtime TARGETS "${Halide_CMAKE_TARGET}")
    add_executable(runtime_internal_vulkan_transfers vulkan_transfers.cpp)
    _set_target_options(runtime_internal_vulkan_transfers)
    target_link_libraries(runtime_internal_vulkan_transfers PRIVATE Halide::Test Vulkan::Headers vulkan_transfers_runtime)
    add_halide_test(runtime_internal_vulkan_transfers GROUPS runtime_internal)
endif ()
