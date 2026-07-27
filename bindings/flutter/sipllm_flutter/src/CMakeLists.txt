# CMakeLists.txt — builds libsipllm_ffi from the SipLLM engine sources +
# the C-ABI shim. Shared by the Android Gradle build (one pass per ABI) and the
# desktop Flutter embedders (macos/linux/windows). No external deps beyond
# pthreads (+ optional Vulkan).
cmake_minimum_required(VERSION 3.14)
project(sipllm_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine root: four levels up from this file (bindings/flutter/sipllm_flutter/src).
# Overridable so the plugin can also be consumed from a vendored copy.
if(NOT DEFINED SIPLLM_ENGINE_ROOT)
  set(SIPLLM_ENGINE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
endif()
get_filename_component(SIPLLM_ENGINE_ROOT "${SIPLLM_ENGINE_ROOT}" ABSOLUTE)

if(NOT EXISTS "${SIPLLM_ENGINE_ROOT}/include/llm/runtime.h")
  message(FATAL_ERROR "SipLLM engine not found at ${SIPLLM_ENGINE_ROOT} "
                      "(expected include/llm/runtime.h). Set -DSIPLLM_ENGINE_ROOT=<path>.")
endif()

# All engine translation units live in src/ (none define main()).
file(GLOB SIPLLM_ENGINE_SOURCES CONFIGURE_DEPENDS "${SIPLLM_ENGINE_ROOT}/src/*.cpp")

add_library(sipllm_ffi SHARED
  ${SIPLLM_ENGINE_SOURCES}
  "${CMAKE_CURRENT_LIST_DIR}/../ffi/sipllm_ffi.cpp"
)

target_include_directories(sipllm_ffi PRIVATE
  "${SIPLLM_ENGINE_ROOT}/include"
  "${SIPLLM_ENGINE_ROOT}"
  "${CMAKE_CURRENT_LIST_DIR}/../ffi"
)

target_compile_options(sipllm_ffi PRIVATE
  -O3 -funroll-loops -fno-math-errno
  -fvisibility=hidden
  -ftrivial-auto-var-init=zero
  -Wno-unused-parameter
)

# Per-ABI SIMD tuning. dotprod (Armv8.2) is universal on the target SoCs
# (Dimensity 8300 A715/A510, Snapdragon W5 A53) and lights up the SDOT kernels
# in the engine's simd.h; everything else keeps the scalar fallback.
if(ANDROID)
  if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_options(sipllm_ffi PRIVATE -march=armv8.2-a+dotprod)
  elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    target_compile_options(sipllm_ffi PRIVATE -mfpu=neon-vfpv4)
  endif()
  find_library(log-lib log)
  if(log-lib)
    target_link_libraries(sipllm_ffi ${log-lib})
  endif()
endif()

# Optional Vulkan matmul backend (-DSIPLLM_VULKAN=ON). The engine falls back to
# CPU when the device or the compiled shader is unavailable, so it never depends
# on the GPU for correctness. When glslc is present we (re)generate the embedded
# SPIR-V header the same way the Makefile does, so a fresh checkout still gets
# GPU offload without a manual `make shader`.
option(SIPLLM_VULKAN "Compile the experimental Vulkan matmul backend" OFF)
if(SIPLLM_VULKAN)
  target_compile_definitions(sipllm_ffi PRIVATE LLM_HAVE_VULKAN=1)

  set(_spv_header "${SIPLLM_ENGINE_ROOT}/shaders/matmul_spv.h")
  find_program(GLSLC_EXE glslc)
  if(GLSLC_EXE AND NOT EXISTS "${_spv_header}"
     AND EXISTS "${SIPLLM_ENGINE_ROOT}/shaders/matmul.comp" AND UNIX)
    message(STATUS "sipllm_ffi: generating ${_spv_header} via ${GLSLC_EXE}")
    execute_process(
      COMMAND ${GLSLC_EXE} -O -o "${CMAKE_CURRENT_BINARY_DIR}/matmul.spv"
              "${SIPLLM_ENGINE_ROOT}/shaders/matmul.comp"
      RESULT_VARIABLE _glslc_rc)
    if(_glslc_rc EQUAL 0)
      execute_process(COMMAND bash -c
        "{ echo '// auto-generated from shaders/matmul.comp'; \
           echo 'static const unsigned char matmul_spv[] = {'; \
           od -An -v -tu1 '${CMAKE_CURRENT_BINARY_DIR}/matmul.spv' | sed 's/[0-9]\\{1,\\}/&,/g'; \
           echo '};'; \
           echo 'static const unsigned int matmul_spv_len = sizeof(matmul_spv);'; } > '${_spv_header}'")
    endif()
  endif()
  if(NOT EXISTS "${_spv_header}")
    message(STATUS "sipllm_ffi: no SPIR-V shader (glslc absent); Vulkan device "
                   "enumeration only, matmul stays on CPU.")
  endif()

  if(ANDROID)
    target_link_libraries(sipllm_ffi vulkan)
  else()
    find_package(Vulkan)
    if(Vulkan_FOUND)
      target_link_libraries(sipllm_ffi Vulkan::Vulkan)
    else()
      message(WARNING "SIPLLM_VULKAN=ON but Vulkan not found; building CPU-only.")
    endif()
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(sipllm_ffi Threads::Threads)

set_target_properties(sipllm_ffi PROPERTIES
  OUTPUT_NAME "sipllm_ffi"
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden)

# Flutter desktop embedders read this list to bundle the shared object.
if(DEFINED FLUTTER_TARGET_PLATFORM OR WIN32 OR APPLE OR UNIX)
  set(sipllm_ffi_bundled_libraries
      $<TARGET_FILE:sipllm_ffi>
      PARENT_SCOPE)
endif()
