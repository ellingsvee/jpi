cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

# nanobind_add_module(_core MODULE src/cpp/csr/csr_module.cc)
nanobind_add_module(backend MODULE
    src/jpi/backend/module.cc
    src/jpi/backend/csr_spmm.cc
)
target_include_directories("backend" PUBLIC ${XLA_DIR})
install(TARGETS backend DESTINATION ${SKBUILD_PROJECT_NAME})
