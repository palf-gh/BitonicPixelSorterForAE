# preprocess_pipl_win.cmake
#
# Preprocess a PiPL .r file for Windows with the MSVC preprocessor (/EP: expand
# to stdout with no #line markers, which PiPLtool cannot parse). Self-contained:
# referenced only by this project's CMakeLists. Run via `cmake -P`.
#
# Required -D variables:
#   BPS_PIPL_COMPILER    full path to cl.exe (CMAKE_CXX_COMPILER)
#   BPS_PIPL_OUTPUT      path to write the preprocessed .i
#   BPS_PIPL_SOURCE      path to the .r source
#   BPS_PIPL_AESDK       After Effects SDK root (Examples/)
#   BPS_PIPL_SOURCE_DIR  project source dir (repo root)
#   BPS_PIPL_BINARY_DIR  build dir
# Optional:
#   BPS_PIPL_DEBUG       if set, defines _DEBUG / BPS_DEBUG_BUILD
#   BPS_PIPL_HAS_CUDA    if set, defines BPS_HAS_CUDA for PiPL flag parity
#   BPS_PIPL_HAS_OPENCL  if set, defines BPS_HAS_OPENCL for PiPL flag parity
#   BPS_PIPL_HAS_HLSL    if set, defines BPS_HAS_HLSL for PiPL flag parity

if(NOT DEFINED BPS_PIPL_COMPILER OR NOT DEFINED BPS_PIPL_OUTPUT OR NOT DEFINED BPS_PIPL_SOURCE)
	message(FATAL_ERROR "preprocess_pipl_win.cmake requires BPS_PIPL_COMPILER, BPS_PIPL_OUTPUT, BPS_PIPL_SOURCE")
endif()

set(_args
	/nologo
	/EP
	"/I${BPS_PIPL_AESDK}/Headers"
	"/I${BPS_PIPL_AESDK}/Resources"
	"/I${BPS_PIPL_BINARY_DIR}"
	"/I${BPS_PIPL_SOURCE_DIR}"
	"/I${BPS_PIPL_SOURCE_DIR}/Source"
	/D "MSWindows"
	/D "AE_OS_WIN"
	/D "AE_PROC_INTELx64"
)
if(BPS_PIPL_DEBUG)
	list(APPEND _args /D "_DEBUG" /D "BPS_DEBUG_BUILD=1")
endif()
if(BPS_PIPL_HAS_CUDA)
	list(APPEND _args /D "BPS_HAS_CUDA=1")
endif()
if(BPS_PIPL_HAS_OPENCL)
	list(APPEND _args /D "BPS_HAS_OPENCL=1")
endif()
if(BPS_PIPL_HAS_HLSL)
	list(APPEND _args /D "BPS_HAS_HLSL=1")
endif()
list(APPEND _args "${BPS_PIPL_SOURCE}")

execute_process(
	COMMAND "${BPS_PIPL_COMPILER}" ${_args}
	OUTPUT_FILE "${BPS_PIPL_OUTPUT}"
	ERROR_VARIABLE _stderr
	RESULT_VARIABLE _result
)

if(NOT _result EQUAL 0)
	message(FATAL_ERROR "PiPL preprocess failed (${_result}): ${_stderr}")
endif()
