# BitonicPixelSorter — self-contained After Effects plugin build.
#
# Self-contained: references ONLY the Adobe After Effects SDK (the Examples tree
# this repo sits in). Place this repository anywhere inside the SDK Examples
# directory and configure; AESDK_ROOT is auto-detected as the parent folder.
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
# Optionally install the built .aex/.plugin straight into an AE plug-ins folder:
#   cmake -S . -B build -DBPS_AE_PLUGIN_DIR="C:/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore"

cmake_minimum_required(VERSION 3.18)
project(BitonicPixelSorter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BPS_RENDER_DIAG "Enable Bitonic Pixel Sorter render diagnostics" OFF)

set(BPS_DIST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dist")

#------------------------------------------------------------------------------
# Optional GPU backends. Each is enabled only when its toolchain is present, so
# the CPU-only build always works. CUDA and OpenCL are wired up here; DirectX /
# Metal are added in later phases.
#------------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
			set(CMAKE_CUDA_ARCHITECTURES all-major)
		else()
			set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86)
		endif()
	endif()
	enable_language(CUDA)
	set(BPS_CUDA_AVAILABLE TRUE)
	message(STATUS "BitonicPixelSorter: CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
	set(BPS_CUDA_AVAILABLE FALSE)
	message(STATUS "BitonicPixelSorter: CUDA toolkit not found — building CPU-only")
endif()

#------------------------------------------------------------------------------
# Locate the After Effects SDK (the Examples tree). Default: parent directory.
#------------------------------------------------------------------------------
if(NOT DEFINED AESDK_ROOT OR NOT EXISTS "${AESDK_ROOT}/Headers")
	get_filename_component(_examples_dir "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
	if(EXISTS "${_examples_dir}/Headers")
		set(AESDK_ROOT "${_examples_dir}" CACHE PATH "After Effects SDK Examples root" FORCE)
	endif()
endif()

if(NOT EXISTS "${AESDK_ROOT}/Headers")
	message(FATAL_ERROR
		"AESDK_ROOT (${AESDK_ROOT}) has no Headers/ directory.\n"
		"Place this repository inside the After Effects SDK 'Examples' tree, "
		"or configure with -DAESDK_ROOT=<path-to-Examples>.")
endif()

message(STATUS "BitonicPixelSorter: AESDK_ROOT = ${AESDK_ROOT}")

# OpenCL is optional. The SDK sample embeds OpenCL source as a C string; follow
# that model so the plugin remains a single self-contained binary.
find_package(OpenCL QUIET)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(OpenCL_FOUND AND Python3_Interpreter_FOUND)
	set(BPS_OPENCL_AVAILABLE TRUE)
	message(STATUS "BitonicPixelSorter: OpenCL enabled")
elseif(OpenCL_FOUND AND NOT Python3_Interpreter_FOUND)
	set(BPS_OPENCL_AVAILABLE FALSE)
	message(STATUS "BitonicPixelSorter: OpenCL found but Python3 is missing — OpenCL disabled")
else()
	set(BPS_OPENCL_AVAILABLE FALSE)
	message(STATUS "BitonicPixelSorter: OpenCL not found")
endif()

if(WIN32)
	find_program(BPS_DXC_EXECUTABLE NAMES dxc)
	if(BPS_DXC_EXECUTABLE AND Python3_Interpreter_FOUND)
		set(BPS_HLSL_AVAILABLE TRUE)
		message(STATUS "BitonicPixelSorter: DirectX/HLSL enabled (${BPS_DXC_EXECUTABLE})")
	elseif(BPS_DXC_EXECUTABLE AND NOT Python3_Interpreter_FOUND)
		set(BPS_HLSL_AVAILABLE FALSE)
		message(STATUS "BitonicPixelSorter: dxc found but Python3 is missing — DirectX disabled")
	else()
		set(BPS_HLSL_AVAILABLE FALSE)
		message(STATUS "BitonicPixelSorter: dxc not found — DirectX disabled")
	endif()
else()
	set(BPS_HLSL_AVAILABLE FALSE)
endif()

set(BPS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

include_directories(
	"${AESDK_ROOT}/Headers"
	"${AESDK_ROOT}/Headers/SP"
	"${AESDK_ROOT}/Util"
	"${AESDK_ROOT}/Resources"
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/Source"
)

#------------------------------------------------------------------------------
# Sources (plugin + the few SDK utility translation units it links).
#------------------------------------------------------------------------------
set(SOURCES
	Source/BitonicPixelSorter.h
	Source/BitonicPixelSorter_Target.h
	Source/BitonicPixelSorter_GpuEligibility.h
	Source/BitonicPixelSorter.cpp
	Source/BitonicPixelSorter_UI.cpp
	Source/BitonicPixelSorter_CPU.cpp
	Source/BitonicPixelSorter_GPU.cpp
	vendor/palf/AELocalise.h
	vendor/palf/AE_SDK.h
	Localise/LocKeys.h
	Localise/Strings_en_US.h
	Localise/Strings_ja_JP.h
	Localise/Strings_zh_CN.h
	Localise/Strings_ko_KR.h

	"${AESDK_ROOT}/Util/AEGP_SuiteHandler.cpp"
	"${AESDK_ROOT}/Util/MissingSuiteError.cpp"
	"${AESDK_ROOT}/Util/AEFX_SuiteHelper.c"
	"${AESDK_ROOT}/Util/Smart_Utils.cpp"
)

# GPU kernel translation units (added when their toolchain is available).
if(BPS_CUDA_AVAILABLE)
	list(APPEND SOURCES "GPU/BitonicPixelSorter_Kernel.cu")
endif()

if(BPS_OPENCL_AVAILABLE)
	set(BPS_OPENCL_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/GPU/BitonicPixelSorter_Kernel.cl")
	set(BPS_OPENCL_KERNEL_HEADER "${BPS_GENERATED_DIR}/BitonicPixelSorter_Kernel.cl.h")
	add_custom_command(
		OUTPUT "${BPS_OPENCL_KERNEL_HEADER}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${BPS_GENERATED_DIR}"
		COMMAND "${Python3_EXECUTABLE}"
			"${AESDK_ROOT}/GPUUtils/CreateCString.py"
			-i "${BPS_OPENCL_KERNEL_SOURCE}"
			-o "${BPS_OPENCL_KERNEL_HEADER}"
			-n kBitonicPixelSorter_Kernel_OpenCLString
		DEPENDS "${BPS_OPENCL_KERNEL_SOURCE}" "${AESDK_ROOT}/GPUUtils/CreateCString.py"
		COMMENT "Embedding OpenCL kernel source..."
		VERBATIM)
	list(APPEND SOURCES "${BPS_OPENCL_KERNEL_HEADER}")
endif()

if(BPS_HLSL_AVAILABLE)
	list(APPEND SOURCES
		Source/BPS_DirectXShaderLoad.h
		Source/BPS_DirectXShaderLoad.cpp
		"${AESDK_ROOT}/Util/DirectXUtils.h"
		"${AESDK_ROOT}/Util/DirectXUtils.cpp")
	set(BPS_HLSL_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/GPU/BitonicPixelSorter_Kernel.hlsl")
	set(BPS_DIRECTX_ASSET_BUILD_DIR "${BPS_GENERATED_DIR}/DirectX_Assets")
	set(BPS_DIRECTX_CSO "${BPS_DIRECTX_ASSET_BUILD_DIR}/BitonicSortKernel.cso")
	set(BPS_DIRECTX_RS  "${BPS_DIRECTX_ASSET_BUILD_DIR}/BitonicSortKernel.rs")
	set(BPS_DIRECTX_CSO_HEADER "${BPS_GENERATED_DIR}/BitonicSortKernel.cso.h")
	set(BPS_DIRECTX_RS_HEADER  "${BPS_GENERATED_DIR}/BitonicSortKernel.rs.h")
	add_custom_command(
		OUTPUT "${BPS_DIRECTX_CSO}" "${BPS_DIRECTX_RS}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${BPS_DIRECTX_ASSET_BUILD_DIR}"
		COMMAND "${BPS_DXC_EXECUTABLE}"
			"${BPS_HLSL_KERNEL_SOURCE}"
			-E main
			-Fo "${BPS_DIRECTX_CSO}"
			-Frs "${BPS_DIRECTX_RS}"
			-T cs_6_5
			-ignore-line-directives
		DEPENDS "${BPS_HLSL_KERNEL_SOURCE}"
		COMMENT "Compiling DirectX HLSL kernel..."
		VERBATIM)
	add_custom_command(
		OUTPUT "${BPS_DIRECTX_CSO_HEADER}"
		COMMAND "${Python3_EXECUTABLE}"
			"${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
			-i "${BPS_DIRECTX_CSO}"
			-o "${BPS_DIRECTX_CSO_HEADER}"
			-n kBitonicSortKernel_cso
		DEPENDS "${BPS_DIRECTX_CSO}"
		COMMENT "Embedding DirectX CSO..."
		VERBATIM)
	add_custom_command(
		OUTPUT "${BPS_DIRECTX_RS_HEADER}"
		COMMAND "${Python3_EXECUTABLE}"
			"${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.py"
			-i "${BPS_DIRECTX_RS}"
			-o "${BPS_DIRECTX_RS_HEADER}"
			-n kBitonicSortKernel_rs
		DEPENDS "${BPS_DIRECTX_RS}"
		COMMENT "Embedding DirectX root signature..."
		VERBATIM)
	add_custom_target(${PROJECT_NAME}_directx_shaders
		DEPENDS
			"${BPS_DIRECTX_CSO}"
			"${BPS_DIRECTX_RS}"
			"${BPS_DIRECTX_CSO_HEADER}"
			"${BPS_DIRECTX_RS_HEADER}")
endif()

set(BPS_PIPL_BACKEND_DEFINES)
if(BPS_CUDA_AVAILABLE)
	list(APPEND BPS_PIPL_BACKEND_DEFINES "-DBPS_HAS_CUDA=1")
endif()
if(BPS_OPENCL_AVAILABLE)
	list(APPEND BPS_PIPL_BACKEND_DEFINES "-DBPS_HAS_OPENCL=1")
endif()
if(BPS_HLSL_AVAILABLE)
	list(APPEND BPS_PIPL_BACKEND_DEFINES "-DBPS_HAS_HLSL=1")
endif()

set(PIPL_R "${CMAKE_CURRENT_SOURCE_DIR}/PiPL/BitonicPixelSorterPiPL.r")
set(TARGET_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Source/BitonicPixelSorter_Target.h")

if(WIN32)
	#--------------------------------------------------------------------------
	# Windows: build PiPL .rc via MSVC /EP preprocess + PiPLtool, per config.
	#--------------------------------------------------------------------------
	set(PIPL_TOOL "${AESDK_ROOT}/Resources/PiPLtool.exe")
	set(PIPL_PRE  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/preprocess_pipl_win.cmake")

	set(PIPL_I_DEBUG    "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Debug.i")
	set(PIPL_I_RELEASE  "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Release.i")
	set(PIPL_RC_DEBUG   "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Debug.rc")
	set(PIPL_RC_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Release.rc")

	add_custom_command(
		OUTPUT "${PIPL_RC_DEBUG}"
		COMMAND ${CMAKE_COMMAND}
			-D "BPS_PIPL_COMPILER=${CMAKE_CXX_COMPILER}"
			-D "BPS_PIPL_OUTPUT=${PIPL_I_DEBUG}"
			-D "BPS_PIPL_SOURCE=${PIPL_R}"
			-D "BPS_PIPL_AESDK=${AESDK_ROOT}"
			-D "BPS_PIPL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
			-D "BPS_PIPL_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
			-D "BPS_PIPL_DEBUG=1"
			-D "BPS_PIPL_HAS_CUDA=${BPS_CUDA_AVAILABLE}"
			-D "BPS_PIPL_HAS_OPENCL=${BPS_OPENCL_AVAILABLE}"
			-D "BPS_PIPL_HAS_HLSL=${BPS_HLSL_AVAILABLE}"
			-P "${PIPL_PRE}"
		COMMAND "${PIPL_TOOL}" "${PIPL_I_DEBUG}" "${PIPL_RC_DEBUG}"
		DEPENDS "${PIPL_R}" "${TARGET_HEADER}" "${PIPL_PRE}"
		COMMENT "Generating PiPL resource (Debug)..."
		VERBATIM)

	add_custom_command(
		OUTPUT "${PIPL_RC_RELEASE}"
		COMMAND ${CMAKE_COMMAND}
			-D "BPS_PIPL_COMPILER=${CMAKE_CXX_COMPILER}"
			-D "BPS_PIPL_OUTPUT=${PIPL_I_RELEASE}"
			-D "BPS_PIPL_SOURCE=${PIPL_R}"
			-D "BPS_PIPL_AESDK=${AESDK_ROOT}"
			-D "BPS_PIPL_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
			-D "BPS_PIPL_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
			-D "BPS_PIPL_HAS_CUDA=${BPS_CUDA_AVAILABLE}"
			-D "BPS_PIPL_HAS_OPENCL=${BPS_OPENCL_AVAILABLE}"
			-D "BPS_PIPL_HAS_HLSL=${BPS_HLSL_AVAILABLE}"
			-P "${PIPL_PRE}"
		COMMAND "${PIPL_TOOL}" "${PIPL_I_RELEASE}" "${PIPL_RC_RELEASE}"
		DEPENDS "${PIPL_R}" "${TARGET_HEADER}" "${PIPL_PRE}"
		COMMENT "Generating PiPL resource (Release)..."
		VERBATIM)

	list(APPEND SOURCES "$<IF:$<CONFIG:Debug>,${PIPL_RC_DEBUG},${PIPL_RC_RELEASE}>")

	add_library(${PROJECT_NAME} SHARED ${SOURCES})

	set_target_properties(${PROJECT_NAME} PROPERTIES
		SUFFIX ".aex"
		LINK_FLAGS "/EXPORT:EffectMain")

	target_compile_definitions(${PROJECT_NAME} PRIVATE MSWindows _WINDOWS)
	if(MSVC)
		# /utf-8 must only reach the C/C++ host compiler, not nvcc.
		target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
	endif()

	# Debug builds get a "_debug" suffix so they can sit beside Release.
	set_target_properties(${PROJECT_NAME} PROPERTIES
		OUTPUT_NAME "$<IF:$<CONFIG:Debug>,BitonicPixelSorter_debug,BitonicPixelSorter>")

elseif(APPLE)
	#--------------------------------------------------------------------------
	# macOS: preprocess the PiPL with clang -E, compile with Rez, embed into the
	# .plugin bundle's executable. Self-contained (no PalfLib).
	#--------------------------------------------------------------------------
	find_program(REZ_TOOL Rez)
	if(NOT REZ_TOOL)
		file(GLOB _xcode_apps "/Applications/Xcode*.app")
		if(_xcode_apps)
			list(GET _xcode_apps 0 _xcode_app)
			set(REZ_TOOL "${_xcode_app}/Contents/Developer/Tools/Rez")
			if(NOT EXISTS "${REZ_TOOL}")
				set(REZ_TOOL "${_xcode_app}/Contents/Developer/usr/bin/Rez")
			endif()
		endif()
	endif()
	if(NOT REZ_TOOL OR NOT EXISTS "${REZ_TOOL}")
		message(FATAL_ERROR "Rez tool not found. Install Xcode or the Command Line Tools.")
	endif()

	set(PIPL_I_DEBUG    "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Debug.i")
	set(PIPL_I_RELEASE  "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Release.i")
	set(PIPL_RSRC_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Debug.rsrc")
	set(PIPL_RSRC_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/BitonicPixelSorterPiPL_Release.rsrc")

	add_custom_command(
		OUTPUT "${PIPL_I_DEBUG}"
		COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c
			-I "${AESDK_ROOT}/Headers"
			-I "${AESDK_ROOT}/Resources"
			-I "${CMAKE_CURRENT_SOURCE_DIR}/Source"
			-D __MACH__ -D AE_OS_MAC -D DEBUG=1 -D BPS_DEBUG_BUILD=1
			${BPS_PIPL_BACKEND_DEFINES}
			-o "${PIPL_I_DEBUG}" "${PIPL_R}"
		DEPENDS "${PIPL_R}" "${TARGET_HEADER}"
		COMMENT "Preprocessing PiPL resource (Debug)...")

	add_custom_command(
		OUTPUT "${PIPL_I_RELEASE}"
		COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c
			-I "${AESDK_ROOT}/Headers"
			-I "${AESDK_ROOT}/Resources"
			-I "${CMAKE_CURRENT_SOURCE_DIR}/Source"
			-D __MACH__ -D AE_OS_MAC
			${BPS_PIPL_BACKEND_DEFINES}
			-o "${PIPL_I_RELEASE}" "${PIPL_R}"
		DEPENDS "${PIPL_R}" "${TARGET_HEADER}"
		COMMENT "Preprocessing PiPL resource (Release)...")

	add_custom_command(
		OUTPUT "${PIPL_RSRC_DEBUG}"
		COMMAND "${REZ_TOOL}" -o "${PIPL_RSRC_DEBUG}" -useDF
			-i "${AESDK_ROOT}/Headers"
			-i "${AESDK_ROOT}/Resources"
			-i "${CMAKE_CURRENT_SOURCE_DIR}/Source"
			"${PIPL_I_DEBUG}"
		DEPENDS "${PIPL_I_DEBUG}"
		COMMENT "Compiling PiPL resource with Rez (Debug)...")

	add_custom_command(
		OUTPUT "${PIPL_RSRC_RELEASE}"
		COMMAND "${REZ_TOOL}" -o "${PIPL_RSRC_RELEASE}" -useDF
			-i "${AESDK_ROOT}/Headers"
			-i "${AESDK_ROOT}/Resources"
			-i "${CMAKE_CURRENT_SOURCE_DIR}/Source"
			"${PIPL_I_RELEASE}"
		DEPENDS "${PIPL_I_RELEASE}"
		COMMENT "Compiling PiPL resource with Rez (Release)...")

	add_custom_target(${PROJECT_NAME}_pipl DEPENDS "${PIPL_RSRC_DEBUG}" "${PIPL_RSRC_RELEASE}")

	add_library(${PROJECT_NAME} MODULE ${SOURCES})
	add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_pipl)

	set(BUNDLE_EXECUTABLE "BitonicPixelSorter")
	set(BUNDLE_NAME "BitonicPixelSorter")
	string(TOLOWER "${BUNDLE_NAME}" BUNDLE_NAME_LOWER)
	configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Mac/Info.plist.in"
				   "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)

	set_target_properties(${PROJECT_NAME} PROPERTIES
		BUNDLE TRUE
		BUNDLE_EXTENSION "plugin"
		OUTPUT_NAME "$<IF:$<CONFIG:Debug>,BitonicPixelSorter_debug,BitonicPixelSorter>"
		MACOSX_BUNDLE_GUI_IDENTIFIER "com.palf.bitonicpixelsorter"
		OSX_ARCHITECTURES "x86_64;arm64"
		XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN "NO")

	set(_exports "${CMAKE_CURRENT_BINARY_DIR}/exported_symbols.txt")
	file(WRITE "${_exports}" "_EffectMain\n_PluginDataEntryFunction2\n")
	target_link_options(${PROJECT_NAME} PRIVATE
		-Wl,-exported_symbols_list,${_exports}
		-Wl,-sectcreate,__TEXT,__plgin,$<IF:$<CONFIG:Debug>,${PIPL_RSRC_DEBUG},${PIPL_RSRC_RELEASE}>)

	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory
			"$<TARGET_BUNDLE_DIR:${PROJECT_NAME}>/Contents/Resources"
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
			"$<TARGET_BUNDLE_DIR:${PROJECT_NAME}>/Contents/Info.plist"
		COMMAND ${CMAKE_COMMAND} -E copy
			"$<IF:$<CONFIG:Debug>,${PIPL_RSRC_DEBUG},${PIPL_RSRC_RELEASE}>"
			"$<TARGET_BUNDLE_DIR:${PROJECT_NAME}>/Contents/Resources/BitonicPixelSorterPiPL.rsrc"
		COMMAND xattr -cr "$<TARGET_BUNDLE_DIR:${PROJECT_NAME}>"
		COMMENT "Assembling .plugin bundle...")

	find_library(COCOA_LIBRARY Cocoa)
	target_link_libraries(${PROJECT_NAME} PRIVATE ${COCOA_LIBRARY})
	target_compile_definitions(${PROJECT_NAME} PRIVATE MAC_ENV)
endif()

# Keep PluginDataEntryFunction2 and PiPL match names aligned per configuration.
target_compile_definitions(${PROJECT_NAME} PRIVATE
	$<$<CONFIG:Debug>:BPS_DEBUG_BUILD=1>)

if(WIN32)
	set(BPS_DIST_PLATFORM_DIR "${BPS_DIST_ROOT}/Win")
elseif(APPLE)
	set(BPS_DIST_PLATFORM_DIR "${BPS_DIST_ROOT}/Mac")
endif()

if(DEFINED BPS_DIST_PLATFORM_DIR)
	foreach(_bps_config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
		string(TOUPPER "${_bps_config}" _bps_config_upper)
		set(_bps_config_output_dir "${BPS_DIST_PLATFORM_DIR}/${_bps_config}")
		set_target_properties(${PROJECT_NAME} PROPERTIES
			"RUNTIME_OUTPUT_DIRECTORY_${_bps_config_upper}" "${_bps_config_output_dir}"
			"LIBRARY_OUTPUT_DIRECTORY_${_bps_config_upper}" "${_bps_config_output_dir}"
			"ARCHIVE_OUTPUT_DIRECTORY_${_bps_config_upper}" "${CMAKE_CURRENT_BINARY_DIR}/${_bps_config}/lib")
	endforeach()
	unset(_bps_config)
	unset(_bps_config_upper)
	unset(_bps_config_output_dir)
endif()

#------------------------------------------------------------------------------
# CUDA backend: define BPS_HAS_CUDA, set architectures, link the CUDA runtime.
#------------------------------------------------------------------------------
if(BPS_RENDER_DIAG)
	target_compile_definitions(${PROJECT_NAME} PRIVATE BPS_RENDER_DIAG=1)
endif()

if(BPS_CUDA_AVAILABLE)
	find_package(CUDAToolkit REQUIRED)
	target_compile_definitions(${PROJECT_NAME} PRIVATE BPS_HAS_CUDA=1)
	set_target_properties(${PROJECT_NAME} PROPERTIES
		CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
		CUDA_RUNTIME_LIBRARY None
		CUDA_SEPARABLE_COMPILATION OFF)
	if(MSVC)
		target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
	endif()
	target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif()

if(BPS_OPENCL_AVAILABLE)
	target_compile_definitions(${PROJECT_NAME} PRIVATE
		BPS_HAS_OPENCL=1
		CL_TARGET_OPENCL_VERSION=120
		CL_MINIMUM_OPENCL_VERSION=120)
	target_include_directories(${PROJECT_NAME} PRIVATE "${BPS_GENERATED_DIR}")
	if(TARGET OpenCL::OpenCL)
		target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
	else()
		target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
		target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})
	endif()
endif()

if(BPS_HLSL_AVAILABLE)
	add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_directx_shaders)
	target_compile_definitions(${PROJECT_NAME} PRIVATE BPS_HAS_HLSL=1)
	target_include_directories(${PROJECT_NAME} PRIVATE "${BPS_GENERATED_DIR}")
	target_link_libraries(${PROJECT_NAME} PRIVATE d3d12 d3dcompiler dxgi)
endif()

#------------------------------------------------------------------------------
# Optional: copy the built plugin into an After Effects plug-ins directory.
#------------------------------------------------------------------------------
if(BPS_AE_PLUGIN_DIR)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"$<TARGET_FILE:${PROJECT_NAME}>" "${BPS_AE_PLUGIN_DIR}/"
		COMMENT "Installing plugin to ${BPS_AE_PLUGIN_DIR}")
endif()
