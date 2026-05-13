cmake_minimum_required(VERSION 3.18)
project(WhisperONNXRuntime)

# Set CMP0144 policy to NEW, using the recommended uppercase <PACKAGENAME>_ROOT variables
#if(POLICY CMP0144)
#    cmake_policy(SET CMP0144 NEW)
#endif()

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------
# Version variables - change here to switch package versions
# ---------------------------------------------------------
set(WINML_VERSION "2.0.300")

# Paths to ONNX Runtime, libsndfile, libtorch and KNF
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
set(LIBSNDFILE_DIR "${CMAKE_SOURCE_DIR}/libsndfile")
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/libtorch")
set(KNF_DIR "${CMAKE_SOURCE_DIR}/kaldi_native_fbank")
set(WINML_DIR "${CMAKE_SOURCE_DIR}/Microsoft.WindowsAppSDK.ML.${WINML_VERSION}")

# ---------------------------------------------------------
# Download Microsoft.WindowsAppSDK.ML via nuget if absent
# ---------------------------------------------------------
if(NOT EXISTS "${WINML_DIR}")
    set(NUGET_EXE "${WINML_DIR}/nuget.exe")
    if(NOT EXISTS "${NUGET_EXE}")
        message(STATUS "Downloading nuget.exe ...")
        file(DOWNLOAD
            "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
            "${NUGET_EXE}"
            SHOW_PROGRESS STATUS NUGET_DL_STATUS)
        list(GET NUGET_DL_STATUS 0 NUGET_DL_ERROR)
        if(NUGET_DL_ERROR)
            list(GET NUGET_DL_STATUS 1 NUGET_DL_MSG)
            message(FATAL_ERROR "Failed to download nuget.exe: ${NUGET_DL_MSG}")
        endif()
    endif()
    message(STATUS "Installing Microsoft.WindowsAppSDK.ML ${WINML_VERSION} ...")
    execute_process(
        COMMAND "${NUGET_EXE}" install Microsoft.WindowsAppSDK.ML
                -version ${WINML_VERSION}
                -OutputDirectory "${WINML_DIR}"
                -NonInteractive
        RESULT_VARIABLE NUGET_RESULT
        OUTPUT_VARIABLE NUGET_OUTPUT
        ERROR_VARIABLE  NUGET_ERROR)
    if(NUGET_RESULT)
        message(FATAL_ERROR "nuget install failed:\n${NUGET_OUTPUT}\n${NUGET_ERROR}")
    endif()
    message(STATUS "Microsoft.WindowsAppSDK.ML installed to ${WINML_DIR}")
endif()

# Find libraries
find_library(WINML_LIB Microsoft.Windows.AI.MachineLearning PATHS ${WINML_DIR}/Microsoft.Windows.AI.MachineLearning.${WINML_VERSION}/lib/native/x64)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${WINML_DIR}/Microsoft.Windows.AI.MachineLearning.${WINML_VERSION}/lib/native/x64)
find_library(LIBSNDFILE_LIB sndfile PATHS ${LIBSNDFILE_DIR}/lib)
find_library(KNF_LIB kaldi-native-fbank-core PATHS ${KNF_DIR}/lib)

if (NOT WINML_LIB)
    message(FATAL_ERROR "Microsoft.Windows.AI.MachineLearning.lib not found")
endif()
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime.lib not found")
endif()
if (NOT LIBSNDFILE_LIB)
    message(FATAL_ERROR "sndfile.lib not found")
endif()
if (NOT KNF_LIB)
    message(FATAL_ERROR "kaldi-native-fbank-core.lib not found")
endif()

# ---------------------------------------------------------
# Download LibTorch if absent
# ---------------------------------------------------------
if(NOT EXISTS "${LIBTORCH_DIR}")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip")
    set(LIBTORCH_ZIP "${CMAKE_BINARY_DIR}/libtorch.zip")
    message(STATUS "LibTorch not found. Downloading from ${LIBTORCH_URL} ...")
    file(DOWNLOAD "${LIBTORCH_URL}" "${LIBTORCH_ZIP}" SHOW_PROGRESS STATUS LIBTORCH_DL_STATUS)
    list(GET LIBTORCH_DL_STATUS 0 LIBTORCH_DL_ERROR)
    if(LIBTORCH_DL_ERROR)
        list(GET LIBTORCH_DL_STATUS 1 LIBTORCH_DL_MSG)
        message(FATAL_ERROR "Failed to download LibTorch: ${LIBTORCH_DL_MSG}")
    endif()
    message(STATUS "Extracting LibTorch ...")
    file(ARCHIVE_EXTRACT INPUT "${LIBTORCH_ZIP}" DESTINATION "${CMAKE_SOURCE_DIR}")
    file(REMOVE "${LIBTORCH_ZIP}")
    message(STATUS "LibTorch extracted to ${LIBTORCH_DIR}")
endif()

# Set the path to libtorch
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)

# Include directories
include_directories(${WINML_DIR}/Microsoft.Windows.AI.MachineLearning.${WINML_VERSION}/include)
include_directories(${WINML_DIR}/Microsoft.Windows.AI.MachineLearning.${WINML_VERSION}/include/winml)
include_directories(${LIBSNDFILE_DIR}/include)
include_directories(${KNF_DIR}/include)
include_directories(include)

# Add the executable
add_executable(whisper_onnx src/main.cpp src/base64.cpp src/base64.h src/whisper.cpp src/whisper.hpp)

# Link the ONNX Runtime, SndFile, libtorch, and FFTW libraries
target_link_libraries(whisper_onnx ${ONNXRUNTIME_LIB} ${WINML_LIB} ${LIBSNDFILE_LIB} ${TORCH_LIBRARIES} ${KNF_LIB} ${VAD_LIB} WindowsApp.lib dxguid.lib )

# ---------------------------------------------------------
# Default install destination: <source_root>/package
# Override at configure time: cmake -DCMAKE_INSTALL_PREFIX=<path> ..
# ---------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/package" CACHE PATH "Install prefix" FORCE)
endif()

# Executable
install(TARGETS whisper_onnx RUNTIME DESTINATION .)

# Root-level data files
install(FILES
    "${CMAKE_SOURCE_DIR}/README.md"
    "${CMAKE_SOURCE_DIR}/how_are_you_doing_today.wav"
    "${CMAKE_SOURCE_DIR}/simple_run.bat"
    DESTINATION .)

# Models and tokenizer from Script/
file(GLOB_RECURSE MODEL_FILES
    "${CMAKE_SOURCE_DIR}/Script/*.onnx"
    "${CMAKE_SOURCE_DIR}/Script/*.weights"
    "${CMAKE_SOURCE_DIR}/Script/*tokens.txt")
install(FILES ${MODEL_FILES} DESTINATION .)

# kaldi_native_fbank DLLs
file(GLOB_RECURSE KNF_DLLS "${KNF_DIR}/*.dll")
install(FILES ${KNF_DLLS} DESTINATION .)

# libsndfile DLLs
file(GLOB_RECURSE LIBSNDFILE_DLLS "${LIBSNDFILE_DIR}/*.dll")
install(FILES ${LIBSNDFILE_DLLS} DESTINATION .)

# libtorch DLLs
file(GLOB_RECURSE LIBTORCH_DLLS "${LIBTORCH_DIR}/*.dll")
install(FILES ${LIBTORCH_DLLS} DESTINATION .)

# WinML DLLs
file(GLOB_RECURSE WINML_DLLS "${WINML_DIR}/Microsoft.Windows.AI.MachineLearning.${WINML_VERSION}/runtimes/win-x64/native/*.dll")
install(FILES ${WINML_DLLS} DESTINATION .)
