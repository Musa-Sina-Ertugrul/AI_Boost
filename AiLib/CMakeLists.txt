cmake_minimum_required(VERSION 3.14)
project(AIBoost VERSION 0.0.0 LANGUAGES CXX)
SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  /opt/OpenBLAS/include
  /usr/local/include/openblas
  /usr/include/openblas
  /usr/local/include/openblas-base
  /usr/include/openblas-base
  /usr/local/include
  /usr/include
  /usr/include/x86_64-linux-gnu
  /usr/lib/x86_64-linux-gnu
  /usr/include/x86_64-linux-gnu/
  /usr/lib/x86_64-linux-gnu/openblas-pthread/
  /usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas/
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        /opt/OpenBLAS/lib
        /usr/local/lib64
        /usr/local/lib
        /lib/openblas-base
        /lib64/
        /lib/
        /usr/lib/openblas-base
        /usr/lib64
        /usr/lib
        /usr/include/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/include/x86_64-linux-gnu/
        /usr/lib/x86_64-linux-gnu/openblas-pthread/
        /usr/lib/x86_64-linux-gnu/openblas-pthread/cmake/openblas/
 )


FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS} NO_DEFAULT_PATH)
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS}  NO_DEFAULT_PATH)

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)

# Tell CMake to also search in the build folder for the "<library>-config.cmake" files
add_library(Model STATIC Model.cpp)
set_target_properties(Model PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(Layer STATIC Layer.cpp)
set_target_properties(Layer PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(Enums STATIC Enums.cpp)
set_target_properties(Enums PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(DataSet STATIC DataSet.cpp)
set_target_properties(DataSet PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Find OpenBLAS
find_package (Python3 REQUIRED Interpreter Development)
# Find Pybind11
find_package(PythonLibs REQUIRED)
find_package(BLAS REQUIRED)
find_package(SWIG REQUIRED)



include(UseSWIG)
include(${SWIG_USE_FILE})
set(SWIG_INTERFACE_FILE AIBoost.i)
set(SOURCE_FILES AIBoost.cpp)

set_source_files_properties(AIBoost.i PROPERTIES CPLUSPLUS ON)
swig_add_library(AIBoost LANGUAGE python SOURCES ${SWIG_INTERFACE_FILE} ${SOURCE_FILES})
swig_link_libraries(AIBoost ${PYTHON_LIBRARIES})
swig_link_libraries(AIBoost ${OpenBLAS_LIB})
swig_link_libraries(AIBoost Model)
swig_link_libraries(AIBoost Layer)
swig_link_libraries(AIBoost Enums)
swig_link_libraries(AIBoost DataSet)
# Create a Pybind11 module
# Link OpenBLAS to the Pybind11 module
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OpenBLAS_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

target_link_libraries(Model ${PYTHON_LIBRARIES})
target_link_libraries(Model ${OpenBLAS_LIB})
target_link_libraries(Layer ${PYTHON_LIBRARIES})
target_link_libraries(Layer ${OpenBLAS_LIB})
target_link_libraries(Enums ${PYTHON_LIBRARIES})
target_link_libraries(Enums ${OpenBLAS_LIB})
target_link_libraries(DataSet ${PYTHON_LIBRARIES})
target_link_libraries(DataSet ${OpenBLAS_LIB})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

install(TARGETS AIBoost LIBRARY DESTINATION /home/musasina/Desktop/AI_Boost/AiLib/build ARCHIVE DESTINATION /home/musasina/Desktop/AI_Boost/AiLib/build)


# Set C++ standard

