cmake_minimum_required(VERSION 3.18)
project(FirstNN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (vcpkg)
find_package(OpenCV REQUIRED)

# Direct LibTorch paths - bypass problematic cmake scripts
set(TORCH_INCLUDE_DIRS 
    "D:/libtorch-cuda/libtorch/include"
    "D:/libtorch-cuda/libtorch/include/torch/csrc/api/include"
)
set(TORCH_LIBRARIES 
    "D:/libtorch-cuda/libtorch/lib/torch.lib"
    "D:/libtorch-cuda/libtorch/lib/torch_cpu.lib"
    "D:/libtorch-cuda/libtorch/lib/torch_cuda.lib"
    "D:/libtorch-cuda/libtorch/lib/c10.lib"
    "D:/libtorch-cuda/libtorch/lib/c10_cuda.lib"
)

# Create executables
add_executable(FirstNN src/main.cpp)
add_executable(BinaryClassifier src/binary_classifier.cpp)
add_executable(ManualTest src/manual_test.cpp)
add_executable(CleanMNIST src/clean_mnist.cpp src/mnist_visualizer.cpp src/hand_drawing.cpp)
add_executable(RealMNIST src/real_mnist.cpp src/mnist_visualizer.cpp src/hand_drawing.cpp src/mnist_data_loader.cpp)

# Include and link
target_include_directories(FirstNN PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(FirstNN ${TORCH_LIBRARIES})
target_include_directories(BinaryClassifier PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(BinaryClassifier ${TORCH_LIBRARIES})
target_include_directories(ManualTest PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(ManualTest ${TORCH_LIBRARIES})
target_include_directories(CleanMNIST PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(CleanMNIST ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(RealMNIST PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(RealMNIST ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET FirstNN PROPERTY CXX_STANDARD 20)
set_property(TARGET BinaryClassifier PROPERTY CXX_STANDARD 20)
set_property(TARGET CleanMNIST PROPERTY CXX_STANDARD 20)
set_property(TARGET RealMNIST PROPERTY CXX_STANDARD 20)

# Define CUDA support
target_compile_definitions(FirstNN PRIVATE USE_CUDA=1)
target_compile_definitions(FirstNN PRIVATE TORCH_CUDA=1)
target_compile_definitions(BinaryClassifier PRIVATE USE_CUDA=1)
target_compile_definitions(BinaryClassifier PRIVATE TORCH_CUDA=1)
target_compile_definitions(CleanMNIST PRIVATE USE_CUDA=1)
target_compile_definitions(CleanMNIST PRIVATE TORCH_CUDA=1)
target_compile_definitions(RealMNIST PRIVATE USE_CUDA=1)
target_compile_definitions(RealMNIST PRIVATE TORCH_CUDA=1)

# Windows CUDA linker fix
if(WIN32)
    target_link_options(FirstNN PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
    target_link_options(BinaryClassifier PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
    target_link_options(CleanMNIST PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
    target_link_options(RealMNIST PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

# DLL copying removed - libraries already present in build/Debug directory

# Set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT FirstNN)
