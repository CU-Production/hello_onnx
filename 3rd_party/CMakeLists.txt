# onnx (with directML)
add_library(onnxrt INTERFACE)
target_include_directories(onnxrt INTERFACE OnnxRuntime.DirectML/include)
target_link_directories(onnxrt INTERFACE OnnxRuntime.DirectML/lib)
target_link_libraries(onnxrt INTERFACE onnxruntime)

# directML
add_library(directML INTERFACE)
target_include_directories(directML INTERFACE directml/include)
target_link_directories(directML INTERFACE directml/lib)
target_link_libraries(directML INTERFACE DirectML)

# onnxruntime extensions
add_library(onnxrtEx INTERFACE)
target_include_directories(onnxrtEx INTERFACE OnnxRuntime.extensions/include)

# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)
