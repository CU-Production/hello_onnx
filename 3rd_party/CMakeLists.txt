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

# sokol
add_library(sokol INTERFACE)
target_include_directories(sokol INTERFACE sokol)

# imgui
add_library(imgui imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp)
target_include_directories(imgui PUBLIC imgui)

# nfd
add_subdirectory(nativefiledialog-extended)
