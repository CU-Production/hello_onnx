add_executable(MNIST MNIST.cpp)
target_link_libraries(MNIST PRIVATE onnxrt)
target_link_options(MNIST PRIVATE "/SUBSYSTEM:WINDOWS")
add_custom_command(
        TARGET MNIST PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/mnist.onnx "${CMAKE_CURRENT_BINARY_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rd_party/OnnxRuntime.DirectML/lib/onnxruntime.dll "${CMAKE_CURRENT_BINARY_DIR}/"
        VERBATIM
)

add_executable(MNIST_DML MNIST_DML.cpp)
target_link_libraries(MNIST_DML PRIVATE onnxrt)
target_link_options(MNIST_DML PRIVATE "/SUBSYSTEM:WINDOWS")
add_custom_command(
        TARGET MNIST_DML PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/mnist.onnx "${CMAKE_CURRENT_BINARY_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rd_party/OnnxRuntime.DirectML/lib/onnxruntime.dll "${CMAKE_CURRENT_BINARY_DIR}/"
        VERBATIM
)
