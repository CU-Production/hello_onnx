add_executable(sd15cpp_txt2img_cli
        txt2img_cli.cpp
        StableDiffusion/TensorType.h
        StableDiffusion/TensorType.cpp
        StableDiffusion/Tensor.h
        StableDiffusion/Tensor.cpp
        StableDiffusion/StableDiffusionConfig.h
        StableDiffusion/UNet.h
        StableDiffusion/UNet.cpp
        StableDiffusion/TextProcessing.h
        StableDiffusion/TextProcessing.cpp
        StableDiffusion/SchedulerBase.h
        StableDiffusion/SchedulerBase.cpp
        StableDiffusion/LMSDiscreteScheduler.h
        StableDiffusion/LMSDiscreteScheduler.cpp
        StableDiffusion/MathUtils.h
        StableDiffusion/VaeDecoder.h
        StableDiffusion/VaeDecoder.cpp

)
target_link_libraries(sd15cpp_txt2img_cli PRIVATE
        directml
        onnxrt
        onnxrtEx
        stb
)
add_custom_command(
        TARGET sd15cpp_txt2img_cli PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/cliptokenizer.onnx "${CMAKE_CURRENT_BINARY_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rd_party/OnnxRuntime.DirectML/lib/onnxruntime.dll "${CMAKE_CURRENT_BINARY_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rd_party/directml/lib/DirectML.dll "${CMAKE_CURRENT_BINARY_DIR}/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../../3rd_party/OnnxRuntime.extensions/lib/ortextensions.dll "${CMAKE_CURRENT_BINARY_DIR}/"
        VERBATIM
)
