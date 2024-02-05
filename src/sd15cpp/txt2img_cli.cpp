#include "StableDiffusion/UNet.h"

#include <iostream>
#include <chrono>

int main()
{
    auto timeStart = std::chrono::high_resolution_clock::now();

    //Default args
    // std::string prompt = "a fireplace in an old cabin in the woods";
    // std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string prompt = "a photo of a gril in a church";
    std::cout << prompt << std::endl;

    StableDiffusionConfig config{};
    config.NumInferenceSteps = 15;
    config.GuidanceScale = 7.5;
    config.ExecutionProviderTarget = StableDiffusionConfig::ExecutionProvider::DirectML;
    config.DeviceId = 0; // on notebook, may have more than one gpu, and discrete gpu may be 1, 0 is integrated GPU

    config.TextEncoderOnnxPath = L"G:/wrokspace2/ML/stable-diffusion-v1-5/text_encoder/model.onnx";
    config.UnetOnnxPath        = L"G:/wrokspace2/ML/stable-diffusion-v1-5/unet/model.onnx";
    config.VaeDecoderOnnxPath  = L"G:/wrokspace2/ML/stable-diffusion-v1-5/vae_decoder/model.onnx";
    config.SafetyModelPath     = L"G:/wrokspace2/ML/stable-diffusion-v1-5/safety_checker/model.onnx";

    config.memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto timeEnd = std::chrono::high_resolution_clock::now();

    UNet::Inference(prompt, config);

    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
    std::cout << "Time taken: " << milliseconds << "ms" << std::endl;

    return 0;
}
