#include "StableDiffusion/UNet.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include <chrono>
#include <ctime>

int main()
{
    auto timeStart = std::chrono::high_resolution_clock::now();

    //Default args
//     std::string prompt = "a fireplace in an old cabin in the woods";
     std::string prompt = "a photo of an astronaut riding a horse on mars";
//    std::string prompt = "a photo of a gril in a church";
//    std::string prompt = " an anime gril in a church";
//    std::string prompt = "An illustration of a young anime girl, clad in her best attire, sitting pensively in a quiet, serene church.";
//    std::string prompt = "A fantastical image of an astronaut gallantly riding a horse across the rust-colored Martian landscape.";
    std::cout << prompt << std::endl;

    StableDiffusionConfig config{};
    config.NumInferenceSteps = 15;
//    config.NumInferenceSteps = 50;
    config.GuidanceScale = 7.5;
    config.ExecutionProviderTarget = StableDiffusionConfig::ExecutionProvider::DirectML;
    config.DeviceId = 0; // on notebook, may have more than one gpu, and discrete gpu may be 1, 0 is integrated GPU

    config.TextEncoderOnnxPath = L"G:/wrokspace2/ML/stable-diffusion-v1-5/text_encoder/model.onnx";
    config.UnetOnnxPath        = L"G:/wrokspace2/ML/stable-diffusion-v1-5/unet/model.onnx";
    config.VaeDecoderOnnxPath  = L"G:/wrokspace2/ML/stable-diffusion-v1-5/vae_decoder/model.onnx";
    config.SafetyModelPath     = L"G:/wrokspace2/ML/stable-diffusion-v1-5/safety_checker/model.onnx";

//    config.TextEncoderOnnxPath = L"E:/workspace/ML/sd1.5onnx/stable-diffusion-v1-5/text_encoder/model.onnx";
//    config.UnetOnnxPath        = L"E:/workspace/ML/sd1.5onnx/stable-diffusion-v1-5/unet/model.onnx";
//    config.VaeDecoderOnnxPath  = L"E:/workspace/ML/sd1.5onnx/stable-diffusion-v1-5/vae_decoder/model.onnx";
//    config.SafetyModelPath     = L"E:/workspace/ML/sd1.5onnx/stable-diffusion-v1-5/safety_checker/model.onnx";

    config.memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//    config.memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    auto rgbaData = UNet::Inference(prompt, config);

    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);

    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", now);

    std::string png_file_path = std::format("sd_image_{}_Steps{}_Scale{}.png", buf, config.NumInferenceSteps, config.GuidanceScale);
    std::string jpg_file_path = std::format("sd_image_{}_Steps{}_Scale{}.jpg", buf, config.NumInferenceSteps, config.GuidanceScale);

    stbi_write_png(png_file_path.c_str(), 512, 512, 4, rgbaData.data(), 512*4);
    stbi_write_jpg(jpg_file_path.c_str(), 512, 512, 4, rgbaData.data(), 100);

    auto timeEnd = std::chrono::high_resolution_clock::now();

    uint64_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
    std::cout << "Time taken: " << milliseconds << "ms" << std::endl;
    return 0;
}
