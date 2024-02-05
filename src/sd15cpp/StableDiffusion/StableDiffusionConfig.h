#pragma once

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

class StableDiffusionConfig
{
public:
    enum class ExecutionProvider : uint32_t
    {
        DirectML = 0,
        Cpu = 1,
//        Cuda = 2,
    };

    // default props
    int NumInferenceSteps = 15;
    ExecutionProvider ExecutionProviderTarget = ExecutionProvider::DirectML;
    double GuidanceScale = 7.5;
    int Height = 512;
    int Width = 512;
    int DeviceId = 0;

    std::wstring OrtExtensionsPath = L"ortextensions.dll";
    std::wstring TokenizerOnnxPath = L"cliptokenizer.onnx";
    std::wstring TextEncoderOnnxPath = L"";
    std::wstring UnetOnnxPath        = L"";
    std::wstring VaeDecoderOnnxPath  = L"";
    std::wstring SafetyModelPath     = L"";

    // default directory for images
    std::string ImageOutputPath = "";

    Ort::SessionOptions GetSessionOptionsForEp() const
    {
        Ort::SessionOptions sessionOptions{};

        switch (this->ExecutionProviderTarget)
        {
//            case ExecutionProvider::Cuda:
//                sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
//                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, this->DeviceId));
//                return sessionOptions;
            case ExecutionProvider::Cpu:
                sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                return sessionOptions;
            case ExecutionProvider::DirectML:
            default:
                sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                sessionOptions.DisableMemPattern();
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, this->DeviceId));
                return sessionOptions;
        }
    }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING};
//    Ort::Env env{ORT_LOGGING_LEVEL_ERROR};
    Ort::MemoryInfo memoryInfo{nullptr};
};
