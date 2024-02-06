#pragma once

#include "StableDiffusionConfig.h"
#include "Tensor.h"

class UNet {
public:
    static std::vector<float> Inference(const std::string& prompt, const StableDiffusionConfig& config);
    static MachineLearning::Tensor GenerateLatentSample(const StableDiffusionConfig& config, int seed, float initNoiseSigma);
    static MachineLearning::Tensor GenerateLatentSample(size_t height, size_t width, int seed, float initNoiseSigma);
    static MachineLearning::Tensor performGuidance(const MachineLearning::Tensor& noisePred, const MachineLearning::Tensor& noisePredText, double guidanceScale);
};

