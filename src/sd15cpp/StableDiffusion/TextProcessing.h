#pragma once

#include "StableDiffusionConfig.h"
#include "Tensor.h"

using namespace MachineLearning;

class TextProcessing {
public:
    static Tensor PreprocessText(const std::string& prompt, const StableDiffusionConfig& config);
    static Tensor CreateUncondInput();
    static Tensor TokenizeText(const std::string& text, const StableDiffusionConfig& config);
    static Tensor TextEncoder(const Tensor& tokenizedInput, const StableDiffusionConfig& config);
};

