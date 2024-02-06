#pragma once

#include "StableDiffusionConfig.h"
#include "Tensor.h"

using namespace MachineLearning;

class TextProcessing {
public:
    static Tensor PreprocessText(const std::string& prompt, const StableDiffusionConfig& config);
    static std::vector<int32_t> CreateUncondInput();
    static std::vector<int32_t> TokenizeText(const std::string& text, const StableDiffusionConfig& config);
    static Tensor TextEncoder(const std::vector<int32_t>& tokenizedInput, const StableDiffusionConfig& config);
};

