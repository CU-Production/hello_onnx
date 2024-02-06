#pragma once
#include "Tensor.h"
#include "StableDiffusionConfig.h"

class VaeDecoder {
public:
    static MachineLearning::Tensor Decoder(const MachineLearning::Tensor& input, const StableDiffusionConfig& config);
};
