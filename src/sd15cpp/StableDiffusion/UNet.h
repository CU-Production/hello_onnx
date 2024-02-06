#pragma once

#include "StableDiffusionConfig.h"

class UNet {
public:
    static void Inference(const std::string& prompt, const StableDiffusionConfig& config);
};

