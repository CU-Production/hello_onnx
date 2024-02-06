#include "VaeDecoder.h"

using namespace MachineLearning;

Tensor VaeDecoder::Decoder(const Tensor &input, const StableDiffusionConfig &config)
{
    // Ort::SessionOptions sessionOptions = config.GetSessionOptionsForEp(); // DirectML EP
    Ort::SessionOptions sessionOptions{}; // DirectML EP

    Ort::Session vaeDecodeSession{config.env, config.VaeDecoderOnnxPath.c_str(), sessionOptions};

    Ort::IoBinding bindings{ vaeDecodeSession };
    bindings.BindInput("latent_sample", input.ToOrtValue());
    bindings.BindOutput("sample", config.memoryInfo);

    vaeDecodeSession.Run({}, bindings);

    auto outputValues = bindings.GetOutputValues();
    auto result = Tensor::FromOrtValue(outputValues[0]);

    return result;
}
