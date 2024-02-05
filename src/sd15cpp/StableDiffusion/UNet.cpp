#include "UNet.h"
#include "TextProcessing.h"

void UNet::Inference(const std::string &prompt, const StableDiffusionConfig &config)
{
    // Preprocess text
    auto textEmbeddings = TextProcessing::PreprocessText(prompt, config);


}
