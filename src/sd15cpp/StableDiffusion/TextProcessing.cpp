#include "TextProcessing.h"
#include <iostream>

Tensor TextProcessing::PreprocessText(const std::string &prompt, const StableDiffusionConfig &config)
{
    auto textTokenized = TokenizeText(prompt, config);
    Tensor textPromptEmbeddings = TextEncoder(textTokenized, config);

    auto uncondInputTokens = CreateUncondInput();
    Tensor uncondEmbedding = TextEncoder(uncondInputTokens, config);

    Tensor textEmbeddings{ TensorType::Single, 2, 77, 768 };

    auto textPromptEmbeddingsSize = textPromptEmbeddings.Size();
    auto textPromptEmbeddingsValue = textPromptEmbeddings.AsPointer<float>();
    auto uncondEmbeddingValue = uncondEmbedding.AsPointer<float>();

    for (int i = 0; i < textPromptEmbeddingsSize; ++i)
    {
//        textEmbeddings[0, i / 768, i % 768] = uncondEmbeddingValue[i];
//        textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddingsValue[i];
        auto tempValue0 = textEmbeddings.AsPointer<float>(0, i / 768);
        auto tempValue1 = textEmbeddings.AsPointer<float>(1, i / 768);

        tempValue0[i % 768]  = uncondEmbeddingValue[i];
        tempValue1[i % 768]  = textPromptEmbeddingsValue[i];
    }

    return textEmbeddings;
}

std::vector<int32_t> TextProcessing::CreateUncondInput()
{
    constexpr int32_t blankTokenValue = 49407;
    constexpr int32_t modelMaxLength = 77;
    std::vector<int> result(modelMaxLength, blankTokenValue);
    result[0] = 49406;
    return result;
}

std::vector<int32_t> TextProcessing::TokenizeText(const std::string &text, const StableDiffusionConfig &config)
{
    Ort::SessionOptions sessionOptions{};
    // sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.RegisterCustomOpsLibrary(config.OrtExtensionsPath.c_str());

    Ort::Session tokenizeSession{config.env, config.TokenizerOnnxPath.c_str(), sessionOptions};

    Ort::Allocator allocator{tokenizeSession, config.memoryInfo};

    const std::vector<const char*>& texts = std::vector<const char*>{ text.data() };

    std::vector<int64_t> inputShape{ int64_t(1) };
    auto inputValue = Ort::Value::CreateTensor(allocator, inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    inputValue.FillStringTensor(texts.data(), texts.size());

    Ort::IoBinding bindings{ tokenizeSession };
    bindings.BindInput("string_input", inputValue);
    bindings.BindOutput("input_ids", config.memoryInfo);
    bindings.BindOutput("attention_mask", config.memoryInfo);

    tokenizeSession.Run({}, bindings);

    auto outputValues = bindings.GetOutputValues();
    auto input_ids = Tensor::FromOrtValue(outputValues[0]);
    auto attention_mask = Tensor::FromOrtValue(outputValues[1]);

    // print input_ids
    for (size_t i = 0; i < input_ids.Shape[0]; i++)
    {
        auto pSource = input_ids.AsPointer<int64_t>(i);
        auto pMask = attention_mask.AsPointer<int64_t>(i);
        for (size_t j = 0; j < input_ids.Shape[1]; j++)
        {
            if(pMask[j])
            {
                std::cout << (int32_t)pSource[j] << " ";
            }
        }
    }
    std::cout << std::endl;

    // Pad results to a fixed size
    constexpr int32_t blankTokenValue = 49407;
    std::vector<int> result(77, blankTokenValue);
    auto inputIdsValue = input_ids.AsPointer<int64_t>();
    for (int i = 0; i < input_ids.Size(); ++i)
    {
        result[i] = (int32_t)inputIdsValue[i];
    }

    return result;
}

Tensor TextProcessing::TextEncoder(const std::vector<int32_t> &tokenizedInput, const StableDiffusionConfig &config)
{
    Tensor input_ids{TensorType::Int32, 1, tokenizedInput.size()};
    memcpy(input_ids.AsPointer<int32_t>(), tokenizedInput.data(), tokenizedInput.size() * sizeof(int32_t));

//    Ort::SessionOptions sessionOptions{};
    Ort::SessionOptions sessionOptions = config.GetSessionOptionsForEp(); // DirectML EP

    Ort::Session encodeSession{config.env, config.TextEncoderOnnxPath.c_str(), sessionOptions};

    Ort::IoBinding bindings{ encodeSession };
    bindings.BindInput("input_ids", input_ids.ToOrtValue());
    bindings.BindOutput("last_hidden_state", config.memoryInfo);

    encodeSession.Run({}, bindings);

    auto outputValues = bindings.GetOutputValues();
    auto lastHiddenState = Tensor::FromOrtValue(outputValues[0]);
    Tensor lastHiddenStateTensor{TensorType::Single, 1, 77, 768};
    memcpy(lastHiddenStateTensor.Buffer.data(), lastHiddenState.Buffer.data(), lastHiddenState.Buffer.size());

    return lastHiddenStateTensor;
}
