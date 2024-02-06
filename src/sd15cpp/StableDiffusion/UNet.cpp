#include "UNet.h"
#include "TextProcessing.h"
#include "LMSDiscreteScheduler.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace MachineLearning;

Tensor UNet::GenerateLatentSample(size_t height, size_t width, int seed, float initNoiseSigma)
{
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> dist; // default [0, 1) range:

    size_t batchSize = 1;
    size_t channels = 4;
    Tensor latents{ TensorType::Single, {batchSize, channels, height / 8, width / 8} };
    auto latentsArray = latents.AsPointer<float>();

    for (int i = 0; i < latents.Size(); ++i)
    {
        // Generate a random number from a normal distribution with mean 0 and variance 1
        double u1 = dist(mt); // Uniform(0,1) random number
        double u2 = dist(mt); // Uniform(0,1) random number
        double radius = std::sqrt(-2.0 * std::log(u1)); // Radius of polar coordinates
        double theta = 2.0 * M_PI * u2; // Angle of polar coordinates
        double standardNormalRand = radius * std::cos(theta); // Standard normal random number

        // add noise to latents with * scheduler.init_noise_sigma
        // generate randoms that are negative and positive
        latentsArray[i] = (float)standardNormalRand * initNoiseSigma;
    }

    return latents;
}

Tensor UNet::GenerateLatentSample(const StableDiffusionConfig& config, int seed, float initNoiseSigma)
{
    return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
}

Tensor UNet::performGuidance(const Tensor &noisePred, const Tensor &noisePredText, double guidanceScale)
{
    Tensor result{TensorType::Single, noisePred.Shape};
    for (int i = 0; i < noisePred.Shape[0]; i++)
    {
        for (int j = 0; j < noisePred.Shape[1]; j++)
        {
            for (int k = 0; k < noisePred.Shape[2]; k++)
            {
                auto noisePredIJKValueArray = noisePred.AsPointer<float>(i, j, k);
                auto noisePredTextIJKValueArray = noisePredText.AsPointer<float>(i, j, k);
                auto resultValueArray = result.AsPointer<float>(i, j, k);
                for (int l = 0; l < noisePred.Shape[3]; l++)
                {
//                    result[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
                    resultValueArray[l] = noisePredIJKValueArray[l] + (float)guidanceScale * (noisePredTextIJKValueArray[l] - noisePredIJKValueArray[l]);
                }
            }
        }
    }
    return result;
}

void UNet::Inference(const std::string &prompt, const StableDiffusionConfig &config)
{
    // Preprocess text
    auto textEmbeddings = TextProcessing::PreprocessText(prompt, config);

    LMSDiscreteScheduler scheduler{};
    auto timesteps = scheduler.SetTimesteps(config.NumInferenceSteps);

    std::srand(std::time(nullptr));

//    int seed = 329922609;
    int seed = std::rand();
    std::cout << "Seed generated: " << seed << std::endl;

    // create latent tensor
    auto latents = GenerateLatentSample(config, seed, scheduler.InitNoiseSigma);

    auto sessionOptions = config.GetSessionOptionsForEp();

    Ort::Session unetSession{config.env, config.UnetOnnxPath.c_str(), sessionOptions};

    for (int t = 0; t < timesteps.size(); t++)
    {
        // torch.cat([latents] * 2)
        auto latentModelInput = latents.Duplicate(2);

        latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);
        auto latentModelInputValuePointer = latentModelInput.AsPointer<float>();
        std::vector<float> latentModelInputValueVector(latentModelInputValuePointer, latentModelInputValuePointer + latentModelInput.Size());

        std::cout << "scaled model input " << latentModelInputValueVector[0]
            << " at step " << t
            << ". Max " << *std::max_element(latentModelInputValueVector.begin(), latentModelInputValueVector.end())
            << ". Min " << *std::min_element(latentModelInputValueVector.begin(), latentModelInputValueVector.end())
            << std::endl;

        Tensor timeStepTensor{TensorType::Int64, 1};
        timeStepTensor.AsPointer<uint64_t >()[0] = timesteps[t];

        Ort::IoBinding bindings{ unetSession };
        bindings.BindInput("encoder_hidden_states", textEmbeddings.ToOrtValue());
        bindings.BindInput("sample", latentModelInput.ToOrtValue());
        bindings.BindInput("timestep", timeStepTensor.ToOrtValue());
        bindings.BindOutput("out_sample", config.memoryInfo);

        unetSession.Run({}, bindings);

        auto outputValues = bindings.GetOutputValues();
        auto outputTensor = Tensor::FromOrtValue(outputValues[0]);

        // Split tensors from 2,4,64,64 to 1,4,64,64
        auto splitTensors = outputTensor.Split(2);
        auto noisePred = splitTensors[0];
        auto noisePredText = splitTensors[1];

        // Perform guidance
        noisePred = performGuidance(noisePred, noisePredText, config.GuidanceScale);

        // LMS Scheduler Step
        latents = scheduler.Step(noisePred, timesteps[t], latents);
        std::cout << "latents result after step {t} min {latents.Min()} max {latents.Max()}" << std::endl;
    }

    // Scale and decode the image latents with vae.
    // latents = 1 / 0.18215 * latents
    latents = latents * (1.0f / 0.18215f);

    // Decode image
}
