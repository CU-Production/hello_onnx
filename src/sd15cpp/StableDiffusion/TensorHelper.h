#pragma once

#include <onnxruntime_cxx_api.h>
#include <random>

constexpr float pi() { return std::atanf(1) * 4; }

class TensorHelper
{
public:
    template<typename T>
    static Ort::Value<T> CreateTensor(std::vector<T> data, std::vector<int> dimensions)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        return Ort::Value::CreateTensor<T>(memory_info, data.data(), data.size(), dimensions.data(), dimensions.size());
    }

    static Ort::Value<float> DivideTensorByFloat(std::vector<float> data, float value, std::vector<int> dimensions)
    {
        for (int i = 0; i < data.size(); ++i)
        {
            data[i] = data[i] / value;
        }
        return CreateTensor(data, dimensions);
    }

    static Ort::Value<float> MultipleTensorByFloat(std::vector<float> data, float value, std::vector<int> dimensions)
    {
        for (int i = 0; i < data.size(); i++)
        {
            data[i] = data[i] * value;
        }

        return CreateTensor(data, dimensions);
    }
    
    static Ort::Value<float> MultipleTensorByFloat(Ort::Value<float> data, float value)
    {
        return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
    }
    
    static Ort::Value<float> AddTensors(std::vector<float> sample, std::vector<float> sumTensor, std::vector<int> dimensions)
    {
        for(auto i=0; i < sample.size(); i++)
        {
            sample[i] = sample[i] + sumTensor[i];
        }
        return CreateTensor(sample, dimensions);
    }

    static Ort::Value<float> AddTensors(Ort::Value<float> sample, Ort::Value<float> sumTensor)
    {
        return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
    }
    
    static std::tuple<Ort::Value, Ort::Value> SplitTensor(Ort::Value<float> tensorToSplit, std::vector<int> dimensions)
    {
        auto tensor1 = new Ort::Value<float>(dimensions);
        auto tensor2 = new Ort::Value<float>(dimensions);

        for (int i = 0; i < 1; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 512 / 8; k++)
                {
                    for (int l = 0; l < 512 / 8; l++)
                    {
                        tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                        tensor2[i, j, k, l] = tensorToSplit[i, j + 4, k, l];
                    }
                }
            }
        }
        return std::tie(tensor1, tensor2);
    }
    
    static Ort::Value<float> SumTensors(std::vector<Ort::Value> tensorArray, std::vector<int> dimensions)
    {
        Ort::Value sumTensor = new Ort::Value<float>(dimensions);
        auto info = sumTensor.GetTensorTypeAndShapeInfo();
        auto sumArray = new std::vector<float>(sumTensor.GetTensorTypeAndShapeInfo().GetElementCount());

        for (int m = 0; m < tensorArray.size(); m++)
        {
            auto& tensorToSum = tensorArray[m];
            for (auto i = 0; i < tensorToSum.GetTensorTypeAndShapeInfo().GetElementCount(); i++)
            {
                sumArray[i] += tensorToSum[i];
            }
        }

        return CreateTensor(sumArray, dimensions);
    }

    static Ort::Value<float> Duplicate(std::vector<float> data, std::vector<int> dimensions)
    {
        data.insert(data.end(), data.begin(), data.end());
        return CreateTensor(data, dimensions);
    }

    static Ort::Value<float> SubtractTensors(std::vector<float> sample, std::vector<float> subTensor, std::vector<int> dimensions)
    {
        for (auto i = 0; i < sample.size(); i++)
        {
            sample[i] = sample[i] - subTensor[i];
        }
        return CreateTensor(sample, dimensions);
    }

    static Ort::Value<float> SubtractTensors(Ort::Value<float> sample, Ort::Value<float> subTensor)
    {
        return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
    }

    static Ort::Value<float> GetRandomTensor(std::vector<int> dimensions)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);

        Ort::Value latents = new Ort::Value<float>(dimensions);

        for (int i = 0; i < latents.GetTensorTypeAndShapeInfo().GetElementCount(); i++)
        {
            // Generate a random number from a normal distribution with mean 0 and variance 1
            auto u1 = dist(gen); // Uniform(0,1) random number
            auto u2 = dist(gen); // Uniform(0,1) random number
            auto radius = std::sqrtf(-2.0 * std::logf(u1)); // Radius of polar coordinates
            auto theta = 2.0 * pi() * u2; // Angle of polar coordinates
            auto standardNormalRand = radius * std::cosf(theta); // Standard normal random number
            latents[i] = (float)standardNormalRand;
        }

        return latents;
    }
};
