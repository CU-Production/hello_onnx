#include "LMSDiscreteScheduler.h"
#include "MathUtils.h"
#include <functional>
#include <iostream>

LMSDiscreteScheduler::LMSDiscreteScheduler(int num_train_timesteps, float beta_start, float beta_end, std::string beta_schedule, std::string prediction_type)
{
    _numTrainTimesteps = num_train_timesteps;
    _predictionType = prediction_type;

    std::vector<float> alphas;
    std::vector<float> betas;

    if (beta_schedule == "linear")
    {
        betas = MathUtils::linspace(beta_start, beta_end, num_train_timesteps);
    }
    else if (beta_schedule == "scaled_linear")
    {
        float start = std::sqrtf(beta_start);
        float end = std::sqrtf(beta_end);
        betas = MathUtils::linspace(start, end, num_train_timesteps);
        for (int i = 0; i < betas.size(); i++)
        {
            betas[i] = betas[i] * betas[i];
        }
    }
    else
    {
        throw std::runtime_error("beta_schedule must be one of 'linear' or 'scaled_linear'");
    }

    alphas.assign(betas.size(), 0);
    this->_alphasCumulativeProducts.assign(betas.size(), 0);
    float cumProd = 1.0f;
    for (int i = 0; i < alphas.size(); i++)
    {
        alphas[i] = 1.0f - betas[i];
        cumProd *= alphas[i];
        this->_alphasCumulativeProducts[i] = cumProd;
    }

    // Create sigmas as a list and reverse it
    float curMax = -10000000.0f;
    std::vector<float> sigmas(this->_alphasCumulativeProducts.size(), 0);
    for (int i = 0; i < this->_alphasCumulativeProducts.size(); i++) {
        float curVal = this->_alphasCumulativeProducts[(this->_alphasCumulativeProducts.size() - 1) - i];
        float newVal = std::sqrtf((1.0f - curVal) / curVal);
        sigmas[i] = newVal;
        if (newVal > curMax) {
            curMax = newVal;
        }
    }

    this->InitNoiseSigma = curMax;
}

double LMSDiscreteScheduler::GetLmsCoefficient(int order, int t, int currentOrder)
{
    // Compute a linear multistep coefficient.
    auto LmsDerivative = [&](double tau) -> double {
        double prod = 1.0;
        for (int k = 0; k < order; k++)
        {
            if (currentOrder == k) continue;

            auto thisSigmasValue = this->Sigmas.AsPointer<float>();
            prod *= (tau - thisSigmasValue[t - k]) / (thisSigmasValue[t - currentOrder] - thisSigmasValue[t - k]);
        }
        return prod;
    };

    auto IntegrateOverInterval = [](const std::function<double(double)>& f, double intervalStart, double intervalEnd)->double {
        int stepCount = 100;
        double stepSize = (intervalEnd - intervalStart) / stepCount;
        // var stepSize = 1e-4;

        double result = 0.0;
        if (stepSize > 0)
        {
            for (double value = intervalStart; value < intervalEnd; value += stepSize)
            {
                result += f(value) * stepSize;
            }
        }
        else // intervalStart > intervalEnd
        {
            for (double value = intervalStart; value > intervalEnd; value += stepSize)
            {
                result += f(value) * stepSize;
            }
        }

        return result;
    };

    auto thisSigmasValue = this->Sigmas.AsPointer<float>();
    double integratedCoeff = IntegrateOverInterval(LmsDerivative, thisSigmasValue[t], thisSigmasValue[t + 1]);

    return integratedCoeff;
}

std::vector<int> LMSDiscreteScheduler::SetTimesteps(int num_inference_steps)
{
    float start = 0;
    float stop = (float)_numTrainTimesteps - 1;
    std::vector<float> timesteps = MathUtils::linspace(start, stop, num_inference_steps, true);

    this->Timesteps.assign(timesteps.size(), 0);
    for (int i = 0; i < timesteps.size(); i++) {
        this->Timesteps[i] = (int) timesteps[(timesteps.size() - 1) - i];
    }

    std::vector<float> sigmas(this->_alphasCumulativeProducts.size(), 0);
    for (int i = 0; i < this->_alphasCumulativeProducts.size(); i++) {
        float curVal = this->_alphasCumulativeProducts[(this->_alphasCumulativeProducts.size() - 1) - i];
        sigmas[i] = std::sqrtf((1.0f - curVal) / curVal);
    }

    auto range = MathUtils::arrange(0, (float)(sigmas.size()), 1.0f);
    std::vector<float> Sigmas = Interpolate(timesteps, range, sigmas);

    MachineLearning::Tensor tmpTensor{MachineLearning::TensorType::Single, Sigmas.size()};
    auto tmpTensorValue = tmpTensor.AsPointer<float>();
    for (int i = 0; i < Sigmas.size(); i++)
    {
        tmpTensorValue[i] = Sigmas[i];
    }
    this->Sigmas = tmpTensor;

    return this->Timesteps;
}

MachineLearning::Tensor LMSDiscreteScheduler::Step(const MachineLearning::Tensor& modelOutput, int timestep, const MachineLearning::Tensor& sample, int order)
{
    using namespace MachineLearning;

    int stepIndex = MathUtils::findIdx(this->Timesteps,timestep);
    auto thisSigmasValue = this->Sigmas.AsPointer<float>();
    float sigma = thisSigmasValue[stepIndex];

    // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    Tensor predOriginalSample{TensorType::Single, modelOutput.Shape};

    // Create array of type float length modelOutput.length
    std::vector<float> predOriginalSampleArray(modelOutput.Size());
    auto modelOutPutArray = modelOutput.AsPointer<float>();
    auto sampleArray = sample.AsPointer<float>();

    if (this->_predictionType == "epsilon")
    {
        for (int i = 0; i < modelOutput.Size(); i++)
        {
            predOriginalSampleArray[i] = sampleArray[i] - sigma * modelOutPutArray[i];
        }
        memcpy(predOriginalSample.Buffer.data(), predOriginalSampleArray.data(), predOriginalSampleArray.size() * sizeof(float));
    }
    else if (this->_predictionType == "v_prediction")
    {
        //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
        throw std::runtime_error("prediction_type given as " + this->_predictionType + " not implemented yet.");
    }
    else
    {
        throw std::runtime_error("prediction_type given as " + this->_predictionType + " must be one of `epsilon`, or `v_prediction`");
    }

    // 2. Convert to an ODE derivative
    Tensor derivativeItems{TensorType::Single, sample.Shape};
    auto derivativeItemsArray = derivativeItems.AsPointer<float>();
    for (int i = 0; i < derivativeItems.Size(); i++)
    {
        derivativeItemsArray[i] = (sampleArray[i] - predOriginalSampleArray[i]) / sigma;
    }

    this->Derivatives.push_back(derivativeItems);

    if (this->Derivatives.size() > 4)
    {
        this->Derivatives.erase(this->Derivatives.begin());
    }

    // 3. compute linear multistep coefficients
    order = std::min(stepIndex + 1, order);
    std::vector<float> lmsCoeffs(order, 0);
    for (int currOrder = 0; currOrder < order; currOrder++)
    {
        lmsCoeffs[currOrder] = (float)GetLmsCoefficient(order, stepIndex, currOrder);
    }

    // 4. compute previous sample based on the derivative path
    // Reverse list of tensors this.derivatives
    std::vector<Tensor> revDerivatives = this->Derivatives;
    std::reverse(revDerivatives.begin(), revDerivatives.end());

    // Create tensor for product of lmscoeffs and derivatives
    std::vector<Tensor> lmsDerProduct(revDerivatives.size(), {TensorType::Single, revDerivatives[0].Shape});

    for(int m = 0; m < revDerivatives.size(); m++) {
        auto curDeriv = revDerivatives[m];
        auto curCoeff = lmsCoeffs[m];
        // Multiply to coeff by each derivative to create the new tensors
        lmsDerProduct[m] = curDeriv * curCoeff;
    }

    // Sum the tensors
    Tensor sumTensor{TensorType::Single, {1,4,64,64}};
    for (int m = 0; m < lmsDerProduct.size(); m++)
    {
        // sumTensor += lmsDerProduct[m];
        sumTensor.UnaryOperation<float>(lmsDerProduct[m], [](float a, float b){ return a + b; });
    }

    // Add the summed tensor to the sample
    auto prevSample = sample;
    // prevSample += sumTensor;
    prevSample.UnaryOperation<float>(sumTensor, [](float a, float b){ return a + b; });

    std::cout << "prevSample[0] : " << prevSample.AsPointer<float>()[0] << std::endl;
    return prevSample;
}
