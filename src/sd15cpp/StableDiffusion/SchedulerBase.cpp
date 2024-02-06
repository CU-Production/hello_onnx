#include "SchedulerBase.h"
#include "MathUtils.h"

std::vector<float> SchedulerBase::Interpolate(const std::vector<float>& timesteps, const std::vector<float>& range, const std::vector<float>& sigmas)
{
    std::vector<float> result((int)timesteps.size() + 1, 0.0);

    for (int i = 0; i < timesteps.size(); ++i)
    {
        auto trainStep = timesteps[i];
        auto previousIndex = std::max(size_t(floor(trainStep)), size_t(0));
        auto nextIndex = std::min(size_t(ceil(trainStep)), sigmas.size() - 1);
        result[i] = std::lerp(sigmas[previousIndex], sigmas[nextIndex], trainStep - floor(trainStep));
    }

    result.push_back(0.000);
    return result;
}

MachineLearning::Tensor SchedulerBase::ScaleInput(const MachineLearning::Tensor& sample, int timestep)
{
    int stepIndex = MathUtils::findIdx(this->Timesteps, timestep);

    auto thisSigmasValue = this->Sigmas.AsPointer<float>();
    auto sigma = thisSigmasValue[stepIndex];
    sigma = (float)std::sqrt((std::pow(sigma, 2) + 1));

    MachineLearning::Tensor result{sample.Type, sample.Shape};

    result = sample / sigma;
    is_scale_input_called = true;
    return result;
}
