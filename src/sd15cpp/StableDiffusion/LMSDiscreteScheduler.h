#pragma once

#include "SchedulerBase.h"

class LMSDiscreteScheduler : public SchedulerBase
{
public:
    explicit LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, std::string beta_schedule = "scaled_linear", std::string prediction_type = "epsilon");
    double GetLmsCoefficient(int order, int t, int currentOrder);
    std::vector<int> SetTimesteps(int num_inference_steps) override;
    MachineLearning::Tensor Step(const MachineLearning::Tensor& modelOutput, int timestep, const MachineLearning::Tensor& sample, int order = 4) override;

    std::vector<MachineLearning::Tensor> Derivatives;

private:
    std::string _predictionType;
};
