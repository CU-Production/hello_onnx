#include "Tensor.h"
#include <vector>
#include <cmath>

// All MachineLearning::Tensor is Tensor<float>
class SchedulerBase
{
public:
    SchedulerBase(int numTrainTimesteps = 1000)
        : _numTrainTimesteps(numTrainTimesteps)
    {}

    static std::vector<float> Interpolate(const std::vector<float>& timesteps, const std::vector<float>& range, const std::vector<float>& sigmas);
    MachineLearning::Tensor ScaleInput(const MachineLearning::Tensor& sample, int timestep);
    virtual std::vector<int> SetTimesteps(int num_inference_steps) = 0;
    virtual MachineLearning::Tensor Step(
            const MachineLearning::Tensor& modelOutput,
            int timestep,
            const MachineLearning::Tensor& sample,
            int order = 4
            ) = 0;

    std::vector<int> Timesteps;
    MachineLearning::Tensor Sigmas;
    float InitNoiseSigma;

protected:
    int _numTrainTimesteps;
    std::vector<float> _alphasCumulativeProducts;
    bool is_scale_input_called;
};
