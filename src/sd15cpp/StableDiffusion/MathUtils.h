#include <vector>
#include <cmath>
#include <stdexcept>

namespace MathUtils
{
    static std::vector<float> linspace(float start, float end, int numSteps, bool includeEnd = true)
    {
        if (end <= start) {
            throw std::runtime_error("Invalid range, end must be strictly greater than start");
        }
        if (numSteps <= 0) {
            throw std::runtime_error("Invalid number of steps, " + numSteps);
        }
        float stepSize = (end - start) / (float)(includeEnd ? numSteps - 1 : numSteps);
        std::vector<float> output(numSteps);
        for (int i = 0; i < numSteps; i++) {
            output[i] = start + ((float)i * stepSize);
        }
        return output;
    }

    static std::vector<float> arrange(float start, float end, float stepSize)
    {
        if (end <= start) {
            throw std::runtime_error("Invalid range, end must be strictly greater than start");
        }
        if (stepSize <= 0.00001f) {
            throw std::runtime_error("Invalid stepSize, must be positive.");
        }
        int numSteps = std::round((float)std::ceilf((end - start)/stepSize));
        std::vector<float> output(numSteps, 0);
        for (int i = 0; i < numSteps; i++) {
            output[i] = start + ((float)i * stepSize);
        }
        return output;
    }

    static int findIdx(std::vector<int> array, int target) {
        int idx = -1;
        for (int i = 0; i < array.size(); i++) {
            if (array[i] == target) {
                idx = i;
            }
        }
        return idx;
    }
}
