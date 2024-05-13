#pragma once

#include "SimpleNetwork.h"
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class SimpleNetworkCUDA :
    public SimpleNetwork<double>
{
protected:
    virtual std::vector<double> forward(const std::vector<double>& input, const SimpleMatrix<double>& layer);
    virtual std::vector<double> ReLU(const std::vector<double>& input);
public:
    virtual void randomize_weights();
    SimpleNetworkCUDA(long input_size) : SimpleNetwork(input_size) {}
    ~SimpleNetworkCUDA() {}
};