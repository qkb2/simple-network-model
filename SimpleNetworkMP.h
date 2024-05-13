#pragma once

#include "SimpleNetwork.h"
#include <omp.h>

template<typename T>
class SimpleNetworkMP :
    public SimpleNetwork<T>
{
protected:
    virtual std::vector<T> forward(const std::vector<T>& input, const SimpleMatrix<T>& layer);
    virtual std::vector<T> ReLU(const std::vector<T>& input);
public:
    virtual void randomize_weights();
    SimpleNetworkMP<T>(long input_size) : SimpleNetwork<T>(input_size) {}
    ~SimpleNetworkMP<T>() {}
};

