#include "SimpleNetworkMP.h"

template<typename T>
std::vector<T> SimpleNetworkMP<T>::forward(const std::vector<T>& input, const SimpleMatrix<T>& layer)
{
	std::vector<T> output(layer.rows);

	// calculate product
#pragma omp parallel for
	for (size_t i = 0; i < layer.rows; i++)
	{
		T res = 0;
		for (size_t j = 0; j < layer.cols; j++)
		{
			res += input[j] * layer.matrix[i][j];
		}
		output[i] = res;
	}

	return output;
}

template<typename T>
std::vector<T> SimpleNetworkMP<T>::ReLU(const std::vector<T>& input)
{
	std::vector<T> output(input.size());

	// calculate Leaky ReLU for every value in vector
#pragma omp parallel for
	for (size_t i = 0; i < input.size(); i++)
	{
		T x = input[i];
		if (x > 0) {
			output[i] = x;
		}
		else
			output[i] = (T) 0.01 * x;
	}
	return output;
}

template<typename T>
void SimpleNetworkMP<T>::randomize_weights()
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<> dist(-1.0, 1.0);

#pragma omp parallel for
	for (size_t layer_it = 0; layer_it < layers.size(); layer_it++)
	{
		auto& layer = layers[layer_it];
		for (size_t row_it = 0; row_it < layer.rows; row_it++)
		{
			for (size_t col_it = 0; col_it < layer.cols; col_it++)
			{
				layer.matrix[row_it][col_it] = (T) dist(rng);
			}
		}
	}
}

template SimpleNetworkMP<double>;
template SimpleNetworkMP<float>;
template SimpleNetworkMP<long>;