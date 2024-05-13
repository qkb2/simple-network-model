#include "SimpleNetwork.h"

template<typename T>
std::vector<T> SimpleNetwork<T>::forward(const std::vector<T> & input, const SimpleMatrix<T> & layer)
{
	std::vector<T> output(layer.rows);

	// calculate product
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
void SimpleNetwork<T>::randomize_weights()
{
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<> dist(-1.0, 1.0);

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

template<typename T>
std::vector<T> SimpleNetwork<T>::ReLU(const std::vector<T> & input)
{
	std::vector<T> output(input.size());

	// calculate Leaky ReLU for every value in vector
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
void SimpleNetwork<T>::add_layer(long size)
{
	size_t len = layers.size();
	if (len > 0) {
		auto& prev_layer = layers[len - 1];
		long prev_rows = prev_layer.rows;
		layers.push_back(SimpleMatrix<T>(size, prev_rows));
	}
	// if no layers exist, calculate matrix size based on input size
	else
		layers.push_back(SimpleMatrix<T>(size, input_size));
}

template<typename T>
void SimpleNetwork<T>::set_input(const std::vector<T> & input)
{
	input_vec = input;
	input_vec.resize(input_size, 0);
}

template<typename T>
std::vector<T> SimpleNetwork<T>::get_output()
{
	std::vector<T> local_out = input_vec;
	for (const auto & layer : layers)
	{
		local_out = forward(local_out, layer);
		local_out = ReLU(local_out);
	}
	output_vec = local_out;

	return output_vec;
}

template SimpleNetwork<double>;
template SimpleNetwork<float>;
template SimpleNetwork<long>;