#include "SimpleNetwork.h"

std::vector<double> SimpleNetwork::forward(const std::vector<double> & input, const SimpleMatrix<double> & layer)
{
	std::vector<double> output;
	output.resize(layer.rows, 0);

	// calculate product
	for (size_t i = 0; i < layer.rows; i++)
	{
		double res = 0;
		for (size_t j = 0; j < layer.cols; j++)
		{
			res += input[j] * layer.matrix[i][j];
		}
		output[i] = res;
	}

	return output;
}

void SimpleNetwork::randomize_weights()
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
				layer.matrix[row_it][col_it] = dist(rng);
			}
		}
	}
}

std::vector<double> SimpleNetwork::ReLU(const std::vector<double> & input)
{
	std::vector<double> output;
	output.resize(input.size(), 0);

	// calculate Leaky ReLU for every value in vector
	for (size_t i = 0; i < input.size(); i++)
	{
		double x = input[i];
		if (x > 0) {
			output[i] = x;
		}
		else
			output[i] = 0.01 * x;
	}
	return output;
}

void SimpleNetwork::add_layer(long size)
{
	size_t len = layers.size();
	if (len > 0) {
		auto& prev_layer = layers[len - 1];
		long prev_rows = prev_layer.rows;
		layers.push_back(SimpleMatrix<double>(size, prev_rows));
	}
	// if no layers exist, calculate matrix size based on input size
	else
		layers.push_back(SimpleMatrix<double>(size, input_size));
}

void SimpleNetwork::set_input(const std::vector<double> & input)
{
	input_vec = input;
	input_vec.resize(input_size, 0);
}

std::vector<double> SimpleNetwork::get_output()
{
	std::vector<double> local_out = input_vec;
	for (const auto & layer : layers)
	{
		local_out = forward(local_out, layer);
		local_out = ReLU(local_out);
	}
	output_vec = local_out;

	return output_vec;
}
