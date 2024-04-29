#pragma once

#include <vector>
#include <random>

template<typename T>
class SimpleMatrix
{
public:
	std::vector<std::vector<T>> matrix;
	long rows;
	long cols;
	SimpleMatrix(long rows, long cols) : rows(rows), cols(cols) { matrix.resize(rows, std::vector<double>(cols)); }
};

class SimpleNetwork
{
	long input_size;
	std::vector<SimpleMatrix<double>> layers;
	std::vector<double> input_vec;
	std::vector<double> output_vec;
	virtual std::vector<double> forward(const std::vector<double> & input, const SimpleMatrix<double> & layer);
	virtual std::vector<double> ReLU(const std::vector<double> & input);
public:
	void add_layer(long size);
	void set_input(const std::vector<double> & input);
	virtual void randomize_weights();
	std::vector<double> get_output();
	SimpleNetwork(long input_size) : input_size(input_size) {}
	virtual ~SimpleNetwork() {}
};