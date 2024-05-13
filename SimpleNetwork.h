#pragma once

#include <vector>
#include <random>
#include <iostream>

template<typename T>
class SimpleMatrix
{
public:
	std::vector<std::vector<T>> matrix;
	long rows;
	long cols;
	SimpleMatrix<T>(long rows, long cols) : rows(rows), cols(cols) { matrix.resize(rows, std::vector<T>(cols)); }
};

template<typename T>
class SimpleNetwork
{
protected:
	long input_size;
	std::vector<SimpleMatrix<T>> layers;
	std::vector<T> input_vec;
	std::vector<T> output_vec;
	virtual std::vector<T> forward(const std::vector<T> & input, const SimpleMatrix<T> & layer);
	virtual std::vector<T> ReLU(const std::vector<T> & input);
public:
	void add_layer(long size);
	void set_input(const std::vector<T> & input);
	virtual void randomize_weights();
	std::vector<T> get_output();
	SimpleNetwork<T>(long input_size) : input_size(input_size) {}
	virtual ~SimpleNetwork<T>() {}
};