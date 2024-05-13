#include <iostream>
#include <string>

#include "SimpleNetwork.h"
#include "SimpleNetworkMP.h"
#include "SimpleNetworkCUDA.cuh"

int main(int argc, char* argv[]) 
{
	if (argc < 3) return -1;
	std::string opt = argv[1];
	long input_size = atol(argv[2]);
	SimpleNetwork<double>* network;
	
	if (opt.compare("MP") == 0) {
		network = new SimpleNetworkMP<double>(input_size);
	}
	else if (opt.compare("CUDA") == 0) {
		network = new SimpleNetworkCUDA(input_size);
	}
	else
		network = new SimpleNetwork<double>(input_size);

	for (int i = 3; i < argc; i++)
	{
		long layer_size = atol(argv[i]);
		network->add_layer(layer_size);
	}

	network->randomize_weights();

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<> dist(-10.0, 10.0);
	std::vector<double> input;
	input.resize(input_size, 0);
	for (long i = 0; i < input_size; i++)
	{
		input[i] = (double) dist(rng);
	}
	network->set_input(input);
	
	auto output = network->get_output();
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << " ";
	}

	return 0;
}