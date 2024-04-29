#include <iostream>
#include "SimpleNetwork.h"

int main() 
{
	SimpleNetwork* network = new SimpleNetwork(10);
	network->add_layer(100);
	network->add_layer(200);
	network->add_layer(100);
	network->add_layer(10);
	network->randomize_weights();

	std::vector<double> input = {0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
	network->set_input(input);
	auto output = network->get_output();
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << " ";
	}

	return 0;
}