#include "SimpleNetworkCUDA.cuh"

__global__ void initialize_kernel(double* d_weights, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state); // Initialize random number generator
        d_weights[idx] = curand_uniform(&state); // Generate random number between 0 and 1
    }
}

std::vector<double> SimpleNetworkCUDA::forward(const std::vector<double>& input, const SimpleMatrix<double>& layer)
{
    std::vector<double> output(layer.rows);

    // calculate product
#pragma omp parallel for
    for (int i = 0; i < layer.rows; i++)
    {
        double res = 0;
        for (int j = 0; j < layer.cols; j++)
        {
            res += input[j] * layer.matrix[i][j];
        }
        output[i] = res;
    }

    return output;
}

std::vector<double> SimpleNetworkCUDA::ReLU(const std::vector<double>& input)
{
    std::vector<double> output(input.size());

    // calculate Leaky ReLU for every value in vector
#pragma omp parallel for
    for (int i = 0; i < input.size(); i++)
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

void SimpleNetworkCUDA::randomize_weights()
{
    std::vector<size_t> sizes(layers.size());
    size_t sum_size = 0;
    for (size_t layer_it = 0; layer_it < layers.size(); layer_it++) {
        auto& layer = layers[layer_it];

        sizes[layer_it] = layer.rows * layer.cols * sizeof(double);
        sum_size += sizes[layer_it];
    }
    double* d_weights;

    cudaMalloc(&d_weights, sum_size);

    double* res = (double*)malloc(sum_size);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (sum_size + threadsPerBlock - 1) / threadsPerBlock;

    initialize_kernel <<< threadsPerBlock, blocksPerGrid >>> (d_weights, sum_size, time(NULL));
    auto err = cudaGetLastError();

    //cudaDeviceSynchronize();

    // Copy data back to host
    cudaMemcpy(res, d_weights, sum_size, cudaMemcpyDeviceToHost);


    cudaFree(d_weights);

    size_t elements = 0;
    for (int ij = 0; ij < layers.size(); ij++) 
    {
        auto& layer = layers[ij];
        for (int i = 0; i < layer.rows; ++i) {
            for (int j = 0; j < layer.cols; ++j) {
                // Calculate the index in the 1D array
                int index = i * layer.cols + j;
                // Assign the value from the 1D array to the matrix
                layer.matrix[i][j] = res[index + elements];
            }
        }
        elements += layer.cols * layer.rows;
    }
    free(res);
}
