#include <chrono>
#include <iostream>

#include "initialization_function.hpp"
#include "activation_function.hpp"
#include "data_reader.hpp"
#include "neural_network.hpp"

class Timer {
    public:
        Timer() {
            start_time = std::chrono::high_resolution_clock::now();
        }

        auto get_elapsed_time() {
            auto current_time = std::chrono::high_resolution_clock::now();

            return std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time;
};

auto time_data_reader() {
    auto timer = Timer();

    auto data_reader = DataReader("train.csv");
    std::vector<training_data_t> training_data;

    for (int i = 0; i < 42000; i++) {
        training_data.push_back(data_reader.get_data_point());
    }

    return timer.get_elapsed_time();
}

auto time_gradient_descent() {
    auto data_reader = DataReader("train.csv");
    std::vector<size_t> layers {10};
    auto neural_net = NeuralNetwork(layers, he_initialization, ReLU, deriv_ReLU, 0.1);

    std::vector<training_data_t> training_data;
    for (int i = 0; i < 10000; i++) {
        training_data.push_back(data_reader.get_data_point());
    }

    auto timer = Timer();

    neural_net.gradient_descent(training_data, 10, false);

    return timer.get_elapsed_time();
}

int main() {
    std::cout << "CSV read time = " << time_data_reader() << "ms\n";
    std::cout << "Gradient descent time for 500 iterations = " << time_gradient_descent() << "ms\n";
    return 0;
}