#include "NeuralNetwork.hpp"

int main(int argc, const char *argv)
{
    // srand(time(NULL));
    std::vector<double> x;
    std::vector<double> t;
    x.push_back(1);
    x.push_back(-2);
    t.push_back(0.1);
    t.push_back(0.2);
    NeuralNetwork nn(x, t);
    nn.train();
    return 0;
}