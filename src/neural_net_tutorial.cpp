File Name "main.cpp"
#include "neural_network.hpp"
int main(int argc, const char * argv[])
{
    std::vector <double> x
    std::vector <double> t
    x.push_back(1)
    x.push_back(-2)
    t.push_back(0.2)
    t.push_back(0.1)
    t.push_back(0.3)
    NeuronNetwork net(x, t)
    net.train()
    return 0
}
File Name "neuron_model.hpp"
#ifndef neural_network_hpp
#define neural_network_hpp
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
class NeuronNetwork


{
    private:
        const double rateOfTraining = 0.3
        // норма обучения
        std::vector <double> hiddenWeight
        // w10, w11
        std::vector <double> outWeight
        // w21, w210, w22, w220, w23, w230
        // w210, w220, w230
        std::vector <double> hiddenOut
        // out1
        std:: vector <double> outOut
        // out2, out3, out4
        std::vector <double> hiddenNet
        // net1
        std:: vector <double> outNet
        // net2, net3, net4
        std::vector <double> hiddenDelta
        // delta1
        std::vector <double> outDelta
        // delta2, delta3, delta4
        std:: vector <double> outX
        // вспомогат для выч-я net1, net2, net3std:: vector <double> hiddenX
        // (1, -2)
        std::vector <double> target
        // (0, 2
            0, 1
            0, 3)
        double fa(double net)
        double derivative(double net)
        std::vector <double> net(std::vector <double> x, std::vector <double> weight)
    public:
        NeuronNetwork(std:: vector <double> x, std::vector <double> t)
        void train()
        ~NeuronNetwork() {}
}
#endif /* neural_network_hpp */
File Name "neuron_model.cpp"
#include "neural_network.hpp"
NeuronNetwork:: NeuronNetwork(std::vector <double> x, std::vector <double> t)
{
    for (int i = 0
         i < x.size()
         i++)
    {
        hiddenX.push_back(x[i])
        // 2
        hiddenWeight.push_back(0.5)
        // 2
    }
    for (int i = 0
         i < t.size()
         i++)
    {
        target.push_back(t[i])
        // 3
        outNet.push_back(0)
        // 3
        outDelta.push_back(0)
        // 3
        outOut.push_back(0)
        // 3
    }
    for (int i = 0
         i < (2 * target.size())
         i++)
    {
        outWeight.push_back(0.5)
        // 6
        if (i % 2 == 1)
        outX.push_back(1)
        else
        outX.push_back(0)
        // 010101
    }
    for (int i = 0
         i < (hiddenX.size() - 1)
         i++){
        hiddenNet.push_back(0)
        // 1
        hiddenOut.push_back(0)
        // 1
        hiddenDelta.push_back(0)
        // 1
    }
}
double NeuronNetwork:: fa(double net)
{
    return (1 - exp(-net)) / (1+exp( - net))
}
double NeuronNetwork:: derivative(double f)
{
    return 0.5 * (1 - (f) * (f))
}
// hiddenX, hiddenWeight OR outX, outWeight
std::vector <double> NeuronNetwork::net(std::vector <double> x, std::vector <double>
                                             weight)
{
    std::vector <double> net
    int c = 0
    for (int i = 0
         i < (weight.size())
         i++)
    {
        if (i % 2 == 0)
        {
            c = i
            c++
            net.push_back((weight[i] * x[i]+weight[c] * x[c]))
        }
    }
    return net
}
void NeuronNetwork:: train()
{
    double error = 1
    int epoch = 1
    while (error > 0.001)
    {
        std:: cout << std::endl << "Эпоха #" << epoch << std::endl
        std:: cout << "Веса скрытого слоя:" << std::endl
        for (int i = 0
             i < hiddenWeight.size()
             i++)
        std::cout << hiddenWeight[i] << " "
        std:: cout << std::endl
        std:: cout << "Веса выходного слоя:" << std::endl
        for (int i = 0
             i < outWeight.size()
             i++)
        std:: cout << outWeight[i] << " "
        std:: cout << std::endl
        // 1st stage
        std:: vector <double> tmpNet = net(hiddenX, hiddenWeight)
        // 1
        for (int i = 0
             i < hiddenOut.size()
             i++)
        {
            hiddenNet[i] = tmpNet[i]
            hiddenOut[i] = fa(hiddenNet[i])
        }
        for (int i = 0
             i < (2 * target.size())
             i++)
        {
            if (i % 2 == 0)
            outX[i] = hiddenOut[(hiddenOut.size() - 1)]
        }
        tmpNet.erase(tmpNet.begin(), tmpNet.end())
        tmpNet = net(outX, outWeight)
        // 3
        for (int i = 0
             i < outOut.size()
             i++)
        {
            outNet[i] = tmpNet[i]
            outOut[i] = fa(outNet[i])
        }
        // 2nd stage
        for (int i = 0
             i < outDelta.size()
             i++)
        {
            outDelta[i] = (target[i] - outOut[i]) * derivative(outOut[i])
        }
        for (int i = 0
             i < hiddenDelta.size()
             i++)
        {
            double e = 0
            for (int j = 0
                 j < outWeight.size()
                 j++)
            {
                if (j % 2 == 0)
                e += outWeight[i] * outDelta[j / 2]
            }
            hiddenDelta[i] = (derivative(hiddenOut[i])) * (e)
        }
        // 3d stage
        for (int i = 0
             i < outWeight.size()
             i++)
        {if (i % 2 == 0)
         outWeight[i] += rateOfTraining *
         hiddenOut[(hiddenOut.size() - 1)] * outDelta[i / 2]
         if (i % 2 == 1)
         outWeight[i] += rateOfTraining * 1 * outDelta[i / 2]
         }
        for (int i = 0
             i < hiddenWeight.size()
             i++)
        {
            hiddenWeight[i] += rateOfTraining * hiddenX[i] *
            hiddenDelta[(hiddenDelta.size() - 1)]
        }
        //
        error = 0
        for (int i = 0
             i < target.size()
             i++)
        error += ((target[i] - outOut[i]) * (target[i] - outOut[i]))
        error = sqrt(error)
        epoch++
        std::cout << "Выход НС:" << std::endl
        for (int i = 0
             i < outOut.size()
             i++)
        std::cout << outOut[i] << " "
        std::cout << std::endl << "Значение ошибки: " << error << std::endl
    }
    std::cout << std::endl << "Обучение завершено за " << --epoch << " эпох!" <<
    std::endl
}
