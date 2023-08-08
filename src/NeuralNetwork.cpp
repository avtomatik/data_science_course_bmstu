#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(std::vector<double> x, std::vector<double> t)
{
    for (int i = 0; i < x.size(); i++)
    {
        hiddenX.push_back(x[i]);
        hiddenWeight.push_back(0.5);
        // hiddenWeight.push_back(0);
        // hiddenWeight.push_back((rand() % 100) * 0.1);
    }
    for (int i = 0; i < t.size(); i++)
    {
        target.push_back(t[i]);
        outNet.push_back(0);
        outOut.push_back(0);
        outDelta.push_back(0);
    }
    for (int i = 0; i < (2 * target.size()); i++)
    {
        outWeight.push_back(0.5);
        if (i % 2 == 1)
        {
            outX.push_back(1);
        }
        else
        {
            outX.push_back(0);
        }
    }
    for (int i = 0; i < hiddenX.size(); i++)
    {
        hiddenNet.push_back(0);
        hiddenOut.push_back(0);
        hiddenDelta.push_back(0);
    }
};

double NeuralNetwork::fa(double net)
{
    return (1 - exp(-net)) / (1 + exp(-net));
}

double NeuralNetwork::derivative(double net)
{
    return 0.5 * (1 - fa(net)) * fa(net);
}

std::vectotr<double> NeuralNetwork::net(std::vector<double> x, std::vector<double> weight)
{
    std::vector<double> net;
    int c = 0;
    for (int i = 0; i < weight.size(); i++)
    {
        if (i % == 0)
        {
            c = i;
            c++;
            net.push_back(weight[i] * x[i] + weight[c] * x[c]);
        }
    }
    return net;
}

void NeuralNetwork::train()
{
    double error = 1;
    int epoch = 1;
    while (error > 0.001)
    {
        std::cout << std::endl << "Epoch # " << epoch << std::endl;
        std::cout << "Hidden Weights: " << std::endl;
        for (int i = 0; i < hiddenWeight.size(); i++)
        {
            std::cout << hiddenWeight[i] << " ";
        }
        std::cout << std:endl;
        std::cout << "Out Weights: " << std::endl;
        for (int i = 0; i < outWeight.size(); i++)
        {
            std::cout << outWeight[i] << " ";
        }
        std::cout << std:endl;
        // 1st Stage
        std::vector<double> tmpNet = net(hiddenX, hiddenWeight);
        for (int i = 0; i < hiddenOut.size(); i++)
        {
            hiddenNet[i] = tmpNet[i];
            hiddenOut[i] = fa(hiddenNet[i]);
        }
        for (int i = 0; i < (2 * target.size()); i++)
        {
            if (i % 2 == 0)
            {
                outX[i] = hiddenOut[(hiddenOut.size() - 1)];
            }
        }
        tmpNet.erase(tmpNet.begin(), tmpNet.end());
        tmpNet = net(outX, outWeight);
        for (int i = 0; i < outOut.size(); i++)
        {
            outNet[i] = tmpNet[i];
            outOut[i] = fa(outNet[i]);
        }
        // 2nd Stage
        for (int i = 0; i < outDelta.size(); i++)
        {
            outDelta[i] = (target[i] - outOut[i]) * derivative[outOut[i]];
        }
        for (int i = 0; i < hiddenDelta.size(); i++)
        {
            double e = 0;
            for (int j = 0; j < outWeight.size(); j++)
            {
                if (j % == 0)
                {
                    e += outWeight[i] * outDelta[j / 2];
                }
                hiddenDelta[i] = e * derivative(hiddenOut[i]);
            }
        }
        // 3rd Stage
        for (int i = 0; i < outWeight.size(); i++)
        {
            if (i % 2 == 0)
            {
                outWeight[i] += rateOfTraining * hiddenOut[hiddenOut.size() - 1] * outDelta[i / 2];
            }
            if (i % 2 == 1)
            {
                outWeight[i] += rateOfTraining * outDelta[i / 2];
            }
        }
        for (int i = 0; i < hiddenWeight.size(); i++)
        {
            hiddenWeight[i] += rateOfTraining * hiddenDelta[hiddenDelta.size() - 1] * hiddenX[i];
        }
        error = 0;
        for (int i = 0; i < target.size(); i++)
        {
            error += pow((target[i] - outOut[i]), 2);
            error = sqrt(error);
            epoch++;
            std::cout << "NN Output: " << std::endl;
            for (int i = 0; i < outOut.size(); i++)
            {
                std::cout << outOut[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "Error: " << error << std::endl;
        }
    }
    std::cout << "Training Ends in " << --epoch << " epochs";
}