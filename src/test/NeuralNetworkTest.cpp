
#define CATCH_CONFIG_MAIN
#include <vector>

#include "../catch.hpp"
#include "../NeuralNetwork.h"

TEST_CASE( "NeuralNetwork can feed forward", "[NeuralNetwork]" ) {

    int nodes[3] = { 2, 2, 1};
    std::vector<int> layout(&nodes[0], &nodes[0]+3);

}
