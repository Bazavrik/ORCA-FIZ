// example_main.cpp
#include <iostream>
#include "fuzzy_sd.hpp"

int main() {
    FuzzySD fis(301);

    // пример входов: V, d, fi, C
    // C: bicycle≈1, human≈2, animal≈3, undef≈4 (как в твоём FIS)
    double V  = 3.0;
    double d  = 2.0;
    double fi = 30.0;
    double C  = 2.0; // human

    double y = fis.eval(V, d, fi, C);
    std::cout << "FuzzySD output = " << y << "\n";
    return 0;
}