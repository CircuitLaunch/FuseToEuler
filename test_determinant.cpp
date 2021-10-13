#include "LinearAlgebra.hpp"
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    float a0[] = { 3.0, 4.0, 5.0, 2.0, 6.0, 1.0, 1.0, 4.0, 4.0, -2.0, 5.0, 8.0, 2.0, 8.0, 7.0, 3.0 };
    mat4 m0(a0);

    cout << m0.determinant() << endl;

    float a1[] = { 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
    mat4 m1(a1);
    mat4 m2(m1.reciprocal());

    int j = 16;
    while(j--) {
        cout << "    " << m2[15 - j];
        if(!(j&3)) cout << endl;
    }
}
