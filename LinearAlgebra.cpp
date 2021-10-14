#include "LinearAlgebra.hpp"

mat4::mat4(int iType, float ia, float x, float y, float z)
{
    a[15] = 1.0;
    float factor = 1.0;
    int i;
    switch(iType) {
        case SCALING:
            factor = ia;
        case IDENTITY:
            i = 15;
            while(i--) a[i] = factor * float ((i % 5) == 0);
            break;
        case TRANSLATION:
            i = 15;
            while(i--) a[i] = 0.0;
            a[3] = ia; a[7] = x; a[11] = y;
            break;
        case ROTATION:
            {
                float c = cos(ia), s = sin(ia);
                float cc = 1 - c, cs = 1 - s;
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, xz = x * z, yz = y * z;
                a[0] = c + xx * cc; a[1] = xy * cc - z * s; a[2] = xz * cc + y * s; a[3] = 0.0;
                a[4] = xy * cc + z * s; a[5] = c + yy * cc; a[6] = yz * cc - x * s; a[7] = 0.0;
                a[8] = xz * cc - y * s; a[9] = yz * cc + x * s; a[10] = c + zz * cc; a[11] = 0.0;
                a[12] = 0.0; a[13] = 0.0; a[14] = 0.0; a[15] = 1.0;
            }
            break;
    }
}

mat4 &mat4::operator*=(const mat4 &m)
{
    mat4 r(*this);
    *this = r * m;
    return *this;
}

mat4 mat4::operator*(const mat4 &m) const
{
    mat4 r;
    int i = 16, c, j;
    while(i--) {
        c = i >> 2;
        j = 4;
        while(j--) {
            r[i] += a[c + j] * m[j << 2 + c];
        }
    }
    return r;
}

vec3 mat4::operator*(const vec3 &v) const
{
    vec3 r;
    int i = 4, j, c;
    while(i--) {
        j = 4;
        c = i << 2;
        while(j--) {
           r[i] += a[c + j] * v[j];
        }
    }
    return r;
}

float mat4::determinant() const
{
    int colIndices[] = { 0, 1, 2, 3 };

    return minorDeterminant(colIndices, 4);
}

float mat4::minorDeterminant(int *colIndices, int level) const
{
    if(level == 2) {
        int i0 = 8 + colIndices[0], i1 = 8 + colIndices[1], i2 = 12 + colIndices[0], i3 = 12 + colIndices[1];
        return a[i0] * a[i3] - a[i2] * a[i1];
    }
    int *minorColIndices = new int[level - 1];
    float det = 0.0;
    int o = (4 - level) << 2;
    int i = level;
    while(i--) {
        int j = level - 1;
        while(j--) minorColIndices[j] = colIndices[(j < i) ? j : j + 1];
        det += a[o + colIndices[i]] * minorDeterminant(minorColIndices, level-1) * float ((((i & 1) == 0) << 1) - 1);
    }
    delete [] minorColIndices;
    return det;
}

mat4 mat4::reciprocal() const
{
    mat4 r;

    // Create augment matrix
    float aug[32];
    int j = 16;
    while(j--) {
        int o = ((j >> 2) << 3) + (j & 3);
        aug[o] = a[j];
        aug[4 + o] = float((j % 5) == 0);
    }
    // Set up array of pointers for efficient row swapping, and simplified indexing
    float *row[] = { aug, aug + 8, aug + 16, aug + 24 };

    // Gauss-Jordan Elimination
    for(j = 0; j < 4; j++) {
        // If the pivot element is zero
        if(row[j][j] == 0.0) 
            // Swap it with a lower row that does not contain zero in the same column
            for(int n = j; n < 4; n++) {
                if(row[n][j] != 0.0) {
                    float * t = row[n];
                    row[n] = row[j];
                    row[j] = t;
                    break;
                }
            }

        // If the pivot element is still zero, skip this pivot
        if(row[j][j] != 0.0) {
            // Normalize row so pivot element is 1.0
            float normalizer = 1.0 / row[j][j];
            for(int i = 0; i < 8; i++) 
                row[j][i] *= normalizer;

            // Use multiple of pivot row to eliminate non-zero elements in the same column in all other rows
            for(int n = 0; n < 4; n++) {
                float eliminator = row[n][j];
                if((n != j) && (eliminator != 0.0)) 
                    // Start from j, as elements in previous columns have already been eliminated
                    for(int i = j; i < 8; i++)
                        row[n][i] -= row[j][i] * eliminator;
            }
        }
    }

    // The augment matrix should now be in reduced echelon form, and the right side square sub-matrix 
    // will hold the reciprocal.

    // Return solution
    j = 16;
    while(j--)
        r[j] = row[j >> 2][4 + (j & 3)];

    return r;
}