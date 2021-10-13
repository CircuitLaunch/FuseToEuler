#ifndef __LINEARALGEBRA_HPP__
#define __LINEARALGEBRA_HPP__

#include <math.h>

typedef struct vec3
{
    vec3(float x = 0.0, float y = 0.0, float z = 0.0) { a[0] = x; a[1] = y; a[2] = z; a[3] = 0.0; }
    vec3(const float *v) { int i = 3; while(i--) a[i] = v[i]; a[3] = 0.0; }
    vec3(const vec3 &v) { int i = 3; while(i--) a[i] = v[i]; a[3] = 0.0; }

    bool operator==(const vec3 &v) const {
        int i = 3; while(i--) if(a[i] != v[i]) return false;
        return true;
    }

    bool operator!=(const vec3 &v) const {
        return !(*this == v);
    }

    float operator[](int i) const {
        return a[i];
    }

    float &operator[](int i) {
        return a[i];
    }

    vec3 &operator+=(const vec3 &o) {
        int i = 3; while(i--) a[i] += o[i];
        return *this;
    }
    vec3 &operator-=(const vec3 &o) {
        int i = 3; while(i--) a[i] -= o[i];
        return *this;
    }

    vec3 operator+(const vec3 &o) const {
        vec3 r(*this);
        r += o;
        return r;
    }

    vec3 operator-(const vec3 &o) const {
        vec3 r(*this);
        r -= o;
        return r;
    }

    vec3 &operator*=(float s) {
        int i = 3; while(i--) a[i] *= s;
        return *this;
    }

    vec3 &operator/=(float s) {
        return *this *= (1.0 / s);
    }

    vec3 operator*(float s) const {
        vec3 r(*this);
        r *= s;
        return r;
    }

    vec3 operator/(float s) const {
        vec3 r(*this);
        r /= s;
        return r;
    }

    vec3 operator^(const vec3 &o) const {
        vec3 r(a[1] * o[2] - a[2] * o[1], a[2] * o[0] - a[0] * o[2], a[0] * o[1] - a[1] * o[0]);
        return r;
    }

    vec3 operator*(const vec3 &o) const {
        return a[0]*o[0] + a[1]*o[1] + a[2]*o[2] + a[3]*o[3];
    }

    float mag() const {
        return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }

    vec3 normalized() const {
        vec3 r(*this);
        float m = mag();
        if(m > 0.0) r /= m;
        return r;
    }

    float a[4];
} vec3;

typedef struct mat4
{
    enum {
        IDENTITY = 1,
        TRANSLATION = 2,
        ROTATION = 3,
        SCALING = 4
    };

    mat4(int iType = IDENTITY, float a = 0.0, float x = 0.0, float y = 0.0, float z = 0.0);
    mat4(const float *v) { int i = 16; while(i--) a[i] = v[i]; }
    mat4(const mat4 &v) { int i = 16; while(i--) a[i] = v[i]; }

    float operator[](int i) const { return a[i]; }
    float &operator[](int i) { return a[i]; }

    mat4 &operator*=(const mat4 &m);
    mat4 operator*(const mat4 &m) const;

    vec3 operator*(const vec3 &v) const;

    float determinant() const;
    float minorDeterminant(int *ci, int level) const;

    mat4 reciprocal() const;

    float a[16];
} mat4;

#endif
