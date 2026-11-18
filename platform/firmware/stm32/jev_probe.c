#include "jev_probe.h"

#include <math.h>
#include <string.h>

static float dot_product(const float *a, const float *b, int n) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void matrix_vector(const float m[RHS_JEV_DIM][RHS_JEV_DIM],
                          const float *v,
                          float *out) {
    int i;
    int j;
    for (i = 0; i < RHS_JEV_DIM; ++i) {
        float sum = 0.0f;
        for (j = 0; j < RHS_JEV_DIM; ++j) {
            sum += m[i][j] * v[j];
        }
        out[i] = sum;
    }
}

void rhs_jev_signature(const float gram[RHS_JEV_DIM][RHS_JEV_DIM], float out[RHS_JEV_K]) {
    float a[RHS_JEV_DIM][RHS_JEV_DIM];
    float eigenvalues[RHS_JEV_K];
    int k;
    int i;
    int j;

    if (gram == 0 || out == 0) {
        return;
    }
    memcpy(a, gram, sizeof(a));

    for (k = 0; k < RHS_JEV_K; ++k) {
        float v[RHS_JEV_DIM];
        float w[RHS_JEV_DIM];
        float lambda = 0.0f;
        int iter;
        for (i = 0; i < RHS_JEV_DIM; ++i) {
            v[i] = 1.0f / sqrtf((float)RHS_JEV_DIM);
        }
        for (iter = 0; iter < 64; ++iter) {
            float norm;
            matrix_vector(a, v, w);
            norm = sqrtf(dot_product(w, w, RHS_JEV_DIM));
            if (norm < 1e-9f) {
                break;
            }
            for (i = 0; i < RHS_JEV_DIM; ++i) {
                v[i] = w[i] / norm;
            }
        }
        matrix_vector(a, v, w);
        lambda = dot_product(v, w, RHS_JEV_DIM);
        eigenvalues[k] = lambda;
        for (i = 0; i < RHS_JEV_DIM; ++i) {
            for (j = 0; j < RHS_JEV_DIM; ++j) {
                a[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    {
        float peak = eigenvalues[0];
        float total = 0.0f;
        for (k = 1; k < RHS_JEV_K; ++k) {
            if (eigenvalues[k] > peak) {
                peak = eigenvalues[k];
            }
        }
        for (k = 0; k < RHS_JEV_K; ++k) {
            out[k] = expf(eigenvalues[k] - peak);
            total += out[k];
        }
        if (total <= 0.0f || !isfinite(total)) {
            for (k = 0; k < RHS_JEV_K; ++k) {
                out[k] = 1.0f / (float)RHS_JEV_K;
            }
            return;
        }
        for (k = 0; k < RHS_JEV_K; ++k) {
            out[k] /= total;
        }
    }
}
