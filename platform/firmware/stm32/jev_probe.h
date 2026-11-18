#ifndef RHS_JEV_PROBE_H
#define RHS_JEV_PROBE_H

#ifdef __cplusplus
extern "C" {
#endif

#define RHS_JEV_DIM 10
#define RHS_JEV_K 4

/* Reduce a 10x10 Hermitian Gram matrix to a 4-bin JEV signature (softmax of
 * the top-4 eigenvalues), matching protocol/jev.py:spectral_signature(). */
void rhs_jev_signature(const float gram[RHS_JEV_DIM][RHS_JEV_DIM], float out[RHS_JEV_K]);

#ifdef __cplusplus
}
#endif

#endif /* RHS_JEV_PROBE_H */
