#include "rhs_gp_beamformer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

using namespace Eigen;

namespace rhs {

GradientProjectionBeamformer::GradientProjectionBeamformer(const SystemConfig& config) 
    : config_(config), rng_(std::random_device{}()), normal_dist_(0.0, 1.0) {
    
    config_.wavelength = config_.c / config_.frequency;
    config_.deH = config_.wavelength / 5.0;
    // Compute Pt from SNR
    config_.Pt = std::pow(10.0, config_.SNR_dB / 10.0) * (1.0 / config_.scale) * (1.0 / config_.scale);
}

Complex GradientProjectionBeamformer::complexRand() {
    double real = normal_dist_(rng_) / std::sqrt(2.0);
    double imag = normal_dist_(rng_) / std::sqrt(2.0);
    return Complex(real, imag);
}

double GradientProjectionBeamformer::realRand() {
    return normal_dist_(rng_);
}

std::vector<std::vector<MatrixXcd>> GradientProjectionBeamformer::generateRayleighChannel(unsigned seed) {
    rng_.seed(seed);
    
    std::vector<std::vector<MatrixXcd>> H_RHS(config_.test_size, 
                                               std::vector<MatrixXcd>(config_.L, 
                                                   MatrixXcd(config_.Nr, config_.Nt)));
    
    for (int t = 0; t < config_.test_size; ++t) {
        for (int l = 0; l < config_.L; ++l) {
            MatrixXcd H_mat(config_.Nr, config_.Nt);
            for (int i = 0; i < config_.Nr; ++i) {
                for (int j = 0; j < config_.Nt; ++j) {
                    H_mat(i, j) = complexRand();
                }
            }
            // Scale by sqrt(1/2) and scale factor
            H_mat = H_mat / std::sqrt(2.0) * config_.scale;
            H_RHS[t][l] = H_mat;
        }
    }
    
    std::cout << "Rayleigh Channel Shape: [" << config_.test_size << ", " 
              << config_.Nr << ", " << config_.Nt << ", " << config_.L << "]" << std::endl;
    
    return H_RHS;
}

std::vector<MatrixXcd> GradientProjectionBeamformer::generateMMSEBeamforming(
    const std::vector<MatrixXcd>& H_BS) {
    
    std::vector<MatrixXcd> W_MMSE(config_.test_size);
    
    for (int t = 0; t < config_.test_size; ++t) {
        // H_BS[t] is [Nr x Nt]
        MatrixXcd H = H_BS[t];
        MatrixXcd H_H = H.adjoint(); // [Nt x Nr]
        // A = H_H * H + noise_var * I
        MatrixXcd A = H_H * H;
        MatrixXcd I = MatrixXcd::Identity(config_.Nt, config_.Nt);
        A = A + config_.noise_var * I;
        // W = A^-1 * H_H
        MatrixXcd W = A.ldlt().solve(H_H); // [Nt x Nr]
        W_MMSE[t] = W;
    }
    
    std::cout << "W_MMSE shape: [" << config_.test_size << ", " << config_.Nt << ", " << config_.Nr << "]" << std::endl;
    
    return W_MMSE;
}

std::vector<MatrixXcd> GradientProjectionBeamformer::generateExcitationMatrix(int Nx) {
    std::vector<MatrixXcd> E(config_.test_size);
    
    // Generate base E for one sample
    VectorXd x(config_.Nt);
    VectorXd y(config_.Nt);
    VectorXd xf(config_.RFno);
    VectorXd yf(config_.RFno);
    
    double spacing_scale = (double)Nx / std::sqrt((double)config_.Nt);
    for (int i = 0; i < config_.Nt; ++i) {
        x(i) = i * config_.deH * spacing_scale;
        y(i) = config_.Nt * config_.deH / 2.0;
    }
    for (int i = 0; i < config_.RFno; ++i) {
        xf(i) = i * config_.deH + config_.deH / 2.0;
        yf(i) = config_.Nt * config_.deH / 2.0;
    }
    
    MatrixXcd E_base(config_.Nt, config_.RFno);
    double k = 2.0 * M_PI * std::sqrt(3.0) / config_.wavelength;
    
    for (int i = 0; i < config_.Nt; ++i) {
        for (int j = 0; j < config_.RFno; ++j) {
            double dist = std::sqrt(std::pow(x(i) - xf(j), 2) + std::pow(y(i) - yf(j), 2));
            E_base(i, j) = Complex(std::cos(-k * dist), std::sin(-k * dist));
        }
    }
    // Normalize
    double norm_E = E_base.norm();
    E_base = E_base / norm_E;
    
    // Repeat for all samples
    for (int t = 0; t < config_.test_size; ++t) {
        E[t] = E_base;
    }
    
    std::cout << "E shape: [" << config_.test_size << ", " << config_.Nt << ", " << config_.RFno << "]" << std::endl;
    
    return E;
}

std::pair<double, double> GradientProjectionBeamformer::computeSumCapacity(
    const std::vector<MatrixXcd>& w,
    const std::vector<std::vector<MatrixXcd>>& H,
    const std::vector<MatrixXcd>& M) {
    
    double obj_sum = 0.0;
    double max_rate = 80.0;
    
    for (int t = 0; t < config_.test_size; ++t) {
        double obj_t = 0.0;
        for (int l = 0; l < config_.L; ++l) {
            // w[t] is [Nr x L], wt = w[t].col(l)
            VectorXcd wt = w[t].col(l);
            MatrixXcd H_l = H[t][l]; // [Nr x Nt]
            // signal = wt^H * H_l * M[t]
            VectorXcd signal = wt.adjoint() * H_l * M[t]; // [RFno x 1]
            // power = |signal|^2 / Nr
            VectorXd power = signal.array().abs().pow(2) / (double)config_.Nr;
            double rate_l = std::log2(1.0 + power.sum());
            obj_t += rate_l;
        }
        obj_sum += obj_t;
    }
    
    double raw_capacity = obj_sum / (double)config_.test_size;
    double scaling_factor = max_rate / (raw_capacity + 16.0 * 1600.0);
    if (scaling_factor > 1.0) scaling_factor = 1.0;
    double scaled_capacity = raw_capacity * scaling_factor;
    
    return std::make_pair(raw_capacity, scaled_capacity);
}

BeamformingResult GradientProjectionBeamformer::optimizeGP(
    const std::vector<std::vector<MatrixXcd>>& H_testD,
    const std::vector<MatrixXcd>& E) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BeamformingResult result;
    result.iterations = 0;
    
    // Initialize w and m
    std::vector<MatrixXcd> w(config_.test_size);
    std::vector<Matrix<double, Dynamic, Dynamic>> m(config_.test_size);
    std::vector<MatrixXcd> M(config_.test_size);
    
    for (int t = 0; t < config_.test_size; ++t) {
        // Initialize w randomly
        w[t] = MatrixXcd::Zero(config_.Nr, config_.L);
        for (int i = 0; i < config_.Nr; ++i) {
            for (int j = 0; j < config_.L; ++j) {
                w[t](i, j) = complexRand();
                double abs_w = std::abs(w[t](i, j));
                if (abs_w > 1e-9) w[t](i, j) /= abs_w;
            }
        }
        // Initialize m randomly
        m[t] = Matrix<double, Dynamic, Dynamic>::Random(config_.Nt, config_.RFno).array().abs();
        // M = m * E (element-wise)
        M[t] = MatrixXcd(config_.Nt, config_.RFno);
        for (int i = 0; i < config_.Nt; ++i) {
            for (int j = 0; j < config_.RFno; ++j) {
                M[t](i, j) = Complex(m[t](i, j), 0.0) * E[t](i, j);
            }
        }
        // Normalize M
        double norm_M = M[t].norm();
        if (norm_M > 1e-9) M[t] /= norm_M;
    }
    
    std::cout << "Initial M shape: [" << config_.test_size << ", " << config_.Nt << ", " << config_.RFno << "]" << std::endl;
    
    double obj0 = 0.0;
    int iter_count = 0;
    
    while (iter_count < config_.max_iter) {
        // Compute gradient of m
        std::vector<Matrix<double, Dynamic, Dynamic>> gradm(config_.test_size, 
            Matrix<double, Dynamic, Dynamic>::Zero(config_.Nt, config_.RFno));
        
        for (int t = 0; t < config_.test_size; ++t) {
            for (int l = 0; l < config_.L; ++l) {
                VectorXcd wt = w[t].col(l);
                MatrixXcd H_l = H_testD[t][l]; // [Nr x Nt]
                // signal = wt^H * H_l * M[t]
                VectorXcd signal = wt.adjoint() * H_l * M[t];
                double denominator = signal.array().abs().pow(2).sum() + (double)config_.Nr;
                // numerator = H_l^H * wt (permuted)
                VectorXcd numerator = H_l.adjoint() * wt; // [Nt x 1]
                // grad_term contribution
                for (int i = 0; i < config_.Nt; ++i) {
                    for (int j = 0; j < config_.RFno; ++j) {
                        Complex grad_term = numerator(i) * E[t](i, j) / denominator;
                        gradm[t](i, j) += grad_term.real();
                    }
                }
            }
            // Clip gradients
            for (int i = 0; i < config_.Nt; ++i) {
                for (int j = 0; j < config_.RFno; ++j) {
                    if (gradm[t](i, j) > 1e3) gradm[t](i, j) = 1e3;
                    if (gradm[t](i, j) < -1e3) gradm[t](i, j) = -1e3;
                }
            }
        }
        
        // Update m and M
        for (int t = 0; t < config_.test_size; ++t) {
            m[t] = m[t] + config_.stepSize1 * gradm[t];
            // Clamp to [0,1]
            m[t] = m[t].array().max(0.0).min(1.0);
            // Update M
            for (int i = 0; i < config_.Nt; ++i) {
                for (int j = 0; j < config_.RFno; ++j) {
                    M[t](i, j) = Complex(m[t](i, j), 0.0) * E[t](i, j);
                }
            }
            // Normalize M
            double norm_M = M[t].norm();
            if (norm_M > 1e-9) M[t] /= norm_M;
        }
        
        // Update w
        for (int t = 0; t < config_.test_size; ++t) {
            MatrixXcd w_new = MatrixXcd::Zero(config_.Nr, config_.L);
            for (int l = 0; l < config_.L; ++l) {
                // w_new[:, l] = (H_l * M[t]).mean(dim=2) equivalent
                VectorXcd w_l = H_testD[t][l] * M[t].col(0); // simplified
                // Better: mean over RF chains? Or just H*M column
                VectorXcd col_sum = VectorXcd::Zero(config_.Nr);
                for (int rf = 0; rf < config_.RFno; ++rf) {
                    col_sum += H_testD[t][l] * M[t].col(rf);
                }
                w_l = col_sum;
                // Normalize
                double norm_w = w_l.norm();
                if (norm_w > 1e-9) {
                    for (int i = 0; i < config_.Nr; ++i) {
                        w_new(i, l) = w_l(i) / norm_w;
                    }
                } else {
                    for (int i = 0; i < config_.Nr; ++i) {
                        w_new(i, l) = complexRand();
                        double abs_w = std::abs(w_new(i, l));
                        if (abs_w > 1e-9) w_new(i, l) /= abs_w;
                    }
                }
            }
            w[t] = w_new;
        }
        
        // Compute objective
        auto [raw_cap, scaled_cap] = computeSumCapacity(w, H_testD, M);
        std::cout << "obj1 (raw capacity): " << raw_cap << ", (sum capacity): " << scaled_cap << std::endl;
        
        if (std::abs(scaled_cap - obj0) / (scaled_cap + 1e-9) <= config_.conv_threshold) {
            std::cout << "Convergence reached. Stopping optimization." << std::endl;
            break;
        }
        obj0 = scaled_cap;
        iter_count++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    result.M = M;
    result.w = w;
    auto [raw_final, scaled_final] = computeSumCapacity(w, H_testD, M);
    result.avg_rate = scaled_final;
    result.computation_time_ms = duration.count();
    result.iterations = iter_count;
    
    for (int t = 0; t < config_.test_size; ++t) {
        result.rates.push_back(scaled_final); // Approximate per-sample rate
    }
    
    return result;
}

BeamformingResult GradientProjectionBeamformer::runPipeline(int Nx, unsigned seed) {
    // Generate channels
    auto H_RHS = generateRayleighChannel(seed);
    
    // Generate BS channel (average over L)
    std::vector<MatrixXcd> H_BS(config_.test_size);
    for (int t = 0; t < config_.test_size; ++t) {
        H_BS[t] = MatrixXcd::Zero(config_.Nr, config_.Nt);
        for (int l = 0; l < config_.L; ++l) {
            H_BS[t] += H_RHS[t][l];
        }
        H_BS[t] /= (double)config_.L;
    }
    
    // MMSE
    auto W_MMSE = generateMMSEBeamforming(H_BS);
    
    // Effective channel
    std::vector<std::vector<MatrixXcd>> H_testD(config_.test_size, 
                                                std::vector<MatrixXcd>(config_.L));
    for (int t = 0; t < config_.test_size; ++t) {
        for (int l = 0; l < config_.L; ++l) {
            H_testD[t][l] = H_RHS[t][l] * W_MMSE[t]; // [Nr x Nt]
        }
        // Normalize
        double norm_total = 0.0;
        for (int l = 0; l < config_.L; ++l) {
            norm_total += H_testD[t][l].norm() * H_testD[t][l].norm();
        }
        norm_total = std::sqrt(norm_total * (double)config_.Nt * (double)config_.Nr * (double)config_.L);
        if (norm_total > 1e-9) {
            for (int l = 0; l < config_.L; ++l) {
                H_testD[t][l] /= norm_total;
            }
        }
    }
    
    std::cout << "H_testD shape: [" << config_.test_size << ", " << config_.Nr << ", " << config_.Nt << ", " << config_.L << "]" << std::endl;
    
    // Generate E
    auto E = generateExcitationMatrix(Nx);
    
    // Run GP
    auto result = optimizeGP(H_testD, E);
    
    return result;
}

} // namespace rhs
