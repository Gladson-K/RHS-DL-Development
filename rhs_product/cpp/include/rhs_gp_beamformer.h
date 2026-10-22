#ifndef RHS_GP_BEAMFORMER_H
#define RHS_GP_BEAMFORMER_H

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <chrono>
#include <random>

using namespace Eigen;

namespace rhs {

/**
 * @brief Complex tensor types for RHS beamforming
 */
using Complex = std::complex<double>;
using MatrixXcd = Matrix<Complex, Dynamic, Dynamic>;
using VectorXcd = Vector<Complex, Dynamic>;
using ArrayXXcd = Array<Complex, Dynamic, Dynamic>;

/**
 * @brief System configuration for RHS beamforming
 */
struct SystemConfig {
    int Nr = 64;           // Number of receive antennas
    int Nt = 64;           // Number of transmit antennas (RHS elements)
    int L = 4;             // Number of subcarriers/path components
    int RFno = 1;          // Number of RF chains
    int test_size = 20;    // Batch size for parallel processing
    double frequency = 30e9; // Carrier frequency in Hz
    double c = 3e8;        // Speed of light
    double wavelength;     // Wavelength
    double deH;            // Element spacing
    double scale = 1e9;    // Channel scaling factor
    double SNR_dB = 30.0;  // SNR in dB
    double Pt;             // Transmit power
    double noise_var = 0.1; // Noise variance
    double stepSize1 = 10.0; // Gradient step size for GP
    int max_iter = 100;    // Maximum iterations for GP
    double conv_threshold = 1e-4; // Convergence threshold
};

/**
 * @brief Result structure for beamforming
 */
struct BeamformingResult {
    std::vector<MatrixXcd> M;      // Beamforming matrices [test_size, Nt x RFno]
    std::vector<MatrixXcd> w;      // Receive combiner [test_size, Nr x L]
    std::vector<double> rates;     // Achievable rates per sample
    double avg_rate;               // Average rate
    double computation_time_ms;    // Computation time
    int iterations;                // Number of iterations for GP
};

/**
 * @brief Gradient Projection Algorithm for RHS Beamforming
 * 
 * Implements the GP algorithm as described in:
 * - Holographic beamforming with reconfigurable holographic surfaces
 * - Gradient projection for amplitude-constrained holographic beamforming
 */
class GradientProjectionBeamformer {
public:
    GradientProjectionBeamformer(const SystemConfig& config);
    
    /**
     * @brief Generate Rayleigh fading channel
     * @param seed Random seed for reproducibility
     * @return H_RHS shape [test_size, Nr, Nt, L]
     */
    std::vector<std::vector<MatrixXcd>> generateRayleighChannel(unsigned seed = 42);
    
    /**
     * @brief Generate MMSE beamforming matrix
     * @param H_BS Channel matrix [test_size, Nr, Nt]
     * @return W_MMSE [test_size, Nt, Nr]
     */
    std::vector<MatrixXcd> generateMMSEBeamforming(const std::vector<MatrixXcd>& H_BS);
    
    /**
     * @brief Generate excitation matrix E
     * @param Nx Number of RHS elements parameter
     * @return E shape [test_size, Nt, RFno]
     */
    std::vector<MatrixXcd> generateExcitationMatrix(int Nx = 64);
    
    /**
     * @brief Run Gradient Projection Algorithm
     * @param H_testD Effective channel [test_size, Nr, Nt, L]
     * @param E Excitation matrix [test_size, Nt, RFno]
     * @return Beamforming result
     */
    BeamformingResult optimizeGP(const std::vector<std::vector<MatrixXcd>>& H_testD,
                                 const std::vector<MatrixXcd>& E);
    
    /**
     * @brief Compute sum capacity/rate
     * @param w Receive combiner
     * @param H Channel
     * @param M Beamforming matrix
     * @return Pair of (raw_capacity, scaled_capacity)
     */
    std::pair<double, double> computeSumCapacity(const std::vector<MatrixXcd>& w,
                                                 const std::vector<std::vector<MatrixXcd>>& H,
                                                 const std::vector<MatrixXcd>& M);
    
    /**
     * @brief Run complete beamforming pipeline
     * @param Nx RHS size parameter
     * @param seed Random seed
     * @return Beamforming result
     */
    BeamformingResult runPipeline(int Nx = 64, unsigned seed = 42);

private:
    SystemConfig config_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    
    Complex complexRand();
    double realRand();
};

} // namespace rhs

#endif // RHS_GP_BEAMFORMER_H
