#include "rhs_gp_beamformer.h"
#include <iostream>
#include <iomanip>

using namespace rhs;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RHS Gradient Projection Beamformer (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    SystemConfig config;
    config.Nr = 64;
    config.Nt = 64;
    config.L = 4;
    config.RFno = 1;
    config.test_size = 20;
    config.frequency = 30e9;
    config.scale = 1e9;
    config.SNR_dB = 30.0;
    config.noise_var = 0.1;
    config.stepSize1 = 10.0;
    config.max_iter = 50; // Reduced for faster testing
    config.conv_threshold = 1e-4;
    
    GradientProjectionBeamformer gp(config);
    
    // Test different Nx values
    std::vector<int> Nx_values = {10, 20, 30, 40};
    
    std::cout << "\n=== Testing GP Algorithm ===" << std::endl;
    for (int Nx : Nx_values) {
        std::cout << "\n--- Nx = " << Nx << " ---" << std::endl;
        BeamformingResult result = gp.runPipeline(Nx, 42);
        std::cout << "Average Rate: " << std::fixed << std::setprecision(4) << result.avg_rate << " bps/Hz" << std::endl;
        std::cout << "Computation Time: " << result.computation_time_ms << " ms" << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "GP Algorithm Implementation Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
