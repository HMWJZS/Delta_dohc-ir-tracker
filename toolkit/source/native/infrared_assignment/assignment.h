#pragma once
#include <cstdint>

extern "C" {
uint32_t ir_assignment_abi();
int ir_assign_f64(int batches, int points, const double* predicted,
    const double* observed, int64_t* indices, double* distances);
}
