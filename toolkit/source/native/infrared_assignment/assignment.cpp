// Adapted from IR-Offline Top6A-R5 assignment.cpp; batched, with compact outputs.
#include "assignment.h"
#include <cmath>
#include <cstddef>

extern "C" uint32_t ir_assignment_abi() { return 1; }

extern "C" int ir_assign_f64(int batches, int points, const double* predicted,
    const double* observed, int64_t* indices, double* distances) {
    if (batches < 1 || batches > 8 || points < 1 || points > 12 ||
        !predicted || !observed || !indices || !distances) return -1;
    for (int batch = 0; batch < batches; ++batch) {
        for (int i = 0; i < points; ++i) {
            const auto row = static_cast<std::size_t>(batch * points + i);
            double best = 0.;
            int index = 0;
            for (int j = 0; j < points; ++j) {
                const double x = predicted[2 * row] - observed[2 * j];
                const double y = predicted[2 * row + 1] - observed[2 * j + 1];
                const double xx = x * x, yy = y * y, sum = xx + yy;
                const double distance = std::sqrt(sum);
                // NumPy argmin keeps the first NaN, otherwise the earliest tie.
                if (j == 0 || (!std::isnan(best) && (std::isnan(distance) || distance < best))) {
                    best = distance;
                    index = j;
                }
            }
            indices[row] = index;
            distances[row] = best;
        }
    }
    return 0;
}
