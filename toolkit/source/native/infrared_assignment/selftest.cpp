#include "assignment.h"
#include <cassert>
#include <cstdint>

int main() {
    double predicted[8 * 12 * 2] = {}, observed[12 * 2] = {};
    int64_t indices[8 * 12 + 2];
    double distances[8 * 12 + 2];
    for (int i = 0; i < 98; ++i) { indices[i] = -7; distances[i] = -8.; }
    assert(ir_assignment_abi() == 1);
    for (int batches = 1; batches <= 8; ++batches) {
        for (int points = 1; points <= 12; ++points) {
            assert(ir_assign_f64(batches, points, predicted, observed, indices + 1, distances + 1) == 0);
            for (int i = 1; i <= batches * points; ++i) {
                assert(indices[i] == 0 && distances[i] == 0.);
            }
        }
    }
    assert(indices[0] == -7 && indices[97] == -7 && distances[0] == -8. && distances[97] == -8.);
    assert(ir_assign_f64(9, 12, predicted, observed, indices + 1, distances + 1) == -1);
    assert(ir_assign_f64(1, 13, predicted, observed, indices + 1, distances + 1) == -1);
    assert(ir_assign_f64(0, 12, predicted, observed, indices + 1, distances + 1) == -1);
    assert(ir_assign_f64(1, 12, nullptr, observed, indices + 1, distances + 1) == -1);
}
