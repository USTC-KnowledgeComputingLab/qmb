#include <torch/extension.h>

namespace {

template <std::int64_t n_qubytes, std::int64_t particle_cut, std::int64_t max_op_number>
auto apply_within_subspace_in_double_side_interface(
    const torch::Tensor& configs_i,
    const torch::Tensor& psi_i,
    const torch::Tensor& configs_j,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    bool configs_i_sorted,
    bool configs_j_sorted,
    int64_t direction) -> torch::Tensor
{
    TORCH_CHECK(false, "CUDA backend not yet implemented.");
    return torch::Tensor{};
}

}  // namespace

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif
#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(nq, pc, mo) qmp_hamiltonian_##nq##_##pc##_##mo
#define QMP_LIBRARY(nq, pc, mo) QMP_LIBRARY_HELPER(nq, pc, mo)

TORCH_LIBRARY_FRAGMENT(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER), m) {
    m.def("apply_within_subspace_in_double_side("
          "Tensor configs_i, Tensor psi_i, Tensor configs_j, "
          "Tensor site, Tensor kind, Tensor coef, "
          "bool configs_i_sorted, bool configs_j_sorted, int direction) -> Tensor");
}

TORCH_LIBRARY_IMPL(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER), CUDA, m) {
    m.impl("apply_within_subspace_in_double_side",
           apply_within_subspace_in_double_side_interface<N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER>);
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif
