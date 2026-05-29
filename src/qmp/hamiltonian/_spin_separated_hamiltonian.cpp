#include <array>
#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace qmp_spin_separated_hamiltonian {

auto prepare(pybind11::dict hamiltonian, std::int64_t n_up, std::int64_t max_op_number)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    std::vector<std::tuple<std::vector<std::int16_t>, std::vector<std::uint8_t>,
                           std::array<double, 2>>> terms;

    for (auto item : hamiltonian) {
        auto key_tuple = item.first.cast<pybind11::tuple>();
        auto value_is_float = pybind11::isinstance<pybind11::float_>(item.second);
        auto value = value_is_float
            ? std::complex<double>(item.second.cast<double>())
            : item.second.cast<std::complex<double>>();

        std::vector<std::int16_t> sites;
        std::vector<std::uint8_t> kinds;
        for (auto op_item : key_tuple) {
            auto op = op_item.cast<pybind11::tuple>();
            std::int64_t interleaved_site = op[0].cast<std::int64_t>();
            std::uint8_t kind = static_cast<std::uint8_t>(op[1].cast<std::int64_t>());

            // Remap from interleaved to block order:
            //   even sites → up block (indices 0..n_up-1)
            //   odd sites  → down block (indices n_up..n_total-1)
            std::int64_t block_site;
            if (interleaved_site % 2 == 0) {
                block_site = interleaved_site / 2;
            } else {
                block_site = n_up + interleaved_site / 2;
            }
            sites.push_back(static_cast<std::int16_t>(block_site));
            kinds.push_back(kind);
        }

        terms.emplace_back(std::move(sites), std::move(kinds),
                           std::array<double, 2>{{value.real(), value.imag()}});
    }

    auto term_number = static_cast<std::int64_t>(terms.size());
    auto site = torch::empty({term_number, max_op_number},
                             torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    auto kind = torch::full({term_number, max_op_number}, 2,
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto coef = torch::empty({term_number, 2},
                             torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto site_accessor = site.accessor<std::int16_t, 2>();
    auto kind_accessor = kind.accessor<std::uint8_t, 2>();
    auto coef_accessor = coef.accessor<double, 2>();

    for (std::int64_t i = 0; i < term_number; ++i) {
        const auto& [s, k, c] = terms[i];
        for (std::size_t j = 0; j < s.size(); ++j) {
            site_accessor[i][j] = s[j];
            kind_accessor[i][j] = k[j];
        }
        coef_accessor[i][0] = c[0];
        coef_accessor[i][1] = c[1];
    }

    return std::make_tuple(site, kind, coef);
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif
#ifndef MAX_OP_NUMBER
#define MAX_OP_NUMBER 0
#endif
#ifndef N_UP
#define N_UP 0
#endif

#define QMP_LIBRARY_HELPER(nq, pc, mo, nu) qmp_spin_separated_hamiltonian_##nq##_##pc##_##mo##_##nu
#define QMP_LIBRARY(nq, pc, mo, nu) QMP_LIBRARY_HELPER(nq, pc, mo, nu)

#if N_QUBYTES == 0
PYBIND11_MODULE(qmp_spin_separated_hamiltonian, m) {
    m.def("prepare", &prepare, pybind11::arg("hamiltonian"), pybind11::arg("n_up"), pybind11::arg("max_op_number"));
}
#else
TORCH_LIBRARY_FRAGMENT(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT, MAX_OP_NUMBER, N_UP), m) {
    m.def("spin_separated_apply_within_subspace_in_double_side("
          "Tensor configs_i, Tensor psi_i, Tensor configs_j, "
          "Tensor site, Tensor kind, Tensor coef, "
          "bool configs_i_sorted, bool configs_j_sorted, int direction) -> Tensor");
}
#endif

#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER

}  // namespace qmp_spin_separated_hamiltonian
