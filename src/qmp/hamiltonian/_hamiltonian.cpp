#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace qmp_hamiltonian {

auto prepare(pybind11::dict hamiltonian, std::int64_t max_op_number)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    std::int64_t term_number = hamiltonian.size();

    auto site = torch::empty({term_number, max_op_number},
                             torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    auto kind = torch::full({term_number, max_op_number}, 2,
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto coef = torch::empty({term_number, 2},
                             torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    auto site_accessor = site.accessor<std::int16_t, 2>();
    auto kind_accessor = kind.accessor<std::uint8_t, 2>();
    auto coef_accessor = coef.accessor<double, 2>();

    std::int64_t index = 0;
    for (const auto& item : hamiltonian) {
        auto key = item.first.cast<pybind11::tuple>();
        auto value_is_float = pybind11::isinstance<pybind11::float_>(item.second);
        auto value = value_is_float
            ? std::complex<double>(item.second.cast<double>())
            : item.second.cast<std::complex<double>>();

        std::int64_t op_number = key.size();
        for (std::int64_t i = 0; i < op_number; ++i) {
            auto op = key[i].cast<pybind11::tuple>();
            site_accessor[index][i] = op[0].cast<std::int16_t>();
            kind_accessor[index][i] = op[1].cast<std::uint8_t>();
        }

        coef_accessor[index][0] = value.real();
        coef_accessor[index][1] = value.imag();

        ++index;
    }

    return std::make_tuple(site, kind, coef);
}

PYBIND11_MODULE(qmp_hamiltonian, m) {
    m.def("prepare", &prepare, pybind11::arg("hamiltonian"), pybind11::arg("max_op_number"));
}

}  // namespace qmp_hamiltonian
