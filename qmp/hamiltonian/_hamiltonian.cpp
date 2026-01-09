#include <pybind11/complex.h>
#include <torch/extension.h>

namespace qmp_hamiltonian {

// `prepare` 函数负责解析代表哈密顿量项的原始 Python 字典，
// 并将其转换为结构化的张量元组。该元组随后存储在 Python 端，
// 并用于后续对 PyTorch 算子的调用中。
//
// 该函数接收一个 Python 字典 `hamiltonian` 作为输入，其中每个键值对代表哈密顿量中的一项。
// 键是一个由元组组成的元组，其中每个内部元组包含两个元素：
// - 第一个元素是一个整数，代表算符的格点索引（site index）。
// - 第二个元素是一个整数，代表算符的类型（0 为湮灭算符，1 为产生算符）。
// 值是浮点实数或浮点复数，代表该项的系数。
//
// 该函数处理字典并构建三个张量：
// - `site`: 一个形状为 [term_number, max_op_number] 的 int16 张量，代表每项中算符的格点索引。
// - `kind`: 一个形状为 [term_number, max_op_number] 的 uint8 张量，代表每项中算符的类型。
//   值编码如下：
//   - 0: 湮灭算符
//   - 1: 产生算符
//   - 2: 空（单位算符）
// - `coef`: 一个形状为 [term_number, 2] 的 float64 张量，代表每项的系数，包含实部和虚部两个元素。
//
// `max_op_number` 模板参数指定了每项中算符的最大数量，对于二体相互作用通常设置为 4。
template<std::int64_t max_op_number>
auto prepare(py::dict hamiltonian) {
    std::int64_t term_number = hamiltonian.size();

    auto site = torch::empty({term_number, max_op_number}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    // 无需初始化
    auto kind = torch::full({term_number, max_op_number}, 2, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    // 默认初始化为 2（单位算符）
    auto coef = torch::empty({term_number, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    // 无需初始化

    auto site_accessor = site.accessor<std::int16_t, 2>();
    auto kind_accessor = kind.accessor<std::uint8_t, 2>();
    auto coef_accessor = coef.accessor<double, 2>();

    std::int64_t index = 0;
    for (auto& item : hamiltonian) {
        auto key = item.first.cast<py::tuple>();
        auto value_is_float = py::isinstance<py::float_>(item.second);
        auto value = value_is_float ? std::complex<double>(item.second.cast<double>()) : item.second.cast<std::complex<double>>();

        std::int64_t op_number = key.size();
        for (std::int64_t i = 0; i < op_number; ++i) {
            auto tuple = key[i].cast<py::tuple>();
            site_accessor[index][i] = tuple[0].cast<std::int16_t>();
            kind_accessor[index][i] = tuple[1].cast<std::uint8_t>();
        }

        coef_accessor[index][0] = value.real();
        coef_accessor[index][1] = value.imag();

        ++index;
    }

    return std::make_tuple(site, kind, coef);
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES == 0
// 将 `prepare` 函数暴露给 Python。
PYBIND11_MODULE(qmp_hamiltonian, m) {
    m.def("prepare", prepare</*max_op_number=*/4>, py::arg("hamiltonian"));
}
#endif

#if N_QUBYTES != 0
#define QMP_LIBRARY_HELPER(x, y) qmp_hamiltonian_##x##_##y
#define QMP_LIBRARY(x, y) QMP_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_FRAGMENT(QMP_LIBRARY(N_QUBYTES, PARTICLE_CUT), m) {
    m.def("apply_within(Tensor configs_i, Tensor psi_i, Tensor configs_j, Tensor site, Tensor kind, Tensor coef) -> Tensor");
    m.def("find_relative(Tensor configs_i, Tensor psi_i, int count_selected, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> Tensor"
    );
    m.def("list_relative(Tensor configs_i, Tensor psi_i, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> (Tensor, Tensor)");
    m.def("diagonal_term(Tensor configs, Tensor site, Tensor kind, Tensor coef) -> Tensor");
}
#undef QMP_LIBRARY
#undef QMP_LIBRARY_HELPER
#endif

} // namespace qmp_hamiltonian
