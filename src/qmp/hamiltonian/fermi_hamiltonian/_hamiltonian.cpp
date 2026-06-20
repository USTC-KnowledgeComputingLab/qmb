#include <bitset>
#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace qmp_hamiltonian {

namespace {

constexpr std::int64_t kMaxOpNumber = 4;  // 最大算符数 (二体相互作用为 4)

std::uint32_t popcount(std::uint32_t x) {
    return __builtin_popcount(x);
}

std::uint64_t low_mask(std::int64_t idx) {
    if (idx <= 0) return 0;
    return (static_cast<std::uint64_t>(1) << idx) - 1;
}

/// 给定产生算符列表和湮灭算符列表，计算一个 term 的 (a, b, t, p, s, coef) 表示。
///
/// 算法来自 BIT.md。假设产生和湮灭算符已按正规序给出 (产生在左、湮灭在右)。
/// config 的位宽为 N (总轨道数)。
///
/// 返回: (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const)
/// 若条件冲突 (项恒为零) 则返回 None。
struct TermParams {
    std::uint64_t create_mask;
    std::uint64_t annihilate_mask;
    std::uint64_t flip_mask;
    std::uint64_t parity_mask;
    std::uint8_t parity_const;
};

std::optional<TermParams> compute_term_64(
    const std::vector<std::int64_t>& creators,
    const std::vector<std::int64_t>& annihilators,
    std::int64_t n_qubits)
{
    // 构造作用序列: 先所有湮灭算符（反序），再所有产生算符（反序）
    // 下标: (idx, is_creation)
    std::vector<std::pair<std::int64_t, bool>> ops;
    for (auto it = annihilators.rbegin(); it != annihilators.rend(); ++it) {
        ops.emplace_back(*it, false);
    }
    for (auto it = creators.rbegin(); it != creators.rend(); ++it) {
        ops.emplace_back(*it, true);
    }

    // 条件: cond[i] = -1 (未设), 0 (须为 0), 1 (须为 1)
    std::vector<std::int8_t> cond(n_qubits, -1);
    std::uint64_t flip = 0;
    std::uint8_t const_s = 0;
    std::uint64_t p_mask = 0;

    for (const auto& [idx, is_cre] : ops) {
        // 1. 作用条件 → 初始构型的约束
        bool flip_bit = (flip >> idx) & 1;
        std::uint8_t required_init = flip_bit ^ (is_cre ? 0 : 1);  // 产生要求 0, 湮灭要求 1

        if (cond[idx] == -1) {
            cond[idx] = required_init;
        } else if (cond[idx] != required_init) {
            return std::nullopt;  // 条件冲突，项恒为零
        }

        // 2. JW 奇偶性常数部分
        std::uint64_t lm = low_mask(idx);
        const_s ^= (popcount(static_cast<std::uint32_t>(flip & lm)) & 1);

        // 3. 翻转
        flip ^= (static_cast<std::uint64_t>(1) << idx);

        // 4. p_mask 累加
        p_mask ^= lm;
    }

    // 构造 create_mask, annihilate_mask
    std::uint64_t create_mask = 0;
    std::uint64_t annihilate_mask = 0;
    for (std::int64_t i = 0; i < n_qubits; ++i) {
        if (cond[i] == 0) {
            create_mask |= (static_cast<std::uint64_t>(1) << i);
        } else if (cond[i] == 1) {
            annihilate_mask |= (static_cast<std::uint64_t>(1) << i);
        }
    }

    return TermParams{
        .create_mask = create_mask,
        .annihilate_mask = annihilate_mask,
        .flip_mask = flip,
        .parity_mask = p_mask,
        .parity_const = const_s,
    };
}

}  // namespace

/// Python 可调用: prepare(hamiltonian, n_qubits)
///
/// 输入 hamiltonian: dict[tuple[tuple[int,int],...], complex]
///   键: 有序的算符序列。每个算符为 (site_index, 0=湮灭/1=产生)
///   值: 复数系数
///
/// 输出: (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef)
///   全部为 numpy arrays。
///
/// create_mask 等的形状为 [T, Q]，Q = ceil(n_qubits/8)。
/// 对于 n_qubits > 64 的情况，使用多字节表示 (需要多次 compute_term_64)。
auto prepare(const py::dict& hamiltonian, std::int64_t n_qubits) -> py::tuple {
    std::int64_t term_number = hamiltonian.size();
    std::int64_t n_qubytes = (n_qubits + 7) / 8;

    // 分配输出 numpy arrays
    auto create_mask_arr = py::array_t<std::uint8_t>({term_number, n_qubytes});
    auto annihilate_mask_arr = py::array_t<std::uint8_t>({term_number, n_qubytes});
    auto flip_mask_arr = py::array_t<std::uint8_t>({term_number, n_qubytes});
    auto parity_mask_arr = py::array_t<std::uint8_t>({term_number, n_qubytes});
    auto parity_const_arr = py::array_t<std::uint8_t>({term_number});
    auto coef_arr = py::array_t<double>({term_number, 2});

    auto create_mask_buf = create_mask_arr.mutable_unchecked<2>();
    auto annihilate_mask_buf = annihilate_mask_arr.mutable_unchecked<2>();
    auto flip_mask_buf = flip_mask_arr.mutable_unchecked<2>();
    auto parity_mask_buf = parity_mask_arr.mutable_unchecked<2>();
    auto parity_const_buf = parity_const_arr.mutable_unchecked<1>();
    auto coef_buf = coef_arr.mutable_unchecked<2>();

    // 清零
    for (std::int64_t t = 0; t < term_number; ++t) {
        for (std::int64_t q = 0; q < n_qubytes; ++q) {
            create_mask_buf(t, q) = 0;
            annihilate_mask_buf(t, q) = 0;
            flip_mask_buf(t, q) = 0;
            parity_mask_buf(t, q) = 0;
        }
        parity_const_buf(t) = 0;
        coef_buf(t, 0) = 0.0;
        coef_buf(t, 1) = 0.0;
    }

    std::int64_t term_idx = 0;
    for (const auto& item : hamiltonian) {
        auto key = item.first.cast<py::tuple>();
        auto value = item.second.cast<std::complex<double>>();

        // 按产生/湮灭分类算符
        std::vector<std::int64_t> creators;
        std::vector<std::int64_t> annihilators;
        std::int64_t op_number = key.size();
        for (std::int64_t i = 0; i < op_number; ++i) {
            auto op = key[i].cast<py::tuple>();
            std::int64_t site = op[0].cast<std::int64_t>();
            std::uint8_t kind = op[1].cast<std::uint8_t>();
            if (kind == 1) {
                creators.push_back(site);    // kind=1 为产生
            } else if (kind == 0) {
                annihilators.push_back(site);  // kind=0 为湮灭
            }
            // kind=2 (identity) 忽略
        }

        // 调用 BIT.md 算法
        auto params = compute_term_64(creators, annihilators, n_qubits);
        if (!params.has_value()) {
            continue;  // 项恒为零，跳过
        }

        // 将 64-bit 掩码写入 uint8 数组 (little-endian, 低字节优先)
        for (std::int64_t q = 0; q < n_qubytes; ++q) {
            create_mask_buf(term_idx, q) = static_cast<std::uint8_t>((params->create_mask >> (q * 8)) & 0xFF);
            annihilate_mask_buf(term_idx, q) = static_cast<std::uint8_t>((params->annihilate_mask >> (q * 8)) & 0xFF);
            flip_mask_buf(term_idx, q) = static_cast<std::uint8_t>((params->flip_mask >> (q * 8)) & 0xFF);
            parity_mask_buf(term_idx, q) = static_cast<std::uint8_t>((params->parity_mask >> (q * 8)) & 0xFF);
        }
        parity_const_buf(term_idx) = params->parity_const;
        coef_buf(term_idx, 0) = value.real();
        coef_buf(term_idx, 1) = value.imag();

        ++term_idx;
    }

    // 截断到实际有效 term 数
    auto slice = [](auto& arr, std::int64_t count) -> py::array {
        return arr[py::slice(0, count, 1)];
    };

    return py::make_tuple(
        slice(create_mask_arr, term_idx),
        slice(annihilate_mask_arr, term_idx),
        slice(flip_mask_arr, term_idx),
        slice(parity_mask_arr, term_idx),
        slice(parity_const_arr, term_idx),
        slice(coef_arr, term_idx));
}

}  // namespace qmp_hamiltonian

PYBIND11_MODULE(_hamiltonian, m) {
    m.doc() = "Hamiltonian term preprocessing";
    m.def("prepare", &qmp_hamiltonian::prepare,
          py::arg("hamiltonian"), py::arg("n_qubits"),
          "Convert a Hamiltonian dictionary to bit-mask representation.");
}
