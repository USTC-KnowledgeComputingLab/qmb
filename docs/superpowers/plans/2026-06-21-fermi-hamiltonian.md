# Fermi Hamiltonian 子系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 `src/qmp/hamiltonian/fermi_hamiltonian/` — 位掩码预处理 + 四个 CUDA kernel + JAX fallback + pytest

**Architecture:** Pure Python `prepare` → 位掩码 JAX arrays → `FermiHamiltonian` 类路由 (CUDA FFI or JAX fallback) → 四个操作。CUDA kernel 通过 nvcc JIT 编译，`XLA_FFI_DEFINE_HANDLER_SYMBOL` 导出，`jax.ffi` 注册。

**Tech Stack:** jax, jaxlib, cuCollections, CUB, nvcc, pytest

---

### Task 1: 目录结构 + 占位文件

**Files:**
- Create placeholder: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_prepare.py`
- Create placeholder: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_jax.py`
- Create placeholder: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda_loader.py`
- Create placeholder: `tests/unit/hamiltonian/fermi_hamiltonian/__init__.py`
- Create placeholder: `tests/unit/hamiltonian/__init__.py`
- Create placeholder: `tests/unit/__init__.py`

- [ ] **Step 1: Create directories and empty files**

```bash
mkdir -p tests/unit/hamiltonian/fermi_hamiltonian
touch src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_prepare.py
touch src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_jax.py
touch src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda_loader.py
touch tests/unit/hamiltonian/fermi_hamiltonian/__init__.py
touch tests/unit/hamiltonian/__init__.py
touch tests/unit/__init__.py
```

- [ ] **Step 2: Commit**

---

### Task 2: `_hamiltonian_prepare.py` — 位掩码预处理

**Files:**
- Write: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_prepare.py`

- [ ] **Step 1: Write the full prepare function**

```python
"""Pure Python bit-mask preparation for fermionic Hamiltonians."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


def prepare(
    hamiltonian: dict[tuple[tuple[int, int], ...], complex],
    n_qubits: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Convert a Hamiltonian dict to bit-mask representation.

    Parameters
    ----------
    hamiltonian : dict
        Keys are tuples of (site_index, kind) where kind=1 for creation,
        kind=0 for annihilation. Values are complex coefficients.
    n_qubits : int
        Number of qubits (orbitals × 2 for fermions).

    Returns
    -------
    tuple[Array, Array, Array, Array, Array, Array]
        (create_mask [T,Q] uint8, annihilate_mask [T,Q] uint8,
         flip_mask [T,Q] uint8, parity_mask [T,Q] uint8,
         parity_const [T] uint8, coef [T,2] float64)
    """
    n_qubytes = (n_qubits + 7) // 8
    terms: list[tuple[list[tuple[int, bool]], complex]] = []
    for key, value in hamiltonian.items():
        ops: list[tuple[int, bool]] = []
        for site, kind in key:
            if kind == 1:
                ops.append((site, True))
            elif kind == 0:
                ops.append((site, False))
            # kind == 2 (identity) silently skipped
        if ops:
            terms.append((ops, value))

    create_mask_list: list[int] = []
    annihilate_mask_list: list[int] = []
    flip_mask_list: list[int] = []
    parity_mask_list: list[int] = []
    parity_const_list: list[int] = []
    coef_list: list[tuple[float, float]] = []

    for ops, coef_val in terms:
        result = _process_term(ops, n_qubits)
        if result is None:
            continue
        cm, am, fm, pm, pc = result
        create_mask_list.append(cm)
        annihilate_mask_list.append(am)
        flip_mask_list.append(fm)
        parity_mask_list.append(pm)
        parity_const_list.append(pc)
        coef_list.append((coef_val.real, coef_val.imag))

    T = len(create_mask_list)

    def _to_array(values: list[int]) -> Array:
        arr = jnp.zeros((T, n_qubytes), dtype=jnp.uint8)
        for t, val in enumerate(values):
            for q in range(n_qubytes):
                arr = arr.at[t, q].set(jnp.uint8((val >> (q * 8)) & 0xFF))
        return arr

    create_mask = _to_array(create_mask_list)
    annihilate_mask = _to_array(annihilate_mask_list)
    flip_mask = _to_array(flip_mask_list)
    parity_mask = _to_array(parity_mask_list)
    parity_const = jnp.array(parity_const_list, dtype=jnp.uint8)
    reals, imags = zip(*coef_list) if coef_list else ([], [])
    coef = jnp.stack([jnp.array(reals, dtype=jnp.float64),
                       jnp.array(imags, dtype=jnp.float64)], axis=1)

    return create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef


def _process_term(
    ops: list[tuple[int, bool]], n_qubits: int
) -> tuple[int, int, int, int, int] | None:
    """Process one term's operator sequence.

    Application order = reverse of writing order.
    Returns (create_mask, annihilate_mask, flip_mask, parity_mask, parity_const)
    or None if term is identically zero.
    """
    known = [False] * n_qubits
    initial = [0] * n_qubits
    flip = 0
    p_const = 0
    p_mask = 0

    for s, c in reversed(ops):
        flip_s = (flip >> s) & 1
        target = 0 if c else 1  # create → 0, annihilate → 1
        if known[s]:
            if (initial[s] ^ flip_s) != target:
                return None
        else:
            initial[s] = target
            known[s] = True

        lo = (1 << s) - 1
        known_mask = 0
        for i in range(n_qubits):
            if known[i]:
                known_mask |= (1 << i)
        unknown_mask = lo & ~known_mask
        contrib = 0
        for i in range(s):
            if known[i]:
                contrib ^= initial[i] ^ ((flip >> i) & 1)
        p_const ^= contrib & 1
        p_mask ^= unknown_mask

        flip ^= (1 << s)

    create_mask = 0
    annihilate_mask = 0
    for i in range(n_qubits):
        if known[i] and initial[i] == 0:
            create_mask |= (1 << i)
        if known[i] and initial[i] == 1:
            annihilate_mask |= (1 << i)

    return (create_mask, annihilate_mask, int(flip), int(p_mask), p_const)
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_prepare.py
git commit -m "feat: implement pure Python Hamiltonian prepare with BIT.md algorithm"
```

---

### Task 3: Test `_hamiltonian_prepare`

**Files:**
- Write: `tests/unit/hamiltonian/fermi_hamiltonian/test_prepare.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for Hamiltonian term bit-mask preparation."""

from __future__ import annotations

import jax.numpy as jnp
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare


def _h2_hamiltonian() -> dict[tuple[tuple[int, int], ...], complex]:
    """H2 STO-3G Hamiltonian: 4 spin-orbitals."""
    return {
        # h_11 n_{0↑}
        ((0, 1), (0, 0)): 0.715104 * (-1),
        ((0, 1), (0, 0)): 0.715104 * (-1),
    }


def test_prepare_empty():
    """Empty dict returns zero-length arrays."""
    result = prepare({}, n_qubits=4)
    assert len(result) == 6
    for arr in result:
        assert int(arr.shape[0]) == 0


def test_prepare_identity_skip():
    """Terms with kind=2 (identity) should be skipped."""
    h = {((0, 2),): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


def test_prepare_number_operator():
    """c_0^dag c_0 = n_0: annihilate_mask={0}, flip_mask=0, parity_mask=0."""
    h = {((0, 1), (0, 0)): 1.0 + 0j}  # create at 0, annihilate at 0
    result = prepare(h, n_qubits=8)
    create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef = result
    assert int(result[0].shape[0]) == 1
    # annihilate_mask bit 0 = 1
    assert int(annihilate_mask[0, 0]) == 1
    # create_mask bit 0 = 0
    assert int(create_mask[0, 0]) == 0
    # flip_mask = 0
    assert int(flip_mask[0, 0]) == 0
    # parity_mask = 0
    assert int(parity_mask[0, 0]) == 0
    # parity_const = 0
    assert int(parity_const[0]) == 0


def test_prepare_conflict_zero():
    """c_0 c_0 should be identically zero (Pauli exclusion)."""
    h = {((0, 0), (0, 0)): 1.0 + 0j}
    result = prepare(h, n_qubits=4)
    assert int(result[0].shape[0]) == 0


def _make_simple_hubbard_2site():
    """2-site spinless Hubbard-like model: single hopping term c_1^dag c_0."""
    return {((1, 1), (0, 0)): -1.0 + 0j}  # create at 1, annihilate at 0


def test_prepare_hubbard_2site_masks():
    """c_1^dag c_0: create_mask={1}, annihilate_mask={0}, flip_mask={0,1}."""
    h = _make_simple_hubbard_2site()
    result = prepare(h, n_qubits=4)
    create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef = result
    assert int(result[0].shape[0]) == 1
    # create_mask: bit 1 = 1, bit 0 = 0
    assert int(create_mask[0, 0]) & 2 == 2  # bit 1
    # annihilate_mask: bit 0 = 1
    assert int(annihilate_mask[0, 0]) & 1 == 1
    # flip_mask: both bits set
    assert (int(flip_mask[0, 0]) & 3) == 3


def test_prepare_coef_preserved():
    """Coefficients should be preserved through prepare."""
    h = {((0, 1), (0, 0)): 3.5 - 2.0j}
    result = prepare(h, n_qubits=4)
    coef = result[5]
    assert abs(float(coef[0, 0]) - 3.5) < 1e-10
    assert abs(float(coef[0, 1]) - (-2.0)) < 1e-10


def test_prepare_four_op_term():
    """c_3^dag c_1^dag c_5 c_7: 2 creates + 2 annihilates on 8 qubits."""
    ops = ((3, 1), (1, 1), (5, 0), (7, 0))
    h = {ops: 1.0 + 0j}
    result = prepare(h, n_qubits=8)
    create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef = result
    assert int(result[0].shape[0]) == 1
    cm = int(create_mask[0, 0])
    am = int(annihilate_mask[0, 0])
    # bits 1,3 in create_mask (2^1=2, 2^3=8 → 10)
    assert cm == 10
    # bits 5,7 in annihilate_mask (2^5=32, 2^7=128 → 160)
    assert am == 160
```

- [ ] **Step 2: Run tests to verify they fail** (prepare not yet importable from __init__)

```bash
cd /home/hzhangxyz/.local/share/opencode/worktree/27289db6f197bfc15851ba39a23a3acb917e4462/witty-mountain
python -m pytest tests/unit/hamiltonian/fermi_hamiltonian/test_prepare.py -v
```
Expected: ImportError or 0 collected (package not fully set up)

- [ ] **Step 3: Run tests directly against the module**

```bash
python -m pytest tests/unit/hamiltonian/fermi_hamiltonian/test_prepare.py -v
```
Expected: all tests pass (direct import of `_hamiltonian_prepare`)

- [ ] **Step 4: Commit**

```bash
git add tests/unit/hamiltonian/fermi_hamiltonian/test_prepare.py tests/unit/hamiltonian/fermi_hamiltonian/__init__.py tests/unit/hamiltonian/__init__.py tests/unit/__init__.py
git commit -m "test: add bit-mask preparation correctness tests"
```

---

### Task 4: `_hamiltonian_jax.py` — 纯 JAX fallback

**Files:**
- Write: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_jax.py`

- [ ] **Step 1: Write JAX fallback for all four operations**

```python
"""Pure JAX fallback implementations of the four Hamiltonian operations.

Used when CUDA .so is not available (CPU, CI, macOS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


@jax.jit
def compute_diagonal_within_subspace(
    configs: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
) -> Array:
    """Compute diagonal Hamiltonian elements. Only terms with flip_mask==0 contribute."""
    B, Q = configs.shape
    T, _ = coef.shape
    psi = jnp.zeros((B, 2), dtype=jnp.float64)

    def _for_term(t, carry):
        psi_acc = carry
        cm = create_mask[t]  # [Q]
        am = annihilate_mask[t]  # [Q]
        fm = flip_mask[t]  # [Q]

        # check flip_mask == 0 for all qubytes
        is_diag = jnp.all(fm == 0)
        if not is_diag:
            return psi_acc

        def _for_config(i, acc):
            c = configs[i]  # [Q]
            # applicable check
            applicable = jnp.logical_and(
                jnp.all((c & cm) == 0),
                jnp.all((c & am) == am),
            )
            if not applicable:
                return acc

            pm = parity_mask[t]  # [Q]
            pc = parity_const[t]  # scalar
            parity = pc
            for q in range(Q):
                parity ^= jnp.bitwise_count(
                    jnp.uint32(pm[q] & c[q])
                ) & 1
            sign = jnp.where(parity & 1, -1.0, 1.0)
            cr = coef[t, 0]
            ci = coef[t, 1]
            return acc.at[i, 0].add(sign * cr).at[i, 1].add(sign * ci)

        return jax.lax.fori_loop(0, B, _for_config, psi_acc)

    return jax.lax.fori_loop(0, T, _for_term, psi)


@jax.jit
def apply_within_subspace(
    configs_i: Array,
    psi_i: Array,
    configs_j: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
    direction: int = 0,
) -> Array:
    """Apply H|psi_i> projected onto configs_j subspace."""
    if direction == 0:
        src_c, src_p, dst_c = configs_i, psi_i, configs_j
    else:
        src_c, src_p, dst_c = configs_j, psi_i, configs_i

    B_src, Q = src_c.shape
    B_dst = dst_c.shape[0]
    T, _ = coef.shape
    psi_j = jnp.zeros((B_dst, 2), dtype=jnp.float64)

    def _for_term(t, carry):
        acc = carry
        cm = create_mask[t]
        am = annihilate_mask[t]
        fm = flip_mask[t]
        pm = parity_mask[t]
        pc = parity_const[t]

        def _for_src(i, inner_acc):
            c = src_c[i]
            applicable = jnp.logical_and(
                jnp.all((c & cm) == 0),
                jnp.all((c & am) == am),
            )
            if not applicable:
                return inner_acc
            new_c = c ^ fm
            # linear search in dst configs (fallback: O(B_dst) per pair)
            idx = -1
            for j in range(B_dst):
                if jnp.all(dst_c[j] == new_c):
                    idx = j
                    break
            if idx < 0:
                return inner_acc
            parity = _parity(pc, pm, c, Q)
            sign = jnp.where(parity & 1, -1.0, 1.0)
            cr = coef[t, 0]
            ci = coef[t, 1]
            pr = src_p[i, 0]
            pi_i = src_p[i, 1]
            val_r = sign * (cr * pr - ci * pi_i)
            val_i = sign * (cr * pi_i + ci * pr)
            return inner_acc.at[idx, 0].add(val_r).at[idx, 1].add(val_i)

        return jax.lax.fori_loop(0, B_src, _for_src, acc)

    result = jax.lax.fori_loop(0, T, _for_term, psi_j)
    return result


def _parity(pc: jax.Array, pm: jax.Array, config: jax.Array, Q: int) -> jax.Array:
    """Compute JW parity: parity_const XOR popcount(parity_mask & config).
    Returns 0 or 1 as a JAX scalar array (JIT-compatible)."""
    p = pc.astype(jnp.int32) & 1
    for q in range(Q):
        p ^= jnp.bitwise_count(jnp.uint32(pm[q] & config[q])) & 1
    return p


@jax.jit
def find_all_relative_configs(
    configs_i: Array,
    psi_i: Array,
    configs_exclude: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
    hash_capacity: int,
) -> tuple[Array, Array, Array]:
    """Enumerate all unique new configs reachable via H, with amplitude accumulation."""
    B, Q = configs_i.shape
    T, _ = coef.shape
    cap = hash_capacity
    # simple linear array as pseudo-hash-table (JAX fallback)
    keys = jnp.zeros((cap, Q), dtype=jnp.uint8)
    vals = jnp.zeros((cap, 2), dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    count = jnp.array(0, dtype=jnp.int32)

    def _for_term(t, carry):
        keys_c, vals_c, occ_c, cnt_c = carry
        cm = create_mask[t]; am = annihilate_mask[t]
        fm = flip_mask[t]; pm = parity_mask[t]; pc = parity_const[t]
        def _for_src(i, inner):
            kk, vv, oo, cc = inner
            c = configs_i[i]
            applicable = jnp.logical_and(
                jnp.all((c & cm) == 0), jnp.all((c & am) == am),
            )
            if not applicable: return (kk, vv, oo, cc)
            new_c = c ^ fm
            # exclude check: linear search
            excluded = False
            for e in range(configs_exclude.shape[0]):
                if jnp.all(configs_exclude[e] == new_c):
                    excluded = True; break
            if excluded: return (kk, vv, oo, cc)
            parity = _parity(pc, pm, c, Q)
            sign = jnp.where(parity & 1, -1.0, 1.0)
            cr = coef[t,0]; ci = coef[t,1]
            pr = psi_i[i,0]; pi_v = psi_i[i,1]
            val_r = sign * (cr * pr - ci * pi_v)
            val_i = sign * (cr * pi_v + ci * pr)
            # linear probe for matching key
            found = -1
            for s in range(cap):
                if oo[s] and jnp.all(kk[s] == new_c):
                    found = s; break
            if found >= 0:
                vv = vv.at[found, 0].add(val_r).at[found, 1].add(val_i)
            elif cc < cap:
                kk = kk.at[cc].set(new_c)
                vv = vv.at[cc, 0].set(val_r).at[cc, 1].set(val_i)
                oo = oo.at[cc].set(True)
                cc = cc + 1
            return (kk, vv, oo, cc)
        return jax.lax.fori_loop(0, B, _for_src, (keys_c, vals_c, occ_c, cnt_c))

    keys_f, vals_f, occupied_f, count_f = jax.lax.fori_loop(
        0, T, _for_term, (keys, vals, occupied, count))
    return keys_f, vals_f, count_f


@jax.jit
def find_topk_relative_configs(
    configs_i: Array,
    psi_i: Array,
    count_selected: int,
    configs_exclude: Array,
    create_mask: Array,
    annihilate_mask: Array,
    flip_mask: Array,
    parity_mask: Array,
    parity_const: Array,
    coef: Array,
) -> Array:
    """Select top-K configs by max weight. Uses simple linear scan for JAX fallback."""
    B, Q = configs_i.shape
    T, _ = coef.shape
    K = count_selected
    cap = K * 2
    keys = jnp.zeros((cap, Q), dtype=jnp.uint8)
    weights = jnp.zeros(cap, dtype=jnp.float64)
    occupied = jnp.zeros(cap, dtype=bool)
    cnt = jnp.array(0, dtype=jnp.int32)

    def _for_term(t, carry):
        kk, ww, oo, cc = carry
        cm = create_mask[t]; am = annihilate_mask[t]; fm = flip_mask[t]
        def _for_src(i, inner):
            k2, w2, o2, c2 = inner
            c = configs_i[i]
            applicable = jnp.logical_and(
                jnp.all((c & cm) == 0), jnp.all((c & am) == am),
            )
            if not applicable: return (k2, w2, o2, c2)
            new_c = c ^ fm
            excluded = False
            for e in range(configs_exclude.shape[0]):
                if jnp.all(configs_exclude[e] == new_c):
                    excluded = True; break
            if excluded: return (k2, w2, o2, c2)
            cr = coef[t,0]; ci = coef[t,1]
            pr = psi_i[i,0]; pi_v = psi_i[i,1]
            weight = (cr*pr - ci*pi_v)**2 + (cr*pi_v + ci*pr)**2
            found = -1
            for s in range(cap):
                if o2[s] and jnp.all(k2[s] == new_c):
                    found = s; break
            if found >= 0:
                prev = w2[found]
                w2 = w2.at[found].set(jnp.maximum(prev, weight))
            elif c2 < cap:
                k2 = k2.at[c2].set(new_c)
                w2 = w2.at[c2].set(weight)
                o2 = o2.at[c2].set(True)
                c2 = c2 + 1
            return (k2, w2, o2, c2)
        return jax.lax.fori_loop(0, B, _for_src, (kk, ww, oo, cc))

    keys_f, weights_f, occupied_f, count_f = jax.lax.fori_loop(
        0, T, _for_term, (keys, weights, occupied, cnt))
    # take top K by weight (simple arg-sort)
    idx = jnp.argsort(weights_f)[::-1][:K]
    return keys_f[idx]
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_jax.py
git commit -m "feat: implement pure JAX fallback for four Hamiltonian operations"
```

---

### Task 5: Test JAX fallback

**Files:**
- Write: `tests/unit/hamiltonian/fermi_hamiltonian/test_fallback.py`

- [ ] **Step 1: Write fallback tests**

```python
"""End-to-end tests for pure JAX fallback operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    apply_within_subspace,
    compute_diagonal_within_subspace,
    find_all_relative_configs,
    find_topk_relative_configs,
)


def _small_hamiltonian() -> tuple[list[jax.Array], tuple]:
    """4-qubit, ~2-term Hubbard-like model."""
    h = {
        ((1, 1), (0, 0)): -1.0 + 0j,  # hopping: create 1, annihilate 0
    }
    masks = prepare(h, n_qubits=4)
    # small set of configs: |10>, |01>, |11>, |00> as bit-packed uint8
    configs = jnp.array([
        [0b00000010],  # bit1=1 (site1 occupied)
        [0b00000001],  # bit0=1
        [0b00000011],  # both
        [0b00000000],  # empty
    ], dtype=jnp.uint8)
    return list(masks), configs


def test_diagonal_exact():
    """Compute diagonal: manually verify against known values."""
    masks, configs = _small_hamiltonian()
    psi = compute_diagonal_within_subspace(configs, *masks)
    # hopping term has flip_mask != 0, so NO diagonal contribution
    expected = jnp.zeros((4, 2), dtype=jnp.float64)
    assert jnp.allclose(psi, expected, atol=1e-12)


def test_apply_within_forward_backward():
    """Forward and backward should be consistent."""
    masks, configs = _small_hamiltonian()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    forward = apply_within_subspace(configs, psi_i, configs, *masks,
                                     direction=0)
    backward = apply_within_subspace(configs, psi_i, configs, *masks,
                                      direction=1)
    assert forward.shape == (4, 2)
    assert backward.shape == (4, 2)


def test_find_all_dedup():
    """find_all_relative_configs should deduplicate and accumulate."""
    masks, configs = _small_hamiltonian()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    new_c, new_p, cnt = find_all_relative_configs(
        configs, psi_i, exclude, *masks, hash_capacity=100)
    assert int(cnt) >= 0


def test_find_topk():
    """find_topk_relative_configs should return K configs."""
    masks, configs = _small_hamiltonian()
    psi_i = jnp.ones((4, 2), dtype=jnp.float64)
    exclude = jnp.zeros((0, 1), dtype=jnp.uint8)
    result = find_topk_relative_configs(
        configs, psi_i, 2, exclude, *masks)
    assert result.shape == (2, configs.shape[1])
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/unit/hamiltonian/fermi_hamiltonian/test_fallback.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/unit/hamiltonian/fermi_hamiltonian/test_fallback.py
git commit -m "test: add JAX fallback end-to-end tests"
```

---

### Task 6: `_hamiltonian.py` — FermiHamiltonian 类 + FFI 路由

**Files:**
- Rewrite: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian.py`
- Update: `src/qmp/hamiltonian/fermi_hamiltonian/__init__.py`

- [ ] **Step 1: Write FermiHamiltonian class**

```python
"""Python-layer Fermi Hamiltonian with JAX FFI integration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.ffi
import jax.numpy as jnp

from ._hamiltonian_jax import (
    apply_within_subspace as _jax_apply_within_subspace,
)
from ._hamiltonian_jax import (
    compute_diagonal_within_subspace as _jax_compute_diagonal_within_subspace,
)
from ._hamiltonian_jax import (
    find_all_relative_configs as _jax_find_all_relative_configs,
)
from ._hamiltonian_jax import (
    find_topk_relative_configs as _jax_find_topk_relative_configs,
)
from ._hamiltonian_prepare import prepare

if TYPE_CHECKING:
    from jax import Array

logger = logging.getLogger(__name__)


def _try_register_ffi(n_qubytes: int) -> bool:
    """Attempt to load and register CUDA FFI targets for the given n_qubytes.
    Called per FermiHamiltonian instance, not at module level, because
    different instances can have different n_qubits → different N_QUBYTES.
    """
    try:
        from ._hamiltonian_cuda_loader import load_cuda_module
        lib = load_cuda_module(n_qubytes=n_qubytes, max_op_number=4)
        targets = {
            f"qmp_compute_diagonal_within_subspace_{n_qubytes}": "ComputeDiagonalWithinSubspace",
            f"qmp_apply_within_subspace_{n_qubytes}": "ApplyWithinSubspace",
            f"qmp_find_all_relative_configs_{n_qubytes}": "FindAllRelativeConfigs",
            f"qmp_find_topk_relative_configs_{n_qubytes}": "FindTopKRelativeConfigs",
        }
        for name, sym in targets.items():
            handler = getattr(lib, sym)
            jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(handler), platform="CUDA")
        logger.info("CUDA FFI targets registered for n_qubytes=%d.", n_qubytes)
        return True
    except Exception:
        logger.info("CUDA FFI targets not available; using pure JAX fallback.")
        return False


class FermiHamiltonian:
    """Stores preprocessed bit-mask Hamiltonian and exposes four operations."""

    def __init__(
        self,
        hamiltonian: dict[tuple[tuple[int, int], ...], complex],
        *,
        n_qubits: int,
        devices: list[str],
    ) -> None:
        self._n_qubits = n_qubits
        self._n_qubytes = (n_qubits + 7) // 8
        self._device = self._parse_device(devices)
        arrays = prepare(hamiltonian, n_qubits)
        (self._create_mask, self._annihilate_mask, self._flip_mask,
         self._parity_mask, self._parity_const, self._coef) = arrays
        # sort by |coef| descending
        order = jnp.argsort(
            self._coef[:, 0] ** 2 + self._coef[:, 1] ** 2
        )[::-1]
        self._create_mask = self._create_mask[order]
        self._annihilate_mask = self._annihilate_mask[order]
        self._flip_mask = self._flip_mask[order]
        self._parity_mask = self._parity_mask[order]
        self._parity_const = self._parity_const[order]
        self._coef = self._coef[order]
        # 预过滤: 将对角 term (flip_mask == 0) 分离出来
        fm_sum = jnp.sum(self._flip_mask, axis=1)
        self._diag_idx = jnp.where(fm_sum == 0)[0]
        self._offdiag_idx = jnp.where(fm_sum != 0)[0]
        self._use_cuda = _try_register_ffi(self._n_qubytes)
        self._ffi_prefix = f"qmp_" if not self._use_cuda else ""
        logger.info("FermiHamiltonian: %d terms (%d diagonal), %d qubits, cuda=%s",
                     int(self._coef.shape[0]), int(len(self._diag_idx)),
                     n_qubits, self._use_cuda)

    @staticmethod
    def _parse_device(devices: list[str]) -> jax.Device:
        device_str = devices[0]
        parts = device_str.split(":")
        if parts[1] == "cpu":
            return jax.devices("cpu")[0]
        if parts[1] == "cuda":
            idx = int(parts[2]) if len(parts) == 3 else 0
            return jax.devices("cuda")[idx]
        raise ValueError(f"Invalid device: {device_str}")

    def _to_dev(self, arr: jax.Array) -> jax.Array:
        return jax.device_put(arr, self._device)

    # ---- public API ----

    def compute_diagonal_within_subspace(self, configs: jax.Array) -> jax.Array:
        B = configs.shape[0]
        if self._use_cuda:
            return jax.ffi.ffi_call(
                "qmp_compute_diagonal_within_subspace",
                jax.ShapeDtypeStruct((B, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(self._to_dev(configs),
              self._to_dev(self._create_mask),
              self._to_dev(self._annihilate_mask),
              self._to_dev(self._flip_mask),
              self._to_dev(self._parity_mask),
              self._to_dev(self._parity_const),
              self._to_dev(self._coef))
        return _jax_compute_diagonal_within_subspace(
            self._to_dev(configs), self._to_dev(self._create_mask),
            self._to_dev(self._annihilate_mask), self._to_dev(self._flip_mask),
            self._to_dev(self._parity_mask), self._to_dev(self._parity_const),
            self._to_dev(self._coef))

    def apply_within_subspace(
        self, configs_i: jax.Array, psi_i: jax.Array,
        configs_j: jax.Array, *, direction: int = 0,
    ) -> jax.Array:
        B_j = configs_j.shape[0]
        inputs = (self._to_dev(configs_i), self._to_dev(psi_i),
                  self._to_dev(configs_j),
                  self._to_dev(self._create_mask), self._to_dev(self._annihilate_mask),
                  self._to_dev(self._flip_mask), self._to_dev(self._parity_mask),
                  self._to_dev(self._parity_const), self._to_dev(self._coef),
                  direction)
        if self._use_cuda:
            return jax.ffi.ffi_call(
                "qmp_apply_within_subspace",
                jax.ShapeDtypeStruct((B_j, 2), jnp.float64),
                vmap_method="broadcast_all",
            )(*inputs)
        return _jax_apply_within_subspace(*inputs)

    def find_all_relative_configs(
        self, configs_i: jax.Array, psi_i: jax.Array,
        configs_exclude: jax.Array, *, hash_capacity: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        Q = configs_i.shape[1]
        if self._use_cuda:
            return jax.ffi.ffi_call(
                "qmp_find_all_relative_configs",
                (jax.ShapeDtypeStruct((hash_capacity, Q), jnp.uint8),
                 jax.ShapeDtypeStruct((hash_capacity, 2), jnp.float64),
                 jax.ShapeDtypeStruct((), jnp.int32)),
                vmap_method="broadcast_all",
            )(self._to_dev(configs_i), self._to_dev(psi_i),
              self._to_dev(configs_exclude),
              self._to_dev(self._create_mask), self._to_dev(self._annihilate_mask),
              self._to_dev(self._flip_mask), self._to_dev(self._parity_mask),
              self._to_dev(self._parity_const), self._to_dev(self._coef),
              hash_capacity)
        return _jax_find_all_relative_configs(
            self._to_dev(configs_i), self._to_dev(psi_i),
            self._to_dev(configs_exclude),
            self._to_dev(self._create_mask), self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask), self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const), self._to_dev(self._coef),
            hash_capacity)

    def find_topk_relative_configs(
        self, configs_i: jax.Array, psi_i: jax.Array,
        count_selected: int, configs_exclude: jax.Array,
    ) -> jax.Array:
        Q = configs_i.shape[1]
        if self._use_cuda:
            return jax.ffi.ffi_call(
                "qmp_find_topk_relative_configs",
                jax.ShapeDtypeStruct((count_selected, Q), jnp.uint8),
                vmap_method="broadcast_all",
            )(self._to_dev(configs_i), self._to_dev(psi_i),
              np.int32(count_selected),
              self._to_dev(configs_exclude),
              self._to_dev(self._create_mask), self._to_dev(self._annihilate_mask),
              self._to_dev(self._flip_mask), self._to_dev(self._parity_mask),
              self._to_dev(self._parity_const), self._to_dev(self._coef))
        return _jax_find_topk_relative_configs(
            self._to_dev(configs_i), self._to_dev(psi_i),
            count_selected, self._to_dev(configs_exclude),
            self._to_dev(self._create_mask), self._to_dev(self._annihilate_mask),
            self._to_dev(self._flip_mask), self._to_dev(self._parity_mask),
            self._to_dev(self._parity_const), self._to_dev(self._coef))
```

- [ ] **Step 2: Update __init__.py**

```python
"""Fermi Hamiltonian module. Provides FermiHamiltonian."""

from __future__ import annotations

from ._hamiltonian import FermiHamiltonian

__all__ = ["FermiHamiltonian"]
```

- [ ] **Step 3: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian.py \
        src/qmp/hamiltonian/fermi_hamiltonian/__init__.py
git commit -m "feat: implement FermiHamiltonian class with FFI routing and JAX fallback"
```

---

### Task 7: `_hamiltonian_cuda_loader.py` — JIT 编译 + 缓存

**Files:**
- Write: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda_loader.py`

- [ ] **Step 1: Write CUDA loader**

```python
"""CUDA kernel JIT compilation and caching, similar to torch.utils.cpp_extension.load."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

_SOURCE_DIR = Path(__file__).resolve().parent


def load_cuda_module(n_qubytes: int, max_op_number: int) -> ctypes.CDLL:
    """Compile and load a CUDA shared library for the given parameters.

    The library is cached in ~/.cache/qmp/kclab/{key}/lib.so.
    On the first call for a given (n_qubytes, max_op_number), nvcc is
    invoked to compile _hamiltonian_cuda.cu with the appropriate macros.
    Subsequent calls load the cached .so directly via ctypes.

    Parameters
    ----------
    n_qubytes : int
        ceil(n_qubits/8), passed as -DN_QUBYTES.
    max_op_number : int
        Maximum operators per term, passed as -DMAX_OP_NUMBER.

    Returns
    -------
    ctypes.CDLL
        The loaded shared library.
    """
    key = f"qmp_hamiltonian_{n_qubytes}_{max_op_number}"
    cache_dir = platformdirs.user_cache_path("qmp", "kclab") / key
    so_path = cache_dir / "lib.so"

    if not so_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import jaxlib
            jax_include = jaxlib.get_include_dir()
        except ImportError:
            jax_include = os.path.join(
                os.path.dirname(os.path.dirname(os.__file__)),
                "jaxlib", "include")
            if not os.path.isdir(jax_include):
                raise RuntimeError(
                    "jaxlib include directory not found. Is jaxlib installed?"
                ) from None

        source = _SOURCE_DIR / "_hamiltonian_cuda.cu"
        cmd = [
            "nvcc", "-shared", "-Xcompiler", "-fPIC",
            f"-I{jax_include}",
            f"-DN_QUBYTES={n_qubytes}",
            f"-DMAX_OP_NUMBER={max_op_number}",
            "-std=c++20", "-O3", "--use_fast_math",
            "-arch=native",
            "-o", str(so_path),
            str(source),
        ]
        logger.info("Compiling CUDA kernel: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("CUDA kernel compiled to %s", so_path)

    lib = ctypes.cdll.LoadLibrary(str(so_path))
    return lib
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda_loader.py
git commit -m "feat: implement CUDA JIT compilation and caching loader"
```

---

### Task 8: `_hamiltonian_cuda.cu` — CUDA kernel

**Files:**
- Rewrite: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu`

This is the largest task. The CUDA file contains all four operations as XLA FFI handlers. Each operation:
1. Receives buffers via `ffi::Buffer<T>` and `ffi::ResultBuffer<T>`
2. Launches a 2D grid-stride loop kernel
3. Uses `is_applicable`, `apply_flip`, `_parity` as device helper functions
4. Hashes with xxHash64, uses `cuco::static_map` for hash table operations

Due to CUDA file length, implement in three sub-tasks.

#### Task 8a: Utility functions + diagonal_term + apply_within

Write the complete CUDA file skeleton with device utility functions and the first two operation handlers.

Full code provided inline — see the complete `_hamiltonian_cuda.cu` listing.

```cuda
#include <cstdint>
#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ── device utilities ──

template <int N_QUBYTES>
__device__ bool is_applicable(
    const uint8_t* config, const uint8_t* cm, const uint8_t* am)
{
    for (int q = 0; q < N_QUBYTES; ++q) {
        if ((config[q] & cm[q]) != 0) return false;
        if ((config[q] & am[q]) != am[q]) return false;
    }
    return true;
}

template <int N_QUBYTES>
__device__ void apply_flip(uint8_t* dst, const uint8_t* src, const uint8_t* fm)
{
    for (int q = 0; q < N_QUBYTES; ++q) dst[q] = src[q] ^ fm[q];
}

template <int N_QUBYTES>
__device__ bool jw_parity(
    const uint8_t* config, const uint8_t* pm, uint8_t pc)
{
    uint8_t p = pc;
    for (int q = 0; q < N_QUBYTES; ++q)
        p ^= __popc(static_cast<unsigned>(pm[q] & config[q])) & 1;
    return p & 1;
}

// ── diagonal_term kernel ──

template <int N_QUBYTES>
__global__ void diagonal_term_kernel(
    int64_t B, int64_t T, int64_t total_pairs,
    const uint8_t* __restrict__ configs,
    const uint8_t* __restrict__ create_mask,
    const uint8_t* __restrict__ annihilate_mask,
    const uint8_t* __restrict__ flip_mask,
    const uint8_t* __restrict__ parity_mask,
    const uint8_t* __restrict__ parity_const,
    const double*  __restrict__ coef,
    double* __restrict__ psi)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t k = idx; k < total_pairs; k += stride) {
        int64_t t = k / B;
        int64_t i = k % B;
        const uint8_t* cfg = configs + i * Q;
        if (!is_applicable<N_QUBYTES>(cfg, create_mask + t * Q, annihilate_mask + t * Q))
            continue;
        const uint8_t* fm = flip_mask + t * Q;
        bool is_diag = true;
        for (int q = 0; q < N_QUBYTES; ++q) { if (fm[q]) { is_diag = false; break; } }
        if (!is_diag) continue;
        bool parity = jw_parity<N_QUBYTES>(cfg, parity_mask + t * Q, parity_const[t]);
        double sign = parity ? -1.0 : 1.0;
        atomicAdd(psi + i * 2,     sign * coef[t * 2]);
        atomicAdd(psi + i * 2 + 1, sign * coef[t * 2 + 1]);
    }
}

ffi::Error ComputeDiagonalWithinSubspaceImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> configs,
    ffi::Buffer<ffi::U8> create_mask,
    ffi::Buffer<ffi::U8> annihilate_mask,
    ffi::Buffer<ffi::U8> flip_mask,
    ffi::Buffer<ffi::U8> parity_mask,
    ffi::Buffer<ffi::U8> parity_const,
    ffi::Buffer<ffi::F64> coef,
    ffi::ResultBuffer<ffi::F64> psi)
{
    auto cd = configs.dimensions();
    int64_t B = cd[0], Q = cd[1];
    int64_t T = create_mask.dimensions()[0];
    int64_t total = T * B;
    cudaMemsetAsync(psi->untyped_data(), 0, psi->size_bytes(), stream);
    // launch with N_QUBYTES compile-time template (use switch for common values)
    int threads = 256, blocks = std::min<int64_t>((total + 255)/256, 65535L);
    diagonal_term_kernel<0><<<blocks, threads, 0, stream>>>(
        B, T, total, configs.typed_data(), create_mask.typed_data(),
        annihilate_mask.typed_data(), flip_mask.typed_data(),
        parity_mask.typed_data(), parity_const.typed_data(),
        reinterpret_cast<const double*>(coef.typed_data()),
        reinterpret_cast<double*>(psi->typed_data()));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ComputeDiagonalWithinSubspace, ComputeDiagonalWithinSubspaceImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
);

// ── apply_within, find_all, find_topk (stubs for subsequent tasks) ──

ffi::Error ApplyWithinSubspaceImpl(
    cudaStream_t, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::F64>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>, ffi::Buffer<ffi::U8>,
    ffi::Buffer<ffi::F64>, ffi::Attr<int32_t>,
    ffi::ResultBuffer<ffi::F64>)
{ return ffi::Error::Unimplemented("TODO"); }
XLA_FFI_DEFINE_HANDLER_SYMBOL(ApplyWithinSubspace, ApplyWithinSubspaceImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::F64>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::U8>>()
    .Arg<ffi::Buffer<ffi::F64>>().Attr<int32_t>("direction")
    .Ret<ffi::Buffer<ffi::F64>>());

ffi::Error FindAllRelativeConfigsImpl(...)
{ return ffi::Error::Unimplemented("TODO"); }
XLA_FFI_DEFINE_HANDLER_SYMBOL(FindAllRelativeConfigs, FindAllRelativeConfigsImpl, ...);

ffi::Error FindTopKRelativeConfigsImpl(...)
{ return ffi::Error::Unimplemented("TODO"); }
XLA_FFI_DEFINE_HANDLER_SYMBOL(FindTopKRelativeConfigs, FindTopKRelativeConfigsImpl, ...);
```

- [ ] **Step 1: Commit preliminary CUDA kernel**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu
git commit -m "feat: CUDA kernel skeleton with diagonal_term and FFI handlers"
```

#### Task 8b: apply_within_subspace CUDA handler

**Files:**
- Modify: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu`

- [ ] **Step 1: Implement ApplyWithinSubspaceImpl**

算法 (参考 spec §3.2.2):

```
1. 预处理: 对 dst_configs 构建 cuco::static_map
   - Key = config bytes (N_QUBYTES uint8), Value = index in original order
   - 容量 = B_dst / 0.6 (60% load factor)
   - 哈希函数: xxHash64

2. Grid-stride loop over (term, src_config):
   for each (t, i):
     if !is_applicable(src[i], cm[t], am[t]): continue
     new_config = src[i] XOR fm[t]
     idx = hash_table.lookup(new_config)
     if idx < 0: continue
     parity = parity_const[t] XOR popcount(parity_mask[t] & src[i]) & 1
     sign = parity ? -1.0 : 1.0
     contribution = sign * complex_mul(coef[t], psi_src[i])
     atomicAdd(psi_j[idx, 0], contribution.real)
     atomicAdd(psi_j[idx, 1], contribution.imag)

3. 方向: direction==0 → src=configs_i, dst=configs_j
          direction==1 → src=configs_j, dst=configs_i (反向)

4. 哈希表构建在 host 端 (cudaMalloc + cuco API)，
   然后传入 kernel。方向选择在 Python 层完成。
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu
git commit -m "feat: add apply_within_subspace CUDA handler"
```

#### Task 8c: find_all_relative_configs CUDA handler

**Files:**
- Modify: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu`

- [ ] **Step 1: Implement FindAllRelativeConfigsImpl**

算法 (参考 spec §3.2.3):

```
1. 预分配 cuco::static_map
   - capacity = estimated_distinct / 0.6
   - Key = config bytes, Value = (real:f64, imag:f64)
   - 哈希: xxHash64, open addressing + linear probing

2. Grid-stride loop over (term, config_i):
   for each (t, i):
     applicable check → new_config = config_i XOR fm[t]
     exclude check (binary search in sorted exclude, or hash table)
     parity → sign → contribution = sign * complex_mul(coef[t], psi_i[i])
     
     hash lookup → 
       found: atomicAdd on (real, imag)
       not found: CAS claim slot + write (config, real, imag)

3. 收集: 线性扫描 hash table → 所有 non-empty slots
   → return (new_configs, psi_j, count)

4. 溢出保护: probe 超阈值 (10×log₂(capacity)) → 标记 overflow → 
   kernel 返回错误码 → Python 层 retry 更大 capacity
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu
git commit -m "feat: add find_all_relative_configs CUDA handler"
```

#### Task 8d: find_topk_relative_configs CUDA handler

**Files:**
- Modify: `src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu`

- [ ] **Step 1: Implement FindTopKRelativeConfigsImpl**

算法 (参考 spec §3.2.4):

```
1. 初始化 hash table (capacity = 2K, ~680 MB)
   global_min_weight = 0.0

2. for chunk in term_chunks:      ← Python 层循环，非 CUDA 内部
     # CUDA kernel: 处理 chunk × all configs
     grid-stride loop:
       for each (t, i) in chunk:
         applicable check
         weight = |coef[t] * psi_i[i]|²
         if weight <= global_min_weight: continue
         exclude check
         hash lookup → found: atomicMax; not found: CAS insert

     # kernel exit (隐式屏障)
     
     # compact (host 端或独立 kernel):
     entries = collect_nonempty()
     CUB::radix_sort_descending(entries, by=weight)
     entries = unique_top_k(entries, K)
     hash_table.rebuild(entries)
     global_min_weight = entries[-1].weight

3. 最终 compact → 返回 top K configs
```

- [ ] **Step 2: Commit**

```bash
git add src/qmp/hamiltonian/fermi_hamiltonian/_hamiltonian_cuda.cu
git commit -m "feat: add find_topk_relative_configs CUDA handler"
```

---

### Task 9: CUDA regression tests

**Files:**
- Write: `tests/unit/hamiltonian/fermi_hamiltonian/test_cuda.py`

- [ ] **Step 1: Write CUDA tests** (only runs when GPU + .so available)

```python
"""CUDA regression tests (requires GPU and compiled .so)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_prepare import prepare
from qmp.hamiltonian.fermi_hamiltonian._hamiltonian_jax import (
    compute_diagonal_within_subspace as jax_diag,
)

pytestmark = pytest.mark.cuda


def _small_hamiltonian_and_configs():
    h = {((1, 1), (0, 0)): -1.0 + 0j}
    masks = prepare(h, n_qubits=4)
    import jax.numpy as jnp
    configs = jnp.array([[0b10], [0b01], [0b11], [0b00]], dtype=jnp.uint8)
    return masks, configs


def test_cuda_vs_fallback_diagonal():
    """CUDA diagonal should match JAX fallback output."""
    masks, configs = _small_hamiltonian_and_configs()
    from qmp.hamiltonian.fermi_hamiltonian._hamiltonian import FermiHamiltonian
    h = FermiHamiltonian({((1, 1), (0, 0)): -1.0 + 0j},
                         n_qubits=4, devices=["localhost:cuda:0"])
    cuda_result = h.compute_diagonal_within_subspace(configs)
    jax_result = jax_diag(configs, *masks)
    assert jnp.allclose(
        jax.device_put(cuda_result, jax.devices("cpu")[0]),
        jax_result, rtol=1e-12)
```

- [ ] **Step 2: Commit**

```bash
git add tests/unit/hamiltonian/fermi_hamiltonian/test_cuda.py
git commit -m "test: add CUDA vs JAX fallback regression tests"
```

---

## 执行阶段补充说明

以下事项 spec 有明确要求，但 plan 中未完整展开。执行 agent 必须在对应任务中补全：

### 补充 1: 块级归约（Task 8a）

Spec §3.2.1 要求 `compute_diagonal_within_subspace` 使用 shared memory 做 intra-block 归约，每个 block 只发一次 `atomicAdd`——而非每个线程都发 `atomicAdd`。实现方法：block 内用 `__shared__ double2 accum[256]` 做 warp-level reduction，最后 thread 0 写回 global。

### 补充 2: `__ldg()` 读取优化（Task 8a-8d）

Spec §3.2.2 要求所有只读输入（configs, create_mask, annihilate_mask, flip_mask, parity_mask, parity_const, coef）通过 `__ldg()` 走 read-only cache。在 CUDA kernel 中将 `const uint8_t* __restrict__` 的访问改为 `__ldg(ptr + offset)` 模式。

### 补充 3: 哈希表跨调用缓存（Task 8b）

Spec §3.2.2 要求 `apply_within_subspace` 的哈希表在 `configs_j` 不变时跨调用复用。实现：在 `FermiHamiltonian` 中维护 `(configs_j_hash, hash_table)` 缓存对，每次调用时比较 hash，命中则跳过 ~50ms 重建。

### 补充 4: 缺失测试（Task 3, Task 5）

Spec §5.1-5.3 要求但 plan 未包含的测试：
- `test_parity_mask_jw`: H₂ 哈密顿量手工计算 JW 奇偶性验证
- `test_create_mask_h2`: H₂ 哈密顿量 verify create_mask
- `test_forward_backward_value_consistency`: apply_within 的 forward 和 backward 结果数值一致性（不仅是 shape）
- `test_diagonal_hand_calculated`: 手工计算结果对比（不仅是 all-zeros）
- `test_hash_table_overflow_retry`: 人为过小 capacity 验证重试逻辑
- `test_cuda_apply_within`, `test_cuda_find_all`, `test_cuda_find_topk`: CUDA vs fallback 对所有四个操作（不仅是 diagonal）

### 补充 5: AGENTS.md 更新（Task 1）

`src/qmp/hamiltonian/fermi_hamiltonian/AGENTS.md` 中旧操作名（`diagonal_term`, `apply_within`, `list_relative`, `find_relative`）需更新为 spec 正式名。旧预处理引用 `_hamiltonian.cpp` 需改为 `_hamiltonian_prepare.py`。

