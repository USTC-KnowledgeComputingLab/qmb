"""
This file implements a trivial Matrix Product State (MPS) network.
"""

import torch
from ..utility.bitspack import pack_int, unpack_int


class WaveFunction(torch.nn.Module):
    """
    A trivial Matrix Product State (MPS) wave function.

    The MPS tensors have shape (physical_dim, virtual_dim, virtual_dim) for each site,
    with open boundary conditions: the left boundary vector is e_0 = [1, 0, ..., 0]
    and the right boundary vector is e_0 = [1, 0, ..., 0].

    The amplitude is computed as:
        Ψ(s) = e_L^T @ A[0][s_0] @ A[1][s_1] @ ... @ A[L-1][s_{L-1}] @ e_R
    where e_L = e_R = [1, 0, ..., 0] (first standard basis vector).
    """

    def __init__(
        self,
        *,
        sites: int,  # Number of sites in the MPS
        physical_dim: int,  # Dimension of the physical (local) Hilbert space at each site
        virtual_dim: int,  # Bond dimension (virtual dimension) of the MPS
    ) -> None:
        super().__init__()
        self.sites: int = sites
        self.physical_dim: int = physical_dim
        self.virtual_dim: int = virtual_dim

        # MPS tensors: (sites, physical_dim, virtual_dim, virtual_dim)
        # All sites use the same tensor shape; open boundary conditions are imposed
        # by contracting with the first standard basis vector e_0 on both ends.
        self.tensors: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(sites, physical_dim, virtual_dim, virtual_dim)
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _bit_size(self) -> int:
        """
        Return the number of bits needed to encode a single physical state.
        """
        if self.physical_dim <= 1 << 1:
            return 1
        if self.physical_dim <= 1 << 2:
            return 2
        if self.physical_dim <= 1 << 4:
            return 4
        if self.physical_dim <= 1 << 8:
            return 8
        raise ValueError("physical_dim should be less than or equal to 256")

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function amplitude for the given configurations.

        Parameters
        ----------
        x : torch.Tensor
            Packed configurations as a 2D uint8 tensor of shape (batch_size, packed_sites).

        Returns
        -------
        torch.Tensor
            Complex128 amplitudes of shape (batch_size,).
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        # x_unpacked: batch_size * sites, each value in [0, physical_dim)
        x_unpacked: torch.Tensor = unpack_int(x, size=self._bit_size(), last_dim=self.sites).view(
            [batch_size, self.sites]
        )

        tensors: torch.Tensor = self.tensors.to(dtype=dtype)

        # v: batch_size * virtual_dim  -- left environment vector, starting from e_L = [1, 0, ..., 0]
        v: torch.Tensor = torch.zeros([batch_size, self.virtual_dim], device=device, dtype=dtype)
        v[:, 0] = 1.0

        for i in range(self.sites):
            # A_i: physical_dim * virtual_dim * virtual_dim
            A_i: torch.Tensor = tensors[i]
            # s_i: batch_size, integer indices for the physical state at site i
            s_i: torch.Tensor = x_unpacked[:, i].to(dtype=torch.int64)
            # A_s_i: batch_size * virtual_dim * virtual_dim
            A_s_i: torch.Tensor = A_i[s_i]
            # v = v @ A_s_i: (batch_size, virtual_dim)
            v = torch.bmm(v.unsqueeze(1), A_s_i).squeeze(1)

        # Project onto right boundary e_R = [1, 0, ..., 0]: extract the first element
        amplitude: torch.Tensor = v[:, 0]

        # Return as complex128 tensor with zero imaginary part
        return torch.view_as_complex(
            torch.stack([amplitude.double(), torch.zeros_like(amplitude).double()], dim=-1)
        )

    @torch.jit.export
    def generate_unique(
        self, batch_size: int, block_num: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate a batch of unique configurations.
        """
        raise NotImplementedError("generate_unique is not implemented for the MPS network")

    @torch.jit.export
    def generate(
        self, batch_size: int, block_num: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Generate a batch of configurations.

        Parameters
        ----------
        batch_size : int
            The number of configurations to generate.
        block_num : int, default=1
            The number of batch block to generate. It is used to split the batch into smaller parts to avoid memory issues.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]
            A tuple containing the generated configurations, their amplitudes, their sample counts, and a None value.
            The configurations are a two-dimensional uint8 tensor with first dimension equal to the number of unique configurations.
            The second dimension contains occupation for each qubit which is bitwise encoded.
            The amplitudes are a one-dimensional complex tensor with the only dimension equal to the number of unique configurations.
            The counts are a one-dimensional int64 tensor recording how many times each unique configuration was sampled.
            The last None value is reserved for future use.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        tensors: torch.Tensor = self.tensors.to(dtype=dtype)

        # ------------------------------------------------------------------ #
        # Pre-compute right-environment density matrices from right to left.  #
        #                                                                      #
        # renv[k] = M[sites - k], where:                                      #
        #   M[sites] = e_R @ e_R^T  (right-boundary density matrix)           #
        #   M[i]     = sum_s A[i][s] @ M[i+1] @ A[i][s]^T  for i < sites     #
        #                                                                      #
        # When sampling at site i we need M[i+1] = renv[sites - i - 1].       #
        # ------------------------------------------------------------------ #
        renv: list[torch.Tensor] = []
        M: torch.Tensor = torch.zeros([self.virtual_dim, self.virtual_dim], device=device, dtype=dtype)
        M[0, 0] = 1.0  # e_R @ e_R^T  (only the [0,0] entry is non-zero)
        renv.append(M)

        for i in range(self.sites - 1, 0, -1):
            A_i: torch.Tensor = tensors[i]  # physical_dim * virtual_dim * virtual_dim
            # M_new[a,d] = sum_{s,b,c} A_i[s,a,b] * M[b,c] * A_i[s,d,c]
            #            = sum_s (A_i[s] @ M @ A_i[s]^T)[a,d]
            M = torch.einsum("sab,bc,sdc->ad", A_i, M, A_i)
            renv.append(M)

        # After the loop renv has length `sites`:
        #   renv[0] = M[sites], renv[1] = M[sites-1], ..., renv[sites-1] = M[1]

        # ------------------------------------------------------------------ #
        # Autoregressive sampling.                                             #
        # ------------------------------------------------------------------ #
        # v: batch_size * virtual_dim  (left state, initialised to e_L)
        v: torch.Tensor = torch.zeros([batch_size, self.virtual_dim], device=device, dtype=dtype)
        v[:, 0] = 1.0

        samples: torch.Tensor = torch.zeros([batch_size, self.sites], device=device, dtype=torch.uint8)

        for i in range(self.sites):
            A_i = tensors[i]  # physical_dim * virtual_dim * virtual_dim
            # M_next is the right-environment density matrix for the positions after site i
            M_next: torch.Tensor = renv[self.sites - i - 1]

            # v_new_all[b, s, j] = sum_k v[b, k] * A_i[s, k, j]  (batch, phys, virt)
            v_new_all: torch.Tensor = torch.einsum("bi,sij->bsj", v, A_i)

            # prob[b, s] = v_new_all[b, s] @ M_next @ v_new_all[b, s]^T  (non-negative)
            Mv: torch.Tensor = torch.einsum("jk,bsk->bsj", M_next, v_new_all)
            prob: torch.Tensor = (v_new_all * Mv).sum(dim=-1)  # batch_size * physical_dim

            # Correct any numerical round-off that may produce tiny negative values
            prob = prob.clamp(min=0.0)

            # Sample a physical state at site i for each batch element
            s_i: torch.Tensor = torch.multinomial(prob, num_samples=1).squeeze(1)  # batch_size
            samples[:, i] = s_i.to(torch.uint8)

            # Advance the left state vector
            arange: torch.Tensor = torch.arange(batch_size, device=device)
            v = v_new_all[arange, s_i, :]  # batch_size * virtual_dim

        # ------------------------------------------------------------------ #
        # Pack, deduplicate, and count.                                        #
        # ------------------------------------------------------------------ #
        x_packed: torch.Tensor = pack_int(samples, size=self._bit_size())

        # torch.unique returns (unique, inverse_indices, counts) in TorchScript
        x_unique, _, counts = torch.unique(
            x_packed, sorted=True, return_inverse=True, return_counts=True, dim=0
        )

        # Compute amplitudes for the unique configurations
        amplitudes: torch.Tensor = self(x_unique)

        return x_unique, amplitudes, counts, None
