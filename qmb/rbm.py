"""
This file defines the RBM based wave function.
Apart from predicting the magnitude, it also has a unique way of sampling.
"""

import torch
from .mlp import MLP
from .bitspack import pack_int, unpack_int


class RBM(torch.nn.Module):
    """The RBM Network.

    It returns the free energy of configurations in forward,
    and also samples configurations according to its weight in the sample function.

    Parameters
    ----------
    visible_dim  : int
        The num of visible nodes.
    hidden_dim  : int
        The num of hidden nodes.
    gamma  : float
        The relative magnitude of initialized weight.
    """

    def __init__(self, visible_dim: int, hidden_dim: int, gamma: float) -> None:

        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.weights = torch.nn.Parameter(torch.zeros([self.visible_dim, self.hidden_dim]))
        init_range = gamma / torch.sqrt(torch.tensor(self.visible_dim))
        torch.nn.init.uniform_(self.weights, -init_range, init_range)
        self.visible_bias = torch.nn.Parameter(torch.zeros(visible_dim))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Returns the predicted free energy for each row.

        Parameters
        ----------
        v  : torch.Tensor
            Batch of configurations with free energy to be determined. Batch first.

        Returns
        -------
        torch.Tensor
            Free energy.
        """
        e1 = (v @ self.visible_bias).view(v.size()[:-1])
        mid2 = torch.nn.functional.linear(v, self.weights.T, self.hidden_bias)  # pylint: disable=not-callable
        e2 = mid2.exp().add(1).div(2).log().sum(dim=-1)
        return e1 + e2

    @torch.jit.export
    def sample(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Sample configurations by gibbs sampling.

        Parameters
        ---------
        v  : torch.Tensor
            The initial configurations to run Gibbs sampling on.
        k  : int, optional
            Rounds of Gibbs sampling (default: 1).

        Returns
        -------
        torch.Tensor
            Configurations sampled as a tensor. Batch first.
        """
        for _ in range(k):
            # samp h
            midh = torch.nn.functional.linear(v, self.weights.T, self.hidden_bias)  # pylint: disable=not-callable
            ph = torch.sigmoid(midh)
            h = torch.bernoulli(ph)
            # samp v
            midv = torch.nn.functional.linear(h, self.weights, self.visible_bias)  # pylint: disable=not-callable
            pv = torch.sigmoid(midv)
            v = torch.bernoulli(pv)
        return v


class WaveFunctionNormal(torch.nn.Module):
    """WaveFunction implemented via RBM.

    (The phase is predicted with MLP, however.)
    """

    def __init__(  # pylint: disable=R0913
            self,
            *,
            sites: int,
            physical_dim: int,
            is_complex: bool,
            rbm_hidden_dim: int,
            rbm_gamma: float,
            mlp_hidden_size: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.sites: int = sites
        assert physical_dim == 2
        assert is_complex == True  # pylint: disable=singleton-comparison
        self.rbm_hidden_dim: int = rbm_hidden_dim
        self.rbm_gamma: float = rbm_gamma
        self.mlp_hidden_size: tuple[int, ...] = mlp_hidden_size

        # Build Networks
        self.magnitude = RBM(self.sites, self.rbm_hidden_dim, rbm_gamma)
        self.phase = MLP(self.sites, 1, self.mlp_hidden_size)

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        """Device of the model's parameters"""
        return self.dummy_param.device

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the model's parameters"""
        return self.dummy_param.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates amplitude

        See model_dict.py
        """
        batch_size: torch.Size = x.shape[:-1]
        x = unpack_int(x, size=1, last_dim=self.sites).view(*batch_size, self.sites)
        x_float: torch.Tensor = x.to(dtype=self.dtype)
        free_energy: torch.Tensor = self.magnitude(x_float).double()
        ln_magnitude: torch.Tensor = free_energy / 2 - free_energy.mean() / 2  # ??
        phase: torch.Tensor = self.phase(x_float).view(*batch_size).double()
        return (ln_magnitude + phase * 1j).exp()

    @torch.jit.export
    def generate_conf(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples configurations(Not unique).

        Parameters
        ----------
        batch_size  : int
            Num of samples

        Returns
        -------
        samples  : torch.Tensor
            Batch of configurations. Batch first.
        amplitude  : torch.Tensor
            The amplitude of each sample.
        """
        samples = self.magnitude.sample(torch.bernoulli(torch.ones((batch_size, self.sites), device=self.device) * 0.5))
        samples = pack_int(samples.byte(), size=1)
        amplitude = self(samples)

        return samples, amplitude
