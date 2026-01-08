"""
This file implements an auto-regressive transformers network with the sampling method introduced in https://arxiv.org/pdf/2408.07625.
This network makes use of DeepSeekMoE architecture introduced in https://arxiv.org/pdf/2401.06066.
"""

import typing
import torch
from ..utility.bitspack import pack_int, unpack_int


class FeedForward(torch.nn.Module):
    """
    A feed-forward layer for transformer architectures.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward layer.
        """
        # x: batch * site * embedding
        x = self.model(x)
        # x: batch * site * embedding
        return x


class SelfAttention(torch.nn.Module):
    """
    Self-attention unit with support for key-value cache.
    """

    def __init__(self, embedding_dim: int, heads_num: int) -> None:
        super().__init__()

        self.heads_num: int = heads_num
        self.heads_dim: int = embedding_dim // heads_num
        assert self.heads_num * self.heads_dim == embedding_dim

        self.qkv: torch.nn.Module = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out: torch.nn.Module = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: typing.Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: typing.Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the self-attention layer.
        """
        # x: batch * site * embedding
        batch_size, sites, embedding_dim = x.shape
        q, k, v = self.qkv(x).split(embedding_dim, dim=-1)
        # q, k, v: batch * site * embedding
        q = q.view([batch_size, sites, self.heads_num, self.heads_dim])
        k = k.view([batch_size, sites, self.heads_num, self.heads_dim])
        v = v.view([batch_size, sites, self.heads_num, self.heads_dim])
        # q, k, v: batch, site, heads_num, heads_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v: batch, heads_num, site, heads_dim
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        # q: batch, heads_num, site, heads_dim
        # k, v: batch, heads_num, total_site, heads_dim
        if mask is None:
            total_sites = k.shape[-2]
            mask = torch.ones(sites, total_sites, dtype=torch.bool, device=x.device).tril(diagonal=total_sites - sites)
        # call scaled dot product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # pylint: disable=not-callable
        # attn: batch, heads_num, site, heads_dim
        out = attn.transpose(1, 2).reshape([batch_size, sites, embedding_dim])
        # out: batch, site, embedding_dim
        return self.out(out), (k, v)


class DecoderUnit(torch.nn.Module):
    """
    A decoder unit within the transformer architecture, integrating both self-attention and feed-forward layers.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        shared_num: int,
        routed_num: int,
        selected_num: int,
    ) -> None:
        super().__init__()
        self.shared_num = shared_num
        self.routed_num = routed_num
        self.selected_num = selected_num

        self.attention: torch.nn.Module = SelfAttention(embedding_dim, heads_num)
        self.norm1: torch.nn.Module = torch.nn.LayerNorm(embedding_dim)
        self.feed_forward_shared: torch.nn.ModuleList = torch.nn.ModuleList(
            [FeedForward(embedding_dim, feed_forward_dim) for _ in range(shared_num)]
        )
        self.feed_forward_routed: torch.nn.ModuleList = torch.nn.ModuleList(
            [FeedForward(embedding_dim, feed_forward_dim) for _ in range(routed_num)]
        )
        self.centroid: torch.nn.Parameter = torch.nn.Parameter(torch.randn(routed_num, embedding_dim))
        self.norm2: torch.nn.Module = torch.nn.LayerNorm(embedding_dim)

        self.bias: torch.Tensor
        self.register_buffer("bias", torch.zeros([self.routed_num]))
        self.accumulater: torch.Tensor
        self.register_buffer("accumulater", torch.zeros([self.routed_num]))
        self.count: torch.Tensor
        self.register_buffer("count", torch.zeros([]))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: typing.Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: typing.Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the decoder unit.
        """
        # x, y: batch * site * embedding
        y, result_cache = self.attention(x, kv_cache, mask)
        x = self.norm1(x + y)

        # The following segmental code implements the DeepSeekMoE architecture.
        y = x
        for i, expert in enumerate(self.feed_forward_shared):
            y = y + expert(x)
        # similarity: batch * site * routed
        similarity = torch.nn.functional.softmax(x @ self.centroid.t(), dim=-1)
        # top_k_indices: batch * site * selected
        _, top_k_indices = torch.topk(similarity + self.bias, self.selected_num, dim=-1)
        # gate_prime, gate: batch * site * routed
        gate_prime = torch.zeros_like(similarity).scatter_(-1, top_k_indices, similarity.gather(-1, top_k_indices))
        gate = gate_prime / gate_prime.sum(dim=-1).unsqueeze(-1)
        for i, expert in enumerate(self.feed_forward_routed):
            y = y + expert(x) * gate[:, :, i].unsqueeze(-1)
        x = self.norm2(y)

        if self.training:
            self.accumulater = self.accumulater + similarity.sum([0, 1])
            self.count = self.count + similarity.size(0) * similarity.size(1)

        return x, result_cache


class Transformers(torch.nn.Module):
    """
    A transformer model consisting of multiple decoder units.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        embedding_dim: int,
        heads_num: int,
        feed_forward_dim: int,
        shared_num: int,
        routed_num: int,
        selected_num: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
            DecoderUnit(
                embedding_dim=embedding_dim,
                heads_num=heads_num,
                feed_forward_dim=feed_forward_dim,
                shared_num=shared_num,
                routed_num=routed_num,
                selected_num=selected_num,
            )
            for _ in range(depth)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: typing.Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        mask: typing.Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the transformer model.
        """
        # x: batch * site * embedding
        result_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            if kv_cache is None:
                x, cache = layer(x, None, mask)
            else:
                x, cache = layer(x, kv_cache[i], mask)
            result_cache.append(cache)
        return x, result_cache


class Tail(torch.nn.Module):
    """
    The Tail layer for the transformer model, responsible for mapping the final embeddings to the desired output dimensions.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Tail layer.
        """
        # x: batch * site * embedding_dim
        x = self.model(x)
        # x: batch * site * output_dim
        return x


class Embedding(torch.nn.Module):
    """
    Embedding layer for transforming input data into a format suitable for the transformer model.
    """

    def __init__(self, sites: int, physical_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.parameter: torch.nn.Parameter = torch.nn.Parameter(torch.randn([sites, physical_dim, embedding_dim]))
        self.sites: int = sites
        self.physical_dim: int = physical_dim
        self.embedding_dim: int = embedding_dim

    def forward(self, x: torch.Tensor, base: int) -> torch.Tensor:
        """
        Forward pass of the Embedding layer.
        """
        # x: batch * sites
        batch, sites = x.shape

        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        # x: batch * sites * config=1 * embedding

        parameter = self.parameter[base : base + sites].unsqueeze(0).expand(batch, -1, -1, -1)
        # parameter: batch * sites * config * embedding

        result = torch.gather(parameter, -2, x.to(dtype=torch.int64))
        # result: batch * site * 1 * embedding

        return result.squeeze(-2)


class WaveFunctionElectronUpDown(torch.nn.Module):
    """
    The wave function for the transformers network.
    This module maintains the conservation of particle number of spin-up and spin-down electrons.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        double_sites: int,  # Number of qubits, where each pair of qubits represents a site
        physical_dim: int,  # Dimension of the physical space, which is always 2
        is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true
        spin_up: int,  # Number of spin-up electrons
        spin_down: int,  # Number of spin-down electrons
        embedding_dim: int,  # Dimension of the embedding space used in the transformer layers
        heads_num: int,  # Number of attention heads in the transformer's self-attention mechanism
        feed_forward_dim: int,  # Dimension of the feed-forward network within the transformer layers
        shared_num: int,  # Number of the shared expert in the DeepSeekMoE architecture
        routed_num: int,  # Number of the routed expert in the DeepSeekMoE architecture
        selected_num: int,  # Number of the selected expert in the DeepSeekMoE architecture
        depth: int,  # Number of decoder layers in the transformer model
        ordering: int
        | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites: int = double_sites
        self.sites: int = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True  # pylint: disable=singleton-comparison
        self.spin_up: int = spin_up
        self.spin_down: int = spin_down
        self.embedding_dim: int = embedding_dim
        self.heads_num: int = heads_num
        self.feed_forward_dim: int = feed_forward_dim
        self.shared_num: int = shared_num
        self.routed_num: int = routed_num
        self.selected_num: int = selected_num
        self.depth: int = depth

        # Embed configurations for each site, considering the four possible states of two qubits.
        self.embedding: torch.nn.Module = Embedding(self.sites, 4, self.embedding_dim)
        # Main body of the wave function computation.
        self.transformers: torch.nn.Module = Transformers(
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_num,
            routed_num=self.routed_num,
            selected_num=self.selected_num,
            depth=self.depth,
        )
        # Tail layer mapping from embedding space to amplitude and phase space.
        # (amplitude and phase) * (4 possible states)
        self.tail: torch.nn.Module = Tail(self.embedding_dim, self.feed_forward_dim, 8)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer(
            "ordering_reversed",
            torch.scatter(
                torch.zeros(self.sites, dtype=torch.int64),
                0,
                self.ordering,
                torch.arange(self.sites, dtype=torch.int64),
            ),
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determined whether we could append spin up or spin down after uncompleted configurations.
        """
        # pylint: disable=too-many-locals

        # x: batch_size * current_site * 2
        # x represents the uncompleted configurations
        current_site = x.shape[1]
        # number: batch_size * 2
        # number denotes the total electron count for each uncompleted configuration
        number = x.sum(dim=1)

        # up/down_electron/hole: batch_size
        # These variables store the count of electrons and holes for each uncompleted configuration
        up_electron = number[:, 0]
        down_electron = number[:, 1]
        up_hole = current_site - up_electron
        down_hole = current_site - down_electron

        # add_up/down_electron/hole: batch_size
        # These variables determine whether it is possible to append an up/down electron/hole
        add_up_electron = up_electron < self.spin_up
        add_down_electron = down_electron < self.spin_down
        add_up_hole = up_hole < self.sites - self.spin_up
        add_down_hole = down_hole < self.sites - self.spin_down

        # add_up: batch_size * 2 * 1
        # add_down: batch_size * 1 * 2
        # These tensors represent the feasibility of adding up/down electrons/holes
        add_up = torch.stack([add_up_hole, add_up_electron], dim=-1).unsqueeze(-1)
        add_down = torch.stack([add_down_hole, add_down_electron], dim=-1).unsqueeze(-2)

        # add: batch_size * 2 * 2
        # add represents the logical AND of add_up and add_down, indicating the feasibility of appending specific electron/hole combinations
        # add[_, 0, 0] indicates the possibility of adding an up hole and a down hole
        # add[_, 0, 1] indicates the possibility of adding an up hole and a down electron
        # add[_, 1, 0] indicates the possibility of adding an up electron and a down hole
        # add[_, 1, 1] indicates the possibility of adding an up electron and a down electron
        add = torch.logical_and(add_up, add_down)

        return add

    @torch.jit.export
    def _normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log amplitude of uncompleted configurations.
        """
        # x : ... * 2 * 2
        # param :  ...
        param = (2 * x).exp().sum(dim=[-2, -1]).log() / 2
        x = x - param.unsqueeze(-1).unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        device: torch.device = self.dummy_param.device

        batch_size: int = x.shape[0]
        # x : batch_size * sites * 2
        x = unpack_int(x, size=1, last_dim=self.double_sites).view([batch_size, self.sites, 2])
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        # x4: batch_size * sites
        # Prepare x4 as the input for the neural network
        # The last configuration is excluded, and a zero vector is prepended at the beginning to represent the initial state.
        x4: torch.Tensor = x[:, :-1, 0] * 2 + x[:, :-1, 1]
        x4 = torch.cat([torch.zeros([batch_size, 1], device=device, dtype=torch.uint8), x4], dim=1)

        # emb: batch_size * sites * embedding
        # Embed the input tensor `x4` using the embedding layer.
        # In embedding layer, 0 for bos, 1 for site 0, ...
        emb: torch.Tensor = self.embedding(x4, 0)

        # post_transformers: batch_size * sites * embedding
        post_transformers, _ = self.transformers(emb, None, None)

        # tail: batch_size * sites * 8
        tail: torch.Tensor = self.tail(post_transformers)

        # amplitude/phase : batch_size * sites * 2 * 2
        amplitude: torch.Tensor = tail[:, :, :4].view(batch_size, self.sites, 2, 2)
        phase: torch.Tensor = tail[:, :, 4:].view(batch_size, self.sites, 2, 2)

        # Apply a filter mask to the amplitude to ensure the conservation of particle number.
        amplitude = amplitude + torch.stack(
            [torch.where(self._mask(x[:, :i]), 0, -torch.inf) for i in range(self.sites)], dim=1
        )

        # Normalize the delta amplitude.
        amplitude = self._normalize_amplitude(amplitude)

        # batch/sites_indices: batch_size * sites
        batch_indices: torch.Tensor = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sites)
        sites_indices: torch.Tensor = torch.arange(self.sites).unsqueeze(0).expand(batch_size, -1)

        # selected_amplitude/phase: batch_size * sites
        x_int32: torch.Tensor = x.to(dtype=torch.int32)
        selected_amplitude: torch.Tensor = amplitude[batch_indices, sites_indices, x_int32[:, :, 0], x_int32[:, :, 1]]
        selected_phase: torch.Tensor = phase[batch_indices, sites_indices, x_int32[:, :, 0], x_int32[:, :, 1]]

        return torch.view_as_complex(
            torch.stack([selected_amplitude.double().sum(dim=1), selected_phase.double().sum(dim=1)], dim=-1)
        ).exp()

    @torch.jit.export
    def _blocked_forward_for_generate_unique(  # pylint: disable=too-many-arguments
        self,
        *,
        x: torch.Tensor,
        cache_input: typing.Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        block_num: int,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # pylint: disable=too-many-locals
        local_batch_size: int = x.size(0)
        local_batch_size_block = local_batch_size // block_num
        remainder = local_batch_size % block_num
        tail_list: list[torch.Tensor] = []
        cache_list: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for j in range(block_num):
            if j < remainder:
                current_local_batch_size_block = local_batch_size_block + 1
            else:
                current_local_batch_size_block = local_batch_size_block
            start_index = j * local_batch_size_block + min(j, remainder)
            end_index = start_index + current_local_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            x_block: torch.Tensor = x[batch_indices_block]
            xi_block: torch.Tensor = x_block[:, -1:, :]
            xi4_block: torch.Tensor = xi_block[:, :, 0] * 2 + xi_block[:, :, 1]
            emb_block: torch.Tensor = self.embedding(xi4_block, i)
            if cache_input is None:
                post_transformers_block, cache_block = self.transformers(emb_block, None, None)
            else:
                cache_block_input = [
                    (cache_layer[0][batch_indices_block], cache_layer[1][batch_indices_block])
                    for cache_layer in cache_input
                ]
                post_transformers_block, cache_block = self.transformers(emb_block, cache_block_input, None)
            tail_block: torch.Tensor = self.tail(post_transformers_block)
            tail_list.append(tail_block)
            cache_list.append(cache_block)
        tail: torch.Tensor = torch.cat(tail_list)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.cat([cache_block[depth][0] for cache_block in cache_list]),
                torch.cat([cache_block[depth][1] for cache_block in cache_list]),
            )
            for depth in range(self.depth)
        ]
        return tail, cache

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        see https://arxiv.org/pdf/2408.07625.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=invalid-name
        # pylint: disable=too-many-statements

        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        cache: typing.Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        # x: local_batch_size * current_site * 2
        x: torch.Tensor = torch.zeros([1, 1, 2], device=device, dtype=torch.uint8)  # site=1, since the first is bos
        # (un)perturbed_log_probability: local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i in range(self.sites):
            local_batch_size: int = x.size(0)

            # # xi: batch * (sites=1) * 2
            # xi: torch.Tensor = x[:, -1:, :]
            # # xi4: batch * (sites=1)
            # xi4: torch.Tensor = xi[:, :, 0] * 2 + xi[:, :, 1]
            # # emb: batch * (sites=1) * embedding
            # emb: torch.Tensor = self.embedding(xi4, i)
            # # post_transformers: batch * (sites=1) * embedding
            # post_transformers, cache = self.transformers(emb, cache, None)
            # # tail: batch * (sites=1) * 8
            # tail: torch.Tensor = self.tail(post_transformers)
            tail, cache = self._blocked_forward_for_generate_unique(
                x=x, cache_input=cache, block_num=block_num, device=device, i=i
            )

            # The first 4 item are amplitude
            # delta_amplitude: batch * 2 * 2
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            delta_amplitude: torch.Tensor = tail[:, :, :4].view([local_batch_size, 2, 2])
            # Apply a filter mask to the amplitude to ensure the conservation of particle number.
            delta_amplitude = delta_amplitude + torch.where(self._mask(x[:, 1:]), 0, -torch.inf)

            # normalized_delta_amplitude: batch_size * 2 * 2
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # The delta unperturbed prob for all batch and 4 adds
            l: torch.Tensor = (2 * normalized_delta_amplitude).view([local_batch_size, 4])
            # and add to get the current unperturbed prob
            l = unperturbed_probability.view([local_batch_size, 1]) + l
            # Get perturbed prob by adding GUMBEL(0)
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()
            # Get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.view([local_batch_size, 1])
            # Evaluate the conditioned prob
            tildeL: torch.Tensor = -torch.log(
                torch.exp(-perturbed_probability.view([local_batch_size, 1])) - torch.exp(-Z) + torch.exp(-L)
            )

            assert cache is not None

            # Calculate appended configurations for 4 adds
            # local_batch_size * current_site * 2 + local_batch_size * 1 * 2
            x0: torch.Tensor = torch.cat(
                [x, torch.tensor([[0, 0]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x1: torch.Tensor = torch.cat(
                [x, torch.tensor([[0, 1]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x2: torch.Tensor = torch.cat(
                [x, torch.tensor([[1, 0]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )
            x3: torch.Tensor = torch.cat(
                [x, torch.tensor([[1, 1]], device=device, dtype=torch.uint8).expand(local_batch_size, -1, -1)], dim=1
            )

            # Cat all configurations to get x : new_local_batch_size * (current_size+1) * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1, x2, x3])
            unperturbed_probability = l.permute(1, 0).reshape([4 * local_batch_size])
            perturbed_probability = tildeL.permute(1, 0).reshape([4 * local_batch_size])
            cache_indices = (
                torch.arange(local_batch_size, device=device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(4, -1)
                .reshape([4 * local_batch_size])
            )
            # kv cache: batch * heads_num * site * heads_dim, so just repeat first dimension

            # Filter results, only use largest batch_size ones
            selected = perturbed_probability.argsort(descending=True)[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache_indices = cache_indices[selected]

            # If prob = 0, filter it forcely
            selected = perturbed_probability.isfinite()
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache_indices = cache_indices[selected]

            cache = [(cache_layer[0][cache_indices], cache_layer[1][cache_indices]) for cache_layer in cache]

        # Apply ordering
        x = torch.index_select(x[:, 1:], 1, self.ordering)
        # Flatten site part of x
        x = x.view([x.size(0), self.double_sites])
        # It should return configurations, amplitudes, probabilities and multiplicities.
        # But it is unique generator, so the last two fields are None
        x = pack_int(x, size=1)

        # Calculate the amplitude for the generated configurations in the batch.
        real_batch_size = len(x)
        real_batch_size_block = real_batch_size // block_num
        remainder = real_batch_size % block_num
        amplitude_list = []
        for j in range(block_num):
            if j < remainder:
                current_real_batch_size_block = real_batch_size_block + 1
            else:
                current_real_batch_size_block = real_batch_size_block
            start_index = j * real_batch_size_block + min(j, remainder)
            end_index = start_index + current_real_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            amplitude_block = self(x[batch_indices_block])
            amplitude_list.append(amplitude_block)
        amplitude: torch.Tensor = torch.cat(amplitude_list)

        return x, amplitude, None, None


class WaveFunctionNormal(torch.nn.Module):
    """
    The wave function for the transformers model.
    This module does not maintain any conservation.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        sites: int,  # Number of qubits
        physical_dim: int,  # Dimension of the physical space,
        is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true
        embedding_dim: int,  # Dimension of the embedding space used in the transformer layers
        heads_num: int,  # Number of attention heads in the transformer's self-attention mechanism
        feed_forward_dim: int,  # Dimension of the feed-forward network within the transformer layers
        shared_num: int,  # Number of the shared expert in the DeepSeekMoE architecture
        routed_num: int,  # Number of the routed expert in the DeepSeekMoE architecture
        selected_num: int,  # Number of the selected expert in the DeepSeekMoE architecture
        depth: int,  # Number of decoder layers in the transformer model
        ordering: int
        | list[int],  # Ordering of sites: +1 for normal order, -1 for reversed order, or a custom order list
    ) -> None:
        super().__init__()
        self.sites: int = sites
        self.physical_dim: int = physical_dim
        assert is_complex == True  # pylint: disable=singleton-comparison
        self.embedding_dim: int = embedding_dim
        self.heads_num: int = heads_num
        self.feed_forward_dim: int = feed_forward_dim
        self.shared_num: int = shared_num
        self.routed_num: int = routed_num
        self.selected_num: int = selected_num
        self.depth: int = depth

        # Embed configurations for each site, considering the physical_dim possible states of each qubit.
        self.embedding: torch.nn.Module = Embedding(self.sites, self.physical_dim, self.embedding_dim)
        # Main body of the wave function computation.
        self.transformers: torch.nn.Module = Transformers(
            embedding_dim=self.embedding_dim,
            heads_num=self.heads_num,
            feed_forward_dim=self.feed_forward_dim,
            shared_num=self.shared_num,
            routed_num=self.routed_num,
            selected_num=self.selected_num,
            depth=self.depth,
        )
        # Tail layer mapping from embedding space to amplitude and phase space.
        # (amplitude and phase) * (all possible states)
        self.tail: torch.nn.Module = Tail(self.embedding_dim, self.feed_forward_dim, 2 * self.physical_dim)

        # Site Ordering Configuration
        # +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.ordering: torch.Tensor
        self.register_buffer("ordering", torch.tensor(ordering, dtype=torch.int64))
        self.ordering_reversed: torch.Tensor
        self.register_buffer(
            "ordering_reversed",
            torch.scatter(
                torch.zeros(self.sites, dtype=torch.int64),
                0,
                self.ordering,
                torch.arange(self.sites, dtype=torch.int64),
            ),
        )

        # Dummy Parameter for Device and Dtype Retrieval
        # This parameter is used to infer the device and dtype of the model.
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def _normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log amplitude of uncompleted configurations.
        """
        # x : ... * physical_dim
        # param :  ...
        param = (2 * x).exp().sum(dim=[-1]).log() / 2
        x = x - param.unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave function psi for the given configurations.
        """
        device: torch.device = self.dummy_param.device

        batch_size: int = x.shape[0]
        # x : batch_size * sites
        x = unpack_int(x, size=self._bit_size(), last_dim=self.sites).view([batch_size, self.sites])
        # Apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        # x_for_emb: batch_size * sites
        # Prepare x for emb as the input for the neural network
        # The last configuration is excluded, and a zero vector is prepended at the beginning to represent the initial state.
        x_for_emb: torch.Tensor = x[:, :-1]
        x_for_emb = torch.cat([torch.zeros([batch_size, 1], device=device, dtype=torch.uint8), x_for_emb], dim=1)

        # emb: batch_size * sites * embedding
        # Embed the input tensor `x_for_emb` using the embedding layer.
        # In embedding layer, 0 for bos, 1 for site 0, ...
        emb: torch.Tensor = self.embedding(x_for_emb, 0)

        # post_transformers: batch_size * sites * embedding
        post_transformers, _ = self.transformers(emb, None, None)

        # tail: batch_size * sites * (2*physical_dim)
        tail: torch.Tensor = self.tail(post_transformers)

        # amplitude/phase : batch_size * sites * physical_dim
        amplitude: torch.Tensor = tail[:, :, : self.physical_dim]
        phase: torch.Tensor = tail[:, :, self.physical_dim :]

        # Normalize the delta amplitude.
        amplitude = self._normalize_amplitude(amplitude)

        # batch/sites_indices: batch_size * sites
        batch_indices: torch.Tensor = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sites)
        sites_indices: torch.Tensor = torch.arange(self.sites).unsqueeze(0).expand(batch_size, -1)

        # selected_amplitude/phase: batch_size * sites
        x_int32: torch.Tensor = x.to(dtype=torch.int32)
        selected_amplitude: torch.Tensor = amplitude[batch_indices, sites_indices, x_int32]
        selected_phase: torch.Tensor = phase[batch_indices, sites_indices, x_int32]

        return torch.view_as_complex(
            torch.stack([selected_amplitude.double().sum(dim=1), selected_phase.double().sum(dim=1)], dim=-1)
        ).exp()

    @torch.jit.export
    def _blocked_forward_for_generate_unique(  # pylint: disable=too-many-arguments
        self,
        *,
        x: torch.Tensor,
        cache_input: typing.Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        block_num: int,
        device: torch.device,
        i: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # pylint: disable=too-many-locals
        local_batch_size: int = x.size(0)
        local_batch_size_block = local_batch_size // block_num
        remainder = local_batch_size % block_num
        tail_list: list[torch.Tensor] = []
        cache_list: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for j in range(block_num):
            if j < remainder:
                current_local_batch_size_block = local_batch_size_block + 1
            else:
                current_local_batch_size_block = local_batch_size_block
            start_index = j * local_batch_size_block + min(j, remainder)
            end_index = start_index + current_local_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            x_block: torch.Tensor = x[batch_indices_block]
            xi_block: torch.Tensor = x_block[:, -1:]
            emb_block: torch.Tensor = self.embedding(xi_block, i)
            if cache_input is None:
                post_transformers_block, cache_block = self.transformers(emb_block, None, None)
            else:
                cache_block_input = [
                    (cache_layer[0][batch_indices_block], cache_layer[1][batch_indices_block])
                    for cache_layer in cache_input
                ]
                post_transformers_block, cache_block = self.transformers(emb_block, cache_block_input, None)
            tail_block: torch.Tensor = self.tail(post_transformers_block)
            tail_list.append(tail_block)
            cache_list.append(cache_block)
        tail: torch.Tensor = torch.cat(tail_list)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.cat([cache_block[depth][0] for cache_block in cache_list]),
                torch.cat([cache_block[depth][1] for cache_block in cache_list]),
            )
            for depth in range(self.depth)
        ]
        return tail, cache

    @torch.jit.export
    def generate_unique(self, batch_size: int, block_num: int = 1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        see https://arxiv.org/pdf/2408.07625.
        """
        # pylint: disable=too-many-locals
        # pylint: disable=invalid-name
        # pylint: disable=too-many-statements

        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        cache: typing.Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        # x: local_batch_size * current_site
        x: torch.Tensor = torch.zeros([1, 1], device=device, dtype=torch.uint8)  # site=1, since the first is bos
        # (un)perturbed_log_probability: local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i in range(self.sites):
            local_batch_size: int = x.size(0)

            # # xi: batch * (sites=1)
            # xi: torch.Tensor = x[:, -1:]
            # # emb: batch * (sites=1) * embedding
            # emb: torch.Tensor = self.embedding(xi, i)
            # # post_transformers: batch * (sites=1) * embedding
            # post_transformers, cache = self.transformers(emb, cache, None)
            # # tail: batch * (sites=1) * (2*physical_dim)
            # tail: torch.Tensor = self.tail(post_transformers)
            tail, cache = self._blocked_forward_for_generate_unique(
                x=x, cache_input=cache, block_num=block_num, device=device, i=i
            )

            # The first physical_dim item are amplitude
            # delta_amplitude: batch * physical_dim
            # delta_amplitude represents the amplitude changes for the configurations at the new site.
            delta_amplitude: torch.Tensor = tail[:, :, : self.physical_dim].view([local_batch_size, self.physical_dim])

            # normalized_delta_amplitude: batch_size * physical_dim
            # Normalize the delta amplitude.
            normalized_delta_amplitude: torch.Tensor = self._normalize_amplitude(delta_amplitude)

            # The delta unperturbed prob for all batch and all new possible states
            l: torch.Tensor = (2 * normalized_delta_amplitude).view([local_batch_size, self.physical_dim])
            # and add to get the current unperturbed prob
            l = unperturbed_probability.view([local_batch_size, 1]) + l
            # Get perturbed prob by adding GUMBEL(0)
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()
            # Get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.view([local_batch_size, 1])
            # Evaluate the conditioned prob
            tildeL: torch.Tensor = -torch.log(
                torch.exp(-perturbed_probability.view([local_batch_size, 1])) - torch.exp(-Z) + torch.exp(-L)
            )

            assert cache is not None

            # Calculate appended configurations for all new possible states
            # local_batch_size * current_site + local_batch_size * 1
            xs: list[torch.Tensor] = [
                torch.cat([x, torch.tensor([j], device=device, dtype=torch.uint8).expand(local_batch_size, -1)], dim=1)
                for j in range(self.physical_dim)
            ]

            # Cat all configurations to get x : new_local_batch_size * (current_size+1)
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat(xs)
            unperturbed_probability = l.permute(1, 0).reshape([self.physical_dim * local_batch_size])
            perturbed_probability = tildeL.permute(1, 0).reshape([self.physical_dim * local_batch_size])
            cache_indices = (
                torch.arange(local_batch_size, device=device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(4, -1)
                .reshape([4 * local_batch_size])
            )
            # kv cache: batch * heads_num * site * heads_dim, so just repeat first dimension

            # Filter results, only use largest batch_size ones
            selected = perturbed_probability.argsort(descending=True)[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache_indices = cache_indices[selected]

            # If prob = 0, filter it forcely
            selected = perturbed_probability.isfinite()
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache_indices = cache_indices[selected]

            cache = [(cache_layer[0][cache_indices], cache_layer[1][cache_indices]) for cache_layer in cache]

        # Apply ordering
        x = torch.index_select(x[:, 1:], 1, self.ordering)
        # Flatten site part of x
        x = x.view([x.size(0), self.sites])
        # It should return configurations, amplitudes, probabilities and multiplicities.
        # But it is unique generator, so the last two fields are None
        x = pack_int(x, size=self._bit_size())

        # Calculate the amplitude for the generated configurations in the batch.
        real_batch_size = len(x)
        real_batch_size_block = real_batch_size // block_num
        remainder = real_batch_size % block_num
        amplitude_list = []
        for j in range(block_num):
            if j < remainder:
                current_real_batch_size_block = real_batch_size_block + 1
            else:
                current_real_batch_size_block = real_batch_size_block
            start_index = j * real_batch_size_block + min(j, remainder)
            end_index = start_index + current_real_batch_size_block
            batch_indices_block = torch.arange(start_index, end_index, device=device, dtype=torch.int64)
            amplitude_block = self(x[batch_indices_block])
            amplitude_list.append(amplitude_block)
        amplitude: torch.Tensor = torch.cat(amplitude_list)

        return x, amplitude, None, None

    def _bit_size(self) -> int:
        if self.physical_dim <= 1 << 1:
            return 1
        if self.physical_dim <= 1 << 2:
            return 2
        if self.physical_dim <= 1 << 4:
            return 4
        if self.physical_dim <= 1 << 8:
            return 8
        raise ValueError("physical_dim should be less than or equal to 256")
