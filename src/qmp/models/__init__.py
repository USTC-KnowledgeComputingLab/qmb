"""Model definitions: translate physical systems into FermiHamiltonian.

Each model constructs a ``{term: complex_coefficient}`` dictionary from its own
input format and wraps it in a FermiHamiltonian, exposing a uniform operator
interface via ModelProto.
"""

from __future__ import annotations
