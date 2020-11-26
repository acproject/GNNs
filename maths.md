* In `N`-ary Tree-LSTM, each unit at node :math:`j` maintains a hidden
* representation :math:`h_j` and a memory cell :math:`c_j`. The unit
* :math:`j` takes the input vector :math:`x_j` and the hidden
* representations of the child units: :math:`h_{jl}, 1\leq l\leq N` as
* input, then update its new hidden representation :math:`h_j` and memory
* cell :math:`c_j` by:
*
###  Tree-LSTM math::
$$
   i_j & = & \sigma\left(W^{(i)}x_j + \sum_{l=1}^{N}U^{(i)}_l h_{jl} + b^{(i)}\right),  & (1)\\
   f_{jk} & = & \sigma\left(W^{(f)}x_j + \sum_{l=1}^{N}U_{kl}^{(f)} h_{jl} + b^{(f)} \right), &  (2)\\
   o_j & = & \sigma\left(W^{(o)}x_j + \sum_{l=1}^{N}U_{l}^{(o)} h_{jl} + b^{(o)} \right), & (3)  \\
   u_j & = & \textrm{tanh}\left(W^{(u)}x_j + \sum_{l=1}^{N} U_l^{(u)}h_{jl} + b^{(u)} \right), & (4)\\    c_j & = & i_j \odot u_j + \sum_{l=1}^{N} f_{jl} \odot c_{jl}, &(5) \\
    h_j & = & o_j \cdot \textrm{tanh}(c_j), &(6)  \\
$$