import torch as th

def src_dot_dst(src_field, dst_field, out_field):
    '''
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function
    :param src_field:
    :param dst_field:
    :param out_field:
    :return:
    '''
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, c):
    '''
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    :param field:
    :param c:
    :return:
    '''
    def func(edges):
        return {field: th.exp((edges.data[field] / c).clamp(-10, 10))}
    return func
