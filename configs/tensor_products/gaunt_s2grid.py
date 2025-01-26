import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Config for the tensor product."""
    tensor_product_config = ml_collections.ConfigDict()
    tensor_product_config.type = "gaunt-s2grid"
    tensor_product_config.res_beta = 100
    tensor_product_config.res_alpha = 99
    tensor_product_config.quadrature = "gausslegendre"
    tensor_product_config.num_channels = 4
    return tensor_product_config
