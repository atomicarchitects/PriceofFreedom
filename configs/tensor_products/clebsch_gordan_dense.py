import ml_collections

def get_config() -> ml_collections.ConfigDict:
    """Config for the tensor product."""
    tensor_product_config = ml_collections.ConfigDict()
    tensor_product_config.type = "clebsch-gordan-dense"
    tensor_product_config.apply_output_linear = True
    tensor_product_config.irrep_normalization = "norm"
    return tensor_product_config