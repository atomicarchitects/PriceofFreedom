import time
import sys

from absl import app, flags
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
from tqdm.auto import tqdm

import e3nn_jax as e3nn

from vector_spherical_harmonics import VSHCoeffs
from simpler_vsh import SimpleVSHCoeffs

flags.DEFINE_integer("num_steps", 1000, "Number of training steps")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the network")
flags.DEFINE_integer("hidden_lmax", 3, "Hidden layer lmax")
flags.DEFINE_integer("sh_lmax", 3, "Spherical harmonics lmax")
flags.DEFINE_integer("multiplicity", 8, "Multiplicity")
flags.DEFINE_integer(
    "num_channels_for_gaunt_TP", 1, "Number of channels for Gaunt tensor product"
)
flags.DEFINE_bool(
    "use_from_s2grid_direct_in_gaunt_TP",
    False,
    "Use from_s2grid_direct in Gaunt tensor product",
)
flags.DEFINE_enum(
    "tensor_product_type",
    "vectorgaunt",
    ["usual", "gaunt", "vectorgaunt"],
    "Type of tensor product",
)
flags.DEFINE_bool("profile", False, "Enable profiling")
FLAGS = flags.FLAGS
FLAGS(sys.argv)


class VectorGauntTensorProduct(nn.Module):

    p_val1: int
    p_val2: int
    res_alpha: int = 19
    res_beta: int = 20
    num_channels: int = FLAGS.num_channels_for_gaunt_TP
    quadrature: str = "gausslegendre"

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        xc = e3nn.flax.Linear(
            SimpleVSHCoeffs.get_vsh_irreps(x.irreps.lmax, parity=self.p_val1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_x",
        )(x)
        xc = xc.mul_to_axis(self.num_channels)

        yc = e3nn.flax.Linear(
            SimpleVSHCoeffs.get_vsh_irreps(y.irreps.lmax, parity=self.p_val2)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_y",
        )(y)
        yc = yc.mul_to_axis(self.num_channels)

        def cross_product_per_channel_per_sample(xc, yc):
            xc = SimpleVSHCoeffs(xc, parity=self.p_val1)
            yc = SimpleVSHCoeffs(yc, parity=self.p_val2)
            zc = xc.reduce_pointwise_cross_product(
                yc,
                res_beta=self.res_beta,
                res_alpha=self.res_alpha,
                quadrature=self.quadrature,
            )
            zc = zc.to_irreps_array()
            return zc

        zc = jax.vmap(jax.vmap(cross_product_per_channel_per_sample))(xc, yc)
        zc = zc.axis_to_mul()
        zc = e3nn.flax.Linear(zc.irreps, name="linear_out_z")(zc)
        return zc


class GauntTensorProduct(nn.Module):

    p_val1: int
    p_val2: int
    res_alpha: int = 19
    res_beta: int = 20
    num_channels: int = FLAGS.num_channels_for_gaunt_TP
    quadrature: str = "gausslegendre"
    use_from_s2grid_direct: bool = FLAGS.use_from_s2grid_direct_in_gaunt_TP

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        xc = e3nn.flax.Linear(
            e3nn.s2_irreps(x.irreps.lmax, p_val=self.p_val1, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_x",
        )(x)
        xc = xc.mul_to_axis(self.num_channels)
        # print(xc.irreps)

        yc = e3nn.flax.Linear(
            e3nn.s2_irreps(y.irreps.lmax, p_val=self.p_val2, p_arg=-1)
            * self.num_channels,
            force_irreps_out=True,
            name="linear_in_y",
        )(y)
        yc = yc.mul_to_axis(self.num_channels)
        # print(yc.irreps)

        xc = e3nn.to_s2grid(
            xc,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val=self.p_val1,
            p_arg=-1,
            fft=False,
        )
        yc = e3nn.to_s2grid(
            yc,
            res_alpha=self.res_alpha,
            res_beta=self.res_beta,
            quadrature=self.quadrature,
            p_val=self.p_val2,
            p_arg=-1,
            fft=False,
        )
        if self.use_from_s2grid_direct:
            zc = jax.vmap(
                jax.vmap(
                    lambda prod: from_s2grid_direct(
                        prod,
                        lmax=x.irreps.lmax + y.irreps.lmax,
                        p_val=self.p_val1 * self.p_val2,
                    )
                )
            )(xc * yc)
        else:
            zc = e3nn.from_s2grid(
                xc * yc,
                irreps=e3nn.s2_irreps(
                    x.irreps.lmax + y.irreps.lmax, p_val=self.p_val1 * self.p_val2
                ),
                fft=False,
            )
        zc = zc.axis_to_mul()
        zc = e3nn.flax.Linear(zc.irreps, name="linear_out_z")(zc)
        return zc


class TensorProduct(nn.Module):

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return e3nn.tensor_product(x, y)


class Layer(nn.Module):

    target_irreps: e3nn.Irreps
    sh_lmax: int
    denominator: float = 1.5
    tensor_product_type = FLAGS.tensor_product_type

    @nn.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            sh = e3nn.spherical_harmonics(
                e3nn.s2_irreps(self.sh_lmax),
                positions[graphs.receivers] - positions[graphs.senders],
                True,
            )
            sender_features = e3nn.as_irreps_array(sender_features)

            if self.tensor_product_type == "usual":
                tp = TensorProduct()(sender_features, sh)

            elif self.tensor_product_type == "gaunt":
                tp1 = GauntTensorProduct(p_val1=1, p_val2=1)(sender_features, sh)
                tp2 = GauntTensorProduct(p_val1=1, p_val2=-1)(sender_features, sh)
                tp3 = GauntTensorProduct(p_val1=-1, p_val2=1)(sender_features, sh)
                tp = e3nn.concatenate([tp1, tp2, tp3])

            elif self.tensor_product_type == "vectorgaunt":
                tp1 = VectorGauntTensorProduct(p_val1=-1, p_val2=-1)(
                    sender_features, sh
                )
                tp2 = VectorGauntTensorProduct(p_val1=1, p_val2=-1)(sender_features, sh)
                tp3 = VectorGauntTensorProduct(p_val1=-1, p_val2=1)(sender_features, sh)
                tp = e3nn.concatenate([tp1, tp2, tp3])

            else:
                raise ValueError(
                    f"Unknown tensor product type: {self.tensor_product_type}"
                )

            return e3nn.concatenate([sender_features, tp]).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats)
            node_feats = e3nn.flax.Linear(
                target_irreps, name="linear_post", force_irreps_out=False
            )(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps, name="shortcut", force_irreps_out=True
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class Model(nn.Module):

    num_layers: int = FLAGS.num_layers
    hidden_lmax: int = FLAGS.hidden_lmax
    sh_lmax: int = FLAGS.sh_lmax
    multiplicity: int = FLAGS.multiplicity

    @nn.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        graphs = graphs._replace(nodes=jnp.ones((len(positions), 1)))

        layer_irreps = 4 * (e3nn.Irreps("0e") + e3nn.Irreps("0o"))
        layer_irreps += e3nn.s2_irreps(lmax=self.hidden_lmax, p_val=1, p_arg=-1)[1:]
        layer_irreps += e3nn.s2_irreps(lmax=self.hidden_lmax, p_val=-1, p_arg=-1)[1:]
        layer_irreps *= self.multiplicity
        layer_irreps = layer_irreps.regroup()

        for irreps in self.num_layers * [layer_irreps] + ["0o + 7x0e"]:
            graphs = Layer(irreps, sh_lmax=self.sh_lmax)(graphs, positions)
            print("Layer output:", graphs.nodes.irreps)

        # Readout logits
        pred = e3nn.scatter_sum(
            graphs.nodes.array, nel=graphs.n_node
        )  # [num_graphs, 1 + 7]
        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)
        assert logits.shape == (
            len(graphs.n_node),
            8,
        ), logits.shape  # [num_graphs, num_classes]

        return logits


def from_s2grid_direct(x, lmax, p_val):
    """Convert a grid of spherical harmonics to a vector of coefficients."""

    def get_clm(l: int, m: int) -> jnp.ndarray:
        Ylm_coeffs = jnp.zeros(2 * l + 1).at[m + l].set(1)
        Ylm = e3nn.IrrepsArray(e3nn.Irrep(l, (-1) ** (l)), Ylm_coeffs)
        Ylm_sig = e3nn.to_s2grid(
            Ylm,
            res_beta=x.res_beta,
            res_alpha=x.res_alpha,
            quadrature=x.quadrature,
            p_val=1,
            p_arg=-1,
        )
        return (x * Ylm_sig).integrate().array[-1] / (4 * jnp.pi)

    coeffs = []
    for l in range(lmax + 1):
        cl = jnp.asarray([get_clm(l, m) for m in range(-l, l + 1)])
        cl = e3nn.IrrepsArray(e3nn.Irrep(l, ((-1) ** (l)) * p_val), cl)
        coeffs.append(cl)
    return e3nn.concatenate(coeffs)


def get_tetris_dataset() -> jraph.GraphsTuple:
    pos = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    pos = jnp.array(pos, dtype=jnp.float32)
    labels = jnp.arange(8)

    graphs = []

    for p, l in zip(pos, labels):
        senders, receivers = e3nn.radius_graph(p, 1.1)

        graphs += [
            jraph.GraphsTuple(
                nodes=p.reshape((4, 3)),  # [num_nodes, 3]
                edges=None,
                globals=l[None],  # [num_graphs]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([len(p)]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)


def train():
    model = Model()

    # Optimizer
    opt = optax.adam(learning_rate=0.01)

    def loss_fn(params, graphs):
        logits = model.apply(params, graphs)
        labels = graphs.globals  # [num_graphs]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @jax.jit
    def update_fn(params, opt_state, graphs):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, graphs)
        labels = graphs.globals
        preds = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(preds == labels)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy, preds

    # Dataset
    graphs = get_tetris_dataset()

    # Init
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(3), graphs)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("Compiling...", flush=True)
    _, _, accuracy, _ = update_fn(params, opt_state, graphs)
    print(f"Compilation took {time.perf_counter() - wall:.1f}s")
    print(f"Initial accuracy = {100 * accuracy:.2f}%", flush=True)

    # Train
    wall = time.perf_counter()
    print("Training...", flush=True)
    with tqdm(range(FLAGS.num_steps)) as bar:
        for step in bar:
            if FLAGS.profile and step == 20:
                from ctypes import cdll
                libcudart = cdll.LoadLibrary("libcudart.so")
                libcudart.cudaProfilerStart()
            
            params, opt_state, accuracy, preds = update_fn(params, opt_state, graphs)

            if FLAGS.profile and step == 25:
                libcudart.cudaProfilerStop()
            
            bar.set_postfix(accuracy=f"{100 * accuracy:.2f}%")
            if accuracy == 1.0:
                break

    print(
        f"Training for tensor_product_type={FLAGS.tensor_product_type} with lmax={FLAGS.sh_lmax} took {time.perf_counter() - wall:.1f}s"
    )
    print(f"Final accuracy = {100 * accuracy:.2f}%")
    print("Final prediction:", preds)

    # Check equivariance.
    print("Checking equivariance...")
    apply = jax.jit(model.apply)
    for key in range(50):
        key = jax.random.PRNGKey(key)
        alpha, beta, gamma = jax.random.uniform(
            key, (3,), minval=-jnp.pi, maxval=jnp.pi
        )

        rotated_nodes = e3nn.IrrepsArray("1o", graphs.nodes)
        rotated_nodes = rotated_nodes.transform_by_angles(alpha, beta, gamma)
        rotated_nodes = rotated_nodes.array
        rotated_graphs = graphs._replace(nodes=rotated_nodes)

        logits = apply(params, graphs)
        rotated_logits = apply(params, rotated_graphs)
        if not jnp.allclose(logits, rotated_logits, atol=1e-4):
            print("Model is not equivariant.")

    print("Model is equivariant.")


if __name__ == "__main__":
    train()
