import time

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
from tqdm.auto import tqdm

import e3nn_jax as e3nn


class GeneralGauntTensorProduct(nn.Module):
    
    p_val1: int
    p_val2: int
    res_alpha: int = 19
    res_beta: int = 20
    num_channels: int = 1
    quadrature: str = "gausslegendre"

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        lmax = x.irreps.lmax + y.irreps.lmax

        xc = e3nn.flax.Linear(e3nn.s2_irreps(x.irreps.lmax, p_val=self.p_val1, p_arg=-1) * self.num_channels, force_irreps_out=True, name="linear_x")(x)
        xc = xc.mul_to_axis(self.num_channels)
        # print(xc.irreps)

        yc = e3nn.flax.Linear(e3nn.s2_irreps(y.irreps.lmax, p_val=self.p_val2, p_arg=-1) * self.num_channels, force_irreps_out=True, name="linear_y")(y)
        yc = yc.mul_to_axis(self.num_channels)
        # print(yc.irreps)

        xc = e3nn.to_s2grid(xc, res_alpha=self.res_alpha, res_beta=self.res_beta, quadrature=self.quadrature, p_val=self.p_val1, p_arg=-1)
        yc = e3nn.to_s2grid(yc, res_alpha=self.res_alpha, res_beta=self.res_beta, quadrature=self.quadrature, p_val=self.p_val2, p_arg=-1)
        zc = e3nn.from_s2grid(xc * yc, irreps=e3nn.s2_irreps(lmax, p_val=self.p_val1 * self.p_val2))
        zc = zc.axis_to_mul()
        # print(zc.irreps)
        # print()
        return zc


class TensorProduct(nn.Module):
    
    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return e3nn.tensor_product(x, y)


class Layer(nn.Module):
    target_irreps: e3nn.Irreps
    denominator: float
    sh_lmax: int
    tensor_product_type = "gaunt"

    @nn.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            sh = e3nn.spherical_harmonics(
                list(range(1, self.sh_lmax + 1)),
                positions[graphs.receivers] - positions[graphs.senders],
                True,
            )
            sender_features = e3nn.as_irreps_array(sender_features)

            if self.tensor_product_type == "usual":
                tp = TensorProduct()(sender_features, sh)
            elif self.tensor_product_type == "gaunt":
                tp1 = GeneralGauntTensorProduct(p_val1=1, p_val2=1)(sender_features, sh)
                tp2 = GeneralGauntTensorProduct(p_val1=1, p_val2=-1)(sender_features, sh)
                tp3 = GeneralGauntTensorProduct(p_val1=-1, p_val2=1)(sender_features, sh)
                tp = e3nn.concatenate([tp1, tp2, tp3])
            else:
                raise ValueError(f"Unknown tensor product type: {self.tensor_product_type}")

            return e3nn.concatenate(
                [sender_features, tp]
            ).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_post", force_irreps_out=True)(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps, name="shortcut", force_irreps_out=True
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class Model(nn.Module):
    @nn.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        graphs = graphs._replace(nodes=jnp.ones((len(positions), 1)))

        layers = 2 * ["32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o"] + ["0o + 7x0e"]
        for irreps in layers:
            graphs = Layer(irreps, denominator=1.5, sh_lmax=3)(graphs, positions)

        # Readout logits
        pred = e3nn.scatter_sum(
            graphs.nodes.array, nel=graphs.n_node
        )  # [num_graphs, 1 + 7]
        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)
        assert logits.shape == (len(graphs.n_node), 8), logits.shape  # [num_graphs, num_classes]

        return logits
    

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




def train(steps=200):
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
    bar = tqdm(range(steps))
    for step in bar:
        params, opt_state, accuracy, preds = update_fn(params, opt_state, graphs)
        
        bar.set_postfix(accuracy=f"{100 * accuracy:.2f}%")    
        if accuracy == 1.0:
            break

    print(f"Final accuracy = {100 * accuracy:.2f}%")
    print("Final prediction:", preds)


if __name__ == "__main__":
    train()
