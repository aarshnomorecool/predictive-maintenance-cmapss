"""The LSTM's internal view of an engine, as vectors you can search and plot.

Before the LSTM predicts a number, its second layer compresses the last 30
cycles of 17 features into 32 numbers. That 32-dimensional vector is the
model's own summary of "what this engine looks like right now", and engines
in a similar state end up with similar vectors. This module:

* encodes every cycle of every engine into that vector space,
* builds a nearest-neighbour index over the *training* engines only, whose
  full run to failure is known, so a test engine can be compared with
  engines that already failed ("engines that looked like this one lasted
  6, 9 and 11 more cycles"),
* projects the vectors to 3D with PCA, fitted on training engines, for the
  dashboard's point cloud,
* exposes every layer's activations and the trained weights for the
  dashboard's network diagram.

Search is exact (sklearn NearestNeighbors). With about 20,000 reference
vectors of 32 numbers it answers in milliseconds, so an approximate index
like FAISS or HNSW, which is what a vector database uses at millions of
vectors, would add complexity for no gain here.

    python -m src.latent_space   # checks the space is meaningful
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data_pipeline import load_train
from src.inference import RulArtifacts
from src.preprocessing import apply_scaler, make_windows_every_cycle

N_NEIGHBORS = 8


def _lstm_layers(model):
    import tensorflow as tf
    return [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]


def layer_model(model):
    """A model returning every computing layer's output in one pass.

    Dropout layers are skipped: at inference they pass values through
    unchanged, so drawing them would show a copy of the previous layer.
    """
    import tensorflow as tf
    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.Dropout)]
    return tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in layers]), layers


def encode(model, X: np.ndarray) -> np.ndarray:
    """The 32-number summary from the last LSTM layer, for every window."""
    import tensorflow as tf
    encoder = tf.keras.Model(inputs=model.inputs, outputs=_lstm_layers(model)[-1].output)
    return encoder.predict(X, batch_size=2048, verbose=0)


@dataclass
class LatentSpace:
    """Training engines encoded, indexed and projected."""

    reference: pd.DataFrame       # unit, cycle, life_left (true, unclipped)
    vectors: np.ndarray           # (n_reference, 32)
    coords: np.ndarray            # (n_reference, 3) PCA projection
    pca: PCA
    index: NearestNeighbors

    @property
    def explained_variance(self) -> list[float]:
        return [float(v) for v in self.pca.explained_variance_ratio_]

    def project(self, vectors: np.ndarray) -> np.ndarray:
        return self.pca.transform(vectors)

    def neighbors(self, vectors: np.ndarray, k: int = N_NEIGHBORS):
        """Indices into ``reference`` and distances, nearest first."""
        distances, indices = self.index.kneighbors(vectors, n_neighbors=k)
        return indices, distances

    def knn_life_estimate(self, vectors: np.ndarray, k: int = N_NEIGHBORS) -> np.ndarray:
        """Average remaining life of the k most similar training moments,
        capped like the LSTM's target so the two are comparable."""
        indices, _ = self.neighbors(vectors, k)
        life = np.minimum(self.reference["life_left"].to_numpy(), config.RUL_CLIP)
        return life[indices].mean(axis=1)


def build_latent_space(rul: RulArtifacts, subset: str = config.SUBSET) -> LatentSpace:
    """Encode every cycle of the 100 training engines, which all ran to failure.

    Includes the units the LSTM used for early stopping: their weights were
    never fitted to them, and their true remaining life is just as known.
    Test engines are never added, so a test engine's neighbours are always
    other engines, never itself.
    """
    train = load_train(subset)
    last = train.groupby(config.UNIT_COL)[config.CYCLE_COL].transform("max")
    train = train.assign(life_left=last - train[config.CYCLE_COL])

    scaled = apply_scaler(train, rul.scaler, rul.feature_cols)
    X, index = make_windows_every_cycle(scaled, rul.feature_cols, rul.window_size)
    reference = index.merge(
        train[[config.UNIT_COL, config.CYCLE_COL, "life_left"]],
        on=[config.UNIT_COL, config.CYCLE_COL], how="left", validate="one_to_one",
    )

    vectors = encode(rul.model, X)
    pca = PCA(n_components=3, random_state=config.RANDOM_SEED).fit(vectors)
    return LatentSpace(
        reference=reference,
        vectors=vectors,
        coords=pca.transform(vectors),
        pca=pca,
        index=NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(vectors),
    )


def encode_frame(rul: RulArtifacts, raw_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Vectors for every cycle of the engines in ``raw_df``, plus unit/cycle."""
    scaled = apply_scaler(raw_df, rul.scaler, rul.feature_cols)
    X, index = make_windows_every_cycle(scaled, rul.feature_cols, rul.window_size)
    return encode(rul.model, X), index


# ------------------------------------------------------- network diagram ----

def connection_strengths(model) -> list[dict]:
    """How strongly each unit feeds each unit of the next layer.

    Dense layers: the absolute weight. LSTM layers have four gates (input,
    forget, cell, output) per unit, so the strength from input i to unit j
    is the summed absolute weight across all four gates. Recurrent weights
    (a unit feeding itself over time) are not drawn.
    """
    import tensorflow as tf
    out = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            kernel = np.abs(layer.get_weights()[0])            # (n_in, 4 * units)
            units = layer.units
            strength = kernel.reshape(kernel.shape[0], 4, units).sum(axis=1)
            out.append({"name": layer.name, "kind": "LSTM", "units": units, "weights": strength})
        elif isinstance(layer, tf.keras.layers.Dense):
            kernel = np.abs(layer.get_weights()[0])
            out.append({"name": layer.name, "kind": "Dense", "units": kernel.shape[1], "weights": kernel})
    return out


def activations(rul: RulArtifacts, window: np.ndarray) -> list[np.ndarray]:
    """Every layer's output for one scaled window of shape (30, n_features)."""
    model, _ = layer_model(rul.model)
    outputs = model.predict(window[None], verbose=0)
    return [np.asarray(o)[0] for o in outputs]


# ------------------------------------------------------------------- cli ----

def main() -> None:
    """Does the space mean anything? Measured, not eyeballed."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from src.data_pipeline import build_dataset
    from src.inference import load_rul_artifacts

    rul = load_rul_artifacts()
    space = build_latent_space(rul)
    print(f"\nReference set: {len(space.reference):,} moments from "
          f"{space.reference[config.UNIT_COL].nunique()} training engines, "
          f"{space.vectors.shape[1]}-dimensional vectors")
    ev = space.explained_variance
    print(f"PCA to 3D keeps {sum(ev):.1%} of the variance "
          f"({', '.join(f'{v:.1%}' for v in ev)})")

    test = build_dataset(rul.metadata["subset"]).test
    vectors, index = encode_frame(rul, test)
    last = index.groupby(config.UNIT_COL).cumcount(ascending=False).to_numpy() == 0
    truth = (test.sort_values([config.UNIT_COL, config.CYCLE_COL])
             .groupby(config.UNIT_COL)[config.RUL_COL].last().to_numpy())

    knn = space.knn_life_estimate(vectors[last])
    rmse = float(np.sqrt(np.mean((knn - truth) ** 2)))
    base = float(np.sqrt(np.mean((truth.mean() - truth) ** 2)))
    print(f"\nTest engines, last cycle: averaging the remaining life of the "
          f"{N_NEIGHBORS} nearest training moments")
    print(f"  kNN in LSTM space    RMSE {rmse:6.2f} cycles")
    print(f"  LSTM itself          RMSE {rul.metadata['metrics']['LSTM']['test']['rmse']:6.2f} cycles")
    print(f"  constant guess       RMSE {base:6.2f} cycles")
    print("If neighbours in this space did not share remaining life, the kNN\n"
          "estimate would be no better than the constant guess.")


if __name__ == "__main__":
    main()
