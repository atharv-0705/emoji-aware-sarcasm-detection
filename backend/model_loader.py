import os, json, pickle, gc

# ══════════════════════════════════════════════════════════════
# CRITICAL: These MUST be set BEFORE importing TensorFlow.
# They prevent memory allocation failures on machines with
# limited available RAM (< 8GB free). Uses direct assignment
# (not setdefault) to guarantee they always take effect.
# ══════════════════════════════════════════════════════════════
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"            # Disable MKL/oneDNN allocator
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"              # Suppress TF info/warnings
os.environ["PYTHONIOENCODING"] = "utf-8"               # Prevent emoji print crashes

import numpy as np
import tensorflow as tf
import h5py
from pathlib import Path
from typing import Any, Dict, Optional
from tensorflow.keras.layers import InputLayer, MultiHeadAttention
from tensorflow.keras.mixed_precision import Policy

# Limit TF threads to reduce memory footprint (~30-40% savings)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ── Paths (relative to this file) ─────────────────────────────
BASE_DIR = Path.cwd()
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH     = ARTIFACTS_DIR / "model.h5"
WORD2IDX_PATH  = ARTIFACTS_DIR / "word2idx.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
CONFIG_PATH    = ARTIFACTS_DIR / "config.json"


# ══════════════════════════════════════════════════════════════
# KERAS 3 → KERAS 2 COMPATIBILITY ADAPTER
# ══════════════════════════════════════════════════════════════
# The model was saved with Keras 3, but this environment runs
# Keras 2 (bundled with TensorFlow 2.15.0). The two formats
# differ in how layers, dtypes, and graph connections are
# serialized inside the .h5 file.
#
# This adapter intercepts the raw JSON config at the h5py
# boundary and translates it before Keras ever sees it.
# ══════════════════════════════════════════════════════════════

class DTypePolicy(Policy):
    """Keras 3 serializes dtype as a DTypePolicy object.
    Keras 2 expects a plain string like 'float32'."""
    def __init__(self, name="float32"):
        if isinstance(name, dict):
            name = name.get("name", "float32")
        super().__init__(name)


# Global registry to store MHA query shapes extracted from
# the Keras 3 tensor history before they are stripped.
MHA_SHAPES = {}


def convert_inbound_nodes(keras3_nodes, class_name=None):
    """Translate Keras 3 inbound_nodes (list-of-dicts with args/kwargs)
    into Keras 2 format (list-of-lists with [layer, node, tensor, kwargs])."""
    if not isinstance(keras3_nodes, list):
        return keras3_nodes

    keras2_nodes = []
    for node in keras3_nodes:
        # Already in Keras 2 format
        if isinstance(node, list):
            keras2_nodes.append(node)
            continue

        if not isinstance(node, dict):
            continue

        args = node.get("args", [])
        kwargs = node.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        else:
            kwargs = dict(kwargs)

        connections = []

        def find_tensors(item):
            if isinstance(item, dict):
                if item.get("class_name") == "__keras_tensor__":
                    history = item.get("config", {}).get("keras_history", None)
                    if isinstance(history, list) and len(history) >= 3:
                        connections.append([history[0], history[1], history[2]])
                else:
                    for val in item.values():
                        find_tensors(val)
            elif isinstance(item, list):
                for val in item:
                    find_tensors(val)

        find_tensors(args)

        if len(connections) == 0:
            continue

        if class_name == "MultiHeadAttention":
            query_conn = connections[0]
            if len(connections) >= 3:
                value_conn = connections[1]
                key_conn = connections[2]
            else:
                value_conn = query_conn
                key_conn = query_conn

            mha_kwargs = dict(kwargs)
            mha_kwargs["value"] = [value_conn[0], value_conn[1], value_conn[2]]
            mha_kwargs["key"] = [key_conn[0], key_conn[1], key_conn[2]]

            keras2_nodes.append([[query_conn[0], query_conn[1], query_conn[2], mha_kwargs]])
        else:
            keras2_connections = []
            for conn in connections:
                keras2_connections.append([conn[0], conn[1], conn[2], kwargs])
            keras2_nodes.append(keras2_connections)

    return keras2_nodes


def clean_layer_config(config, class_name=None):
    """Remove or translate Keras 3-only keys from a layer config dict."""
    if not isinstance(config, dict):
        return

    # 1. Remove Keras 3-only keys
    config.pop("quantization_config", None)
    config.pop("rms_scaling", None)

    # 2. Translate DTypePolicy object → plain string
    if "dtype" in config and isinstance(config["dtype"], dict):
        dtype_dict = config["dtype"]
        if isinstance(dtype_dict, dict) and dtype_dict.get("class_name") == "DTypePolicy":
            config["dtype"] = dtype_dict.get("config", {}).get("name", "float32")
        else:
            config.pop("dtype", None)

    # 3. InputLayer: batch_shape → input_shape, remove 'optional'
    if class_name == "InputLayer" or config.get("name") == "Input":
        if "batch_shape" in config:
            batch_shape = config.pop("batch_shape")
            if batch_shape and len(batch_shape) > 1:
                config["input_shape"] = batch_shape[1:]
        config.pop("optional", None)

    # 4. MultiHeadAttention: inject shapes, remove 'seed'
    if "Attention" in str(class_name) or class_name == "MultiHeadAttention":
        layer_name = config.get("name")
        shape = MHA_SHAPES.get(layer_name, [None, 25, 128])
        print(f"  [PATCH] Injecting shape {shape} for MHA layer '{layer_name}'")

        if "query_shape" not in config:
            config["query_shape"] = shape
        if "value_shape" not in config:
            config["value_shape"] = shape
        if "key_shape" not in config:
            config["key_shape"] = shape
        config.pop("seed", None)

    # 5. Recurse into nested layer configs (e.g. TimeDistributed)
    if "layer" in config and isinstance(config["layer"], dict):
        inner_class = config["layer"].get("class_name")
        inner_config = config["layer"].get("config")
        if isinstance(inner_config, dict):
            clean_layer_config(inner_config, inner_class)
        else:
            clean_layer_config(config["layer"])


def clean_model_config(config):
    """Walk the full model config and translate every layer."""
    if not isinstance(config, dict):
        return

    model_config = config.get("config", config) if "class_name" in config else config

    # Fix input/output layers (Keras 3 may omit wrapping list)
    input_layers = model_config.get("input_layers")
    if isinstance(input_layers, list) and len(input_layers) > 0 and not isinstance(input_layers[0], list):
        model_config["input_layers"] = [input_layers]

    output_layers = model_config.get("output_layers")
    if isinstance(output_layers, list) and len(output_layers) > 0 and not isinstance(output_layers[0], list):
        model_config["output_layers"] = [output_layers]

    # Iterate every layer and translate
    for layer_cfg in model_config.get("layers", []):
        class_name = layer_cfg.get("class_name")
        layer_name = layer_cfg.get("name")

        # Extract MHA shapes from Keras 3 tensor history BEFORE conversion
        if "inbound_nodes" in layer_cfg:
            orig = layer_cfg["inbound_nodes"]

            if class_name == "MultiHeadAttention" and isinstance(orig, list) and len(orig) > 0:
                node = orig[0]
                if isinstance(node, dict):
                    args = node.get("args", [])
                    if isinstance(args, list) and len(args) > 0:
                        q_tensor = args[0]
                        if isinstance(q_tensor, dict):
                            q_shape = q_tensor.get("config", {}).get("shape")
                            if q_shape:
                                MHA_SHAPES[layer_name] = q_shape
                                print(f"  [PATCH] Extracted MHA shape {q_shape} for '{layer_name}'")

            layer_cfg["inbound_nodes"] = convert_inbound_nodes(orig, class_name)

        # Clean inner config
        inner_config = layer_cfg.get("config", {})
        clean_layer_config(inner_config, class_name)


# ── Monkeypatch h5py to intercept model_config reads ──────────
_original_h5_getitem = h5py.AttributeManager.__getitem__

def _patched_h5_getitem(self, name):
    val = _original_h5_getitem(self, name)
    if name == "model_config":
        print("  [PATCH] Intercepted model_config from .h5 file")
        if isinstance(val, bytes):
            val_str = val.decode("utf-8")
        else:
            val_str = val

        config = json.loads(val_str)
        clean_model_config(config)

        new_val_str = json.dumps(config)
        print("  [PATCH] Config translated to Keras 2 format successfully")
        if isinstance(val, bytes):
            return new_val_str.encode("utf-8")
        return new_val_str
    return val

h5py.AttributeManager.__getitem__ = _patched_h5_getitem

_original_h5_get = h5py.AttributeManager.get

def _patched_h5_get(self, name, default=None):
    if name == "model_config":
        try:
            return self[name]
        except KeyError:
            return default
    return _original_h5_get(self, name, default)

h5py.AttributeManager.get = _patched_h5_get

# ── Monkeypatch from_config for extra safety ──────────────────
_original_input_from_config = InputLayer.from_config

def _patched_input_from_config(config):
    clean_layer_config(config, "InputLayer")
    return _original_input_from_config.__func__(InputLayer, config)

InputLayer.from_config = _patched_input_from_config

_original_mha_from_config = MultiHeadAttention.from_config

def _patched_mha_from_config(config):
    clean_layer_config(config, "MultiHeadAttention")
    return _original_mha_from_config.__func__(MultiHeadAttention, config)

MultiHeadAttention.from_config = _patched_mha_from_config

_original_layer_from_config = tf.keras.layers.Layer.from_config

@classmethod
def _patched_layer_from_config(cls, config):
    clean_layer_config(config, cls.__name__)
    return _original_layer_from_config.__func__(cls, config)

tf.keras.layers.Layer.from_config = _patched_layer_from_config

_original_deserialize = tf.keras.layers.deserialize

def _patched_deserialize(config, custom_objects=None):
    if isinstance(config, dict):
        inner_config = config.get("config", None)
        if isinstance(inner_config, dict):
            clean_layer_config(inner_config, config.get("class_name"))
        else:
            clean_layer_config(config, config.get("class_name"))
    return _original_deserialize(config, custom_objects=custom_objects)

tf.keras.layers.deserialize = _patched_deserialize


# ══════════════════════════════════════════════════════════════
# SINGLETON MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Singleton class to hold all loaded artifacts.
    Avoids reloading on every request.
    """
    model:     Optional[tf.keras.Model] = None
    word2idx:  Optional[Dict[str, int]] = None
    threshold: float = 0.449
    config:    Dict[str, Any] = {}
    is_ready:  bool = False
    load_error: Optional[str] = None


registry = ModelRegistry()


def load_all_artifacts() -> None:
    """
    Load model + all artifacts.
    Called once at FastAPI startup event.
    """
    print("\n" + "=" * 55)
    print("  LOADING SARCOJI MODEL ARTIFACTS")
    print("=" * 55)

    errors = []

    # ── 1. Load Config ─────────────────────────────────────────
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                registry.config = json.load(f)
            print(f"  [OK] config.json loaded  (v{registry.config.get('version', '?')})")
        else:
            print("  [WARN] config.json not found — using defaults")
            registry.config = {"max_len": 50, "version": "1.0.0"}
    except Exception as e:
        print(f"  [ERROR] config.json error: {e}")
        errors.append(str(e))
        registry.config = {"max_len": 50}

    MAX_LEN = registry.config.get("max_len", 50)

    # ── 2. Load word2idx ──────────────────────────────────────
    try:
        if not WORD2IDX_PATH.exists():
            raise FileNotFoundError(
                f"word2idx.pkl not found at {WORD2IDX_PATH}\n"
                "  → Run save_artifacts.py in Colab first!"
            )
        with open(WORD2IDX_PATH, "rb") as f:
            registry.word2idx = pickle.load(f)

        # Also inject into preprocess module
        from preprocess import load_vocab
        load_vocab(str(WORD2IDX_PATH))

        print(f"  [OK] word2idx.pkl loaded  ({len(registry.word2idx):,} tokens)")
    except Exception as e:
        print(f"  [ERROR] word2idx.pkl error: {e}")
        errors.append(str(e))

    # ── 3. Load Keras model ────────────────────────────────────
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"model.h5 not found at {MODEL_PATH}\n"
                "  → Copy your trained model.h5 to backend/artifacts/"
            )

        # Free as much memory as possible before loading the model
        gc.collect()

        # Attempt to load with retry on transient memory errors
        model_loaded = False
        for attempt in range(3):
            try:
                registry.model = tf.keras.models.load_model(
                    str(MODEL_PATH),
                    compile=False,       # Don't need optimizer for inference
                    custom_objects={"DTypePolicy": DTypePolicy}
                )
                model_loaded = True
                break
            except Exception as load_err:
                err_str = str(load_err).lower()
                if "bad allocation" in err_str or "oom" in err_str or "memory" in err_str:
                    print(f"  [WARN] Model load attempt {attempt+1}/3 failed (memory): {load_err}")
                    gc.collect()
                    import time; time.sleep(2)
                    continue
                else:
                    raise  # Non-memory errors should propagate immediately

        if not model_loaded:
            raise MemoryError(
                "Failed to load model after 3 attempts due to insufficient memory. "
                "Close other applications to free RAM and restart the server."
            )

        params = registry.model.count_params()
        print(f"  [OK] model.h5 loaded  ({params:,} parameters)")
        print(f"     Input  shape : {registry.model.input_shape}")
        print(f"     Output shape : {registry.model.output_shape}")

        # Warmup pass (optional — reduces first-prediction latency)
        try:
            dummy = np.zeros((1, MAX_LEN), dtype=np.int32)
            _ = registry.model.predict(dummy, verbose=0)
            print(f"  [OK] Warmup prediction succeeded")
        except Exception as warmup_err:
            print(f"  [WARN] Warmup prediction failed (non-fatal): {warmup_err}")
            print(f"     First real prediction may be slightly slower.")

    except Exception as e:
        print(f"  [ERROR] model.h5 error: {e}")
        errors.append(str(e))

    # ── 4. Load Threshold ──────────────────────────────────────
    try:
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                t_data = json.load(f)
            registry.threshold = float(t_data.get("threshold", 0.449))
            method = t_data.get("method", "unknown")
            print(f"  [OK] threshold.json loaded  ({registry.threshold:.4f} — {method})")
        else:
            print("  [WARN] threshold.json not found — using default 0.449")
            registry.threshold = 0.449
    except Exception as e:
        print(f"  [WARN] threshold.json error: {e} — using 0.449")
        registry.threshold = 0.449

    # ── Final status ───────────────────────────────────────────
    if errors:
        registry.is_ready   = False
        registry.load_error = " | ".join(errors)
        print(f"\n  [WARN] Model NOT ready — {len(errors)} error(s)")
    else:
        registry.is_ready   = True
        registry.load_error = None
        print("\n  [OK] ALL ARTIFACTS LOADED — Model is READY")

    print("=" * 55 + "\n")


def get_registry() -> ModelRegistry:
    """Dependency injection for FastAPI routes."""
    return registry
