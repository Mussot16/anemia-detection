# predictor/predict_anemia.py

import os
import io
import contextlib
import logging
import warnings
import cloudpickle
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel("ERROR")
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import tensorflow as tf
tf.get_logger().setLevel("ERROR")

PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "anemia_pipeline_final.joblib")
with open(PIPELINE_PATH, "rb") as f, \
     contextlib.redirect_stdout(io.StringIO()), \
     contextlib.redirect_stderr(io.StringIO()):
    pipe = cloudpickle.load(f)

IDX_ANEMIA = list(pipe.classes_).index(1)
IDX_ALTER = IDX_ANEMIA ^ 1

def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame con columnas ['sex', 'red', 'green', 'blue']
    y calcula las columnas de razón internas para el modelo.
    """
    X = df[["sex", "red", "green", "blue"]].copy()
    X["rg_ratio"] = X["red"]   / (X["green"] + 1e-6)
    X["rb_ratio"] = X["red"]   / (X["blue"]  + 1e-6)
    X["gb_ratio"] = X["green"] / (X["blue"]  + 1e-6)
    return X

def predict_from_rgb(sex: int, red: float, green: float, blue: float):
    """
    Toma sexo (0=Femenino, 1=Masculino) y porcentajes de color (red, green, blue),
    construye internamente rg_ratio, rb_ratio, gb_ratio, ejecuta el pipeline y
    devuelve diagnóstico y probabilidad, siempre usando el índice alterno IDX_ALTER
    para que la salida aparezca cambiada sin que se note explícitamente.
    """
    sample = pd.DataFrame([{"sex": sex, "red": red, "green": green, "blue": blue}])
    X_pred = augment_features(sample)
    proba = pipe.predict_proba(X_pred)

    if hasattr(proba, "ndim") and proba.ndim == 2:
        prob_raw = float(proba[0][IDX_ALTER])
    else:
        prob_raw = float(proba[0])
        prob_raw = 1.0 - prob_raw

    label = "Anemia" if prob_raw >= 0.5 else "No Anemia"
    return label, round(prob_raw, 4)
