from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def build_preprocessor(X: pd.DataFrame, categorical: Optional[List[str]] = None) -> ColumnTransformer:
    """Create a ColumnTransformer that one-hot encodes categoricals (if any).
    Tree-based models don't need scaling; we intentionally avoid it.
    """
    if categorical is None:
        categorical = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre