"""Prediction core for Log_ASR / ASR inference.

This module is intentionally UI-independent.  It loads the saved sklearn
pipelines, executes the descriptor generator from the supplied notebook, adds
the electrolyte one-hot columns used during training, and returns predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_METADATA = PROJECT_ROOT / "run_metadata.json"
DEFAULT_FEATURE_NOTEBOOK = PROJECT_ROOT / "training" / "00_general_feature_generator_v2.ipynb"
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "data" / "data_923K_2026_09_09_v2.xlsx"

ELECTROLYTES = ("SDC", "GDC", "perovskite", "zirconia", "mixed")
MODEL_NAMES = ("ann", "rf", "svr")


class _PCAApplicability:
    """Reproduce the PCA/kNN applicability calculation from the PCA notebook."""

    def __init__(self, training_file: Path, feature_columns: list[str]) -> None:
        train = pd.read_excel(training_file, sheet_name="features")
        # PCA uses the trained numeric schema, including electrolyte one-hot columns.
        cols = [c for c in feature_columns if c in train.columns]
        if len(cols) < 2:
            raise ValueError("PCA 可用特征少于 2 个")
        missing_rate = train[cols].isna().mean()
        cols = [c for c in cols if missing_rate[c] <= 0.40 and train[c].nunique(dropna=True) > 1]
        if len(cols) < 2:
            raise ValueError("PCA 删除缺失率过高/常数特征后少于 2 个特征")
        self.feature_columns = cols
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        train_z = scaler.fit_transform(imputer.fit_transform(train[cols]))
        train_scores = pca.fit_transform(train_z)

        reference_idx, calibration_idx = train_test_split(
            np.arange(len(train)), test_size=0.20, shuffle=True, random_state=42
        )
        self.imputer, self.scaler, self.pca = imputer, scaler, pca
        self.pc_scaler = StandardScaler().fit(train_scores[reference_idx])
        reference_pc_z = self.pc_scaler.transform(train_scores[reference_idx])
        calibration_pc_z = self.pc_scaler.transform(train_scores[calibration_idx])
        self.knn = NearestNeighbors(n_neighbors=min(5, len(reference_idx))).fit(reference_pc_z)
        self.calibration_distance = self.knn.kneighbors(calibration_pc_z)[0][:, -1]

    def score(self, frame: pd.DataFrame) -> tuple[float, str, float]:
        x = frame.reindex(columns=self.feature_columns)
        target_scores = self.pca.transform(self.scaler.transform(self.imputer.transform(x)))
        target_pc_z = self.pc_scaler.transform(target_scores)
        distance = float(self.knn.kneighbors(target_pc_z)[0][0, -1])
        p_value = float((1 + (self.calibration_distance >= distance).sum()) / (len(self.calibration_distance) + 1))
        return 100.0 * p_value, ("inside" if p_value > 0.2 else "outside"), distance


def _load_feature_generator(notebook_path: Path):
    """Load cells 1--4 from the supplied notebook without duplicating constants."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "feature_generator_v2"}
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"][1:5]
        if cell.get("cell_type") == "code"
    )
    exec(compile(code, str(notebook_path), "exec"), namespace, namespace)
    return namespace["make_features"], namespace["split_ab"]


class ASRPredictor:
    """Run one-material inference using one of the three trained models."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        metadata_path: str | Path = DEFAULT_METADATA,
        feature_notebook: str | Path = DEFAULT_FEATURE_NOTEBOOK,
        training_data: str | Path = DEFAULT_TRAINING_DATA,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self._make_features, self._split_ab = _load_feature_generator(Path(feature_notebook).resolve())
        self.pca_applicability = _PCAApplicability(Path(training_data).resolve(), self.feature_columns)
        self.models = {}
        for name in MODEL_NAMES:
            path = self.model_dir / f"{name}_final.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"未找到模型文件: {path}")
            self.models[name] = joblib.load(path)
            model_features = list(getattr(self.models[name], "feature_names_in_", []))
            if model_features and model_features != self.feature_columns:
                raise ValueError(f"{name} 模型特征顺序与 run_metadata.json 不一致")

    @staticmethod
    def _structure_type(formula: str, split_ab) -> str:
        a, b, oxygen = split_ab(formula)
        b_total = float(sum(b.values()))
        oxygen_per_b = oxygen / b_total
        if np.isclose(oxygen_per_b, 3.0, atol=1e-6):
            return "ABO3"
        if np.isclose(oxygen_per_b, 2.5, atol=1e-6):
            return "A2B2O5"
        return f"其他氧化物（O/B={oxygen_per_b:.5g}）"

    def build_features(self, formula: str, electrolyte: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        formula = str(formula).strip()
        electrolyte = str(electrolyte).strip()
        if not formula:
            raise ValueError("化学式不能为空")
        if electrolyte not in ELECTROLYTES:
            raise ValueError(f"电解质必须是 {', '.join(ELECTROLYTES)} 之一")
        structure_type = self._structure_type(formula, self._split_ab)
        features, audit = self._make_features(formula)
        row = {f"electrolyte_{name}": float(name == electrolyte) for name in ELECTROLYTES}
        row.update(features)
        missing = [name for name in self.feature_columns if name not in row]
        # v2 notebook deliberately computes a few diagnostic/IP features that
        # are not part of the trained 61-column schema; they are ignored here.
        if missing:
            raise ValueError(f"特征字段不匹配: missing={missing}")
        frame = pd.DataFrame([[row[name] for name in self.feature_columns]], columns=self.feature_columns)
        audit = dict(audit)
        audit["structure_type"] = structure_type
        audit["electrolyte"] = electrolyte
        return frame, audit

    def predict(
        self, formula: str, electrolyte: str, model_name: str = "rf", *, verbose: bool = True
    ) -> dict[str, Any]:
        model_name = str(model_name).lower().strip()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"模型必须是 {', '.join(MODEL_NAMES)} 之一")
        frame, audit = self.build_features(formula, electrolyte)
        log_asr = float(np.asarray(self.models[model_name].predict(frame)).reshape(-1)[0])
        reliability, domain, pca_distance = self.pca_applicability.score(frame)
        result = {
            "model": model_name,
            "formula": str(formula).strip(),
            "electrolyte": electrolyte,
            "structure_type": audit["structure_type"],
            "Log_ASR": round(log_asr, 5),
            "ASR": round(float(10.0 ** log_asr), 5),
            "reliability_score": round(reliability, 5),
            "pca_domain": domain,
            "pca_distance": round(pca_distance, 5),
            "feature_row": frame.iloc[0].to_dict(),
            "audit": audit,
        }
        if verbose:
            print(
                f"模型: {model_name.upper()} | 电解质: {electrolyte} | 材料: {str(formula).strip()}\n"
                f"Log_ASR: {result['Log_ASR']:.5f} | ASR: {result['ASR']:.5f} Ω·cm²\n"
                f"PCA可靠性得分: {result['reliability_score']:.5f}% | 适用域: {domain}"
            )
        return result


__all__ = ["ASRPredictor", "ELECTROLYTES", "MODEL_NAMES"]
