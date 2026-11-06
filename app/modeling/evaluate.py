"""
Métricas e gráficos de avaliação — mesmo repertório do Ex1.py (briefing
seção 9): matriz de confusão, curva ROC, curva Precision-Recall, efeito
do threshold, calibração de probabilidades, importância de variáveis
(permutation importance).

Este módulo não seleciona "equipamentos prioritários" nem produz
qualquer ranking — isso é deliberadamente omitido (briefing seção 10).
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # scripts/servidor sem display interativo
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance

from app.modeling.features import FEATURES


def avaliar(y_true, y_pred, prob) -> dict:
    """ROC-AUC e PR-AUC exigem as duas classes presentes em `y_true`. Com
    poucos equipamentos no escopo (ex.: monta-cargas, 6 TAGs), o conjunto
    de teste pode ficar com um único TAG cuja taxa de falha no horizonte é
    0% ou 100% — nesse caso essas duas métricas ficam indefinidas
    (`None`), em vez de estourar uma exceção.
    """
    duas_classes = pd.Series(y_true).nunique() >= 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob) if duas_classes else None,
        "PR-AUC": average_precision_score(y_true, prob) if duas_classes else None,
    }


def gerar_graficos(
    *,
    horizonte: int,
    nome_modelo: str,
    modelo_final,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prob_test: np.ndarray,
    pred_050: np.ndarray,
    pred_otimo: np.ndarray,
    threshold_otimo: float,
    thresholds: np.ndarray,
    precision_threshold: np.ndarray,
    recall_threshold: np.ndarray,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    prefixo = os.path.join(out_dir, f"falha_{horizonte}d")

    duas_classes_teste = y_test.nunique() >= 2
    if not duas_classes_teste:
        print(
            f"[aviso] y_test de falha_{horizonte}d tem uma única classe "
            f"({y_test.iloc[0]}) — provável com poucos TAGs no escopo. "
            "ROC, Precision-Recall, calibração e permutation importance "
            "(que dependem de duas classes) serão pulados para este horizonte."
        )

    # -- Matrizes de confusão ------------------------------------------------
    cm_050 = confusion_matrix(y_test, pred_050)
    cm_otimo = confusion_matrix(y_test, pred_otimo)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cm, titulo in zip(
        axes,
        [cm_050, cm_otimo],
        ["Threshold = 0.50", f"Threshold operacional = {threshold_otimo:.3f}"],
    ):
        ax.imshow(cm)
        ax.set_title(titulo)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1], ["Normal", "Falha"])
        ax.set_yticks([0, 1], ["Normal", "Falha"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    fig.suptitle(f"Matriz de confusão — falha_{horizonte}d ({nome_modelo})")
    plt.tight_layout()
    plt.savefig(f"{prefixo}_matriz_confusao.png", dpi=120)
    plt.close(fig)

    # -- ROC (exige as duas classes em y_test) -------------------------------
    if duas_classes_teste:
        plt.figure(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_test, prob_test, name=nome_modelo)
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"ROC — falha_{horizonte}d (teste)")
        plt.tight_layout()
        plt.savefig(f"{prefixo}_roc.png", dpi=120)
        plt.close()

    # -- Precision-Recall (idem) --------------------------------------------
    if duas_classes_teste:
        plt.figure(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(y_test, prob_test, name=nome_modelo)
        baseline = y_test.mean()
        plt.axhline(baseline, linestyle="--", label=f"Baseline = {baseline:.3f}")
        plt.title(f"Precision-Recall — falha_{horizonte}d (teste)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefixo}_precision_recall.png", dpi=120)
        plt.close()

    # -- Efeito do threshold (curva calculada no dev/OOF) ------------------
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precision_threshold, label="Precision")
    plt.plot(thresholds, recall_threshold, label="Recall")
    plt.axvline(threshold_otimo, linestyle="--", label=f"Threshold escolhido = {threshold_otimo:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Métrica")
    plt.title(f"Trade-off Precision × Recall (OOF, dev) — falha_{horizonte}d")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{prefixo}_threshold.png", dpi=120)
    plt.close()

    # -- Calibração (idem) ----------------------------------------------------
    if duas_classes_teste:
        plt.figure(figsize=(7, 6))
        CalibrationDisplay.from_predictions(y_test, prob_test, n_bins=8, strategy="quantile")
        plt.title(f"Calibração das probabilidades — falha_{horizonte}d")
        plt.tight_layout()
        plt.savefig(f"{prefixo}_calibracao.png", dpi=120)
        plt.close()

    # -- Permutation importance (average_precision também exige duas classes) --
    if duas_classes_teste:
        importancia = permutation_importance(
            modelo_final,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importancia_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": importancia.importances_mean,
                "std": importancia.importances_std,
            }
        ).sort_values("importance")

        plt.figure(figsize=(9, 6))
        plt.barh(importancia_df["feature"], importancia_df["importance"], xerr=importancia_df["std"])
        plt.xlabel("Redução de Average Precision após permutação")
        plt.title(f"Permutation Importance — falha_{horizonte}d")
        plt.tight_layout()
        plt.savefig(f"{prefixo}_permutation_importance.png", dpi=120)
        plt.close()
