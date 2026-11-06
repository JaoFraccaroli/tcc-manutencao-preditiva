"""
Definição dos modelos candidatos — mesma estrutura de comparação do
Ex1.py (Regressão Logística, Random Forest, Gradient Boosting), adaptada
para ter um pré-processador com colunas categóricas (Área, Sist Oper),
que o Ex1.py não precisava (só tinha sensores numéricos).
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from app.modeling.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

RANDOM_STATE = 42


def _preprocessador(escalar: bool) -> ColumnTransformer:
    passos_num = [("imputer", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(passos_num), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def construir_modelos() -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("preprocessador", _preprocessador(escalar=True)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("preprocessador", _preprocessador(escalar=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        [
            ("preprocessador", _preprocessador(escalar=False)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.07,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
    }


PARAM_GRIDS = {
    # "Logistic Regression" fica de fora: seu grid usa np.logspace e é
    # montado diretamente em train.py.
    "Random Forest": {
        "classifier__n_estimators": [200, 300, 400, 600],
        "classifier__max_depth": [None, 6, 10, 14, 20],
        "classifier__min_samples_leaf": [1, 2, 3, 5, 8],
        "classifier__max_features": ["sqrt", 0.5, 0.75],
    },
    "Gradient Boosting": {
        "classifier__learning_rate": [0.02, 0.04, 0.06, 0.08, 0.10],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__l2_regularization": [0, 0.5, 1, 2, 5],
        "classifier__min_samples_leaf": [10, 20, 30, 50],
    },
}
