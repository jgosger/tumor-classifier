from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_and_param_grid(model_name: str):
    """
    Returns (pipeline, param_grid) for GridSearchCV.
    model_name: 'knn' | 'logreg' | 'dtree'
    """
    model_name = model_name.lower().strip()

    if model_name == "knn":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ])
        grid = {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # 1=Manhattan, 2=Euclidean
        }
        return pipe, grid

    if model_name == "logreg":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])
        grid = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__solver": ["lbfgs"],
        }
        return pipe, grid

    if model_name == "dtree":
        pipe = Pipeline([
            ("clf", DecisionTreeClassifier(random_state=42))
        ])
        grid = {
            "clf__max_depth": [None, 3, 5, 8, 12],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }
        return pipe, grid

    raise ValueError("model_name must be one of: 'knn', 'logreg', 'dtree'")