# Gridsearch and cross-validation can be added later for hyperparameter tuning if needed.

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def GridS(x_train, y_train, x_test, y_test):
    param_grid = {'n_neighbors': np.arange(1, 20),
                'metric': ['euclidean', 'manhattan']}

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv= 5)

    grid.fit(x_train, y_train)

    print("Best score (CV):", grid.best_score_)
    print("Best parameters:", grid.best_params_)

    model = grid.best_estimator_

    test_score = model.score(x_test, y_test)
    print("Test set score:", test_score)

    model.score(x_test, y_test)

    return model, grid.best_params_, test_score

# Exemple d'utilisation 

# model, best_params, test_score = GridS(X_train, y_train, X_test, y_test)