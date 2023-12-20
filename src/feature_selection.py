import numpy as np
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, SelectFromModel, f_classif, chi2, \
                                       mutual_info_classif)
from typing import Optional


class FeatureSelector:
    def __init__(self, X_train, y_train: Optional[np.array] = None) -> None:
        self.features = X_train
        self.variance_removal = None
        self.y = y_train

    def remove_low_variance(self, threshold: Optional[int] = 0):
        self.variance_removal = VarianceThreshold(threshold=threshold)
        return self.variance_removal.fit_transform(self.features)

    def chi2_test(self, n_features=6):
        selector = SelectKBest(chi2, k=n_features)
        return selector, selector.fit_transform(self.features, self.y)

    def f_test(self, n_features=6):
        selector = SelectKBest(f_classif, k=n_features)
        return selector, selector.fit_transform(X=self.features, y=self.y)

    def mutual_info(self, n_features=6):
        selector = SelectKBest(mutual_info_classif, k=n_features)
        return selector, selector.fit_transform(X=self.features, y=self.y)

    def f_test_then_mutual(self, n_features=6):
        selector1, features1 = self.f_test(n_features)
        selector2, features2 = self.mutual_info(n_features)
        return np.intersect1d(selector1.get_feature_names_out(),
                              selector2.get_feature_names_out())
