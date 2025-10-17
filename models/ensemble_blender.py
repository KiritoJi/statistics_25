import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

class EnsembleBlender:
    def __init__(self, weights=None, method='weighted'):
        self.weights = np.array(weights or [0.25,0.25,0.25,0.15,0.10])
        self.method = method

    def blend_weighted(self, preds):
        return np.average(preds, axis=0, weights=self.weights)

    def blend_stacking(self, oof_preds, y_true, test_preds):
        oof_stack = np.vstack(oof_preds).T
        test_stack = np.vstack(test_preds).T
        meta = LogisticRegression(max_iter=1000, class_weight='balanced')
        meta.fit(oof_stack, y_true)
        return meta.predict_proba(test_stack)[:,1]

    def run(self, oof_preds, test_preds, y_true=None):
        if self.method == 'stacking' and y_true is not None:
            return self.blend_stacking(oof_preds, y_true, test_preds)
        else:
            return self.blend_weighted(test_preds)
