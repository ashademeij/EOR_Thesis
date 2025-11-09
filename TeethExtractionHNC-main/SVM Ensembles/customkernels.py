import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


# Basic overlap kernel for categorical features (Hamming)
class MeanOverlapKernel(BaseEstimator):

    def __init__(self, continuous_vars=None, rbf_gamma=None):
        self.continuous_vars = continuous_vars or []
        self.rbf_gamma = rbf_gamma

    def fit(self, X, y=None):
        # Store column names and split variables
        if not hasattr(X, 'columns'):
            raise ValueError("X must be a pandas DataFrame")
        self.columns_ = X.columns.tolist()
        self.cont_vars_ = [c for c in self.columns_ if c in self.continuous_vars]
        self.cat_vars_ = [c for c in self.columns_ if c not in self.continuous_vars]
        return self

    def _cat_kernel(self, X_cat, Y_cat):
        # Mean-overlap kernel for categorical vars
        n_X, n_Y = len(X_cat), len(Y_cat)
        m = len(self.cat_vars_)
        if m == 0:
            return 0
        Kc = np.zeros((n_X, n_Y), dtype=float)
        for col in self.cat_vars_:
            xv = X_cat[col].values[:, None]
            yv = Y_cat[col].values[None, :]
            Kc += (xv == yv)
        return Kc / m

    def kernel(self, X, Y):
        # Ensure DataFrame format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, columns=self.columns_)

        # Continuous RBF part
        if self.cont_vars_:
            X_cont = X[self.cont_vars_].astype(float).values
            Y_cont = Y[self.cont_vars_].astype(float).values
            gamma = self.rbf_gamma or (1.0 / X_cont.shape[1])
            Kr = rbf_kernel(X_cont, Y_cont, gamma=gamma)
        else:
            Kr = 0

        # Categorical mean-overlap part
        if self.cat_vars_:
            Kc = self._cat_kernel(X[self.cat_vars_], Y[self.cat_vars_])
        else:
            Kc = 0

        # Combine kernels
        return Kr + Kc

    __call__ = kernel

def compute_univariate_kernel(df, alpha=1.0):
    n = len(df)
    probs = {}
    hvals = {}
    for col in df.columns:
        counts = df[col].value_counts()
        pz = (counts / n).to_dict()
        probs[col] = pz
        hvals[col] = {val: (1 - (pz[val]**alpha))**(1/alpha) for val in pz}
    return probs, hvals



# ------------------ Kernel1 ------------------


# Kernel1 proposed by Belanche and Villegas (2013) 
class Kernel1Full(BaseEstimator):
    def __init__(self, continuous_vars=None,
        alpha=1.0, rbf_gamma=None,
        composition='mean', 
        cat_gamma=1.0
    ):

        self.continuous_vars = continuous_vars 
        self.alpha = alpha
        self.rbf_gamma = rbf_gamma
        self.composition = composition
        self.cat_gamma = cat_gamma

    def fit(self, X, y=None):
        # split feature lists
        self.columns_ = X.columns.tolist()
        self.cat_vars_ = [c for c in self.columns_ if c not in self.continuous_vars]
        self.cont_vars_ = self.continuous_vars
        
        # compute categorical h‑values
        if self.cat_vars_:
            _, self.hvals_ = compute_univariate_kernel(X[self.cat_vars_], alpha=self.alpha)
        return self

    def _cat_kernel(self, X_cat, Y_cat):
        # compute categorical Gram
        n_X, n_Y = len(X_cat), len(Y_cat)
        Kc = np.zeros((n_X, n_Y))
        m = len(self.cat_vars_)
        for col in self.cat_vars_:
            x_vals = X_cat[col].values
            y_vals = Y_cat[col].values
            h_col = self.hvals_[col]
            row_h = np.array([h_col.get(v, 0.0) for v in x_vals])
            eq = x_vals[:, None] == y_vals[None, :]
            Kc += eq * row_h[:, None]
        # the mean of all categorical kernel variables
        Kc = Kc / m
        Kc = np.exp(self.cat_gamma * Kc)

        return Kc

    def kernel(self, X, Y):
        # ensure DataFrame format
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, columns=self.columns_)
        
        # continuous part
        if self.cont_vars_:
            X_cont = X[self.cont_vars_].values.astype(float)
            Y_cont = Y[self.cont_vars_].values.astype(float)
            gamma = self.rbf_gamma or (1.0 / X_cont.shape[1])
            Kr = rbf_kernel(X_cont, Y_cont, gamma=gamma)
        else:
            Kr = 0
        
        # categorical part
        if self.cat_vars_:
            Kc = self._cat_kernel(X[self.cat_vars_], Y[self.cat_vars_])
        else:
            Kc = 0
        
        # sum both
        return Kr + Kc

    __call__ = kernel



# ------------------ Kernel2 ------------------
class Kernel2Full(BaseEstimator):
    def __init__(self, continuous_vars=None,
        alpha=1.0, sigma=1.0,
        rbf_gamma=None, gamma=1.0):

        self.continuous_vars = continuous_vars or []
        self.alpha = alpha
        self.sigma = sigma
        self.rbf_gamma = rbf_gamma
        self.gamma = gamma

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()
        self.cont_vars_ = [c for c in self.columns_ if c in self.continuous_vars]
        self.cat_vars_ = [c for c in self.columns_ if c not in self.cont_vars_]

        n = len(X)
        self.counts_ = {
            col: (X[col].value_counts() / n).to_dict()
            for col in self.cat_vars_
        }
        self.hvals_ = {
            col: {val: (1 - p**self.alpha)**(1/self.alpha)
                for val, p in self.counts_[col].items()}
            for col in self.cat_vars_
        }
        return self

    def _cat_kernel(self, X_cat, Y_cat):
        n_X, n_Y = X_cat.shape[0], Y_cat.shape[0]
        Kc = np.zeros((n_X, n_Y))
        d_cat = len(self.cat_vars_)

        for col in self.cat_vars_:
            x = X_cat[col].values
            y = Y_cat[col].values
            p_x = np.array([self.counts_[col].get(v,0) for v in x])
            h_x = np.array([self.hvals_[col].get(v,0) for v in x])
            p_y = np.array([self.counts_[col].get(v,0) for v in y])
            h_y = np.array([self.hvals_[col].get(v,0) for v in y])

            eq = x[:,None] == y[None,:]
            K_eq = h_x[:,None]
            dp = p_x[:,None] - p_y[None,:]
            K_neq = np.minimum(p_x[:,None], p_y[None,:]) * np.exp(- (dp**2) / (self.sigma**2))
        
            Kc += np.where(eq, K_eq, K_neq)

        if d_cat>0:
            Kc = np.exp((self.gamma/d_cat) * Kc)

        return Kc

    def kernel(self, X, Y):
        X = pd.DataFrame(X, columns=self.columns_)
        Y = pd.DataFrame(Y, columns=self.columns_)

        # continuous RBF part
        if self.cont_vars_:
            Xc = X[self.cont_vars_].values.astype(float)
            Yc = Y[self.cont_vars_].values.astype(float)
            gamma = self.rbf_gamma or (1.0 / Xc.shape[1])
            Kr = rbf_kernel(Xc, Yc, gamma=gamma)
        else:
            Kr = 0

        # categorical kernel 
        if self.cat_vars_:
            Kc = self._cat_kernel(X[self.cat_vars_], Y[self.cat_vars_])
        else:
            Kc = 0

        # sum both
        return Kr + Kc

    __call__ = kernel



# ------------------ Grid Search for Hyperparameter Tuning ------------------
def GridSearch_k1_k3(X_train, y_train, cont_cols, class_weights={0: 1, 1: 4}):

    gammas = [2**e for e in range(-3, 3)]          
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
    Cs = [0.1, 1, 10]

    # Initialize variables to track the best model
    best_accuracy = 0
    best_params = {}
    best_model = None

    # Add cross-validation (5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for γ in gammas:
        print(f"Testing gamma={γ}")
        for α in alphas:
            for C in Cs:
                accuracies = []
                
                # Cross-validation loop
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Train kernel and SVC
                    k1 = Kernel1Full(
                        continuous_vars=cont_cols,
                        alpha=α,
                        rbf_gamma=γ,
                        cat_gamma=γ
                    ).fit(X_tr, y_tr)
                    
                    svc1 = SVC(kernel=k1, class_weight=class_weights, C=C)
                    svc1.fit(X_tr, y_tr)
                    y_pred = svc1.predict(X_val)
                    accuracies.append(accuracy_score(y_val, y_pred))
                
                # Average accuracy across folds
                mean_accuracy = np.mean(accuracies)
                
                # Update best model if current is better
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_params = {'gamma': γ, 'alpha': α, 'C': C}
                    best_model = svc1  # Note: This is trained on a subset of data

    print("Best accuracy:", best_accuracy)
    print("Best params:", best_params)
    
    return best_model, best_params, best_accuracy


def GridSearch_k2(X_train, y_train, cont_cols, class_weights={0: 1, 1: 4}):

    gammas = [2**e for e in range(-3, 3)]          
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
    Cs = [0.1, 1, 10]
    sigmas = [0.001, 0.01, 0.05, 0.1]  # Specific to Kernel2Full

    # Initialize variables to track the best model
    best_accuracy = 0
    best_params = {}
    best_model = None

    # Add cross-validation (5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for γ in gammas:
        print(f"Testing gamma={γ}")
        for α in alphas:
            for C in Cs:
                for σ in sigmas:  # Additional loop for sigma parameter
                    accuracies = []
                    
                    # Cross-validation loop
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        # Train kernel and SVC
                        k2 = Kernel2Full(
                            continuous_vars=cont_cols,
                            alpha=α,
                            sigma=σ,  
                            rbf_gamma=γ,
                            gamma=γ
                        ).fit(X_tr, y_tr)
                        
                        svc2 = SVC(kernel=k2, class_weight=class_weights, C=C)
                        svc2.fit(X_tr, y_tr)
                        y_pred = svc2.predict(X_val)
                        accuracies.append(accuracy_score(y_val, y_pred))
                    
                    # Average accuracy across folds
                    mean_accuracy = np.mean(accuracies)
                    
                    # Update best model if current is better
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_params = {
                            'gamma': γ, 
                            'alpha': α, 
                            'C': C,
                            'sigma': σ  
                        }
                        best_model = svc2  

    print("Best accuracy:", best_accuracy)
    print("Best params:", best_params)
    
    return best_model, best_params, best_accuracy