# File: ML.py
# Import necessary packages
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy as cp

# from tensorflow.keras.callbacks import EarlyStopping


from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import Ridge

from sklearn.base import is_classifier

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Neural Network MLP
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
# from keras import backend as K

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator, RegressorMixin

# SVM
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)
# Decision Tree

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold)

import ML


from pathlib import Path
import os.path
from datetime import datetime

#Linear Regression
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA
from paretochart.paretochart import pareto

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_style("ticks")
sns.set_context("paper")

random_state = 42

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import Lars


from sklearn.ensemble import GradientBoostingRegressor


from datetime import datetime
import os
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score, precision_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LassoLars

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import PoissonRegressor

from lightgbm import LGBMRegressor
from sklearn.svm import NuSVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge


class RidgeWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'RidgeCVRegressor'

    def build_model(self):
        model = Ridge(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)



class RandomForestRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'RandomForestRegressor'

    def build_model(self):
        model = RandomForestRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class NuSVR_RegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'NuSVR'

    def build_model(self):
        model = NuSVR(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)    


    

class DecisionTreeRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'decision_tree_regression'

    def build_model(self):
        model = DecisionTreeRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)    



class LGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'LGBMRegressor'

    def build_model(self):
        model = LGBMRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class LassoLarsRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'LassoLarsRegression'

    def build_model(self):
        model = LassoLars(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)



class LarsRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'LarsRegression'

    def build_model(self):
        model = Lars(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)




class RANSACRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'ransac_regression'

    def build_model(self):
        model = RANSACRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)






class RandomForestRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'random_forest_regression'

    def build_model(self):
        model = RandomForestRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)





class GradientBoostingRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'gradient_boosting_regression'

    def build_model(self):
        model = GradientBoostingRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)



class BayesianRidgeRegressionWrapper(BaseEstimator,RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'gradient_boosting_regression'

    def build_model(self):
        model = BayesianRidge(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)



class KNeighborsRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'k_neighbors_regression'

    def build_model(self):
        model = KNeighborsRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    


    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class ElasticNetCVWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'elastic_net_cv'

    def build_model(self):
        model = ElasticNetCV(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class SVRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'svr'

    def build_model(self):
        model = SVR(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

class HuberRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'HuberRegressor'

    def build_model(self):
        model = HuberRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
        
    def get_params(self, deep=True): 
        return self.model.get_params(deep) 
    
    def set_params(self, **params): 
        return self.model.set_params(**params)


class LinearSVRWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'linear_svr'

    def build_model(self):
        model = LinearSVR(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'mlp_regressor'

    def build_model(self):
        model = MLPRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class PoissonRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'poisson_regression'

    def build_model(self):
        model = PoissonRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class RidgeRegressionWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'ridge_regression'

    def build_model(self):
        model = Ridge(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

class LarsCV(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'LarsCV'

    def build_model(self):
        model = LarsCV(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
        
    def get_params(self, deep=True): 
        return self.model.get_params(deep) 
    
    def set_params(self, **params): 
        return self.model.set_params(**params)


class linear_Regression_Wrapper(BaseEstimator,  RegressorMixin):
    def __init__(self, **kwargs):
        # Initialize model
        self.model = self.build_model()
        self.model_name = 'linear_regression'


    def build_model(self):

        model = LinearRegression()

        return model
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class AdaBoostRegressionWrapper(BaseEstimator, RegressorMixin):
    def _init_(self, **kwargs):
        self.kwargs = kwargs
        self.model = self.build_model()
        self.model_name = 'AdaBoost_Regression'

    def build_model(self):
        model = AdaBoostRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, deep=True): 
        return self.model.get_params(deep) 
    
    def set_params(self, **params): 
        return self.model.set_params(**params)



# class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
#     """
#     Keras-based MLP (multi-layer perceptron) regressor
#     """

#     def __init__(self, **kwargs):
#         # Initialize model
#         self._estimator_type = 'regressor'
#         self.n_neurons = kwargs.get('n_neurons', 2)
#         self.input_dim = kwargs.get('input_dim', None)
#         self.output_dim = kwargs.get('output_dim', 1)
#         self.hidden_neurons_activation_function = kwargs.get('hidden_neurons_activation_function', 'sigmoid')
#         self.output_neurons_activation_function = kwargs.get('output_neurons_activation_function', 'linear')
#         self.epochs = kwargs.get('epochs', 100)
#         self.model = self.build_model()
#         self.estimator = self.model

#         # Model checkpoint
#         self.checkpoint_path = kwargs.get('checkpoint', None)
#         self.monitor = kwargs.get('monitor', 'val_loss')
#         self.mode = kwargs.get('mode', 'auto')
#         self.best_model_checkpoint = None

#         if self.checkpoint_path is None:
#             # Set a default checkpoint file path if None is provided
#             self.checkpoint_path = 'model_checkpoint.h5'

#     def build_model(self):
#         """
#         Build MLP model
#         """
#         print("self.n_neurons:", self.n_neurons)
#         print("self.input_dim:", self.input_dim)
#         model = Sequential()
#         model.add(Dense(self.n_neurons, input_dim=self.input_dim, activation=self.hidden_neurons_activation_function))
#         model.add(Dense(self.output_dim, activation=self.output_neurons_activation_function))
#         model.compile(
#             optimizer=optimizers.SGD(learning_rate=0.1),
#             loss="mean_squared_error",
#             metrics=["mse"]
#         )
#         return model

#     def fit(self, X_train, y_train, X_val, y_val):
#         """
#         Fit model using training and validation data
#         The EarlyStopping callback is defined with the specified parameters.
#         The ModelCheckpoint callback is created with the monitor set to 'val_loss' and the mode set to 'min' 
#         to save the best model based on validation loss.
#         Both the early_stopping_callback and checkpoint_callback are included in the callbacks list.
#         The callbacks list is passed to the fit method of the model.
#         The checkpoint_callback.best value is used to find the best model checkpoint file.
#         The code checks if self.best_model_checkpoint is a string or float and assigns the appropriate value.
#         The weights of the best model are loaded if a valid checkpoint file is available.
#         The history returned by the fit method is assigned to self.history.
#         Now, the fit method includes both the EarlyStopping and ModelCheckpoint callbacks to monitor the training process, 
#         save the best model, and restore the best weights if available.
#                 """

#         early_stopping_callback = EarlyStopping(
#             min_delta=0.001,
#             patience=20,
#             restore_best_weights=True
#         )

#         checkpoint_callback = ModelCheckpoint(
#             self.checkpoint_path,
#             monitor='val_loss',
#             mode='min',
#             save_best_only=True
#         )

#         callbacks = [early_stopping_callback, checkpoint_callback]

#         history = self.model.fit(
#             X_train,
#             y_train,
#             validation_data=(X_val, y_val),
#             callbacks=callbacks,
#             verbose=0,
#             epochs=self.epochs
#     )

#         # Find the best model checkpoint file
#         self.best_model_checkpoint = checkpoint_callback.best
#         if isinstance(self.best_model_checkpoint, str):
#             self.best_model_checkpoint = str(self.best_model_checkpoint)
#         elif isinstance(self.best_model_checkpoint, float):
#             self.best_model_checkpoint = self.checkpoint_path

#         # Load weights if the best model checkpoint is available
#         if self.best_model_checkpoint and isinstance(self.best_model_checkpoint, str):
#             self.model.load_weights(self.best_model_checkpoint)

#         self.history = history

#     def plot_loss(self, )

#     def predict(self, X):
#         """
#         Make regression predictions
#         """
#         return self.model.predict(X)

#     def score(self, X, y):
#         """
#         Compute mean squared error score
#         """
#         y_pred = self.predict(X)
#         return -mean_squared_error(y, y_pred)  # Negate the MSE to align with higher-is-better scoring

#     def get_params(self, deep=True):
#         return self.model.get_params(deep)

#     def set_params(self, **params):
#         return self.model.set_params(**params)

class MLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Keras-based MLP (multi-layer perceptron) regressor
    """

    def __init__(self, **kwargs):
        # Initialize model
        self._estimator_type = 'regressor'
        self.n_neurons = kwargs.get('n_neurons', 2)
        self.input_dim = kwargs.get('input_dim', None)
        self.output_dim = kwargs.get('output_dim', 1)
        self.hidden_neurons_activation_function = kwargs.get('hidden_neurons_activation_function', 'sigmoid')
        self.output_neurons_activation_function = kwargs.get('output_neurons_activation_function', 'linear')
        self.epochs = kwargs.get('epochs', 100)
        self.model = self.build_model()
        self.estimator = self.model

        # Model checkpoint
        self.checkpoint_path = kwargs.get('checkpoint', None)
        self.monitor = kwargs.get('monitor', 'val_loss')
        self.mode = kwargs.get('mode', 'auto')
        self.best_model_checkpoint = None

        if self.checkpoint_path is None:
            # Set a default checkpoint file path if None is provided
            self.checkpoint_path = 'model_checkpoint.h5'

    def build_model(self):
        """
        Build MLP model
        """
        print("self.n_neurons:", self.n_neurons)
        print("self.input_dim:", self.input_dim)
        model = Sequential()

        #### maybe add flattens at some point??

        #Reshaping the Input: The Flatten layer takes the input tensor from the preceding layer and reshapes it into a single vector. 
        # This means that any structure or dimensions in the input data are flattened and combined into a single dimension. 
        # For example, if the input is a 2D image with dimensions (height, width), the Flatten layer would convert it into a 1D vector by
        #  flattening all the pixels.
        # Suitable for Fully Connected Layers: The purpose of the Flatten layer is to convert the multi-dimensional output from the previous layer 
        #(in this case, the pre-trained base model) into a one-dimensional format that can be processed by fully connected layers. 
        # Fully connected layers require a one-dimensional input, where each neuron is connected to every neuron in the previous layer
        # Preserving Spatial Structure: By flattening the input, the Flatten layer removes the spatial structure of the data. 
        # However, this is often desirable when transitioning from convolutional or pre-trained base models to fully connected layers, 
        # as the subsequent layers can learn to capture higher-level abstract features from the flattened representation.


        model.add(Dense(self.n_neurons, input_dim=self.input_dim, activation=self.hidden_neurons_activation_function))
        model.add(Dense(self.output_dim, activation=self.output_neurons_activation_function))
        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.01),
            loss="mean_squared_error",
            metrics=["mse"]
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit model using training and validation data
        O código define um método chamado fit em uma classe de modelo personalizado.

        `Dentro do método fit, é criado um callback de checkpoint usando a classe ModelCheckpoint. 
        Esse callback será usado para salvar o melhor modelo durante o treinamento. 
        São fornecidos parâmetros para o callback, como o caminho onde o checkpoint será salvo, 
        a métrica a ser monitorada para determinar o melhor modelo e o modo da métrica (por exemplo, se deve ser minimizada ou maximizada).

        O método fit então usa a função fit do modelo Keras subjacente (self.model) para treinar o modelo. 
        Ele passa os dados de treinamento (X_train e y_train), dados de validação (X_val e y_val) e outros parâmetros, como o callback (checkpoint_callback), nível de verbosidade (verbose) e número de épocas (self.epochs).

        Após o treinamento, o método captura o histórico de treinamento retornado pela função fit. 
        O histórico de treinamento contém informações sobre as métricas e valores de perda registrados durante o processo de treinamento.

        Em seguida, o método determina o melhor arquivo de checkpoint do modelo com base no callback ModelCheckpoint. 
        Ele verifica se há um arquivo de checkpoint do melhor modelo disponível e atribui o caminho desse arquivo à variável self.best_model_checkpoint.

        Se um arquivo de checkpoint válido do melhor modelo for encontrado (representado como uma string de caminho), 
        o método carrega os pesos do melhor modelo em self.model usando a função load_weights.

        Por fim, o método atribui o histórico de treinamento (history) à variável self.history. 
        Isso permite que você acesse o histórico de treinamento fora do método fit para análises ou visualizações adicionais.

        Resumindo, o código configura um callback para salvar o melhor modelo durante o treinamento, 
        treina o modelo usando os dados fornecidos, acompanha o arquivo de checkpoint do melhor modelo e o histórico de treinamento, `
        e carrega os pesos do melhor modelo, se disponível
        """

        checkpoint_callback = ModelCheckpoint(
            self.checkpoint_path,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback],
            verbose=0,
            epochs=self.epochs
        )

        # Find the best model checkpoint file
        self.best_model_checkpoint = checkpoint_callback.best
        if isinstance(self.best_model_checkpoint, str):
            self.best_model_checkpoint = str(self.best_model_checkpoint)
        elif isinstance(self.best_model_checkpoint, float):
            self.best_model_checkpoint = self.checkpoint_path

        # Load weights if the best model checkpoint is available
        if self.best_model_checkpoint and isinstance(self.best_model_checkpoint, str):
            self.model.load_weights(self.best_model_checkpoint)

        self.history = history.history

    def predict(self, X):
        """
        Make regression predictions
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Compute mean squared error score
        """
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)  # Negate the MSE to align with higher-is-better scoring

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)
    
    def plot_loss(self):
        fig = plt.Figure()
        plt.plot(self.history['loss'], label='Loss')
        plt.plot(self.history['val_loss'], label= 'Validation Loss')
        plt.legend()
        

class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Keras-based MLP (multi-layer perceptron) classifier
    """

    def __init__(self, **kwargs):
        # Initialize model
        self._estimator_type = 'classifier'
        self.response_method = 'predict_proba'
        self.classes_ = [0, 1]
        self.threshold = 0.5
        self.n_neurons = kwargs.get('n_neurons',2)
        self.input_dim = kwargs.get('input_dim',None)
        self.output_dim = kwargs.get('output_dim',1)
        self.hidden_neurons_activation_function = kwargs.get('hidden_neurons_activation_function', 'tanh')
        self.output_neurons_activation_function = kwargs.get('output_neurons_activation_function', 'sigmoid')
        self.epochs = kwargs.get('epochs', 100)
        self.model = self.build_model()
        self.estimator = self.model

        # Model checkpoint
        self.checkpoint_path = kwargs.get('checkpoint', None)
        self.monitor = kwargs.get('monitor', 'val_loss')
        self.mode = kwargs.get('mode', 'auto')
        self.best_model_checkpoint = None

        if self.checkpoint_path is None:
        # Set a default checkpoint file path if None is provided
            self.checkpoint_path = 'model_checkpoint.h5'

    def build_model(self):
        """
        Build MLP model

        ### should I add units=1?define how many outputs we want.
        """
        print("self.n_neurons:", self.n_neurons)
        print("self.input_dim:", self.input_dim)
        model = Sequential()
        model.add(Dense(self.n_neurons, input_dim=self.input_dim, activation=self.hidden_neurons_activation_function))
        model.add(Dense(self.output_dim, activation=self.output_neurons_activation_function))
        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.1),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit model using training and validation data
        
        """

        checkpoint_callback = ModelCheckpoint(
            self.checkpoint_path,
            monitor=self.monitor,
            mode=self.mode,
            save_best_only=True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback],
            verbose=0,
            epochs=self.epochs
        )

        # Find the best model checkpoint file
        self.best_model_checkpoint = checkpoint_callback.best
        if isinstance(self.best_model_checkpoint, str):
            self.best_model_checkpoint = str(self.best_model_checkpoint)
        elif isinstance(self.best_model_checkpoint, float):
            self.best_model_checkpoint = self.checkpoint_path

        # Load weights if the best model checkpoint is available
        if self.best_model_checkpoint and isinstance(self.best_model_checkpoint, str):
            self.model.load_weights(self.best_model_checkpoint)

        self.history = history

    def predict_proba(self, X):
        """
        Make probability predictions
        """
        y_pred = self.model.predict(X)
        return y_pred

    def predict(self, X):
        """
        Make binary predictions
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba > self.threshold).astype(int)

    def get_sensitivity(self, X, y):
        """
        Compute sensitivity/recall score
        """
        y_pred = self.predict(X)
        return recall_score(y, y_pred)

    def get_precision(self, X, y):
        """
        Compute precision score
        """
        y_pred = self.predict(X)
        return precision_score(y, y_pred)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)
    


def create_classifier(x,column,classifier):
    if x[column]<=classifier:
        return 0
    else:
        return 1
    
def get_PCA_comp(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = X.shape[1]
    pca= PCA(n_components)
    pca.fit(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    var = pca.explained_variance_ratio_.cumsum()

    for i in range(len(var)):
        if var[i]>=0.90:
            num_comp = i
            fig, ax = plt.subplots(figsize=(11, 5))
            pareto(pca.explained_variance_ratio_)
            ax.grid();
            break

    return var, var_ratio, num_comp

def plot_distribution_base(data):
    num_plots = data.shape[1]
    num_rows = (num_plots + 3) // 4
    fig = plt.figure(figsize=(12, num_rows*2))

    for i, column in enumerate(data.columns):
        ax = fig.add_subplot(num_rows, 4, i+1)

        sns.histplot(data[column], ax=ax)
        
        ax.grid()
        ax.set_title(f'{column} Distribution')
        ax.set_xlabel(f'{column}')
        ax.set_ylabel('Count')

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.tight_layout()
    plt.show()

def log_transform(X):
    X_log= np.log(X)
    return X_log

def normalize_L1_L2(X_Train, X_Test, L_type):
    transformer = preprocessing.Normalizer(norm=L_type)
    normalized_Xtrain = transformer.transform(X_Train)
    normalized_Xtest = transformer.transform(X_Test)
    return normalized_Xtrain, normalized_Xtest

def get_train_test(X, y, test_size, random_state,**kwargs):
    """
    Method for training multiple models
    """
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X.values,
                                                        y.values,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        **kwargs)

    return X_train_cv,X_test_cv, y_train_cv,y_test_cv

def interpolation(fpr, tpr):
    interp_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(interp_fpr, fpr, tpr)
    interp_tpr[0] = 0.
    return interp_fpr, interp_tpr

def check_vif(columns, df):
    features = columns
    variables = df.loc[:, features]
    vifs_runs = []

    cond = True
    while cond:
        vif_temp = pd.DataFrame()
        vif_temp['VIF'] = 0.0
        vif_temp['Features'] = ''

        for i in range(variables.shape[1]):
            # Calculate the VIF for the current column
            vif_value = variance_inflation_factor(variables.values, i)
            # Assign the VIF value directly to the temporary DataFrame
            vif_temp.loc[i, 'VIF'] = vif_value
            vif_temp.loc[i, 'Features'] = variables.columns[i]

        vifs_runs.append(vif_temp)

        if (vif_temp['VIF'] > 10).any():
            vif_filtered = vif_temp[vif_temp['VIF'] <= 10]
            variables_filtered = df.loc[:, vif_filtered['Features']]
            variables = variables_filtered.copy()
            continue
        else:
            final_vif = vifs_runs[-1]
            initial_vif = vifs_runs[0]
            cond = False

    return initial_vif, final_vif


# def train(X, X_test, y, y_test, model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="", **kwargs):

#     cv = StratifiedKFold(n_splits=n_splits)
#     #Several empty lists are initialized to store the results for each fold:


#     f1_score_val_list   = []
#     f1_score_train_list = []
#     fprs_list           = [] # fprs_list: List to store the false positive rates.
#     tprs_list           = [] # tprs_list: List to store the true positive rates.
#     auc_list            = [] # auc_list: List to store the area under the curve (AUC) scores.
#     scaler_list         = [] # scaler_list: List to store the StandardScaler objects used for feature scaling.
#     model_list          = []

#     X_train_scaled_list = [] # Initialize list to store scaled training data
#     y_train_list        = [] # Initialize list to store scaled training data
#     X_train_list        = [] # Initialize list to store scaled training data

#     precision_list      = []
#     sensitivity_list    = []
#     pcas                = []
#     best_pca = 'Not Applicable'
#     num_comp_list       = []

#     # Create the figure and axes for plotting the ROC curves
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))

#     # Validação cruzada só em Training Data
#     for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):

#         X_train = X[train_idx, :]
#         y_train = y[train_idx]

#         X_val = X[val_idx, :]
#         y_val = y[val_idx]

#         # Escala
#         scaler = StandardScaler()
#         scaler_list.append(scaler)
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_val_scaled = scaler.transform(X_val)

#         #PCA
#         if use_pca:
#             var, var_ratio, num_comp = get_PCA_comp(X_train)
#             pca = PCA(n_components=num_comp)
#             X_train_scaled = pca.fit_transform(X_train_scaled)
#             X_val_scaled = pca.transform(X_val_scaled)
#             print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")

#             # Update the model's input_dim parameter if it exists in the kwargs
#             if 'input_dim' in kwargs:
#                 kwargs['input_dim'] = num_comp

#         X_train_scaled_list.append(X_train_scaled)
#         X_train_list.append(X_train)
#         y_train_list.append(y_train)
        
        
#         f1_score_val=0
#         model = None

#         for idx in range(n_init):

#             _model = model_klass(**kwargs)
#             _model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

#             _y_pred = _model.predict(X_train_scaled)
#             _y_pred_val = _model.predict(X_val_scaled)

#             _f1_score_val = f1_score(y_val, _y_pred_val)
#             if _f1_score_val > f1_score_val:
#                 y_pred_val = _y_pred_val
#                 y_pred = _y_pred
#                 model = _model
#             clear_session()

#         sensitivity = 100 *  recall_score(y_val, y_pred_val)
#         precision= 100 *  precision_score(y_val, y_pred_val)

#         precision_list.append(precision)
#         sensitivity_list.append(sensitivity)

#         print(f"========================= FOLD {fold} ==========================")

#         print(f"The results of this train's F1-Score is {f1_score(y_train, y_pred):.2}")
#         print(f"The results of this validation's F1-Score is {f1_score(y_val, y_pred_val):.2}")

#         f1_score_val_list.append(f1_score(y_val, y_pred_val))
#         f1_score_train_list.append(f1_score(y_train, y_pred))
#         model_list.append(model)

#         if use_pca:
#             pcas.append(pca)
#             num_comp_list.append(num_comp)
        

#         y_hat_val = model.predict_proba(X_val_scaled)

#         viz = RocCurveDisplay.from_predictions(
#             y_val,
#             y_hat_val,
#             ax = ax,
#             alpha=0.3,
#             lw=1
#         )


#         interp_fpr, interp_tpr = interpolation(viz.fpr, viz.tpr)
#         fprs_list.append(interp_fpr)
#         tprs_list.append(interp_tpr)
#         auc_list.append(viz.roc_auc)

#     mean_fpr = np.mean(fprs_list, axis=0)
#     mean_tpr = np.mean(tprs_list, axis=0)
#     mean_auc = np.mean(auc_list)
#     std_auc  = np.std(auc_list)
#     mean_val = np.mean(f1_score_val_list)
#     std_val  = np.std(f1_score_val_list)
#     mean_pre = np.mean(precision_list, axis=0)
#     mean_sen = np.mean(sensitivity_list, axis=0)
#     std_pre  = np.std(precision_list)
#     std_sen  = np.std(precision_list)
#     mean_f1  = np.mean(f1_score_val_list, axis=0)
#     std_f1   = np.std(f1_score_val_list)

#     ax.plot(
#         mean_fpr,
#         mean_tpr,
#         color='blue',
#         lw=2,
#         label=r"Mean ROC (AUC = %.2f $\pm$ %.2f)" %(mean_auc, std_auc)
#     )

#     ax.plot(np.linspace(0, 1, 100),
#             np.linspace(0, 1, 100),
#             color='g',
#             ls=":",
#             lw=0.5)
#     ax.legend()

#     if fold == n_splits-1:
#         print(f"The average F1-Score of the train set is {np.mean(f1_score_train_list): .2} +- {np.std(f1_score_train_list): .2} ")
#         print(f"The average F1-Score of the validation set is é {mean_val: .2} +- {std_val: .2} ")
#         best_model_idx = np.argmax(f1_score_val_list)
#         print(f"The best fold was: {best_model_idx} ")
#         best_model  = model_list[best_model_idx]
#         best_scaler = scaler_list[best_model_idx]
    

#         # Fazer a inferência em Test Data
   
#         X_test_scaled       = best_scaler.transform(X_test)
#         if use_pca:
#             best_pca        = pcas[best_model_idx]
#             X_test_scaled   = best_pca.transform(X_test_scaled)
#             best_num_comp = num_comp_list[best_model_idx]
            

#         y_pred_test         = best_model.predict(X_test_scaled)
#         X_train_scaled_best = X_train_scaled_list[best_model_idx]
#         X_train_best        = X_train_list[best_model_idx]
#         y_train_best        = y_train_list[best_model_idx]

#         print(f"The F1-Score for the test data is: {f1_score(y_test, y_pred_test):.2} with the best model")

#         print('===============================Summary of analysis====================================')
#         if use_pca:
#             print(f'The amount of components required to explain 80% variance of the best model is {best_num_comp}')
#         print(f"The Average F1-Score of this test is {mean_f1:.2f} %")
#         print(f"The Deviation of F1-Score is: {std_f1:.2}")
#         print(f"The Average sensitivity of this test is {mean_sen:.2f} %")
#         print(f"The Deviation of sensitivity is: {std_sen:.2} ")
#         print(f"The Average Precision of this model is { mean_pre:.2f} %")
#         print(f"The Deviation of Precision is: {std_pre:.2} ")
#         print(f"The Average accuracy of this model is {mean_auc:.2f} %")
#         print(f"The Deviation of accuracy is: {std_auc:.2} ")
        
#     return {
#         'model': best_model,
#         'kwargs': str(kwargs),
#         'scaler': best_scaler,
#         'X_train': X_train_best,
#         'X_train_scaled': X_train_scaled_best,
#         'X_test': X_test,
#         'X_test_scaled': X_test_scaled,  # Add X_test_scaled to the returned dictionary
#         'y_train': y_train_best,
#         'y_test': y_test,
#         'mean_val': mean_val,
#         'std_val': std_val,
#         'pca_used': use_pca,
#         'pca': best_pca,
#         'transformation': transformation,
#         'tag': tag
#     }

def roc_auc_models(model_name_class):
    num_rows = (len(model_name_class) + 1) // 2
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, num_rows*6))
    axs = axs.flatten()

    for i, model in enumerate(model_name_class):
        model_name = model.model_name
        ax = axs[i]
        # row = i // 2
        # col = i % 2
        # ax = axs[row, col]

        model.plot_roc_train(ax=ax)
        model.plot_roc_test(ax=ax, color='orange')
        
        ax.grid()
        ax.set_title(f'{model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(['Train', 'Test'])

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.tight_layout()
    plt.show()

def get_validation_info(model_name_class):

    model_dicts_storage = []
    model_pca_dicts_storage = []

    # Iterate over each model instance
    for i, model in enumerate(model_name_class):
        model_type = model.model_name
        Parameters = model.kwargs
        validation_f1 = model.mean_val
        validation_f1_deviation = model.std_val 
        precision = model.get_precision()
        sensitivity = model.get_sensitivity()

    
        if model_type.endswith('_pca'):
            model_pca_dict = {
                'Model': model_type[:-4],
                # 'Parameters': Parameters,
                'Precision PCA': precision,
                'Sensitivity PCA':sensitivity,
                'Validation F1 PCA': validation_f1,
                'Validation F1 deviation PCA': validation_f1_deviation,
            }
            model_pca_dicts_storage.append(model_pca_dict)
            
        else:
            model_dict = {
                            'Model': model_type,
                            # 'Parameters': Parameters,
                            'Precision': precision,
                            'Sensitivity': sensitivity,
                            'Validation F1': validation_f1,
                            'Validation F1 deviation': validation_f1_deviation,
                        }
            model_dicts_storage.append(model_dict)

    model_no_pca = pd.DataFrame(model_dicts_storage)
    model_pca = pd.DataFrame(model_pca_dicts_storage)

    # Merge dataframes and keep all rows
    results = model_no_pca.merge(model_pca, how='outer')

    return results, model_no_pca, model_pca

def get_validation_info_reg(model_name_class):

    model_dicts_storage = []
    model_pca_dicts_storage = []

    # Iterate over each model instance
    for i, model in enumerate(model_name_class):
        model_type = model.model_name
        validation_MSE = model.mean_val
        validation_MSE_deviation = model.std_val
        Adjusted_R_Squared= model.calculate_adjusted_r2()
        R2 = model.calculate_r2()
        RMSE = model.calculate_rmse()

    
        if model_type.endswith('_pca'):
            model_pca_dict = {
                'Model': model_type[:-4],
                'Adjusted R-Squared PCA' : Adjusted_R_Squared,
                'R2 PCA': R2,
                'RMSE PCA': RMSE,
                'Validation MSE PCA': validation_MSE,
                'Validation MSE deviation PCA': validation_MSE_deviation,
            }
            model_pca_dicts_storage.append(model_pca_dict)
            
        else:
            model_dict = {
                            'Model': model_type,
                            'Adjusted R-Squared' : Adjusted_R_Squared,
                            'R2': R2,
                            'RMSE': RMSE,
                            'Validation MSE': validation_MSE,
                            'Validation MSE deviation': validation_MSE_deviation,
                        }
            model_dicts_storage.append(model_dict)

    model_no_pca = pd.DataFrame(model_dicts_storage)
    model_pca = pd.DataFrame(model_pca_dicts_storage)

    # Merge dataframes and keep all rows
    results = model_no_pca.merge(model_pca, how='outer')

    return results, model_no_pca, model_pca
















def train_regressor_multiple(X, X_test, y, y_test, model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="", **kwargs):
    cv = KFold(n_splits=n_splits)
    mse_val_list = []
    mse_train_list = []
    scaler_list = []
    model_list = []
    X_train_scaled_list = []
    X_train_list = []
    y_train_list = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx, :]
        X_val = X[val_idx, :]
        y_val = y[val_idx, :]

        scaler = StandardScaler()
        scaler_list.append(scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_pca:
            var, var_ratio, num_comp = get_PCA_comp(X_train)
            pca = PCA(n_components=num_comp)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")

        X_train_scaled_list.append(X_train_scaled)
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        mse_val = np.inf
        model = None

        for idx in range(n_init):
            _model = MultiOutputRegressor(model_klass(**kwargs))
            _model.fit(X_train_scaled, y_train)

            _y_pred = _model.predict(X_train_scaled)
            _y_pred_val = _model.predict(X_val_scaled)

            _mse_val = mean_squared_error(y_val, _y_pred_val)
            if np.isnan(_mse_val) or np.isinf(_mse_val):
                continue

            if _mse_val < mse_val:
                y_pred_val = _y_pred_val
                y_pred = _y_pred
                model = _model
                mse_val = _mse_val

        mse_train = mean_squared_error(y_train, y_pred)
        mse_val_list.append(mse_val)
        mse_train_list.append(mse_train)
        model_list.append(model)

        print(f"========================= FOLD {fold} ==========================")
        print(f"The results of this train's MSE is {mse_train:.2f}")
        print(f"The results of this validation's MSE is {mse_val:.2f}")

        plt.figure(figsize=(10, 6))
        for i in range(y_train.shape[1]):
            plt.scatter(y_train[:, i], y_pred[:, i], color='blue', label=f'Target {i+1}')
        for i in range(y_val.shape[1]):
            plt.scatter(y_val[:, i], y_pred_val[:, i], color='red', label=f'Validation Target {i+1}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Fold {fold+1} - Actual vs Predicted')
        plt.legend()
        plt.show()

    mean_mse_val = np.mean(mse_val_list)
    mean_mse_train = np.mean(mse_train_list)

    print(f"\nMean MSE Train: {mean_mse_train:.2f}")
    print(f"Mean MSE Validation: {mean_mse_val:.2f}")

    best_model_idx = np.argmin(mse_val_list)
    best_model = model_list[best_model_idx]
    scaler = scaler_list[best_model_idx]
    X_train_scaled = X_train_scaled_list[best_model_idx]
    X_train = X_train_list[best_model_idx]
    y_train = y_train_list[best_model_idx]

    X_test_scaled = scaler.transform(X_test)
    if use_pca:
        X_test_scaled = pca.transform(X_test_scaled)

    y_test_pred = best_model.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f"\nBest Model Test MSE: {mse_test:.2f}")
    print(f"Best Model:\n{best_model}")

    plt.figure(figsize=(10, 6))
    for i in range(y_test.shape[1]):
        plt.scatter(y_test[:, i], y_test_pred[:, i], color='green', label=f'Test Target {i+1}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Test Set - Actual vs Predicted')
    plt.legend()
    plt.show()

    return {
        'model': best_model,
        'kwargs': str(kwargs),
        'scaler': scaler,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'mean_val': mean_mse_val,
        'std_val': np.std([mse for mse in mse_val_list if not np.isnan(mse) and not np.isinf(mse)]),
        'pca_used': use_pca,
        'pca': pca if use_pca else None,
        'transformation': transformation,
        'tag': tag
    }


def train_keras_regressor(X, X_test, y, y_test, model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="", **kwargs):
    cv = KFold(n_splits=n_splits)
    mse_val_list = []
    mse_train_list = []
    scaler_list = []
    model_list = []
    X_train_scaled_list = []
    X_train_list = []
    y_train_list = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        scaler = StandardScaler()
        scaler_list.append(scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_pca:
            var, var_ratio, num_comp = get_PCA_comp(X_train)
            pca = PCA(n_components=num_comp)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")

        X_train_scaled_list.append(X_train_scaled)
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        mse_val = np.inf
        model = None

        for idx in range(n_init):
            _model = model_klass(**kwargs)
            _model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

            _y_pred_val = _model.predict(X_val_scaled)
            _mse_val = mean_squared_error(y_val, _y_pred_val)
            if np.isnan(_mse_val) or np.isinf(_mse_val):
                continue

            if _mse_val < mse_val:
                y_pred_val = _y_pred_val
                model = _model
                mse_val = _mse_val

        y_pred = model.predict(X_train_scaled)
        mse_train = mean_squared_error(y_train, y_pred)
        mse_val_list.append(mse_val)
        mse_train_list.append(mse_train)
        model_list.append(model)

        print(f"========================= FOLD {fold} ==========================")
        print(f"The results of this train's MSE is {mse_train:.2f}")
        print(f"The results of this validation's MSE is {mse_val:.2f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_pred, color='blue', label='Train')
        plt.scatter(y_val, y_pred_val, color='red', label='Validation')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Fold {fold+1} - Actual vs Predicted')
        plt.legend()
        plt.show()

    mean_mse_val = np.mean(mse_val_list)
    mean_mse_train = np.mean(mse_train_list)

    print(f"\nMean MSE Train: {mean_mse_train:.2f}")
    print(f"Mean MSE Validation: {mean_mse_val:.2f}")

    best_model_idx = np.argmin(mse_val_list)
    best_model = model_list[best_model_idx]
    scaler = scaler_list[best_model_idx]
    X_train_scaled = X_train_scaled_list[best_model_idx]
    X_train = X_train_list[best_model_idx]
    y_train = y_train_list[best_model_idx]

    X_test_scaled = scaler.transform(X_test)

    if use_pca:
        X_test_scaled = pca.transform(X_test_scaled)

    y_pred = best_model.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_pred)

    print(f"\nBest Model MSE Validation: {mse_val_list[best_model_idx]:.2f}")
    print(f"Best Model MSE Test: {mse_test:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green', label='Test')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Test Set - Actual vs Predicted')
    plt.legend()
    plt.show()

    return {
    'model': best_model,
    'kwargs': str(kwargs),
    'scaler': scaler,
    'X_train': X_train,
    'X_train_scaled': X_train_scaled,
    'X_test': X_test,
    'X_test_scaled': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'mean_val': mean_mse_val,
    'std_val': np.std([mse for mse in mse_val_list if not np.isnan(mse) and not np.isinf(mse)]),
    'pca_used': use_pca,
    'pca': pca if use_pca else None,
    'transformation': transformation,
    'tag': tag
}


# # Reproducability
# def set_seed(seed=31415):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'
# set_seed(31415)

def train(X, X_test, y, y_test, model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="", random_seed=None, **kwargs):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Initialize empty lists to store the results for each fold
    f1_score_val_list = []
    f1_score_train_list = []
    fprs_list = []
    tprs_list = []
    auc_list = []
    scaler_list = []
    model_list = []
    X_train_scaled_list = []
    y_train_list = []
    X_train_list = []
    precision_list = []
    sensitivity_list = []
    pcas = []
    best_pca = 'Not Applicable'
    num_comp_list = []
    
    # Create the figure and axes for plotting the ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Validation using cross-validation on training data
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]
        
        # Scaling
        scaler = StandardScaler()
        scaler_list.append(scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # PCA
        if use_pca:
            var, var_ratio, num_comp = get_PCA_comp(X_train)
            pca = PCA(n_components=num_comp)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")
            
            # Update the model's input_dim parameter if it exists in the kwargs
            if 'input_dim' in kwargs:
                kwargs['input_dim'] = num_comp
        
        X_train_scaled_list.append(X_train_scaled)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        
        f1_score_val = 0
        model = None
        
        for idx in range(n_init):
            _model = model_klass(**kwargs)
            _model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
            
            _y_pred = _model.predict(X_train_scaled)
            _y_pred_val = _model.predict(X_val_scaled)
            
            _f1_score_val = f1_score(y_val, _y_pred_val, average='weighted')
            
            if _f1_score_val > f1_score_val:
                y_pred_val = _y_pred_val
                y_pred = _y_pred
                model = _model
                K.clear_session()
        
        sensitivity = 100 * recall_score(y_val, y_pred_val)
        precision = 100 * precision_score(y_val, y_pred_val)
        
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        
        print(f"========================= FOLD {fold} ==========================")
        print(f"The results of this train's F1-Score is {f1_score(y_train, y_pred, average='weighted'):.2}")
        print(f"The results of this validation's F1-Score is {f1_score(y_val, y_pred_val, average='weighted'):.2}")
        
        f1_score_val_list.append(f1_score(y_val, y_pred_val, average='weighted'))
        f1_score_train_list.append(f1_score(y_train, y_pred, average='weighted'))
        model_list.append(model)
        
        if use_pca:
            pcas.append(pca)
            num_comp_list.append(num_comp)
        
        y_hat_val = model.predict_proba(X_val_scaled)
        
        viz = RocCurveDisplay.from_predictions(y_val, y_hat_val, ax=ax, alpha=0.3, lw=1)
        
        interp_fpr, interp_tpr = interpolation(viz.fpr, viz.tpr)
        fprs_list.append(interp_fpr)
        tprs_list.append(interp_tpr)
        auc_list.append(viz.roc_auc)
    
    mean_fpr = np.mean(fprs_list, axis=0)
    mean_tpr = np.mean(tprs_list, axis=0)
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)
    mean_val = np.mean(f1_score_val_list)
    std_val = np.std(f1_score_val_list)
    mean_pre = np.mean(precision_list, axis=0)
    mean_sen = np.mean(sensitivity_list, axis=0)
    std_pre = np.std(precision_list)
    std_sen = np.std(precision_list)
    mean_f1 = np.mean(f1_score_val_list, axis=0)
    std_f1 = np.std(f1_score_val_list)
    
    ax.plot(mean_fpr, mean_tpr, color='blue', lw=2, label=r"Mean ROC (AUC = %.2f $\pm$ %.2f)" %(mean_auc, std_auc))
    
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='g', ls=":", lw=0.5)
    ax.legend()
    
    if fold == n_splits-1:
        print(f"The average F1-Score of the train set is {np.mean(f1_score_train_list): .2} +- {np.std(f1_score_train_list): .2} ")
        print(f"The average F1-Score of the validation set is é {mean_val: .2} +- {std_val: .2} ")
        best_model_idx = np.argmax(f1_score_val_list)
        print(f"The best fold was: {best_model_idx} ")
        best_model = model_list[best_model_idx]
        best_scaler = scaler_list[best_model_idx]
        
        # Perform inference on the test data
        X_test_scaled = best_scaler.transform(X_test)
        
        if use_pca:
            best_pca = pcas[best_model_idx]
            X_test_scaled = best_pca.transform(X_test_scaled)
            best_num_comp = num_comp_list[best_model_idx]
        
        y_pred_test = best_model.predict(X_test_scaled)
        X_train_scaled_best = X_train_scaled_list[best_model_idx]
        X_train_best = X_train_list[best_model_idx]
        y_train_best = y_train_list[best_model_idx]
        
        print(f"The F1-Score for the test data is: {f1_score(y_test, y_pred_test):.2} with the best model")
        
        print('===============================Summary of analysis====================================')
        if use_pca:
            print(f'The amount of components required to explain 80% variance of the best model is {best_num_comp}')
        
        print(f"The Average F1-Score of this test is {mean_f1:.2f} %")
        print(f"The Deviation of F1-Score is: {std_f1:.2}")
        print(f"The Average sensitivity of this test is {mean_sen:.2f} %")
        print(f"The Deviation of sensitivity is: {std_sen:.2} ")
        print(f"The Average Precision of this model is { mean_pre:.2f} %")
        print(f"The Deviation of Precision is: {std_pre:.2} ")
        print(f"The Average accuracy of this model is {mean_auc:.2f} %")
        print(f"The Deviation of accuracy is: {std_auc:.2} ")
    
    return {
        'model': best_model,
        'kwargs': str(kwargs),
        'scaler': best_scaler,
        'X_train': X_train_best,
        'X_train_scaled': X_train_scaled_best,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train_best,
        'y_test': y_test,
        'mean_val': mean_val,
        'std_val': std_val,
        'pca_used': use_pca,
        'pca': best_pca,
        'transformation': transformation,
        'tag': tag
    }











def train_regressor(X, X_test, y, y_test, model_klass, n_splits=5, n_init=1, use_pca=False, tag="", transformation="", **kwargs):
    cv = KFold(n_splits=n_splits)
    mse_val_list = []
    mse_train_list = []
    scaler_list = []
    model_list = []
    X_train_scaled_list = []
    X_train_list = []
    y_train_list = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        scaler = StandardScaler()
        scaler_list.append(scaler)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_pca:
            var, var_ratio, num_comp = get_PCA_comp(X_train)
            pca = PCA(n_components=num_comp)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_val_scaled = pca.transform(X_val_scaled)
            print(f"The number of components required to explain 80% of the variance for this model was {num_comp}")

        X_train_scaled_list.append(X_train_scaled)
        X_train_list.append(X_train)
        y_train_list.append(y_train)

        mse_val = np.inf
        model = None

        for idx in range(n_init):
            _model = model_klass(**kwargs)
            _model.fit(X_train_scaled, y_train)

            _y_pred = _model.predict(X_train_scaled)
            _y_pred_val = _model.predict(X_val_scaled)

            _mse_val = mean_squared_error(y_val, _y_pred_val)
            if np.isnan(_mse_val) or np.isinf(_mse_val):
                continue

            if _mse_val < mse_val:
                y_pred_val = _y_pred_val
                y_pred = _y_pred
                model = _model
                mse_val = _mse_val

        mse_train = mean_squared_error(y_train, y_pred)
        mse_val_list.append(mse_val)
        mse_train_list.append(mse_train)
        model_list.append(model)

        print(f"========================= FOLD {fold} ==========================")
        print(f"The results of this train's MSE is {mse_train:.2f}")
        print(f"The results of this validation's MSE is {mse_val:.2f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_pred, color='blue', label='Train')
        plt.scatter(y_val, y_pred_val, color='red', label='Validation')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Fold {fold+1} - Actual vs Predicted')
        plt.legend()
        plt.show()

    mean_mse_val = np.mean(mse_val_list)
    mean_mse_train = np.mean(mse_train_list)

    print(f"\nMean MSE Train: {mean_mse_train:.2f}")
    print(f"Mean MSE Validation: {mean_mse_val:.2f}")

    best_model_idx = np.argmin(mse_val_list)
    best_model = model_list[best_model_idx]
    scaler = scaler_list[best_model_idx]
    X_train_scaled = X_train_scaled_list[best_model_idx]
    X_train = X_train_list[best_model_idx]
    y_train = y_train_list[best_model_idx]

    X_test_scaled = scaler.transform(X_test)
    if use_pca:
        X_test_scaled = pca.transform(X_test_scaled)

    y_test_pred = best_model.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f"\nBest Model Test MSE: {mse_test:.2f}")
    print(f"Best Model:\n{best_model}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='green', label='Test')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Test Set - Actual vs Predicted')
    plt.legend()
    plt.show()

    return {
        'model': best_model,
        'kwargs': str(kwargs),
        'scaler': scaler,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'mean_val': mean_mse_val,
        'std_val': np.std([mse for mse in mse_val_list if not np.isnan(mse) and not np.isinf(mse)]),
        'pca_used': use_pca,
        'pca': pca if use_pca else None,
        'transformation': transformation,
        'tag': tag
    }

def plot_final_results(results_final):
    sns.set_style("ticks")
    sns.set_context("talk")
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot without PCA analysis
    ax[0].errorbar(range(results_final.shape[0]),
                   results_final['Validation F1'],
                   results_final['Validation F1 deviation'])
    ax[0].grid(True)
    sns.despine(offset=5, ax=ax[0])

    ax[0].set_title("Models performance in the validation set -- No PCA Analysis")
    ax[0].set_ylabel("F1 Score")
    ax[0].set_xlabel("Model")
    ax[0].set_xticks(range(results_final.shape[0]))
    ax[0].set_xticklabels(results_final['Model'], rotation=90)
    
    # Plot with PCA analysis
    ax[1].errorbar(range(results_final.shape[0]),
                   results_final['Validation F1 PCA'],
                   results_final['Validation F1 deviation PCA'])
    ax[1].grid(True)
    sns.despine(offset=5, ax=ax[1])

    ax[1].set_title("Models performance in the validation set PCA")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_xlabel("Model")
    ax[1].set_xticks(range(results_final.shape[0]))
    ax[1].set_xticklabels(results_final['Model'], rotation=90)
    
    plt.tight_layout()
    plt.show()

class ModelResults:
    def __init__(self, results):
        self.model = results['model']
        self.kwargs = results['kwargs']
        self.using_pca = results['pca_used']
        self.transformation = results['transformation']
        self.tag = results['tag']
        concat_string = '_model'+ str(self.tag) + str(self.transformation)
        if self.using_pca: concat_string += '_pca'
        self.model_name = self.model.__class__.__name__ + concat_string
        self.scaler = results['scaler']
        self.X_train = results['X_train']
        self.X_train_scaled = results['X_train_scaled']
        self.X_test = results['X_test']
        self.X_test_scaled = results['X_test_scaled']
        self.y_train = results['y_train']
        self.y_test = results['y_test']
        self.mean_val = results['mean_val']
        self.std_val = results['std_val']
        self.pca = results['pca']
        self.y_pred = None  # Initialize y_pred attribute as None

    def plot_hist(self, estimator_name : str = "test", **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        plt.hist(self.y_pred)
        plt.xlabel('Predictions')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {self.model_name} Predictions')
        plt.show()
    
    def plot_roc(self, X, y, estimator_name: str = "train", **kwargs):
        y_hat = self.model.predict_proba(X)
        
        # Check if y_hat has only one column, if so, create a dummy column of zeros
        if y_hat.shape[1] == 1:
            zeros = np.zeros_like(y_hat)
            y_hat = np.concatenate([zeros, y_hat], axis=1)

        fpr, tpr, thresholds = roc_curve(y, y_hat[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)
        return RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc_score,
            estimator_name=estimator_name
        ).plot(**kwargs)

    def plot_roc_train(self, **kwargs):
        self.plot_roc(self.X_train_scaled, self.y_train, estimator_name="train", **kwargs)

    def plot_roc_test(self, **kwargs):
        self.plot_roc(self.X_test_scaled, self.y_test, estimator_name="test", **kwargs)

    def plot_distribution(self, X, y, ax=None, estimator_name: str = "train", **kwargs):
        y = pd.Series(y)  # Convert y to pandas Series
        y_hat = self.model.predict_proba(X)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.distplot(y_hat[y.values == 1, 1], label="Good", ax=ax)
        ax.set_xlim([0, 1])
        sns.distplot(y_hat[y.values == 0, 1], label="Bad", ax=ax)
        ax.legend()
        return ax

    def plot_distribution_train(self, **kwargs):
        self.plot_distribution(self.X_train_scaled, self.y_train, estimator_name="train", **kwargs)

    def plot_distribution_test(self, **kwargs):
        self.plot_distribution(self.X_test_scaled, self.y_test, estimator_name="test", **kwargs)

    def correlation_matrix_train(self, **kwargs):
        self.y_pred = self.model.predict(self.X_train_scaled)
        cm = confusion_matrix(self.y_train, self.y_pred)
        ax = sns.heatmap(cm, cmap ="BuGn", annot=True, fmt='g')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Bad Wine(0)', 'Good Wine(1)'])
        ax.set_yticklabels(['Bad Wine(0)', 'Good Wine(1)'])

    def correlation_matrix_test(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, self.y_pred)
        ax = sns.heatmap(cm, cmap ="BuGn", annot=True, fmt='g')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['Bad Wine(0)', 'Good Wine(1)'])
        ax.set_yticklabels(['Bad Wine(0)', 'Good Wine(1)'])
    
    def get_sensitivity(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        return (f" {100 *  recall_score(self.y_test, self.y_pred):.2f} %")

    def get_precision(self, **kwargs):
        self.y_pred = self.model.predict(self.X_test_scaled)
        return (f" {100 *  precision_score(self.y_test, self.y_pred):.2f} %")
    
    def calculate_r2(self):
        y_pred = self.model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        return r2
    
    def calculate_rmse(self):
        y_pred = self.model.predict(self.X_test_scaled)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return rmse
    
    def calculate_adjusted_r2(self):
        y_pred = self.model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        n = len(self.X_test_scaled)
        p = self.X_test_scaled.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def plot_residuals(self, **kwargs):
        y_pred = self.model.predict(self.X_train_scaled)
        residuals = self.y_train - y_pred
        model_name = self.model_name
        sns.distplot(residuals)
        plt.title(f'{model_name} Residuals PDF', size=18)
        plt.show()