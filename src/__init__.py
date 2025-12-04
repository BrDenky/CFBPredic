import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
# ...

class LSTMPredictor:    
    def __init__(self, timesteps: int = 12, n_features: int = None, seed: int = 42):
        # Semillas para reproducibilidad
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.timesteps = timesteps
        self.n_features = n_features
        self.model = None
        self.history = None
