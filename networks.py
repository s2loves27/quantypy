import os
import threading
import numpy as np


class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

def set_session(sess): pass


graph = DummyGraph()
sess = None


if os.environ['KERAS_BACKEND'] == 'tensorflow':
    import tensorflow.keras as keras
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    # from tensorflow.keras.backend import set_session
    # from tensorflow.keras import backend as K

    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.Session()
    # tf.compat.v1.experimental.output_all_intermediates(Tru
    # e)
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    #샘플에 대한 행동 가치 또는 확률을 예측

    """
        Numpy 다차원 배열을 1차원으로 바꾸는 것을 지원하는 3개의 함수.
        1. revel() 2. reshape() 3. flatten()
        
        ex, 
        import numpy as np
        a1 = np.array([[1,2],[3,4]])
        a2 = a1.ravel() # 또는 a2 = a1.reshape(-1) 또는 a2 = a1.flatten()   
        
        1. numpy.revel() - 1차원 배열을 반환합니다.
        2. numpy.reshape() - 데이터 변경 없이 형상만 변경하여 반환합니다.
        3. numpy.ndarray.flatten() - 1차원 배열의 복사본을 반환합니다.
    """
    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    """
    학습 데이터와 레이블 x, y를 입력으로 받아서 모델을 학습 시킵니다
    A3C에서는 여러 스레드가 병렬로 신경망을 사용할 수 있기 때문에 충돌이
    일어나지 않게 스레드들의 동시 사용을 막습니다. 
    """

    """
    Keras 클래스의 함수인 train_on_batch()는 입력으로 들어온 학습 데이터 집합(배치, Batch)
    으로 신경망을 한 번 학습합니다.
    """
    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x, y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lstm':
                return LSTMNetwork.get_network_head(
                    Input((num_steps, input_dim)))
            elif net == 'cnn':
                return CNN.get_network_head(
                    Input((1, num_steps, input_dim)))


class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation, 
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    """
    train_on_batch()
    함수는 학습 데이터나 샘플의 형태(shape)를 적절히 변경하고 
    상위 클래스의 함수를 그대로 호출.
    DNN - 배치크기, 자질 텍터 차원의 모양을 가진다.
    """
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    """
    train_on_batch()
    여러 샘플을 한꺼번에 받아서 신경망의 출력을 반환
    하나의 샘플에 대한 결과만 받고 싶어도 샘플의 배열로 입력값을 구성해야 하기 때문에
    2차원 배열로 재구성.
    (1행 28열인 2차원 배열로 만든다.)
    """

    """
    Numpy ndarray는 reshape()함수로 배열을 다른 차원으로 변환 할 수 있습니다.
    [1,2,3,4,5,6]이 있을때 이 배열의 shape(모양)은 (6,)입니다. 이를 (3,2)로 만들면
    [[1,2], [3,4], [5,6]]이 됩니다. 
    이때 유의할 점은 배열의 총 크기는 변하지 않아야 한다.
    """
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)
    

class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation, 
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = LSTM(256, dropout=0.1, 
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
            stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    #배치 학습을 위한 데이터 생성
    """
    DNN과 전체적으로 비슷하지만 속성의 차이로 LSTMNetwork클래스는 num_steps변수를 가지고 있다.
    몇개의 샘플을 묶어서 LSTM 신경망의 입력으로 사용할지 결정하는 것이다.
    따라서 train_on_batch() 함수와 pridect함수에서 학습데이터와 샘플의 형태를 변경할 때 num_steps 변수를 사용하게 된다.
    
    (배치크기, 스텝수, 자질 벡터 차원)
    """
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)


class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                K.set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim, 1))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(128, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    """
    2차원 합성곱 신경망이므로 (배치 크기, 스텝 수, 자질 벡터 차원, 1)의 모양으로 학습 데이터를 가진다.
    
    보통 합성곱 신경망은 이미지 데이터를 취급해 마지막 차원으로 RGB와 같은 이미지 채널이 들어 갑니다.
    
    주식 데이터에는 채널이라 할 것 없으므로 1로 고정 했습니다. 
    
    학습데이터와 샘플의 모양을 바꾸고 Network클래스의 함수를 그래도 호출 합니다. 
    """
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)
