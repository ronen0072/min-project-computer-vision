import keras
import numpy as np

class SplitLayer(keras.layers.Layer):
    """
    Layer expects a tensor (multi-dimensonal array) of shape (samples, views, ...)
    and returns a list of #views elements, each of shape (samples, ...)
    """
    
    def __init__(self, num_splits, **kwargs):
        self.num_splits = num_splits
        super(SplitLayer, self).__init__(**kwargs)
    
    def call(self, x):
        return [x[:, i] for i in range(self.num_splits)]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],) + input_shape[2:]]*self.num_splits


def get_test_shared_model():    
    num_channels = 32 # for example
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(num_channels, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(32,32,1)))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(keras.layers.Conv2D(num_channels*2, (3, 3), activation='relu'))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))    
    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(128, activation='relu'))
    return cnn



def test_split_layer():
    num_views = 4 # or any other number ...
    cnn = get_test_shared_model()
    input = keras.layers.Input(shape=(num_views, 32, 32, 1))
    views = SplitLayer(num_views)(input) # list of keras-tensors
    processed_views = [] # empty list
    for view in views:
        x = cnn(view)
        processed_views.append(x)

    
    pooled_views = keras.layers.Maximum()(processed_views)
    
    x = keras.layers.Dense(64, activation='relu')(x)
    prediction = keras.layers.Dense(10)(x)
    model = keras.models.Model(input, prediction)
    
    a = np.random.rand(5,num_views,32,32,1) #batch of 5 samples, each with 4 views
    b = model.predict(a)
    assert b.shape == (5, 10)
    return model
    
    