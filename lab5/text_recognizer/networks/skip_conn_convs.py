from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.layers import add, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model


def skip_conn_convs(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    
    input_tensor = Input(shape=input_shape)
    pre_res_input = Conv2D(32, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv1')(input_tensor)
    pre_res_input = BatchNormalization(axis=3, name='bn_conv1')(pre_res_input)
    pre_res_input = Activation('relu')(pre_res_input)

    x = Conv2D(32, (1, 1), strides=(2, 2),
                      kernel_initializer='he_normal')(pre_res_input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(32, (1, 1), strides=(2, 2), kernel_initializer='he_normal')(pre_res_input)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)

    model = Model(input_tensor, x, name='skip_conn_convs')
    ##### Your code above (Lab 2)

    return model

