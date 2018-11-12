from setting import TrainingSetting as set
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D, LeakyReLU

# Q-network: guess how many score it will make
def QNet(scrshot_size, action_size) :
    input_scrshots = Input(scrshot_size) # screen shot image
    x = Conv2D(32, (6, 6), padding = "valid", activation = "relu")(input_scrshots)
    x = MaxPooling2D((4, 4), padding = "same")(x)
    x = Conv2D(64, (4, 4), padding = "valid", activation = "relu")(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(128, (3, 3), padding = "valid", activation = "relu")(x)
    x = Flatten()(x)
    x = Dense(256, activation = "relu")(x)
    scores = Dense(action_size, activation = "relu")(x)
    model = Model(input_scrshots, scores)
    model.summary()
    return model
