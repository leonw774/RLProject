from setting import TrainingSetting as set
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D, LeakyReLU

# Q-network: guess how many score it will make
def QNet(images_size, action_size) :
    input_scrshots = Input(images_size) # screen shot image
    x = Conv2D(32, (8, 8), padding = "valid", activation = LeakyReLU(0.0))(input_scrshots)
    x = MaxPooling2D((4, 4), padding = "same")(x)
    x = Conv2D(64, (4, 4), padding = "valid", activation = LeakyReLU(0.0))(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(128, (3, 3), padding = "valid", activation = LeakyReLU(0.0))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    scores = Dense(action_size)(x)
    model = Model(input_scrshots, scores)
    model.summary()
    return model
