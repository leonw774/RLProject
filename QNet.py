from setting import TrainingSetting as set
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D, LeakyReLU

# Q-network: guess how many score it will make
def QNet(scrshot_size, action_size) :
    input_image = Input(scrshot_size) # image
    x = Conv2D(32, (8, 8), padding = "valid", activation = LeakyReLU(0.0))(input_image)
    x = MaxPooling2D((4, 4), padding = "same")(x)
    x = Conv2D(64, (4, 4), padding = "valid", activation = LeakyReLU(0.0))(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(128, (3, 3), padding = "valid", activation = LeakyReLU(0.0))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    scores = Dense(action_size)(x)
    model = Model(input_image, scores)
    model.summary()
    return model
