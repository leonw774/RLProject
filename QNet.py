from setting import TrainingSetting as set
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D, LeakyReLU

# Q-network: guess how many score it will make
def QNet(scrshot_size, action_size) :
    input_image = Input(scrshot_size) # image
    input_action = Input((action_size,)) # one-hot
    x = Conv2D(8, (3, 3), padding = "valid", activation = LeakyReLU(0.2))(input_image)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(16, (3, 3), padding = "valid", activation = LeakyReLU(0.2))(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(32, (3, 3), strides = (2, 2), padding = "valid", activation = LeakyReLU(0.2))(x)
    flat_image = Flatten()(x)
    #flat_action = Flatten()(input_action)
    conc = Concatenate()([flat_image, input_action])
    score = Dense(1)(conc)
    model = Model([input_image, input_action], score)
    model.summary()
    return model
