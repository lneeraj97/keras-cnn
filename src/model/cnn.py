from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential


TRAIN = 'data/train'
TEST = 'data/test'
JSON_FILE = './model/model.json'
HDF5_FILE = './model/model.h5'
EPOCHS = 15
BATCH_SIZE = 32
STEPS_PER_EPOCH = 130
VALIDATION_STEPS = 22
POOL_SIZE = (2, 2)
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (224, 224, 3)
TARGET_SIZE = (224, 224)


def create_model():

    # Initialise a model
    model = Sequential()

    # First conv layer
    model.add(Conv2D(64, KERNEL_SIZE, input_shape=INPUT_SHAPE,
                     activation='relu', use_bias=True, strides=1, padding='same'))

    # Second conv layer
    model.add(Conv2D(64, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))

    # First pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))

    # Third conv layer
    model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))

    # Fourth conv layer
    model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))

    # Second pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))

    # Flattening
    model.add(Flatten())

    # FC layers
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=4, activation='sigmoid'))

    return model


def init_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_set = train_datagen.flow_from_directory(
        TRAIN, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    test_set = test_datagen.flow_from_directory(
        TEST, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    model.fit_generator(train_set, steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS, validation_steps=VALIDATION_STEPS, validation_data=test_set)

    return model


def save_model(model):
    model_json = model.to_json()
    with open(JSON_FILE, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(HDF5_FILE)
    print("Model saved...! Ready to go.")
