from model.cnn import create_model
from model.cnn import init_model
from keras.utils import plot_model
from contextlib import redirect_stdout

TEXT_FILE = './model/model.txt'
IMAGE_FILE = './model/model.png'


def to_text(model):
    with open(TEXT_FILE, 'w') as my_file:
        with redirect_stdout(my_file):
            model.summary()

        print("Exported to {0} successfully".format(TEXT_FILE))


def to_image(model):
    plot_model(model, to_file=IMAGE_FILE, show_layer_names=True,
               show_shapes=True)
    print("Exported to {0} successfully".format(IMAGE_FILE))


my_model = create_model()
my_model = init_model(my_model)
to_text(my_model)
to_image(my_model)
