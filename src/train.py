from model.cnn import *

my_model = create_model()
my_model = init_model(my_model)
print(my_model.summary())
trained_model = train_model(my_model)
save_model(trained_model)
