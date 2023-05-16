from numpy.random import seed
from keras.callbacks import ModelCheckpoint
from keras import backend as keras
from utils import generate_data
from unet import *
import flwr as fl

params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
seismPathT = "../../data_seismic/fl_test/client2/train/seis/"
faultPathT = "../../data_seismic/fl_test/client2/train/fault/"
seismPathV = "../../data_seismic/fl_test/client2/validation/seis/"
faultPathV = "../../data_seismic/fl_test/client2/validation/fault/"
train_ID = range(100, 201)
valid_ID = range(10, 21)
x_train, y_train = generate_data(dpath=seismPathT, fpath=faultPathT, data_IDs=train_ID, **params)
x_test, y_test = generate_data(dpath=seismPathV, fpath=faultPathV, data_IDs=valid_ID, **params)

print("Finish loading dataset")
model = unet(input_size=(None, None, None,1))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
              metrics=['accuracy'])


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())