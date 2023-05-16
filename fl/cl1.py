from numpy.random import seed
from keras.callbacks import ModelCheckpoint
from keras import backend as keras
from utils import DataGenerator
from unet import *
import flwr as fl

params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
seismPathT = "../../data_seismic/fl_test/client1/train/seis/"
faultPathT = "../../data_seismic/fl_test/client1/train/fault/"
seismPathV = "../../data_seismic/fl_test/client1/validation/seis/"
faultPathV = "../../data_seismic/fl_test/client1/validation/fault/"
train_ID = range(100)
valid_ID = range(10)
train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                data_IDs=train_ID,**params)
valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                data_IDs=valid_ID,**params)

print("Finish loading dataset")
model = unet(input_size=(None, None, None,1))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
              metrics=['accuracy'])


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.train_generator, epochs=1)
        new_parameters = model.get_weights()
        num_examples = len(self.train_generator) * self.train_generator.batch_size
        return new_parameters, num_examples, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.valid_generator)
        num_examples = len(self.valid_generator) * self.valid_generator.batch_size
        return loss, num_examples, {"accuracy": accuracy}


flower_client = FlowerClient()
flower_client.train_generator = train_generator
flower_client.valid_generator = valid_generator

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=flower_client)
