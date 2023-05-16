import os
import numpy as np
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf



class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(128,128,128), 
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    gx  = np.fromfile(self.dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    fx  = np.fromfile(self.fpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    gx = np.reshape(gx,self.dim)
    fx = np.reshape(fx,self.dim)
    #gmin = np.min(gx)
    #gmax = np.max(gx)
    #gx = gx-gmin
    #gx = gx/(gmax-gmin)
    #gx = gx*255
    xm = np.mean(gx)
    xs = np.std(gx)
    gx = gx-xm
    gx = gx/xs
    gx = np.transpose(gx)
    fx = np.transpose(fx)
    #in seismic processing, the dimensions of a seismic array is often arranged as
    #a[n3][n2][n1] where n1 represnts the vertical dimenstion. This is why we need 
    #to transpose the array here in python 
    # Generate data
    X = np.zeros((2, *self.dim, self.n_channels),dtype=np.single)
    Y = np.zeros((2, *self.dim, self.n_channels),dtype=np.single)
    X[0,] = np.reshape(gx, (*self.dim,self.n_channels))
    Y[0,] = np.reshape(fx, (*self.dim,self.n_channels))  
    X[1,] = np.reshape(np.flipud(gx), (*self.dim,self.n_channels))
    Y[1,] = np.reshape(np.flipud(fx), (*self.dim,self.n_channels))  
    '''
    for i in range(4):
      X[i,] = np.reshape(np.rot90(gx,i,(2,1)), (*self.dim,self.n_channels))
      Y[i,] = np.reshape(np.rot90(fx,i,(2,1)), (*self.dim,self.n_channels))  
    '''
    return X,Y


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./log1', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super().__init__(log_dir=training_log_dir, **kwargs)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
        super().set_model(model)
        
    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        with self.val_writer.as_default():
            for name, value in val_logs.items():
                tf.summary.scalar(name, value, step=epoch)
        self.val_writer.flush()
        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        print('check point')
        logs.update({'lr': float(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.val_writer.close()
