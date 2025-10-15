import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np 
import scipy.io
import tensorflow as tf
import keras
from keras import optimizers
from keras.layers import Conv2DTranspose, Conv2D, ConvLSTM2D, TimeDistributed, LeakyReLU, LayerNormalization, ReLU
from keras.layers import BatchNormalization, Dense, Flatten, Reshape, Permute, Input, Lambda, Add, Dropout, RNN, Softmax, Concatenate
from keras import activations
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.saving import register_keras_serializable
import scipy
import h5py
import glob
from scipy import optimize
import gc


# LOAD COORDINATES 
def load_coords():
  with h5py.File('coordinates.mat', 'r') as mat_file:
  
    x = mat_file['x'][:].flatten()
    z = mat_file['z'][:].flatten()
    t = mat_file['t'][:].flatten()
    
  return x, z, t

# LOAD DNS DATA
def load_uwpT(n=2500):
  files = sorted([f for f in os.listdir('./data/') if 'RB' in f])
  data = []
  for f in files[:n]:
    with h5py.File(f, 'r') as h5f:
        data.append(np.array(h5f['uwpT']))

  uwpT = np.concatenate(data, axis=0)  
  x, z, t = load_coords()
  return uwpT, x, z, t

# LOAD PHYSICAL CONSTANTS
def load_constants():
  const_dict = {}
  with h5py.File('physical_constants.mat', 'r') as mat_file:
    for key, value in mat_file.items():
      const_dict[key] = np.array(value, dtype=np.float32)  
  return const_dict

# EXTRACT CONSTANTS
def get_model_constants(const_dict):
  _, z, _ = load_coords()
  Lz = z[-1] - z[0]
    
  delta_T = const_dict['T_bot']-const_dict['T_top']
  Uf = tf.math.sqrt(const_dict['alpha']*const_dict['g']*Lz*delta_T)
  P = const_dict['rho_0'] * (Uf**2)
  
  Pr = const_dict['visco'] / const_dict['kappa']
  Ra = (const_dict['alpha']*delta_T*const_dict['g']*Lz) / (const_dict['visco'] * const_dict['kappa'])
   
  return Uf, P, const_dict['T_bot'], const_dict['T_top'], np.array(Pr, np.float32), np.array(Ra, np.float32)

# NONDIMENSIONALIZE
def nondim(U, Uf, P, T_h, T_0):
  u, w, p, T = U[...,0,tf.newaxis], U[...,1,tf.newaxis], U[...,2,tf.newaxis], U[...,3,tf.newaxis]
  
  u, w = u/Uf, w/Uf
  p = p/P
  T = (T-T_0) / (T_h-T_0) - 0.5 
  return u, w, p, T


# HELPER FUNCTIONS TO GENERATE dx, dz, dt
def get_grads(x, z, const_dict, Uf):
  Lz = z[-1] - z[0]
  
  x = x / Lz
  z = z / Lz
  
  dx = x[2:] - x[:-2] 
  dz = z[2:] - z[:-2]
  
  dx = np.concatenate((x[1:2] - x[:1], dx, x[-2:-1] - x[-1:]))
  dz = np.concatenate((z[1:2] - z[:1], dz, z[-2:-1] - z[-1:]))
  
  dt = const_dict['plot_interval'] * Uf / Lz
  return tf.cast(dx, tf.float32), tf.cast(dz, tf.float32), tf.cast(dt, tf.float32) 
   
   
# LOAD TRAIN-VAL DATA SPLIT
def load_data(train_size, val_size, Uf, P, T_h, T_0):
  data_dim, x, z, t = load_uwpT() 

  u, w, p, T = nondim(data_dim, Uf, P, T_h, T_0)
  data = np.concatenate((u, w, p, T), axis=-1)

  data_train = data[:train_size]
  data_val = data[train_size:(train_size+val_size)]
  
  return np.array(data_train, dtype=np.float32), np.array(data_val, dtype=np.float32), x, z, t
  
# LOAD DATA FOR AE TRAINING
def load_ae_data(train_size, val_size, batch_size, Uf, P, T_h, T_0): 

  data_train, data_val, x, z, t = load_data(train_size, val_size, Uf, P, T_h, T_0)
  
  data_train_tf = tf.data.Dataset.from_tensor_slices((data_train, data_train))
  data_train_tf = data_train_tf.shuffle(buffer_size=train_size)
  data_train_tf = data_train_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  
  data_val_tf = tf.data.Dataset.from_tensor_slices((data_val, data_val))
  data_val_tf = data_val_tf.batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return data_train_tf, data_val_tf, x, z


# BUILD CAE
def build_ae(nodes_enc, kernel_size, activation):
  nodes_dec = list(reversed(nodes_enc))
  inputs = Input(shape=(256,256,4), name='inputs')
  x = inputs
     
  for i, k in enumerate(nodes_enc):
    x = Conv2D(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'ConvENC{i+1}')(x)
    x = LayerNormalization(name=f'ENCNorm{i+1}')(x)
    x = LeakyReLU(0.2, name=f'ReLUENC{i+1}')(x)
    
  for i, k in enumerate(nodes_dec):
    x = Conv2DTranspose(k, kernel_size=kernel_size, strides=2, activation=activation, padding='same', name=f'ConvDEC{len(nodes_enc)-i}')(x)
    x = LayerNormalization(name=f'DECNorm{len(nodes_enc)-i}')(x)
    x = LeakyReLU(0.2, name=f'ReLUDEC{len(nodes_enc)-i}')(x)
    
  x = Conv2D(4, kernel_size=1, activation='tanh', padding='same', name='ConvDECOut')(x)
  
  return tf.keras.Model(inputs, x, name='Autoencoder')

  
def get_ae_layers():
  autoencoder = tf.keras.models.load_model('ae.keras')

  enc_layers = []
  dec_layers = []
  for l in autoencoder.layers:
    if 'ENC' in l.name:
      enc_layers.append(l)
    elif 'DEC' in l.name:
      dec_layers.append(l)
        
  return enc_layers, dec_layers

# BUILD SPATIAL ENCODER TO REDUCE DIMENSION OF INPUT SEQUENCES TO PI-CRNN
def build_ae_encoder():
  enc_layers, _ = get_ae_layers()
  inputs = Input(shape=(None,256,256,4), name='inputs')
  x_enc = inputs
  for l in enc_layers: x_enc = TimeDistributed(l, name=l.name)(x_enc)
  return tf.keras.Model(inputs, x_enc, name='AE_Encoder')

# LOAD TRAINED SPATIAL DECODER 
def build_ae_decoder():
  _, dec_layers = get_ae_layers()
  inputs = Input(shape=(None,16,16,64), name='inputs')
  x = inputs
  for l in dec_layers: x = TimeDistributed(l, name=l.name)(x)
  return tf.keras.Model(inputs, x, name='AE_Decoder')


# PREPARE DATA FOR PI-CRNN
def load_lstm_data(train_size, val_size, look_b, look_f, Uf, P, T_h, T_0, seqs):
    np.random.seed(1)
    ae_encoder = build_ae_encoder()
    data_train, data_val, x, z, _ = load_data(train_size, val_size, Uf, P, T_h, T_0)
    data_train = tf.convert_to_tensor(data_train)
    data_val = tf.convert_to_tensor(data_val)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, look_b, 256, 256, 4], dtype=tf.float32)])
    def compress_in(x):
      return ae_encoder(x)
      
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, look_f, 256, 256, 4], dtype=tf.float32)])
    def compress_out(x):
      return ae_encoder(x)

    bsize = 1
    def create_dataset(starts, data):
      def generator():
        for t in starts:
          input_temp = compress_in(tf.expand_dims(data[(t-look_b):t,...], axis=0))
          input_dec_temp = compress_out(tf.expand_dims(data[t:(t+look_f),...], axis=0)) 
          output_temp = data[t:(t+look_f),...]
          yield input_temp[0], input_dec_temp[0], output_temp
              
      output_types = (tf.float32, tf.float32, tf.float32)
      output_shapes = (
          tf.TensorShape([look_b, 16, 16, 64]),
          tf.TensorShape([look_f, 16, 16, 64]),
          tf.TensorShape([look_f] + list(data.shape[1:])))
          
      return tf.data.Dataset.from_generator(generator, output_types, output_shapes)
    
    # Create datasets
    train_starts = np.random.choice(np.arange(look_b, train_size-look_f), size=seqs, replace=False)
    data_train_tf = create_dataset(train_starts, data_train)
    data_train_tf = data_train_tf.shuffle(buffer_size=seqs).batch(bsize).prefetch(tf.data.AUTOTUNE)
    
    val_starts = np.random.choice(np.arange(look_b, val_size-look_f), size=5, replace=False)
    data_val_tf = create_dataset(val_starts, data_val)
    data_val_tf = data_val_tf.batch(1).prefetch(tf.data.AUTOTUNE)
    
    return data_train_tf, data_val_tf, x, z
