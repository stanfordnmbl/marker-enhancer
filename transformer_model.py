# Code inspired from https://www.tensorflow.org/text/tutorials/transformer.
# Neural machine translation with a Transformer and Keras

import tensorflow as tf
import numpy as np

from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

# %% Positional Encoding
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

# %% Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, in_dimension, d_model):
    super().__init__()
    self.d_model = d_model
    # Original line:
    # self.embedding = tf.keras.layers.Embedding(in_dimension, d_model, mask_zero=True)
    # Embedding layer does not seem suited for type of problem with 2D array.
    # TODO: does it make sense to use Dense alyer instead? Reasonning is that
    # it acts as a linear layer, mapping to the latent space. Tom used 
    # nn.Linear(self.number_of_inputs, self.number_of_outputs) in pytorch,
    # which I believe is doing the same. Dimensions seem to make sense, we are
    # mapping from the input dimension (eg, 64,59,47) to the d_model dimension
    # (eg, 64,59,512).
    self.embedding = tf.keras.layers.Dense(d_model)    
    self.pos_encoding = positional_encoding(length=10000, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    # In other words, it scales the embedding vectors to ensure stability
    # during training and to improve the conditioning of the optimization process.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

# %% Attention
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    # Print the kwargs, so we know what they are.
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    
# Global self-attention is used in the encoder.
# This layer is responsible for processing the context sequence, and propagating
# information along its length. Since the context sequence is fixed while the
# translation is being generated, information is allowed to flow in both
# directions. 
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

# Causal attention is used in the decoder.
# Transformers are an "autoregressive" model: They generate the text one token
# at a time and feed that output back to the input. To make this efficient, 
# these models ensure that the output for each sequence element only depends on
# the previous sequence elements; the models are "causal".
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

# Cross attention to connect the encoder and decoder.
# To implement this you pass the target sequence x as the query and the context
# sequence as the key/value when calling the mha layer
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
    
    # Cache the attention scores for plotting later.
    # self.last_attn_scores = attn_scores
    
    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

# %% Feedforward
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, d_ff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(d_ff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  
# %% Encoder
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, d_ff, d_k, d_v, dropout_rate=0.1,
               attention_axes=None):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_k,
        value_dim=d_v,
        dropout=dropout_rate,
        attention_axes=attention_axes)

    self.ffn = FeedForward(d_model, d_ff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               d_ff, d_k, d_v, feature_dimension, dropout_rate=0.1,
               attention_axes=None):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        in_dimension=feature_dimension, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     d_ff=d_ff,
                     d_k=d_k,
                     d_v=d_v,
                     dropout_rate=dropout_rate,
                     attention_axes=attention_axes)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)
    
    return x  # Shape `(batch_size, seq_len, d_model)`.

# %% Decoder
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, d_ff, dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, d_ff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    # self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, d_ff, label_dimension,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(in_dimension=label_dimension,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     d_ff=d_ff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    # self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    # self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

# %% Custom learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# %% Transformer
class Transformer_EncoderOnly(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, d_ff, d_k, d_v,
               feature_dimension, label_dimension, dropout_rate=0.1,
               attention_axes=None):
    super().__init__()
    
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, d_ff=d_ff,
                           d_k=d_k, d_v=d_v,
                           feature_dimension=feature_dimension,
                           dropout_rate=dropout_rate,
                           attention_axes=attention_axes)

    self.final_layer = tf.keras.layers.Dense(label_dimension)

  def call(self, inputs):
    inputs = self.encoder(inputs)  # (batch_size, context_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(inputs)  # (batch_size, context_len, output_dim)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits


def get_transformer_encoderonly_model(
      input_dim, output_dim, num_layers=6, num_heads=8, d_model=512, d_ff=2048,
      d_k=64, d_v=64, loss_f='mean_squared_error', dropout=0.1,
      attention_axes=None, weights=None, learning_r='custom'):
    
    # inputs = Input(shape=(None, input_dim))
    # targets = Input(shape=(None, output_dim))
  
    transformer = Transformer_EncoderOnly(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        d_k=d_k,
        d_v=d_v,
        feature_dimension=input_dim,
        label_dimension=output_dim,
        dropout_rate=dropout,
        attention_axes=attention_axes)
    
    if learning_r == 'custom':
      learning_rate = CustomSchedule(d_model)
    else:
      learning_rate = learning_r

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, 
                                         beta_2=0.98, epsilon=1e-9)    

    # Loss function.
    if loss_f == "weighted_l2_loss":
        transformer.compile(
            optimizer=optimizer,
            loss=weighted_l2_loss(weights))
            # metrics=[MeanSquaredError(), RootMeanSquaredError()])
    else:
        transformer.compile(
            optimizer=optimizer,
            loss=loss_f)
            # metrics=[MeanSquaredError(), RootMeanSquaredError()])
    
    return transformer

# %% Inference
# Note: The model is optimized for efficient training and makes a next-token
# prediction for each token in the output simultaneously. This is redundant 
# during inference, and only the last prediction is used. This model can be
# made more efficient for inference if you only calculate the last prediction
# when running in inference mode (training=False).     
class Augmenter_EncoderOnly(tf.Module):
    def __init__(self, transformer):
        self.transformer = transformer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, input_array):
        # Expand to match expected dimensions
        input_array = tf.expand_dims(input_array, axis=0)
        
        # Generate target sequence
        output_array = self.transformer(input_array, training=False)[0,:,:]

        return output_array

# %% Export  
class ExportAugmenter_EncoderOnly(tf.Module):
  def __init__(self, augmenter):
    self.augmenter = augmenter
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
  def __call__(self, input_array):
      
      output_array = self.augmenter(input_array)
        
      return output_array
  
# %% Helper functions.
def weighted_l2_loss(weights):
    def loss(y_true, y_pred):      
        squared_difference = tf.square(y_true - y_pred)        
        weighted_squared_difference = weights * squared_difference  
        return tf.reduce_mean(weighted_squared_difference, axis=-1)
    return loss
