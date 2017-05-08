import tensorflow as tf
class AutoEncoder:

    input_dimension = None
    output_dimesion = None
    '''
        Convert D(dimension of embedding) dimension to D' dimension.
        By Default D' = 0.5 * D
    '''
    def encode(self, embedding):
        with tf.name_scope("encoder"):
            W = tf.get_variable("W",
                                dtype= tf.float32,
                                initializer= tf.truncated_normal(
                                    shape= [self.input_dimension, self.output_dimesion],
                                    stddev= 0.01))
            B = tf.get_variable("B",
                                dtype= tf.float32,
                                initializer=tf.truncated_normal(shape= [self.output_dimesion], stddev= 0.01))
            op_ = tf.nn.xw_plus_b(embedding, W, B)
            op = tf.nn.tanh(op_, name= "output")
            return op

    '''
            Convert D' dimension to D dimension.
            By Default D' = 0.5 * D
    '''
    def decode(self, embedding):
        with tf.name_scope("decoder"):
            W = tf.get_variable("W_",
                                dtype= tf.float32,
                                initializer=tf.truncated_normal(
                                    shape= [self.output_dimesion, self.input_dimension], stddev= 0.01))
            B = tf.get_variable("B_",
                                dtype= tf.float32,
                                initializer=tf.truncated_normal(shape= [self.input_dimension], stddev= 0.01))
            op_ = tf.nn.xw_plus_b(embedding, W, B)
            op = tf.nn.tanh(op_, name= "output")
            return op

    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.output_dimesion = int(0.5*self.input_dimension)
        self.input_embeddings = tf.placeholder(shape=[None, input_dimension], dtype= tf.float32)

        # self.encode(self.input_embeddings)
        W1 = tf.Variable(tf.truncated_normal(
                                shape=[self.input_dimension, self.output_dimesion],
                                stddev=0.01))
        B1 = tf.Variable(tf.truncated_normal(shape=[self.output_dimesion], stddev=0.01))
        op_ = tf.nn.xw_plus_b(self.input_embeddings, W1, B1)
        self.reduced_embeddings =tf.nn.relu(op_, name="encoded_output")

        # self.decode(self.reduced_embeddings)
        W2 = tf.Variable(tf.truncated_normal(shape=[self.output_dimesion, self.input_dimension], stddev=0.01))
        B2 = tf.Variable(tf.truncated_normal(shape=[self.input_dimension], stddev=0.01))
        op_1 = tf.nn.xw_plus_b(self.reduced_embeddings, W2, B2)
        self.predicted_embedding = tf.nn.relu(op_1, name="decoded_output")

        self.loss = tf.reduce_mean(tf.square(tf.sub(self.input_embeddings, self.predicted_embedding)))