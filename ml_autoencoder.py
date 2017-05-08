import tensorflow as tf
class ml_autoencoder:

    ml_encoder = None
    ml_decoder = None
    input_dimension = None
    output_dimension = None
    def __init__(self, input_dimension, reduced_dimension = None, output_dimension = None, hidden_layer_size = None):

        self.input_dimension = input_dimension
        if reduced_dimension == None:
            self.output_dimension = int(0.7*input_dimension)
        if hidden_layer_size is None:
            self.hidden_layer_size = 2*input_dimension
        if output_dimension is None:
            self.output_dimension = 2*self.hidden_layer_size

        self.input_embeddings = tf.placeholder(shape=[None, input_dimension], dtype= tf.float32)
        self.dropout = tf.placeholder(dtype= tf.float32)
        ip_drop = tf.nn.dropout(self.input_embeddings, self.dropout)

        with tf.name_scope('Encoder'):
            # self.encode(self.input_embeddings)
            W1_e = tf.Variable(tf.truncated_normal(
                shape=[self.input_dimension, self.hidden_layer_size],
                stddev=0.01))
            B1_e = tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size], stddev=0.01))
            op_1e = tf.nn.xw_plus_b(ip_drop, W1_e, B1_e)
            op1e =  tf.nn.relu(op_1e)

            W2_e = tf.Variable(tf.truncated_normal(
                shape=[self.hidden_layer_size, self.output_dimension],
                stddev=0.01))
            B2_e = tf.Variable(tf.truncated_normal(shape=[self.output_dimension], stddev=0.01))
            op_2e = tf.nn.xw_plus_b(op1e, W2_e, B2_e)
            op2e = tf.nn.relu(op_2e)

        self.reduced_embeddings = op2e

        with tf.name_scope('Decoder'):
            reduce_drop = tf.nn.dropout(self.reduced_embeddings, keep_prob= self.dropout)

            # self.decode(self.reduced_embeddings)
            W1_d = tf.Variable(tf.truncated_normal(
                shape=[self.output_dimension, self.hidden_layer_size],
                stddev=0.01))
            B1_d = tf.Variable(tf.truncated_normal(shape=[self.hidden_layer_size], stddev=0.01))
            op_1d = tf.nn.xw_plus_b(reduce_drop, W1_d, B1_d)
            op1d = tf.nn.relu(op_1d)

            W2_d = tf.Variable(tf.truncated_normal(
                shape=[self.hidden_layer_size, self.input_dimension],
                stddev=0.01))
            B2_d = tf.Variable(tf.truncated_normal(shape=[self.input_dimension], stddev=0.01))
            op_2d = tf.nn.xw_plus_b(op1d, W2_d, B2_d)
            op2d = tf.nn.relu(op_2d)


        self.predicted_embedding = op2d
        # with tf.name_scope('Final_Layer'):
        #     self.predicted_embedding = op2d
        #

        self.loss = tf.reduce_mean(tf.square(tf.sub(self.input_embeddings, self.predicted_embedding)))