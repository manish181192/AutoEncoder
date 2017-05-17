import tensorflow as tf


class relation_classifier_cpu(object):
    lstm_hidden_size = 100
    Da = 50
    ip_dimension = 300

    def attentive_sum(self, inputs, input_dim, hidden_dim):
        with tf.variable_scope("attention"):
            seq_length = len(inputs)
            W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[input_dim, hidden_dim]))
            U = tf.get_variable(name="U", initializer=tf.truncated_normal(shape=[hidden_dim, 1]))
            tf.get_variable_scope().reuse_variables()
            temp1 = [tf.nn.tanh(tf.matmul(inputs[i], W)) for i in range(seq_length)]
            temp2 = [tf.matmul(temp1[i], U) for i in range(seq_length)]
            pre_activations = tf.concat(1, temp2)
            attentions = tf.split(1, seq_length, tf.nn.softmax(pre_activations))
            weighted_inputs = [tf.mul(inputs[i], attentions[i]) for i in range(seq_length)]
            output = tf.add_n(weighted_inputs)
        return output, attentions
    def get_context_vector(self, x):
        # Context Vector
        with tf.name_scope('relation_vector'):
            with tf.name_scope('bi-lstm'):
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_size,
                                                  initializer=tf.truncated_normal_initializer(stddev=0.001,
                                                                                              dtype=tf.float32))
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob= self.dropout_keep_prob)

                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_size,
                                                  initializer=tf.truncated_normal_initializer(stddev=0.001,
                                                                                              dtype=tf.float32))
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob= self.dropout_keep_prob)

                self.outputs, fw_states, bw_states = tf.nn.bidirectional_rnn(cell_fw=fw_cell,
                                                                        cell_bw=bw_cell,
                                                                        inputs=x,
                                                                        dtype=tf.float32)
            Vc, attentions = self.attentive_sum(self.outputs,
                                           input_dim= 2* self.lstm_hidden_size,
                                           hidden_dim= self.Da)
            # with tf.name_scope('attention_layer'):
            #     # Left Attention Output
            #     att_outputs = []
            #     h_concat_list = []
            #
            #     for i in range(self.sequence_length):
            #         # fw_h = outputs_left[i][0]
            #         # bw_h = outputs_left[i][0]
            #         h = tf.concat(concat_dim=3, values=outputs[i])
            #         h_concat_list.append(h)
            #
            #         # LAYER 1
            #         We = tf.get_variable("We", shape=[2 * self.lstm_hidden_size, self.Da],
            #                              initializer=tf.truncated_normal_initializer(stddev=0.001),
            #                              dtype=tf.float32)
            #         # Be = tf.get_variable("Be", shape=[data_reader.Da],
            #         #                      initializer=tf.truncated_normal_initializer(stddev=0.001),
            #         #                      dtype=tf.float32)
            #         #
            #         # op_1 = tf.nn.xw_plus_b(x= h, weights= We, biases= Be)
            #         op_1 = tf.matmul(h, We)
            #         op1 = tf.tanh(op_1)
            #         # LAYER 2
            #         Wa = tf.get_variable("Wa", shape=[self.Da, 1],
            #                              initializer=tf.truncated_normal_initializer(stddev=0.001),
            #                              dtype=tf.float32)
            #         # Ba = tf.get_variable("Ba", shape=[1],
            #         #                      initializer=tf.truncated_normal_initializer(stddev=0.001),
            #         #                      dtype=tf.float32)
            #         # op_2 = tf.nn.xw_plus_b(x= op1, weights= Wa, biases= Ba)
            #         op_2 = tf.matmul(op1, Wa)
            #         op2 = tf.exp(op_2)
            #
            #         att_outputs.append(op2)
            #         tf.get_variable_scope().reuse_variables()
            #
            #     att_tensor = tf.pack(att_outputs)
            #     total_att_sum = tf.reduce_sum(input_tensor=att_tensor, axis=0)
            #
            #     # Normalize Attention
            #     norm_att_output = []
            #     for i in range(self.sequence_length):
            #         norm_att_output.append(tf.div(att_outputs[i], total_att_sum))
            #
            #     # PATTERN VECTOR
            #     temp_list = []
            #     for i in range(self.sequence_length):
            #         temp = tf.mul(norm_att_output[i], h_concat_list[i])
            #         # temp2 = tf.mul(norm_right_att_output[i], right_h_concat_list[i])
            #         # temp = tf.add(temp1, temp2)
            #         temp_list.append(temp)
            #         # sum_temp = tf.add(Vc, temp)
            #         # tf.assign(Vc, sum_temp)
            #
            #     temp_tensor = tf.pack(temp_list)
            #     Vc = tf.reduce_sum(temp_tensor, axis=0)
        return Vc

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size,  filter_sizes, num_filters, batch_size=64, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, self.ip_dimension], name="input_x")

        # self.input_left_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_left_x")
        # self.input_right_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_right_x")
        # self.input_center_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_center_x")
        # self.entityTypePId = tf.placeholder(tf.int32, [None, type_sequence_length], name="enitty_types")
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # with tf.device('/cpu:0'), tf.name_scope("entity_type_embedding"):
        # self.lookup_table_etype = tf.Variable(tf.random_uniform([type_size, type_embedding_size], -1.0, 1.0),name="etype_lookup")
        # embedded_type = tf.nn.embedding_lookup(self.lookup_table_etype, self.entityTypePId)
        # embedded_type_expanded = tf.reshape(embedded_type,[-1, type_sequence_length * type_embedding_size])

        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     self.lookup_table = tf.Variable(
        #         tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        #         name="lookup")
        #     embedded_chars_left = tf.nn.embedding_lookup(self.lookup_table, self.input_x)
        #     # embedded_chars_right = tf.nn.embedding_lookup(self.lookup_table, self.input_right_x)
        #     # embedded_chars_center = tf.nn.embedding_lookup(self.lookup_table, self.input_center_x)
        #
        #     embedded_chars_expanded = tf.expand_dims(embedded_chars_left, -1)
        embedded_chars_expanded = self.input_x
        # embedded_chars_expanded = tf.nn.dropout( embedded_chars_expanded, keep_prob= self.dropout_keep_prob)

        # x_1 = tf.transpose(embedded_chars_right, perm=[1, 0, 2])
        # x_2 = tf.reshape(x_1, [-1, embedding_size])
        # x_right = tf.split(value=x_2,
        #              num_split=sequence_length,
        #              split_dim=0)
        # x_1 = tf.transpose(embedded_chars_center, perm=[1, 0, 2])
        # x_2 = tf.reshape(x_1, [-1, embedding_size])
        # x_center = tf.split(value=x_2,
        #              num_split=sequence_length,
        #              split_dim=0)
        # with tf.name_scope("Attn-layer"):
        #     Wa = tf.Variable(
        #         tf.truncated_normal([type_sequence_length * type_embedding_size, sequence_length * embedding_size],
        #                             stddev=0.1), name="Wa")
        #
        #     lay = tf.matmul(embedded_type_expanded, Wa)
        #     sgm = tf.sigmoid(lay)
        #     reshaped_sgm = tf.reshape(sgm, [-1, sequence_length, embedding_size])
        #     reshaped_sgm_exp = tf.expand_dims(reshaped_sgm, -1)
        #     x_left = tf.mul(embedded_chars_expanded, reshaped_sgm_exp)

        # x_ = tf.reshape(embedded_chars_expanded, [tf.shape(self.input_x)[0], sequence_length, embedding_size])
        # self.x_1 = tf.transpose(embedded_chars_expanded, perm=[1, 0, 2])
        # self.x_2 = tf.reshape(self.x_1, [-1, embedding_size])
        # self.x_left = tf.split(value=self.x_2,
        #                   num_split=sequence_length,
        #                   split_dim=0)
            # x_right = tf.mul(x_right, reshaped_sgm_exp)
            # x_center = tf.mul(x_center, reshaped_sgm_exp)
        self.x_left = tf.unstack(embedded_chars_expanded, 14, 1)
        Vc_left = self.get_context_vector(self.x_left)
        # Vc_right = self.get_context_vector(x_right)
        # Vc_center = self.get_context_vector(x_center)

        # Vc = tf.concat(values=[Vc_left, Vc_right, Vc_center], concat_dim= 1, name="context_vector")
        self.Vc = Vc_left
        # # Add dropout
        # with tf.name_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(Vc, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions


        with tf.name_scope('auto_encoder'):
            self.input_dimension = 2*self.lstm_hidden_size
            self.output_dimesion = 50
            self.input_embeddings = self.Vc

            # self.encode(self.input_embeddings)
            W1 = tf.Variable(tf.truncated_normal(
                shape=[self.input_dimension, self.output_dimesion],
                stddev=0.01))
            B1 = tf.Variable(tf.truncated_normal(shape=[self.output_dimesion], stddev=0.01))
            op_ = tf.nn.xw_plus_b(self.input_embeddings, W1, B1)
            self.reduced_embeddings = tf.nn.relu(op_, name="encoded_output")

            # self.decode(self.reduced_embeddings)
            W2 = tf.Variable(tf.truncated_normal(shape=[self.output_dimesion, self.input_dimension], stddev=0.01))
            B2 = tf.Variable(tf.truncated_normal(shape=[self.input_dimension], stddev=0.01))
            op_1 = tf.nn.xw_plus_b(self.reduced_embeddings, W2, B2)
            self.predicted_embedding = tf.nn.relu(op_1, name="decoded_output")

        self.loss_vector = tf.square(tf.sub(self.input_embeddings, self.predicted_embedding))
        self.loss = tf.reduce_mean(tf.reduce_sum(self.loss_vector, axis= 1), axis= 0)

        #
        # with tf.name_scope("output"):
        #     W = tf.Variable(tf.truncated_normal(shape=[2 * self.lstm_hidden_size, num_classes], name="W"))
        #     # "W",
        #     # shape=[2*embedding_size, num_classes],
        #     # initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #     # l2_loss += tf.nn.l2_loss(W)
        #     # l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(self.Vc, W, b, name="scores")
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #
        # # CalculateMean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        #
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")