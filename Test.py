import datetime
import tensorflow as tf
import numpy as np
from CNN import TextCNN
from tensorflow.contrib import learn
import DataExtractor
import pickle
from AutoEncoder import AutoEncoder
import time
import os
from sklearn import svm

'''
    train_data = numpy array of embeddings.
'''

istrain = False
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

run_path = "/home/mvidyasa/Documents/1488370322"
# file = open("/home/rpothams/PycharmProjects/Relation_Classifier_whole_pattern/resources/rel_pickle_id", 'rb')
# rel_id_Map = pickle.load(file)
# file.close()
TrainDatapath = "/home/mvidyasa/Documents/freepal_large_data_test"
TestDatapath = "/home/mvidyasa/Documents/freepal_large_data_test"
# rel_id_Map = {}
# rel_id_Map[0] = "REL$/business/company/founders"
# rel_id_Map[1] = "REL$/people/person/nationality"
# rel_id_Map[2] = "REL$/organization/parent/child"
# rel_id_Map[3] = "REL$/location/neighborhood/neighborhood_of"
# rel_id_Map[4] = "REL$/people/person/parents"

x_text_train, y_train = DataExtractor.load_data_and_labels_new(TrainDatapath, rel=1)
max_document_length = max([len(x.split(" ")) for x in x_text_train])

vocab_processor = learn.preprocessing.VocabularyProcessor.restore('/home/mvidyasa/Documents/vocab')

x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))

session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(config=session_conf)

# sess = tf.Session()
cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )


def Test_pattern(sess,x_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.dropout_keep_prob: 1.0
    }
    prediction = sess.run(
        cnn.h_pool_flat,
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    # if writer:
        # writer.add_summary(summaries, step)
    return prediction


global_step = tf.Variable(0, name="global_step", trainable=False)
sent_embedding_len = len(list(map(int, FLAGS.filter_sizes.split(","))))*FLAGS.num_filters

saver = tf.train.Saver()
saver.restore(sess, "/home/mvidyasa/Documents/checkpoints"+"/model-200")
print " Model restored."
count = 0

lines = open(TrainDatapath).readlines()
train_data = np.zeros([len(lines), sent_embedding_len])
train_patterns = []
for i, line in enumerate(lines):
    splt = line.split("\t")
    pattern = splt[0]
    list_sents = []
    list_sents.append(pattern)
    x_train = np.array(list(vocab_processor.fit_transform(list_sents)))
    train_data[i, :] = Test_pattern(sess, x_train)
    train_patterns.append(pattern)

print "Train data prepared for AutoEncoder"

###### AUTO ENCODER ##########
ae_train_saver = tf.train.Saver()
save_path = None
ae_model_path = "AE_Models/Model_13_March/model.ckpt"
min_loss = 4
auto_encoder = AutoEncoder(input_dimension= sent_embedding_len)
ae_train_saver.restore(sess, save_path= ae_model_path)
f = open("autoEncoder_loss_trainData","w")
# optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(auto_encoder.loss)
sess.run(tf.global_variables_initializer())
count =0
for i, pat in enumerate(train_patterns):
    # print "Epoch : "+str(i)
    input_emb = np.reshape(train_data[i,:],newshape= [1,384])
    loss = sess.run([auto_encoder.loss], feed_dict= {auto_encoder.input_embeddings: input_emb})
    print "Loss : "+str(loss)
    f.write(pat+"\t"+str(loss)+"\n")
    if loss[0]<1:
        count+=1
    # if loss >100000:
    #     sess.run(tf.global_variables_initializer())
    #     print "Reinitialized"
    # if loss<(min_loss*0.9):
    #     save_path = ae_train_saver.save(sess, save_path= ae_model_path)
    #     print "Model Saved"
    #     min_loss = loss
print "Accuracy: "+str(count)+" / "+ str(len(train_patterns))
f.close()