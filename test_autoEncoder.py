import datetime
import tensorflow as tf
import numpy as np
# from CNN_with_Attn import TextCNN_with_Attn
from AutoEncoder import relation_classifier_cpu
# from lstm_ep_att import relation_classifier_cpu
from tensorflow.contrib import learn
import DataExtractor
import pickle
lstm_hidden_size = 100
Da = 50

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("type_embedding_size", 300, "Dimensionality of type pair embedding (default:128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes2", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

run_path = "runs/1494345152"
file = open("resources/rel_pickle_id", 'rb')
rel_id_Map = pickle.load(file)
file.close()
TrainDataPath = "resources/train_web_freepal"
TestDatapath = "resources/test_web_freepal"
# rel_id_Map = {}
# rel_id_Map[0] = "REL$/business/company/founders"
# rel_id_Map[1] = "REL$/people/person/nationality"
# rel_id_Map[2] = "REL$/organization/parent/child"
# rel_id_Map[3] = "REL$/location/neighborhood/neighborhood_of"
# rel_id_Map[4] = "REL$/people/person/parents"

x_text_train, y_train = DataExtractor.load_data_and_labels_new(TestDatapath)
max_document_length = max([len(x.split(" ")) for x in x_text_train])

vocab_processor = learn.preprocessing.VocabularyProcessor.restore('runs/1494343499/vocab')

f = open(run_path+'/vocab','w')
word_to_id_map = vocab_processor.vocabulary_._mapping
word_to_id_map["max_sequence_length"] = max_document_length
pickle.dump(word_to_id_map,f)
f.close()

x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))

session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(config=session_conf)

# sess = tf.Session()
cnn = relation_classifier_cpu(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_) -1,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )


def Test_pattern(sess,x_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict1 = {
        cnn.input_x: x_batch,
        cnn.dropout_keep_prob: 1.0
    }
    loss_vector = sess.run(
        [cnn.loss],
        feed_dict1)

    return loss_vector


global_step = tf.Variable(0, name="global_step", trainable=False)

saver = tf.train.Saver()
saver.restore(sess, run_path+"/checkpoints/model-100")
# sess.run(tf.global_variables_initializer())
count = 0
mis_match_count = 0
cor_count = 0
match_loss_mean = 0
mis_match_loss_mean = 0
lines = open(TrainDataPath).readlines()
fw = open('results_in_domain_data_test','w')
indomain_count = 0
threshold = 0.000001

ip_autoEncoder = np.zeros([len(lines), 2*lstm_hidden_size] )

for i,line in enumerate(lines):
    if line.__contains__(".xml"):
        continue

    splt = line.strip("\n").split("\t")
    pattern = splt[0]
    # act_rel_id = int(splt[1].strip("\n"))
    # actual_relation = rel_id_Map[act_rel_id]
    list_sents = []
    list_sents.append(pattern)
    x_test = np.array(list(vocab_processor.fit_transform(list_sents)))
    loss_vector = Test_pattern(sess,x_test)

    if loss_vector > threshold:
        indomain_count+=1
    else:
        count+=1

    # if predicted_relation == actual_relation:
    #     match_loss_mean += loss
    #     count += 1
    #     if loss < 0.45:
    #         cor_count += 1
    # else:
    #     mis_match_loss_mean += loss
    #     mis_match_count += 1
    # domain = "in-domain"
    # if loss > 0.45:
    #     count += 1
    #     domain = "out-domain"
    # else:
    #     indomain_count += 1
    # fw.write(pattern + "\t" + domain + "\t" + str(loss) + "\t" + str(score[0][prediction[0]])+"\n")
    # result.append(predicted_relation)
print "Out domain count:" + str(count) ##+ "\t" + str(cor_count)
print "in domain count" + str(indomain_count)
# print "match loss mean : " + str(match_loss_mean/count)
# print "mis match loss mean " + str(mis_match_loss_mean/mis_match_count)
fw.close()