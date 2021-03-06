import tensorflow as tf
import numpy as np
import os
import time
import datetime
import DataExtractor
# from lstm_ep_att import relation_classifier_cpu
# from AutoEncoder import relation_classifier_cpu
from GloveAutoEncoder import relation_classifier_cpu
from tensorflow.contrib import learn
import pickle
from tensorflow.contrib.session_bundle import exporter
import shutil

# Parameters
# ==================================================

with tf.device('/cpu:0'):
    max_entityTypes = 42
    Cv_filepath = "resources/test_web_freepal"
    TrainDatapath = "resources/train_web_freepal"
    TestDataPath = "resources/test_web_freepal"
    # Data loading params
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
    tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

# Load data

# Data Preparation

    # ==================================================
    x_text_train, y_train, max_document_length = DataExtractor.load_data_and_labels_glove(TrainDatapath)
    x_text_cv, y_dev, _ = DataExtractor.load_data_and_labels_glove(Cv_filepath)
    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text_train])
    # max_document_length = DataExtractor.g.max_sequence_length
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))
    # x_dev = np.array(list(vocab_processor.fit_transform(x_text_cv)))
    x_train = x_text_train
    x_dev = x_text_cv

    # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    print("Loading data...")
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]



    # print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    file = open("resources/rel_pickle_id", 'rb')
    id_rel_Map = pickle.load(file)
    file.close()

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = relation_classifier_cpu(
                sequence_length=x_train.shape[1],
                # type_sequence_length = entityType_train_Arry.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = 0.05
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate,
                global_step,
                FLAGS.batch_size,
                0.97,
                staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            # train_summary_dir = os.path.join(out_dir, "summaries", "train")
            # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            export_dir = os.path.abspath(os.path.join(out_dir,"export"))
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # model_exporter = exporter.Exporter(saver)
            # model_exporter.init(
            #     sess.graph.as_graph_def(),
            #     named_graph_signatures={
            #         'inputs': exporter.generic_signature(
            #             {
            #                 'input_x': cnn.input_x,
            #                 'dropout_keep_prob': cnn.dropout_keep_prob
            #             }),
            #         'outputs': exporter.generic_signature(
            #             {'predictions': cnn.predictions})})

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  # cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss = sess.run(
                    [train_op, global_step, cnn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                # train_summary_writer.add_summary(summaries, step)
                return  loss
            # def dev_step(x_batch, y_batch, writer=None):
            #     """
            #     Evaluates model on a dev set
            #     """
            #     feed_dict = {
            #       cnn.input_x: x_batch,
            #       cnn.input_y: y_batch,
            #       cnn.dropout_keep_prob: 1.0
            #     }
            #     step, loss = sess.run(
            #         [global_step,  cnn.loss, ],
            #         feed_dict)
            #     time_str = datetime.datetime.now().isoformat()
            #     print("{}: step {}, loss {:g}".format(time_str, step, loss))
            #     if writer:
            #         writer.add_summary(step)
            #     return predictions


            max_accuracy = 100
            # Generate batches
            batches = DataExtractor.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Generate Test batches
            test_batches = DataExtractor.batch_iter(
                list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
            n_samples = len(x_train)
            # Training loop. For each batch...
            for batch in batches:
                avg_cost = 0.
                x_batch, y_batch = zip(*batch)
                loss = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                avg_cost += loss / n_samples * FLAGS.batch_size
                # if current_step % FLAGS.evaluate_every == 0:
                #     print("\nEvaluation:")
                #     predictions,accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #
                #     f = open(os.path.join(out_dir, "cv_results"),'w')
                #     for prediction in predictions:
                #         f.write(id_rel_Map[prediction] + "\n")
                #     f.close()
                #     print predictions
                #     print("")
                #     print("Max accuracy :" + str(max_accuracy))
                if current_step % FLAGS.checkpoint_every == 0:
                    min_loss = loss
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    if os.path.exists(export_dir):
                        try:
                            shutil.rmtree(export_dir)
                        except OSError as ex:
                            print(ex)
                    else:
                        os.makedirs(export_dir)
                    # try:
                    #     model_exporter.export(export_dir, tf.constant(1), sess)
                #     except:
                #         print "export error"
            print "Max accuracy model:" + str(max_accuracy)
            print "Average Loss : "+str(avg_cost)
# export CUDA_VISIBLE_DEVICES=
