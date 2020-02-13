# -*- coding: utf-8 -*-
"""

@author: Francesco
"""

import network
import Path
import tensorflow as tf
import json
import os 
import vocab_parser
import tensorflow_hub as hub
from tqdm import tqdm
# =============================================================================
# BiLSTM Parameters
# ============================================================================= 

LEARNING_RATE = 0.01 
HIDDEN_SIZE = 64  
EPOCHS = 100
SAVE_CHECKPOINT = 1
SUMMARY_CKPT = 500 # number of steps before saving summaries.
N_CKPT = 5 #max number of checkpoints to keep
BATCH_SIZE = 10

# =============================================================================
# Functions
# =============================================================================
def save_summaries(sess,feed_dict,writer,summary,gs):
    """ save summaries
        :param sess current tensorflow session
        :param feed_dict(list) list of summaries to save
        :param writer (tf.summary.FileWriter) 
        :param summary(tf.summary.merge) summary to save
        :param global_step (tf.Variable) the current training step 
    """
    summary_to_save = sess.run(
            summary,
            feed_dict=feed_dict,
            )
    #gs = tf.train.global_step(sess, global_step)
    writer.add_summary(summary_to_save, global_step=gs)
    writer.add_graph(graph=sess.graph, global_step = gs)
def save_model(sess,saver,global_step,ckpt_file):
    """ save the model variables 
        :param sess current tensorflow session
        :param saver (tf.train.Saver)
        :param global_step (tf.Variable) the current training step 
    """
    saver.save(sess,ckpt_file, global_step=global_step)


if __name__ == "__main__":
# =============================================================================
#   PATHS
# =============================================================================
    Path.Paths()
    xml_train_file = Path.Paths.xml_train_file
    word_vocab_file = Path.Paths.word_vocab_file
    wnid_vocab_file = Path.Paths.wnid_vocab_file
    skipped_file = Path.Paths.skipped_file
    gold_train_file = Path.Paths.gold_train_file
    wordid_to_wnsynset_file = Path.Paths.wordid_to_wnsynset_vocab_file
    fine_out_vocab_file = Path.Paths.fine_out_vocab_file
    pos_out_vocab_file = Path.Paths.pos_out_vocab_file
    bn_to_wn_file = Path.Paths.bn_to_wn_file
    bn_to_wnd_file = Path.Paths.bn_to_wnd_file
    bn_to_lex_file = Path.Paths.bn_to_lex_file
    wnd_out_vocab_file = Path.Paths.wnd_out_vocab_file
    lex_out_vocab_file = Path.Paths.lex_out_vocab_file
    
    MODEL_NAME = "MultiTaskBiLSTM"
    CKPT_DIR = Path.Paths.CHECKPOINT_DIR
    CKPT_FILE = os.path.join(CKPT_DIR,MODEL_NAME+".ckpt")
    SUMMARIES_DIR = Path.Paths.SUMMARIES_DIR
    epochs_file = Path.Paths.epochs_file
# =============================================================================
#   VOCABS
# =============================================================================
    bn_to_wn,wn_to_bn = vocab_parser.read_tsv_file(bn_to_wn_file)
    bn_to_wnd, _ = vocab_parser.read_tsv_file(bn_to_wnd_file)
    bn_to_lex, _ = vocab_parser.read_tsv_file(bn_to_lex_file)   
    output_vocab = vocab_parser.read_vocab(fine_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    wnd_vocab = vocab_parser.read_vocab(wnd_out_vocab_file)
    lex_vocab = vocab_parser.read_vocab(lex_out_vocab_file)
    pos_dim = len(pos_vocab)
    coarse_dim = len(wnd_vocab)    
    output_dim = len(output_vocab)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    
    tf.reset_default_graph()

    # variables used for keep track of the training
    global_step = tf.Variable(0, trainable=False, name='global_step')
    increase_gs = tf.assign(global_step, global_step+1)
    # =============================================================================
    # DEFINE VARIABLES TO SAVE
    # =============================================================================
    with tf.name_scope("Summaries"):
        train_loss = tf.placeholder(tf.float32, shape=(), name="train_loss")
        train_accuracy = tf.placeholder(tf.float32, shape=(),name="train_accuracy")
        dev_loss = tf.placeholder(tf.float32, shape=(), name="dev_loss")
        dev_accuracy = tf.placeholder(tf.float32, shape=(), name="dev_accuracy")
        fg_train_loss = tf.placeholder(tf.float32,shape=(),name="fg_train_loss")
        cg_train_loss = tf.placeholder(tf.float32,shape=(),name="cg_train_loss")
        pos_train_loss =tf.placeholder(tf.float32,shape=(),name="pos_train_loss") 
        fg_dev_loss = tf.placeholder(tf.float32,shape=(),name="fg_dev_loss")
        cg_dev_loss =tf.placeholder(tf.float32,shape=(),name="cg_dev_loss")
        pos_dev_loss = tf.placeholder(tf.float32,shape=(),name="pos_dev_loss")
    
    metrics_train = [
        tf.summary.scalar("train_loss", train_loss),
        tf.summary.scalar("train_accuracy",train_accuracy),
        tf.summary.scalar("dev_loss", dev_loss),
        tf.summary.scalar("dev_accuracy",dev_accuracy),
        tf.summary.scalar("fg_train_loss",fg_train_loss),
        tf.summary.scalar("cg_train_loss",cg_train_loss),
        tf.summary.scalar("pos_train_loss",pos_train_loss),
        tf.summary.scalar("fg_dev_loss",fg_dev_loss),
        tf.summary.scalar("cg_dev_loss",cg_dev_loss),
        tf.summary.scalar("pos_dev_loss",pos_dev_loss),
    ]
 
    merged_train = tf.summary.merge(metrics_train)

    # =============================================================================
    # RECOVER PREVIOUS TRAINING DATA
    # =============================================================================
    if os.path.exists(epochs_file):
        with open(epochs_file) as json_file:
            try:
                data = json.load(json_file)
                actual_epoch = data["step_number"]                  
            except:
                actual_epoch = 0
                print("\nActual epoch = 0 \n")    
    else:
        actual_epoch = 0
        print("\nActual step = 0 \n") 
        
    update_gs = tf.assign(global_step, actual_epoch)
    
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) 
    MultiTaskBiLSTM = network.MultiTask_BiLSTM (output_dim,pos_dim,coarse_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    saver = tf.train.Saver(max_to_keep=N_CKPT)
    writer = tf.summary.FileWriter(SUMMARIES_DIR,graph = graph)
    
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))
            print("\nModel restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            
        print("***************************TRAIN STARTED***************************************\n")
        for epoch in range(actual_epoch,EPOCHS):
            print("actual epoch: " + str(actual_epoch) + " of " + str(EPOCHS) + "\n")
            count = 0 
            total_loss_train = 0.0
            fg_loss_train = 0.0
            cg_loss_train = 0.0
            pos_loss_train = 0.0
            fg_loss_dev = 0.0
            cg_loss_dev = 0.0
            pos_loss_dev = 0.0
            total_accuracy_train = 0.0
            train_count = 0
            total_loss_dev = 0.0
            total_accuracy_dev = 0.0
            dev_count = 0
            for sentences, labels, synsets, lengths, pos, coarse_grained in \
                tqdm(MultiTaskBiLSTM.load_sentences_train(BATCH_SIZE,xml_train_file, word_vocab_file,\
                            bn_to_wn_file,bn_to_wnd_file,wnd_out_vocab_file,pos_out_vocab_file,\
                            skipped_file,fine_out_vocab_file,wordid_to_wnsynset_file,True)):
                # 8 step for training and 2 for dev
                count = count + 1
                
                if count % 10 < 8:
                    train_predictions, train_loss_val,train_loss_fg,train_loss_cg,train_loss_pos, _ =\
                        sess.run([MultiTaskBiLSTM.Dense,MultiTaskBiLSTM.loss,\
                        MultiTaskBiLSTM.loss_fg,MultiTaskBiLSTM.loss_cg,MultiTaskBiLSTM.loss_pos,MultiTaskBiLSTM.train],\
                        feed_dict = {MultiTaskBiLSTM.input: sentences, MultiTaskBiLSTM.labels: labels,MultiTaskBiLSTM.nopad_len: lengths,\
                                     MultiTaskBiLSTM.coarse_labels: coarse_grained,MultiTaskBiLSTM.pos_labels:pos})
                    train_accuracy_fg = MultiTaskBiLSTM.accuracy(train_predictions,lengths,labels,synsets)
                    
                    total_loss_train += train_loss_val
                    fg_loss_train += train_loss_fg
                    cg_loss_train += train_loss_cg
                    pos_loss_train +=  train_loss_pos
                    total_accuracy_train += train_accuracy_fg
                    train_count += 1
                else:
                    dev_predictions, dev_loss_val,dev_loss_fg,dev_loss_cg,dev_loss_pos =\
                        sess.run([MultiTaskBiLSTM.Dense,MultiTaskBiLSTM.loss,\
                        MultiTaskBiLSTM.loss_fg,MultiTaskBiLSTM.loss_cg,MultiTaskBiLSTM.loss_pos],\
                        feed_dict = {MultiTaskBiLSTM.input: sentences, MultiTaskBiLSTM.labels: labels,MultiTaskBiLSTM.nopad_len: lengths,\
                                     MultiTaskBiLSTM.coarse_labels: coarse_grained,MultiTaskBiLSTM.pos_labels:pos})
                                                         
                    dev_accuracy_fg = MultiTaskBiLSTM.accuracy(dev_predictions,lengths,labels,synsets)
                    fg_loss_dev += dev_loss_fg
                    cg_loss_dev += dev_loss_cg
                    pos_loss_dev += dev_loss_pos
                    total_loss_dev += dev_loss_val
                    total_accuracy_dev += dev_accuracy_fg
                    dev_count += 1
                    
                    
                if count == SUMMARY_CKPT:
                    print("Saving Summaries...\n")
                    if train_count != 0:
                        total_loss_train = total_loss_train/train_count
                        total_accuracy_train= total_accuracy_train/train_count
                        fg_loss_train = fg_loss_train/train_count
                        cg_loss_train = cg_loss_train/train_count
                        pos_loss_train = pos_loss_train/train_count
                        print("Actual loss: " + str(total_loss_train) + "\n")
                        print("Actual accuracy: " + str(total_accuracy_train) + "\n")
                    else:
                        print("Error, train counter is 0\n")
                    if dev_count != 0:
                        total_loss_train = total_loss_train/dev_count
                        total_accuracy_dev = total_accuracy_dev/dev_count
                        fg_loss_dev = fg_loss_dev/dev_count
                        cg_loss_dev = cg_loss_dev/dev_count
                        pos_loss_dev = pos_loss_dev/dev_count
                    else:
                        print("Error, dev counter is 0\n")
                        
                    train_count=0
                    dev_count=0
                    feed_dict_train = {
                        train_loss: float(total_loss_train),
                        train_accuracy : float(total_accuracy_train),
                        dev_loss: float(total_loss_dev),
                        dev_accuracy: float(total_accuracy_dev),
                        fg_train_loss: float(fg_loss_train),
                        cg_train_loss: float(cg_loss_train),
                        pos_train_loss: float(pos_loss_train),
                        fg_dev_loss: float(fg_loss_dev),
                        cg_dev_loss: float(cg_loss_dev),
                        pos_dev_loss: float(pos_loss_dev)
                     }
                    global_step_val = global_step.eval()
                    save_summaries(sess, feed_dict_train,writer, merged_train,global_step_val)
                    total_loss_train = 0.0
                    total_loss_dev = 0.0
                    total_accuracy_train = 0.0
                    total_accuracy_dev = 0.0
                    fg_loss_train = 0.0
                    cg_loss_train = 0.0
                    pos_loss_train = 0.0
                    fg_loss_dev = 0.0
                    cg_loss_dev = 0.0
                    pos_loss_dev = 0.0
                                
            # save model and global step number 
            global_step_val = global_step.eval()
            if global_step_val % SAVE_CHECKPOINT == 0 and global_step_val != 0:
                 save_model(sess, saver, global_step_val)
                 step_count = {"step_number": int(global_step_val)}
                 with open(epochs_file, 'w') as outfile:
                     json.dump(step_count, outfile)
            sess.run(increase_gs)
            global_step_val = global_step.eval()
            print(global_step_val)
    
    
    