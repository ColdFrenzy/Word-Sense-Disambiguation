# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:44:09 2020

@author: Francesco
"""




"""
SCRIPT ONLY USED TO TEST THE MULTILINGUAL NETWORK
"""

from elmoformanylangs import Embedder
import Path
import torch
import tensorflow as tf
from tensorflow.keras import layers
import utilities
import os
#import network
torch.cuda.current_device()
torch.cuda._initialized = True

it_model_path = Path.Paths.IT_MODEL_PATH

# =============================================================================
# PARAMETERS
# =============================================================================
LEARNING_RATE = 0.01 
HIDDEN_SIZE = 64  
EPOCHS = 100
SAVE_CHECKPOINT = 1
SUMMARY_CKPT = 500 # number of steps before saving summaries.
N_CKPT = 5 #max number of checkpoints to keep
BATCH_SIZE = 10
OUTPUT_SIZE = 500 
POS_SIZE = 8 
COARSE_SIZE = 15 
EMB_SIZE = 1024


class MultiLing_BiLSTM:
    """
    network with a simple bilstm layer used for pos tagging,fine and 
    coarse-grained classification
    """
    def __init__(self, output_size,pos_size,coarse_size, learning_rate, h_size):
        """ 
            :input output_size: size of the vocabulary containing the fine-grained words
            :input lr: learning rate 
            :input h_size: size of the biLSTM hidden layer
        
        """
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.h_size = h_size
        
        with tf.variable_scope("MultiTask_BiLSTM"):

            self.elmo_emb = tf.placeholder(tf.float32,name="elmo_emb",shape=[None,None,EMB_SIZE])
            self.labels = tf.placeholder(tf.int32,name = "labels",shape=[None,None] )
            self.pos_labels = tf.placeholder(tf.int32,name = "pos_labels", shape=[None,None])
            self.coarse_labels = tf.placeholder(tf.int32,name = "coarse_labels",shape=[None,None])
            # length of the real sequences without padding
            self.nopad_len = tf.placeholder(tf.int32, name = "sequence_len",shape=[None])
            
            # elmo pre-trained module for sense embeddings.
            
            self.input_mask = tf.sequence_mask(self.nopad_len)            
            # Bidirectional take the input lstm and create a copy that goes in the
            # opposite direction.
            self.LSTM = layers.LSTM(self.h_size,return_sequences=True)
            #self.bwd_lstm = layers.LSTM(self.h_size,return_sequences=True,go_backwards=True)
            self.biLSTM = layers.Bidirectional(self.LSTM, merge_mode='concat')(self.elmo_emb)#, mask=self.input_mask)
            self.Dense = layers.Dense(output_size)(self.biLSTM)
            self.Dense_pos = layers.Dense(pos_size)(self.biLSTM)
            self.Dense_coarse = layers.Dense(coarse_size)(self.biLSTM)
        
        #fine-grained loss
        with tf.variable_scope("fine_grained_loss"):
            # sparse_softmax labels must have the shape [batch_size], while
            # softmax labels must have shapes [batch_size,num_classes] (one_hot)
            loss_fg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.Dense,\
                                                                          labels = self.labels)
            # use the input mask to exclude padding tokens in the optimization phase.
            loss_fg = tf.boolean_mask(loss_fg,self.input_mask)
            self.loss_fg = tf.reduce_mean(loss_fg)
        with tf.variable_scope("coarse_grained_loss"):
            loss_cg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.Dense_coarse,\
                                                                          labels = self.coarse_labels)
            # use the input mask to exclude padding tokens in the optimization phase.
            loss_cg = tf.boolean_mask(loss_cg,self.input_mask)
            self.loss_cg = tf.reduce_mean(loss_cg)
            
        with tf.variable_scope("pos_loss"):
            loss_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.Dense_pos,\
                                                                          labels = self.pos_labels)
            # use the input mask to exclude padding tokens in the optimization phase.
            loss_pos = tf.boolean_mask(loss_pos,self.input_mask)
            self.loss_pos = tf.reduce_mean(loss_pos)            
        with tf.variable_scope("loss"):
            self.loss = self.loss_fg + self.loss_cg + self.loss_pos
            
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            
            self.train = self.optimizer.minimize(self.loss_fg)  
            
            
if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    tf.reset_default_graph()
    CKPT_DIR = Path.Paths.CHECKPOINT_DIR
    xml_path = os.path.join(Path.Paths.MULTILING_DIR,"eval_dataset","semeval2013.it.data.xml")
    elmo = Embedder(it_model_path,batch_size= BATCH_SIZE)
    network=MultiLing_BiLSTM(OUTPUT_SIZE,POS_SIZE,COARSE_SIZE,LEARNING_RATE,HIDDEN_SIZE)
    with tf.Session() as sess:


# =============================================================================
#    CHECK VARIABLE NAMES AND VALUES
# =============================================================================

#    all_vars = tf.global_variables()
#    for var in all_vars:
#        print(var.name)
#    all_vars = tf.global_variables()
#    initial_val = []
#    for var in all_vars:
#        initial_val.append(var.eval())
            
# =============================================================================
#       Restore variables of the english MultiTask_BiLSTM
# =============================================================================
        vars_to_restore=[v for v in tf.global_variables() if "MultiTask_BiLSTM" in v.name or  "optimizer" in v.name]
        vars_to_restore_dict = {}
        for v in vars_to_restore:
            vars_to_restore_dict[v.name[:-2]] = v
        
        saver = tf.train.Saver(vars_to_restore_dict)
        saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))
     

        #tf.global_variables_initializer().run() 
        for sent,_,_ in utilities.load_sentences(BATCH_SIZE,xml_path):
                my_inp = elmo.sents2elmo(sent)
                yolo = sess.run(network.Dense,feed_dict = {network.elmo_emb:my_inp})
                break