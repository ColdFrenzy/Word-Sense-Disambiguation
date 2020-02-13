# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:57:37 2020

@author: Francesco
"""


"""
SCRIPT USED TO COMPUTE THE ACCURACY OF OUR NETWORK OVER THE EVALUATION DATASET

"""


import network
import Path
import tensorflow as tf
import os 
import vocab_parser
import tensorflow_hub as hub
# =============================================================================
# BiLSTM Parameters
# ============================================================================= 

LEARNING_RATE = 0.01 
HIDDEN_SIZE = 64  
EPOCHS = 100
BATCH_SIZE = 10

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
    
    EVAL_DIR = Path.Paths.EVAL_DIR
    MODEL_DIR = Path.Paths.MODEL_DIR
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


    
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) 
    MultiTaskBiLSTM = network.MultiTask_BiLSTM (output_dim,pos_dim,coarse_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))
            print("\nModel restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            




        eval_dirs = [os.path.join(EVAL_DIR, o) for o in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR,o))]
        eval_paths = []

        for eval_dir in eval_dirs:
            print(eval_dir)
            eval_files = os.listdir(eval_dir)
            for eval_file in eval_files:
                print(eval_file.split(".")[-1])
                if eval_file.split(".")[-1] == "xml":
                    eval_file = os.path.join(eval_dir,eval_file)
                    eval_paths.append(eval_file)
    
        for eval_path in eval_paths:
            accuracy_total = 0
            steps = 0
            for sentences, labels, synsets, lengths, pos, coarse_grained in \
                MultiTaskBiLSTM.load_sentences_train(BATCH_SIZE,eval_path, word_vocab_file,\
                            bn_to_wn_file,bn_to_wnd_file,wnd_out_vocab_file,pos_out_vocab_file,\
                            skipped_file,fine_out_vocab_file):
              
                    
                    train_predictions = sess.run(MultiTaskBiLSTM.Dense,\
                        feed_dict = {MultiTaskBiLSTM.input: sentences,MultiTaskBiLSTM.nopad_len: lengths})
                    accuracy_fg = MultiTaskBiLSTM.accuracy(train_predictions,lengths,labels,synsets)
                    accuracy_total += accuracy_fg
                    steps += 1
 
            accuracy_total = float(accuracy_total/steps)  
            print("accuracy for " + str(eval_path.split("/")[-1] + " is "  + str(accuracy_total)))
            
                                
 
    
    
    