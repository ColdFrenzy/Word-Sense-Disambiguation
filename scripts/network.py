# -*- coding: utf-8 -*-
"""

@author: Francesco
"""

import tensorflow as tf
from tensorflow.keras import layers  
import tensorflow_hub as hub
import vocab_parser
from random import shuffle
import numpy as np
import utilities
import Path

# =============================================================================
# BILSTM FOR FINE-GRAINED CLASSIFICATION
# =============================================================================
class BiLSTM_FG:
    """
    network with a simple bilstm layer used for fine-grained
    classification
    """
    def __init__(self, output_size, learning_rate, h_size,elmo):
        """ 
            :input output_size: size of the vocabulary containing the fine-grained words
            :input lr: learning rate 
            :input h_size: size of the biLSTM hidden layer
        
        """
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.h_size = h_size
        
        with tf.variable_scope("BiLSTM_FG"):

            self.input = tf.placeholder(tf.string,name="input", shape=[None,None])
            self.labels = tf.placeholder(tf.int32,name = "labels",shape=[None,None] )
            # length of the real sequences without padding
            self.nopad_len = tf.placeholder(tf.int32, name = "sequence_len",shape=[None])
            
            # elmo pre-trained module for sense embeddings.
            self.sense_embedding = elmo(
                    inputs={
                            "tokens": self.input,
                            "sequence_len": self.nopad_len
                            },
                    signature="tokens",
                    as_dict=True)["elmo"]
            self.input_mask = tf.sequence_mask(self.nopad_len)            
            # Bidirectional take the input lstm and create a copy that goes in the
            # opposite direction.
            self.LSTM = layers.LSTM(self.h_size,return_sequences=True)
            #self.bwd_lstm = layers.LSTM(self.h_size,return_sequences=True,go_backwards=True)
            self.biLSTM = layers.Bidirectional(self.LSTM, merge_mode='concat')\
                                (self.sense_embedding, mask=self.input_mask)
            self.Dense = layers.Dense(output_size)(self.biLSTM)
        
        #fine-grained loss
        with tf.variable_scope("fine_grained_loss"):
            # sparse_softmax labels must have the shape [batch_size], while
            # softmax labels must have shapes [batch_size,num_classes] (one_hot)
            loss_fg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.Dense,\
                                                                          labels = self.labels)
            # use the input mask to exclude padding tokens in the optimization phase.
            loss_fg = tf.boolean_mask(loss_fg,self.input_mask)
            self.loss_fg = tf.reduce_mean(loss_fg)
            
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            
            self.train = self.optimizer.minimize(self.loss_fg)       
        
        
       
    def load_sentences_train(self,batch_size,xml_path,word_vocab_path,wordid_to_wnsynset_path,\
                             skipped_vocab_path, output_vocab_path, to_shuffle = False):
        """
        return the batch of sentences needed for training
        :input batch_size: number of sentence to retrieve
        :input xml_path: path of the train file .xml in the Raganato's format
        :input word_vocab_path: path to the file created by the vocab_from_xml function
        :input wordid_to_wnsysset_path: path to the file created by the wordid_to_wnid_map function 
        :input skipped_vocab_path: path to the file of the skipped terms
        :input output_vocab_path: path to the file created by the fine_out_vocab function 
        :input shuffle: if we want to shuffle the dataset or not 
        :return : a list of sentences,a list of labels , a list of all the synsets
        for every word and the length of every sentence
        """
        # sentence: list of input words
        # label: first wnid associated to this word or the word itself 
        # if it has no instance (<unk> or <skp> otherwise )
        # synset: list of wnid associated with that word
        # lenghts: actual len of the sequence 
        sentences = []
        labels = []
        synsets = []
        lengths = []
        actual_max_len = 0
        
        # We need to add the start and ending token to every sentence 
        # padding the shorter sentences
        corpora = vocab_parser.get_xmlsentences(xml_path)
        word_vocab,inv_word,skipped_vocab = vocab_parser.read_semcor_vocab(word_vocab_path,skipped_vocab_path)
        wordid_to_wnsynset = vocab_parser.read_wordid_to_wnid_map(wordid_to_wnsynset_path)
        output_vocab = vocab_parser.read_vocab(output_vocab_path)
        # if to_shuffle is true
        sen_keys = list(corpora.keys())
        if to_shuffle:
            shuffle(sen_keys)
        
        
        for sentence in sen_keys:
            # add start and end symbol to the sentence length
            lengths.append(int(len(corpora[sentence])/3) + 2 )
            if lengths[-1] > actual_max_len:
                actual_max_len = lengths[-1]
            #print(lengths[-1])
            sentences.append([])
            synsets.append([])
            labels.append([])
            for i in range(len(corpora[sentence])):
                if i % 3 != 0:
                    continue
                if corpora[sentence][i+2] == "NUM":
                    corpora[sentence][i+1] = "<NUM>"
                elif corpora[sentence][i+2] == ".":
                    corpora[sentence][i+1] = "<PUNCT>"
                # if we don't have an instance for that word and the word is in the skipped 
                # vocab we just substitute it with the <SKP> token
                if corpora[sentence][i] is None and corpora[sentence][i+1] in skipped_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<SKP>"])
                    synsets[-1].append([output_vocab["<SKP>"]])
                # if word  has no instance and it is not in skipped but it's in word vocab then 
                # the label is just the word itself
                elif corpora[sentence][i] is None and corpora[sentence][i+1] in word_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab[corpora[sentence][i+1]])
                    synsets[-1].append([output_vocab[corpora[sentence][i+1]]])
                # if word  has no instance and it is not in skipped neither in word vocab then 
                # the label is just the unk token
                elif corpora[sentence][i] is None:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])
                  
                # if the word has an instance, it cannot be in the skipped vocab
                if corpora[sentence][i] is not None and corpora[sentence][i+1] in word_vocab:
                    actual_keys = wordid_to_wnsynset[corpora[sentence][i]]
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab[actual_keys[0]])
                    temp_synsets = utilities.wnid_from_lemma(corpora[sentence][i+1])
                    synsets[-1].append([output_vocab[x] if x in output_vocab else output_vocab["<UNK>"] for x in temp_synsets])
                    
                elif corpora[sentence][i] is not None:   
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])   
            
            # let's add start and end symbol 
            sentences[-1] = ["<S>"] + sentences[-1] + ["</S>"]
            labels[-1] = [output_vocab["<S>"]] + labels[-1] + [output_vocab["</S>"]]
            synsets[-1] = [[output_vocab["<S>"]]] + synsets[-1] + [[output_vocab["</S>"]]]
            
            # if the number of sentences is enough we create a batch
            if len(sentences) % batch_size == 0:
                # let's add padding
                for sent in range(len(sentences)):    
                    actual_len = len(sentences[sent])
                    sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                    labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                    synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
                yield sentences, labels, synsets, lengths
                
                sentences = []
                labels = []
                synsets = []
                lengths = []
                actual_max_len = 0
             
                
        if len(sentences) > 0 :
            remaining = len(sentences)
            for sent in range(remaining):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
            for sent in range(batch_size - remaining):
                lengths.append(0)
                sentences[-1].append(["<PAD>"] * actual_max_len)
                labels[-1].append([output_vocab["<PAD>"]] * actual_max_len)
                synsets[-1].append([[output_vocab["<PAD>"]]] * actual_max_len)                
            
            yield sentences, labels, synsets, lengths

    def accuracy(self,predictions,sen_lengths,labels, synsets):
        """
            compute the accuracy of network given its predictions 
            :input predictions: list of dimension [batch_size,sentence_len,output_len]
            :input labels: list of label, same dimension as predictions
            :input sen_lenghts: length of the single sentences, without padding
            :input synsets: list of possible synsets associated with each word
                [batch_size,sentence_len,synset_number]
            :return accuracy: number of words correctly classified / number of words
        """
        
        accuracy = 0.0
        total_words = 0
        
        for i in range(len(predictions)):
            actual_predictions = predictions[i]
            actual_labels = labels[i]
            actual_synsets = synsets[i]
            best_prediction = []
            
            # we ignore <S> and </S> symbols
            for j in range(1,sen_lengths[i]-1):
                word_out_distr = actual_predictions[j]
                word_synsets = actual_synsets[j]

                max_prob_synsets = word_out_distr[word_synsets]          
                max_prob_synset = int(np.argmax(max_prob_synsets))
                best_prediction = word_synsets[max_prob_synset]
                total_words = total_words + 1
                
                if best_prediction == actual_labels[j]:
                    accuracy = accuracy + 1
        
        # print("number of correct predictions = " + str(accuracy) + "\n")
        # print("over " + str(total_words) + " words\n")
        
        with tf.variable_scope("fine_grained_accuracy"):
            self.fg_accuracy = accuracy/total_words
            
        return self.fg_accuracy  
    
       
    
  
# =============================================================================
# MULTITASK_BILSTM
# =============================================================================   
class MultiTask_BiLSTM:
    """
    network with a simple bilstm layer used for pos tagging,fine and 
    coarse-grained classification
    """
    def __init__(self, output_size,pos_size,coarse_size, learning_rate, h_size,elmo):
        """ 
            :input output_size: size of the vocabulary containing the fine-grained words
            :input lr: learning rate 
            :input h_size: size of the biLSTM hidden layer
        
        """
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.h_size = h_size
        
        with tf.variable_scope("MultiTask_BiLSTM"):

            self.input = tf.placeholder(tf.string,name="input", shape=[None,None])
            self.labels = tf.placeholder(tf.int32,name = "labels",shape=[None,None] )
            self.pos_labels = tf.placeholder(tf.int32,name = "pos_labels", shape=[None,None])
            self.coarse_labels = tf.placeholder(tf.int32,name = "coarse_labels",shape=[None,None])
            # length of the real sequences without padding
            self.nopad_len = tf.placeholder(tf.int32, name = "sequence_len",shape=[None])
            
            # elmo pre-trained module for sense embeddings.
            self.sense_embedding = elmo(
                    inputs={
                            "tokens": self.input,
                            "sequence_len": self.nopad_len
                            },
                    signature="tokens",
                    as_dict=True)["elmo"]
            self.input_mask = tf.sequence_mask(self.nopad_len)            
            # Bidirectional take the input lstm and create a copy that goes in the
            # opposite direction.
            self.LSTM = layers.LSTM(self.h_size,return_sequences=True)
            #self.bwd_lstm = layers.LSTM(self.h_size,return_sequences=True,go_backwards=True)
            self.biLSTM = layers.Bidirectional(self.LSTM, merge_mode='concat')\
                                (self.sense_embedding, mask=self.input_mask)
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


    def load_sentences_train(self,batch_size,xml_path,word_vocab_path,\
                             bn_to_wnid_file, bn_to_cg, cg_vocab_file,pos_vocab_file, \
                             skipped_vocab_path, output_vocab_path,wordid_to_wnsynset_path = None, to_shuffle = False):
        """
        return the batch of sentences needed for training
        :input batch_size: number of sentence to retrieve
        :input xml_path: path of the train file .xml in the Raganato's format
        :input word_vocab_path: path to the file created by the vocab_from_xml function
        :input wordid_to_wnsysset_path: path to the file created by the wordid_to_wnid_map function 
        :input bn_to_wnid_file: file containing the map from babelnet to wordnet ids
        :input bn_to_cg_file: file .tsv containing the mapping from babelnet to a coarse_grained inventory 
        :input cg_vocab_file: file containing the coarse-grained vocab(can be lexnames or wndomain)
        :input pos_vocab_file: file containing the available pos
        :input skipped_vocab_path: path to the file of the skipped terms
        :input output_vocab_path: path to the file created by the fine_out_vocab function 
        :input shuffle: if we want to shuffle the dataset or not 
        :return : a list of sentences,a list of labels , a list of all the synsets
        for every word, the length of every sentence, a list of pos and the list
        of coarse-grained associated with the sentence
        """

        sentences = []
        labels = []
        synsets = []
        lengths = []
        coarse_grained = []
        pos = []
        actual_max_len = 0
        
        
        corpora = vocab_parser.get_xmlsentences(xml_path)
        word_vocab,inv_word,skipped_vocab = vocab_parser.read_semcor_vocab(word_vocab_path,skipped_vocab_path)
        if wordid_to_wnsynset_path is not None:
            wordid_to_wnsynset = vocab_parser.read_wordid_to_wnid_map(wordid_to_wnsynset_path)
        output_vocab = vocab_parser.read_vocab(output_vocab_path)
        # we only care about the reverse mapping (from wnid to babelnet)
        _ , wnid_to_bn = vocab_parser.read_tsv_file(bn_to_wnid_file)
        pos_vocab = vocab_parser.read_vocab(pos_vocab_file)
        coarse_grained_vocab = vocab_parser.read_vocab(cg_vocab_file)
        bn_to_cg, _ = vocab_parser.read_tsv_file(bn_to_cg)
        
        # if to_shuffle is true
        sen_keys = list(corpora.keys())
        if to_shuffle:
            shuffle(sen_keys)
        
        
        for sentence in sen_keys:
            # add start and end symbol to the sentence length
            lengths.append(int(len(corpora[sentence])/3) + 2 )
            if lengths[-1] > actual_max_len:
                actual_max_len = lengths[-1]
            #print(lengths[-1])
            sentences.append([])
            synsets.append([])
            labels.append([])
            pos.append([])
            coarse_grained.append([])
            
            for i in range(len(corpora[sentence])):
                if i % 3 != 0:
                    continue
                if corpora[sentence][i+2] == "NUM":
                    corpora[sentence][i+1] = "<NUM>"
                elif corpora[sentence][i+2] == ".":
                    corpora[sentence][i+1] = "<PUNCT>"
                    
                # POS
                if corpora[sentence][i+2] in pos_vocab:
                    pos[-1].append(pos_vocab[corpora[sentence][i+2]])
                else:
                    pos[-1].append(pos_vocab["<UNK>"])
                    
                    
                # if we don't have an instance for that word and the word is in the skipped 
                # vocab we just substitute it with the <SKP> token
                if corpora[sentence][i] is None and corpora[sentence][i+1] in skipped_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<SKP>"])
                    synsets[-1].append([output_vocab["<SKP>"]])
                    coarse_grained[-1].append(coarse_grained_vocab["<SKP>"])
                # if word  has no instance and it is not in skipped but it's in word vocab then 
                # the label is just the word itself
                elif corpora[sentence][i] is None and corpora[sentence][i+1] in word_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    if corpora[sentence][i+1] in output_vocab: 
                        labels[-1].append(output_vocab[corpora[sentence][i+1]])
                        synsets[-1].append([output_vocab[corpora[sentence][i+1]]])                    
                    else:
                        labels[-1].append(output_vocab["<UNK>"])
                        synsets[-1].append([output_vocab["<UNK>"]])                        
                    coarse_grained[-1].append(coarse_grained_vocab["<SKP>"])
                # if word  has no instance and it is not in skipped neither in word vocab then 
                # the label is just the unk token
                elif corpora[sentence][i] is None:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])
                    coarse_grained[-1].append(coarse_grained_vocab["<UNK>"])
                    
                # if the word has an instance, it cannot be in the skipped vocab
                if corpora[sentence][i] is not None and corpora[sentence][i+1] in word_vocab:
                    if wordid_to_wnsynset_path is not None:
                        actual_keys = wordid_to_wnsynset[corpora[sentence][i]]
                    else:
                        actual_keys = utilities.wnid_from_lemmapos(corpora[sentence][i+1],corpora[sentence][i+2])[0]
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab[actual_keys[0]])
                    temp_synsets = utilities.wnid_from_lemma(corpora[sentence][i+1])
                    synsets[-1].append([output_vocab[x] if x in output_vocab else output_vocab["<UNK>"] for x in temp_synsets])
                    # Add the coarse_grained
                    
                    if actual_keys[0] in wnid_to_bn:
                        wn_coarse = wnid_to_bn[actual_keys[0]][0]
                    else:
                        wn_coarse = None
                    if wn_coarse is not None:   
                        if wn_coarse in bn_to_cg:
                            coarse_grained[-1].append(coarse_grained_vocab[bn_to_cg[wn_coarse][0]])
                        else:
                            coarse_grained[-1].append(coarse_grained_vocab["factotum"])
                    else:
                        coarse_grained[-1].append(coarse_grained_vocab["<UNK>"])                        
                    
                    
                elif corpora[sentence][i] is not None:   
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])   
                    coarse_grained[-1].append([coarse_grained_vocab["<UNK>"]])
                    
            
            # let's add start and end symbol 
            sentences[-1] = ["<S>"] + sentences[-1] + ["</S>"]
            labels[-1] = [output_vocab["<S>"]] + labels[-1] + [output_vocab["</S>"]]
            synsets[-1] = [[output_vocab["<S>"]]] + synsets[-1] + [[output_vocab["</S>"]]]
            pos[-1] = [pos_vocab["<S>"]] + pos[-1] + [pos_vocab["</S>"]]
            coarse_grained[-1] =  [coarse_grained_vocab["<S>"]] + coarse_grained[-1] + [coarse_grained_vocab["</S>"]]
            
            # if the number of sentences is enough we create a batch
            if len(sentences) % batch_size == 0:
                # let's add padding
                for sent in range(len(sentences)):    
                    actual_len = len(sentences[sent])
                    sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                    labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                    synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
                    coarse_grained[sent] = coarse_grained[sent] + [coarse_grained_vocab["<PAD>"]]*(actual_max_len - actual_len)
                    pos[sent] = pos[sent] + [pos_vocab["<PAD>"]]*(actual_max_len - actual_len)
                yield sentences, labels, synsets, lengths, pos, coarse_grained
                
                sentences = []
                labels = []
                synsets = []
                lengths = []
                pos = []
                coarse_grained = []
                actual_max_len = 0
                
        

        if len(sentences) > 0 :
            remaining = len(sentences)
            for sent in range(remaining):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
                coarse_grained[sent] = coarse_grained[sent] + [coarse_grained_vocab["<PAD>"]]*(actual_max_len - actual_len)
                pos[sent] = pos[sent] + [pos_vocab["<PAD>"]]*(actual_max_len - actual_len)
#            for sent in range(batch_size - remaining):
#                lengths.append(0)
#                sentences.append([])
#                sentences[-1].append(["<PAD>"] * actual_max_len)
#                labels.append([])
#                labels[-1].append([output_vocab["<PAD>"]] * actual_max_len)
#                synsets.append([])
#                synsets[-1].append([[output_vocab["<PAD>"]]] * actual_max_len)
#                coarse_grained.append([])
#                coarse_grained[-1].append([coarse_grained_vocab["<PAD>"]]*actual_max_len)
#                pos.append([])
#                pos[-1].append([pos_vocab["<PAD>"]]*actual_max_len)
                
            yield sentences, labels, synsets, lengths, pos, coarse_grained
           

    def load_sentences_val(self,batch_size,xml_path):
        """
        return the batch of sentences needed for training
        :input batch_size: number of sentence to retrieve
        :input xml_path: path of the train file .xml in the Raganato's format
       
        :return : a list of sentences, the length of every sentence
        a list of words id and the pos
        """

        sentences = []
        lengths = []
        words_id = []
        pos_list = []
        actual_max_len = 0
        
        
        corpora = vocab_parser.get_xmlsentences(xml_path)
        sen_keys = list(corpora.keys())
        
        
        for sentence in sen_keys:
            # add start and end symbol to the sentence length
            if len(corpora[sentence]) == 0:
                continue
            lengths.append(int(len(corpora[sentence])/3) + 2 )
            if lengths[-1] > actual_max_len:
                actual_max_len = lengths[-1]
            sentences.append([])
            words_id.append([])
            pos_list.append([])
            for i in range(len(corpora[sentence])):
                if i % 3 != 0:
                    continue
                pos_list[-1].append(corpora[sentence][i+2])
                if corpora[sentence][i+2] == "NUM":
                    corpora[sentence][i+1] = "<NUM>"
                elif corpora[sentence][i+2] == ".":
                    corpora[sentence][i+1] = "<PUNCT>"
                
                if corpora[sentence][i] == None:
                    words_id[-1].append(None)
                else:
                    words_id[-1].append(corpora[sentence][i])
                    
                sentences[-1].append(corpora[sentence][i+1])                         
                # let's add start and end symbol 
            sentences[-1] = ["<S>"] + sentences[-1] + ["</S>"]            
            words_id[-1] = [None] + words_id[-1] + [None]
            pos_list[-1] = [None] + pos_list[-1] + [None]
            if len(sentences) % batch_size == 0:
                # let's add padding
                for sent in range(len(sentences)):    
                    actual_len = len(sentences[sent])
                    sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                    words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
                    pos_list[sent] = pos_list[sent] + [None] * (actual_max_len - actual_len)
                yield sentences, lengths, words_id,pos_list
                
                sentences = []
                lengths = []
                words_id = []
                pos_list = []
                actual_max_len = 0
                
        if len(sentences) > 0 :
            remaining = len(sentences)
            for sent in range(remaining):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
                pos_list[sent] = pos_list[sent] + [None] * (actual_max_len - actual_len)
#            for sent in range(batch_size - remaining):
#                lengths.append(0)
#                sentences.append([])
#                sentences[-1].append(["<PAD>"] * actual_max_len)
#                words_id.append([])
#                words_id[-1].append([None] * actual_max_len)                
            yield sentences,lengths, words_id,pos_list
 


                    
    def accuracy(self,predictions,sen_lengths,labels, synsets):
        """
            compute the accuracy of network given its predictions 
            :input predictions: list of dimension [batch_size,sentence_len,output_len]
            :input labels: list of label, same dimension as predictions
            :input sen_lenghts: length of the single sentences, without padding
            :input synsets: list of possible synsets associated with each word
                [batch_size,sentence_len,synset_number]
            :return accuracy: number of words correctly classified / number of words
        """
        
        accuracy = 0.0
        total_words = 0
        
        for i in range(len(predictions)):
            actual_predictions = predictions[i]
            actual_labels = labels[i]
            actual_synsets = synsets[i]
            best_prediction = []
            
            # we ignore <S> and </S> symbols
            for j in range(1,sen_lengths[i]-3):
                
                word_out_distr = actual_predictions[j]
                word_synsets = actual_synsets[j]

                max_prob_synsets = word_out_distr[word_synsets]          
                max_prob_synset = int(np.argmax(max_prob_synsets))
                best_prediction = word_synsets[max_prob_synset]
                total_words = total_words + 1
                
                if best_prediction == actual_labels[j]:
                    accuracy = accuracy + 1
        
        with tf.variable_scope("fine_grained_accuracy"):
            self.fg_accuracy = accuracy/total_words
            
        return self.fg_accuracy     
    
# =============================================================================
# MULTILING_BILSTM
# =============================================================================   
class MultiLing_BiLSTM:
    """
    network for multilingual wsd with a simple bilstm layer used for pos tagging,fine and 
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

            self.elmo_emb = tf.placeholder(tf.float32,name="elmo_emb",shape=[None,None,1024])
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
      


    def load_sentences_train(self,batch_size,xml_path,word_vocab_path,\
                             bn_to_wnid_file, bn_to_cg, cg_vocab_file,pos_vocab_file, \
                             skipped_vocab_path, output_vocab_path,wordid_to_wnsynset_path = None, to_shuffle = False):
        """
        return the batch of sentences needed for training
        :input batch_size: number of sentence to retrieve
        :input xml_path: path of the train file .xml in the Raganato's format
        :input word_vocab_path: path to the file created by the vocab_from_xml function
        :input wordid_to_wnsysset_path: path to the file created by the wordid_to_wnid_map function 
        :input bn_to_wnid_file: file containing the map from babelnet to wordnet ids
        :input bn_to_cg_file: file .tsv containing the mapping from babelnet to a coarse_grained inventory 
        :input cg_vocab_file: file containing the coarse-grained vocab(can be lexnames or wndomain)
        :input pos_vocab_file: file containing the available pos
        :input skipped_vocab_path: path to the file of the skipped terms
        :input output_vocab_path: path to the file created by the fine_out_vocab function 
        :input shuffle: if we want to shuffle the dataset or not 
        :return : a list of sentences,a list of labels , a list of all the synsets
        for every word, the length of every sentence, a list of pos and the list
        of coarse-grained associated with the sentence
        """

        sentences = []
        labels = []
        synsets = []
        lengths = []
        coarse_grained = []
        pos = []
        actual_max_len = 0
        
        
        corpora = vocab_parser.get_xmlsentences(xml_path)
        word_vocab,inv_word,skipped_vocab = vocab_parser.read_semcor_vocab(word_vocab_path,skipped_vocab_path)
        if wordid_to_wnsynset_path is not None:
            wordid_to_wnsynset = vocab_parser.read_wordid_to_wnid_map(wordid_to_wnsynset_path)
        output_vocab = vocab_parser.read_vocab(output_vocab_path)
        # we only care about the reverse mapping (from wnid to babelnet)
        _ , wnid_to_bn = vocab_parser.read_tsv_file(bn_to_wnid_file)
        pos_vocab = vocab_parser.read_vocab(pos_vocab_file)
        coarse_grained_vocab = vocab_parser.read_vocab(cg_vocab_file)
        bn_to_cg, _ = vocab_parser.read_tsv_file(bn_to_cg)
        
        # if to_shuffle is true
        sen_keys = list(corpora.keys())
        if to_shuffle:
            shuffle(sen_keys)
        
        
        for sentence in sen_keys:
            # add start and end symbol to the sentence length
            lengths.append(int(len(corpora[sentence])/3) + 2 )
            if lengths[-1] > actual_max_len:
                actual_max_len = lengths[-1]
            #print(lengths[-1])
            sentences.append([])
            synsets.append([])
            labels.append([])
            pos.append([])
            coarse_grained.append([])
            
            for i in range(len(corpora[sentence])):
                if i % 3 != 0:
                    continue
                if corpora[sentence][i+2] == "NUM":
                    corpora[sentence][i+1] = "<NUM>"
                elif corpora[sentence][i+2] == ".":
                    corpora[sentence][i+1] = "<PUNCT>"
                    
                # POS
                if corpora[sentence][i+2] in pos_vocab:
                    pos[-1].append(pos_vocab[corpora[sentence][i+2]])
                else:
                    pos[-1].append(pos_vocab["<UNK>"])
                    
                    
                # if we don't have an instance for that word and the word is in the skipped 
                # vocab we just substitute it with the <SKP> token
                if corpora[sentence][i] is None and corpora[sentence][i+1] in skipped_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<SKP>"])
                    synsets[-1].append([output_vocab["<SKP>"]])
                    coarse_grained[-1].append(coarse_grained_vocab["<SKP>"])
                # if word  has no instance and it is not in skipped but it's in word vocab then 
                # the label is just the word itself
                elif corpora[sentence][i] is None and corpora[sentence][i+1] in word_vocab:
                    sentences[-1].append(corpora[sentence][i+1])
                    if corpora[sentence][i+1] in output_vocab: 
                        labels[-1].append(output_vocab[corpora[sentence][i+1]])
                        synsets[-1].append([output_vocab[corpora[sentence][i+1]]])                    
                    else:
                        labels[-1].append(output_vocab["<UNK>"])
                        synsets[-1].append([output_vocab["<UNK>"]])                        
                    coarse_grained[-1].append(coarse_grained_vocab["<SKP>"])
                # if word  has no instance and it is not in skipped neither in word vocab then 
                # the label is just the unk token
                elif corpora[sentence][i] is None:
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])
                    coarse_grained[-1].append(coarse_grained_vocab["<UNK>"])
                    
                # if the word has an instance, it cannot be in the skipped vocab
                if corpora[sentence][i] is not None and corpora[sentence][i+1] in word_vocab:
                    if wordid_to_wnsynset_path is not None:
                        actual_keys = wordid_to_wnsynset[corpora[sentence][i]]
                    else:
                        actual_keys = utilities.wnid_from_lemmapos(corpora[sentence][i+1],corpora[sentence][i+2])[0]
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab[actual_keys[0]])
                    temp_synsets = utilities.wnid_from_lemma(corpora[sentence][i+1])
                    synsets[-1].append([output_vocab[x] if x in output_vocab else output_vocab["<UNK>"] for x in temp_synsets])
                    # Add the coarse_grained
                    
                    if actual_keys[0] in wnid_to_bn:
                        wn_coarse = wnid_to_bn[actual_keys[0]][0]
                    else:
                        wn_coarse = None
                    if wn_coarse is not None:   
                        if wn_coarse in bn_to_cg:
                            coarse_grained[-1].append(coarse_grained_vocab[bn_to_cg[wn_coarse][0]])
                        else:
                            coarse_grained[-1].append(coarse_grained_vocab["factotum"])
                    else:
                        coarse_grained[-1].append(coarse_grained_vocab["<UNK>"])                        
                    
                    
                elif corpora[sentence][i] is not None:   
                    sentences[-1].append(corpora[sentence][i+1])
                    labels[-1].append(output_vocab["<UNK>"])
                    synsets[-1].append([output_vocab["<UNK>"]])   
                    coarse_grained[-1].append([coarse_grained_vocab["<UNK>"]])
                    
            
            # let's add start and end symbol 
            sentences[-1] = ["<S>"] + sentences[-1] + ["</S>"]
            labels[-1] = [output_vocab["<S>"]] + labels[-1] + [output_vocab["</S>"]]
            synsets[-1] = [[output_vocab["<S>"]]] + synsets[-1] + [[output_vocab["</S>"]]]
            pos[-1] = [pos_vocab["<S>"]] + pos[-1] + [pos_vocab["</S>"]]
            coarse_grained[-1] =  [coarse_grained_vocab["<S>"]] + coarse_grained[-1] + [coarse_grained_vocab["</S>"]]
            
            # if the number of sentences is enough we create a batch
            if len(sentences) % batch_size == 0:
                # let's add padding
                for sent in range(len(sentences)):    
                    actual_len = len(sentences[sent])
                    sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                    labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                    synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
                    coarse_grained[sent] = coarse_grained[sent] + [coarse_grained_vocab["<PAD>"]]*(actual_max_len - actual_len)
                    pos[sent] = pos[sent] + [pos_vocab["<PAD>"]]*(actual_max_len - actual_len)
                yield sentences, labels, synsets, lengths, pos, coarse_grained
                
                sentences = []
                labels = []
                synsets = []
                lengths = []
                pos = []
                coarse_grained = []
                actual_max_len = 0
                
        

        if len(sentences) > 0 :
            remaining = len(sentences)
            for sent in range(remaining):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                labels[sent] = labels[sent] + [output_vocab["<PAD>"]] * (actual_max_len - actual_len)
                synsets[sent] = synsets[sent] + [[output_vocab["<PAD>"]]] * (actual_max_len - actual_len)
                coarse_grained[sent] = coarse_grained[sent] + [coarse_grained_vocab["<PAD>"]]*(actual_max_len - actual_len)
                pos[sent] = pos[sent] + [pos_vocab["<PAD>"]]*(actual_max_len - actual_len)
#            for sent in range(batch_size - remaining):
#                lengths.append(0)
#                sentences.append([])
#                sentences[-1].append(["<PAD>"] * actual_max_len)
#                labels.append([])
#                labels[-1].append([output_vocab["<PAD>"]] * actual_max_len)
#                synsets.append([])
#                synsets[-1].append([[output_vocab["<PAD>"]]] * actual_max_len)
#                coarse_grained.append([])
#                coarse_grained[-1].append([coarse_grained_vocab["<PAD>"]]*actual_max_len)
#                pos.append([])
#                pos[-1].append([pos_vocab["<PAD>"]]*actual_max_len)
                
            yield sentences, labels, synsets, lengths, pos, coarse_grained
           

    def load_sentences_val(self,batch_size,xml_path):
        """
        return the batch of sentences needed for training
        :input batch_size: number of sentence to retrieve
        :input xml_path: path of the train file .xml in the Raganato's format
       
        :return : a list of sentences, the length of every sentence
        a list of words id and the pos
        """

        sentences = []
        lengths = []
        words_id = []
        pos_list = []
        actual_max_len = 0
        
        
        corpora = vocab_parser.get_xmlsentences(xml_path)
        sen_keys = list(corpora.keys())
        
        
        for sentence in sen_keys:
            # add start and end symbol to the sentence length
            if len(corpora[sentence]) == 0:
                continue
            lengths.append(int(len(corpora[sentence])/3) + 2 )
            if lengths[-1] > actual_max_len:
                actual_max_len = lengths[-1]
            sentences.append([])
            words_id.append([])
            pos_list.append([])
            for i in range(len(corpora[sentence])):
                if i % 3 != 0:
                    continue
                pos_list[-1].append(corpora[sentence][i+2])
                if corpora[sentence][i+2] == "NUM":
                    corpora[sentence][i+1] = "<NUM>"
                elif corpora[sentence][i+2] == ".":
                    corpora[sentence][i+1] = "<PUNCT>"
                
                if corpora[sentence][i] == None:
                    words_id[-1].append(None)
                else:
                    words_id[-1].append(corpora[sentence][i])
                    
                sentences[-1].append(corpora[sentence][i+1])                         
                # let's add start and end symbol 
            sentences[-1] = ["<S>"] + sentences[-1] + ["</S>"]            
            words_id[-1] = [None] + words_id[-1] + [None]
            pos_list[-1] = [None] + pos_list[-1] + [None]
            if len(sentences) % batch_size == 0:
                # let's add padding
                for sent in range(len(sentences)):    
                    actual_len = len(sentences[sent])
                    sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                    words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
                    pos_list[sent] = pos_list[sent] + [None] * (actual_max_len - actual_len)
                yield sentences, lengths, words_id,pos_list
                
                sentences = []
                lengths = []
                words_id = []
                pos_list = []
                actual_max_len = 0
                
        if len(sentences) > 0 :
            remaining = len(sentences)
            for sent in range(remaining):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
                pos_list[sent] = pos_list[sent] + [None] * (actual_max_len - actual_len)
#            for sent in range(batch_size - remaining):
#                lengths.append(0)
#                sentences.append([])
#                sentences[-1].append(["<PAD>"] * actual_max_len)
#                words_id.append([])
#                words_id[-1].append([None] * actual_max_len)                
            yield sentences,lengths, words_id,pos_list
 


                    
    def accuracy(self,predictions,sen_lengths,labels, synsets):
        """
            compute the accuracy of network given its predictions 
            :input predictions: list of dimension [batch_size,sentence_len,output_len]
            :input labels: list of label, same dimension as predictions
            :input sen_lenghts: length of the single sentences, without padding
            :input synsets: list of possible synsets associated with each word
                [batch_size,sentence_len,synset_number]
            :return accuracy: number of words correctly classified / number of words
        """
        
        accuracy = 0.0
        total_words = 0
        
        for i in range(len(predictions)):
            actual_predictions = predictions[i]
            actual_labels = labels[i]
            actual_synsets = synsets[i]
            best_prediction = []
            
            # we ignore <S> and </S> symbols
            for j in range(1,sen_lengths[i]-3):
                
                word_out_distr = actual_predictions[j]
                word_synsets = actual_synsets[j]

                max_prob_synsets = word_out_distr[word_synsets]          
                max_prob_synset = int(np.argmax(max_prob_synsets))
                best_prediction = word_synsets[max_prob_synset]
                total_words = total_words + 1
                
                if best_prediction == actual_labels[j]:
                    accuracy = accuracy + 1
        
        with tf.variable_scope("fine_grained_accuracy"):
            self.fg_accuracy = accuracy/total_words
            
        return self.fg_accuracy     
    
    
    
if __name__ == "__main__":
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
# =============================================================================
#     create vocabularies
# =============================================================================
#    vocab_from_xml(xml_train_file,word_vocab_file,skipped_file)
#    wordid_to_wnid_map(gold_train_file,wordid_to_wnsynset_file)
#    vocab_from_gold(gold_train_file,wnid_vocab_file)
#    fine_out_vocab(wnid_vocab_file,xml_train_file,skipped_file,fine_out_vocab_file)
#    pos_out_vocab(xml_train_file,pos_out_vocab_file)

#    bn_to_wn,wn_to_bn = vocab_parser.read_tsv_file(bn_to_wn_file)
#    bn_to_wnd, _ = vocab_parser.read_tsv_file(bn_to_wnd_file)
#    bn_to_lex, _ = vocab_parser.read_tsv_file(bn_to_lex_file)   
    
    
# =============================================================================
#   test the network
# =============================================================================
    LEARNING_RATE = 0.01 
    HIDDEN_SIZE = 32
    BATCH_SIZE = 20 
    output_vocab = vocab_parser.read_vocab(fine_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    wnd_vocab = vocab_parser.read_vocab(wnd_out_vocab_file)
    lex_vocab = vocab_parser.read_vocab(lex_out_vocab_file)
    pos_dim = len(pos_vocab)
    output_dim = len(output_vocab)
    coarse_dim = len(wnd_vocab)
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True) 
    My_model = MultiTask_BiLSTM (output_dim,pos_dim,coarse_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    i = 0
    for sentences, labels, synsets, lengths, pos, coarse_grained in \
        My_model.load_sentences_train(BATCH_SIZE,xml_train_file, word_vocab_file,wordid_to_wnsynset_file,\
                            bn_to_wn_file,bn_to_wnd_file,wnd_out_vocab_file,pos_out_vocab_file,\
                            skipped_file,fine_out_vocab_file):
        if i == 0:
            i = 1
        else:
            break             
     
        

        

       
        
    
