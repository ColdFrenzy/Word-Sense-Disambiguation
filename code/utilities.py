# -*- coding: utf-8 -*-
"""
@author: Francesco
"""
import Path
from nltk.corpus import wordnet as wn 
import vocab_parser





def wnid_from_synset(synset):
    offset = synset.offset()
    wn_id = "wn:" + str(offset).zfill( 8) + synset.pos()
    return wn_id

def wnid_from_sensekey(sensekey):
    # offset should be a sequence of 8 numbers that represent bytes
    # Since the offset function return just an int, we need to pad 
    # with zeroes on the left
    synset = wn.lemma_from_key(sensekey).synset()
    offset = synset.offset()
    wn_id = "wn:" + str(offset).zfill( 8) + synset.pos()
    return wn_id

def wnid_from_lemma(lemma):
    """
        retrieve the wnid given the lemma
        :input lemma: wordn lemma
        :return wn_id: list of wordnet id or return the word if no wnid was found
    """
        
    synsets = wn.synsets(lemma)
    # if no synset found, just return the lemma
    if len(synsets) == 0:
        return [lemma]
    return [wnid_from_synset(syn) for syn in synsets]           
            

def wnid_from_lemmapos(lemma,pos):
    """
        Retrieves the wnid given the lemma and pos
        :input lemma: word lemma
        :input pos: word pos
        :return wn_id: list of wordnet id or return the word if no wnid was found
    """

    pos_dict = {"ADJ": "a", "ADV": "r", "NOUN": "n", "VERB": "v"}
    if pos == "." or pos == "PUNCT":
        return ["<PUNCT>"]
    elif pos == "NUM":
        return ["<NUM>"]
    
    elif pos in pos_dict:
        synsets = wn.synsets(lemma, pos=pos_dict[pos])
    else:
        synsets = wn.synsets(lemma)

    if len(synsets) == 0:
        return [lemma]
    return [wnid_from_synset(syn) for syn in synsets]
           

def get_wnpos(pos):
    pos_dict = {"ADJ": "a", "ADV": "r", "NOUN": "n", "VERB": "v"}
    if pos == "." or pos == "PUNCT":
        return ["<PUNCT>"]
    elif pos == "NUM":
        return ["<NUM>"]
    
    elif pos in pos_dict:
        return pos_dict[pos]  

 
def load_sentences(batch_size,xml_path):
    """
    return the batch of sentences needed for training
    :input batch_size: number of sentence to retrieve
    :input xml_path: path of the train file .xml in the Raganato's format
   
    :return : a list of sentences, the length of every sentence
    :return : a list of words id
    """

    sentences = []
    lengths = []
    words_id = []
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
        
        for i in range(len(corpora[sentence])):
            if i % 3 != 0:
                continue
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
        if len(sentences) % batch_size == 0:
            # let's add padding
            for sent in range(len(sentences)):    
                actual_len = len(sentences[sent])
                sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
                words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
            yield sentences, lengths, words_id
            
            sentences = []
            lengths = []
            words_id = []
            actual_max_len = 0
            
    if len(sentences) > 0 :
        remaining = len(sentences)
        for sent in range(remaining):    
            actual_len = len(sentences[sent])
            sentences[sent] = sentences[sent] + ["<PAD>"] * (actual_max_len - actual_len)
            words_id[sent] = words_id[sent] + [None] * (actual_max_len - actual_len)
#            for sent in range(batch_size - remaining):
#                lengths.append(0)
#                sentences.append([])
#                sentences[-1].append(["<PAD>"] * actual_max_len)
#                words_id.append([])
#                words_id[-1].append([None] * actual_max_len)                
        yield sentences,lengths, words_id        
            
if __name__ == "__main__":
    Path.Paths()
    xml_train_file = Path.Paths.xml_train_file
    word_vocab_file = Path.Paths.word_vocab_file
    wnid_vocab_file = Path.Paths.wnid_vocab_file
    skipped_file = Path.Paths.skipped_file
    gold_train_file = Path.Paths.gold_train_file
    wordid_to_wnsynset_file = Path.Paths.wordid_to_wnsynset_vocab_file
    fine_out_vocab_file = Path.Paths.fine_out_vocab_file
    batch_size = 5
#    i = 0
#    for sentences,labels, synsets,lengths in \
#        load_sentences_train(batch_size,xml_train_file, word_vocab_file,\
#                             wordid_to_wnsynset_file,skipped_file,fine_out_vocab_file,True):
#            if i == 0:
#                i = 1
#            else:
#                break
