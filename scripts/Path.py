# -*- coding: utf-8 -*-
"""

@author: Francesco
"""

import os
from os.path import join 
from os import curdir

class Paths:
    DATA_DIR = join(curdir,"../resources/Dataset")
    TRAIN_DIR = join(DATA_DIR,"Train")
    VOCAB_DIR = join(DATA_DIR,"Vocabs")
    EVAL_DIR = join(DATA_DIR,"Eval")
    MODEL_DIR = join(DATA_DIR,"Model")
    MULTILING_DIR = join(DATA_DIR,"Multilingual")
    CHECKPOINT_DIR = join(MODEL_DIR,"Checkpoint")
    SUMMARIES_DIR = join(MODEL_DIR,"Summaries")
    IT_MODEL_PATH = join(MULTILING_DIR,"it")
    DE_MODEL_PATH = join(MULTILING_DIR,"de")
    ES_MODEL_PATH = join(MULTILING_DIR,"es")
    FR_MODEL_PATH = join(MULTILING_DIR,"fr")
    xml_train_file = join(TRAIN_DIR,"semcor.data.xml")
    gold_train_file = join(TRAIN_DIR, "semcor.gold.key.txt")
    word_vocab_file = join(VOCAB_DIR,"semcor_vocab.txt")
    wnid_vocab_file = join(VOCAB_DIR,"wnid_vocab.txt")    
    wordid_to_wnsynset_vocab_file = join(VOCAB_DIR,"wordid_to_wnsynset.txt")
    multilingual_word2bn_file= join(MULTILING_DIR,"lemma2synsets4.0.xx.wn.ALL.txt")
    multilingual_word2bn_vocab_file = join(VOCAB_DIR,"word2babelnet")
    bn_to_wn_file = join(DATA_DIR,"babelnet2wordnet.tsv")
    bn_to_wnd_file = join(DATA_DIR,"babelnet2wndomains.tsv") 
    bn_to_lex_file = join(DATA_DIR,"babelnet2lexnames.tsv")
    fine_out_vocab_file = join(VOCAB_DIR,"fine_out_vocab.txt")
    pos_out_vocab_file = join(VOCAB_DIR,"pos_out_vocab.txt")
    wnd_out_vocab_file = join(VOCAB_DIR,"wnd_out_vocab.txt")
    lex_out_vocab_file = join(VOCAB_DIR,"lex_out_vocab.txt")
    skipped_file = join(VOCAB_DIR, "semcor_excluded.txt")
    epochs_file = join(MODEL_DIR,"epochs.txt") #keep track of the training time
    def __init__(self):
        self.initialize_dirs()
        
    def initialize_dirs(self):
        """
            create the directories listed above if they do not exists
        """
        dirs = [attr for attr in dir(self) if not callable(getattr(self, attr))\
                and not attr.startswith("__") and attr.isupper() ]
        for dir_ in dirs:
            path = getattr(self, dir_)
            if os.path.exists(path):
                print(dir_ + " already exists\n")
                continue
            else:
                os.mkdir(path)
                print(dir_ + " has been created\n")
            
