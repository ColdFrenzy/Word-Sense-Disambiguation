# -*- coding: utf-8 -*-
"""

@author: Francesco
"""


from lxml import etree
import math 
from random import uniform
import Path
import utilities


def get_xmlsentences(xml_file):
    """ get sentence from the xml file
        :input xml_file: file in the Raganato's format
        :return a vocab of lists {sentence_id: [word1_id,lemma1,pos1,word2_id,lemma2,...]}
    """
    vocab_sentences = {}
    for event, sentence in etree.iterparse(xml_file,tag="sentence"):
        sent_id = sentence.get("id")
        vocab_sentences[sent_id] = []
        #vocab_sentences[sent_id].append([])
        
        if event == "end":
            for word in sentence:
                if word.tag == "instance":
                    vocab_sentences[sent_id].append(word.get("id"))
                    vocab_sentences[sent_id].append(word.get("lemma"))
                    vocab_sentences[sent_id].append(word.get("pos"))
                
                elif word.tag == "wf":
                    vocab_sentences[sent_id].append(None)
                    vocab_sentences[sent_id].append(word.get("lemma"))
                    vocab_sentences[sent_id].append(word.get("pos"))
  
    return vocab_sentences
                

def vocab_from_xml(in_path, out_path,skipped_path ,subsampling=1e-5,min_num=6):
    """
    Create a vocabulary in the out_path file from the xml file in the 
    Raganato's format. The file will have one 1 word every line
    :input in_path: path of the xml_file
    :input out_path: path of the vocabulary output in txt format
    :input skipped_path: path of the vocabulary containing the skipped words
    :input subsampling: subsampling rate
    :input min_num: minimun number of occurrence of a word in order to be added
        to the vocabulary
    """
    n_words = 0
    #vocab containing all the "instance" words
    inst_vocab = []
    occ_vocab  = {}
    #vocab after subsampling
    parsed_vocab = []
    # vocab of the skipped terms
    skipped_vocab = []
    vocab_sentences = get_xmlsentences(in_path)
 
    print("Initializing vocabulary...")
    
    for sentence in vocab_sentences:
        actual_sent = []                
        for elem_in_sent in range(len(vocab_sentences[sentence])):
            if elem_in_sent % 3 == 0:
                actual_sent.append([])
            actual_sent[-1].append(vocab_sentences[sentence][elem_in_sent])
            
        for word in range(len(actual_sent)):
            n_words += 1
            lemma = actual_sent[word][1]
            
            #group all punctuations, numbers and symbols in three classes 
            if actual_sent[word][2] == "NUM":
                lemma = "<NUM>"
            elif actual_sent[word][2] == ".":
                lemma = "<PUNCT>"
                
            if actual_sent[word][0] is not None:
        
                if lemma not in inst_vocab:
                    inst_vocab.append(lemma)
            occ_vocab[lemma] = occ_vocab.get(lemma,0) + 1 
        
    print("found " + str(n_words) + " words over "+ str(len(vocab_sentences))\
                         + " sentences")
    
    #apply subsampling
    for word in occ_vocab:
        #if we have less than min_num occurrences we skip the word
        if word in inst_vocab:
            if word not in parsed_vocab:
                parsed_vocab.append(word)
        elif occ_vocab[word] > min_num:
            word_prob = occ_vocab[word] / n_words
            skip_prob = 1.0 - math.sqrt(subsampling / word_prob)
            
            if uniform(0,1) >= skip_prob:
                if word not in parsed_vocab:
                    parsed_vocab.append(word)
            else:
                if word not in skipped_vocab:
                    skipped_vocab.append(word)

    print("the parsed vocabulary has " + str(len(parsed_vocab)) + " words\n")
    print("the skipped vocabulary has " + str(len(skipped_vocab)) + " words\n")
    out_f = open(out_path, "w")
    for elem in parsed_vocab:
        out_f.write(elem + "\n")
    out_f.close()
    skipped_f = open(skipped_path, "w")
    for elem in skipped_vocab:
        skipped_f.write(elem + "\n")
    skipped_f.close()       
    print("the word vocabulary has been written at " + out_path + "\n")
    print("the skipped vocabulary has been written at " + skipped_path + "\n")     
        
def vocab_from_gold(in_path,out_path):
    """
    create a vocabulary with all the wordnet_synset_ids in our gold.key.txt file.
    one wn_synset per line
    :input in_path: gold.key.txt path
    :input out_path: wordnet_synset_ids path
    """
    print("Creating wordnet_synset_id vocabulary\n")
    vocab = []
    with open(in_path,"r") as f_in:
        for line in f_in:
            line = line.strip()
            line = line.split(" ")
            if len(line) < 2:
                continue 
            
            for sen_key in line[1:]:
                synset_id = utilities.wnid_from_sensekey(sen_key)
                if synset_id not in vocab:
                    vocab.append(synset_id)
            
    with open(out_path,"w") as f:
        for elem in vocab:
            f.write(elem + '\n')

    print("vocabulary has been created at " + out_path + "\n")
                    
    
def wordid_to_wnid_map(in_path,out_path):          
    """
    create a map between words_id and their synsets from the wordnet, given 
    the gold.key.txt file. Output vocabulary is written as
    word_id *space* series of wn_synset_id separated by a comma.
    :input in_path: gold.key.txt path
    :input out_path: word_id to wordnet_synsets_id vocabulary path
    """
    
    word_to_wn_vocab = {}
    with open(in_path,"r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            # if a line has less than 2 element than we have no key
            if len(line) < 2:
                continue
            
            wordid = line[0]
            word_to_wn_vocab[wordid] = []
            for sen_key in line[1:]:
                synset_id = utilities.wnid_from_sensekey(sen_key)
                word_to_wn_vocab[wordid].append(synset_id)
                
    with open(out_path,"w") as f:
        for elem in word_to_wn_vocab:
            f.write(elem + " ")
            for i,subelem in enumerate(word_to_wn_vocab[elem]):
                if i == len(word_to_wn_vocab[elem]) - 1:
                    f.write(subelem + "\n")
                else:
                    f.write(subelem + ",")                
        
    print("word_id to wordnet_synsets_id vocabulary has been written at " +\
          str(out_path) + "\n")
      
def read_wordid_to_wnid_map(wordid_to_wnid_path):
    """
    read the file created by the wordid_to_wnid_map function
    :input wordid_to_wnid_path: path to the file created by the wordid_to_wnid_map function 
    :return wordid_to_wnid: a dictionary {wordid: [wnid1,...,wnidn]}
    """
    word_id_to_wnid = {}
    with open(wordid_to_wnid_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            word_id_to_wnid[line[0]] = line[1].split(",")
            
    return word_id_to_wnid


def read_map(in_path):
    """
    read generic map where first word is separated by tab and all others by comma
    :input inp_path: path to the file created by the wordid_to_wnid_map function 
    :return out_dict: a dictionary {wordid: [wnid1,...,wnidn]}
    """
    map_to = {}
    with open(in_path, "r",encoding="utf8") as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            map_to[line[0]] = line[1].split(",")[:-1]
            
    return map_to           
    
    
def read_semcor_vocab(vocab_path,skipped_path=None):
    """ 
    read the vocabulary generated by the semcor.xml datas
    :input vocab_path: path of the vocabulary generated by the vocab_from_xml function
    :input skipped_path: path of the skipped terms generated by the vocab_from_xml function
    :return word_to_id: dict of words to ids
    :return id_to_word: dict of ids to words
    :return skipped_vocab: list of skipped terms
    """
    # initialize our vocabulary with some token, <S> start and <\S> of a sequence token
    # <pad> for padding on shorter sequences, <skp> for terms in skipped_vocab and <unk>
    # for never seen token
    word_to_id = {"<S>" : 0, "</S>" : 1, "<PAD>" : 2, "<SKP>" : 3, "<UNK>" : 4}
    id_to_word = {0 : "<S>", 1 : "</S>", 2 : "<PAD>", 3 : "<SKP>", 4: "<UNK>"}
    skipped_vocab = []
    with open(vocab_path, "r") as vocab_file:
        for word in vocab_file:
            word = word.strip()
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                id_to_word[len(word_to_id)-1] = word
    
    if skipped_path is not None:
        with open(skipped_path, "r") as skipped_file:
            for word in skipped_file:
                word = word.strip()
                if word not in skipped_vocab:
                    skipped_vocab.append(word)
        return word_to_id, id_to_word, skipped_vocab
    else:
        return word_to_id, id_to_word
        
def read_vocab(vocab_path):
    """
    read a generic vocabulary
    :input vocab_path: path to the input vocabulary where every line contains 
    a word
    :return word_to_id: dict of words to ids
    """
    word_to_id = {"<S>" : 0, "</S>" : 1, "<PAD>" : 2, "<SKP>" : 3, "<UNK>" : 4}
    with open(vocab_path, "r") as vocab_file:
        for word in vocab_file:
            word = word.strip()
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)

    return word_to_id


def read_tsv_file(tsv_file):
    """
        read the mapping in the tsv files("wn_to_bn,bn_to_wnd,bn_to_lexname ecc..)
        separated by tabs
        :input tsv_file: file where elements are separated by tabs
        :input reverse: return the inverse mapping, works if we have a one to one map
        :return output_dict: str-> list of str
        :return inv_output_dict: str->list of str
    """
    
    output_dict = {}
    inv_output_dict = {}
    with open(tsv_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
    
            if len(line) < 2:
                continue
            
            elem = line[0]
            if elem not in output_dict:
                output_dict[elem] = []
                for subelem in line[1:]:
                    output_dict[elem].append(subelem)
            else:
                for subelem in line[1:]:
                    output_dict[elem].append(subelem)
            
            for x in line[1:]:
                
                if x not in inv_output_dict:
                    inv_output_dict[x] = [elem]
                else:
                    inv_output_dict[x].append(elem)
            
    return output_dict,inv_output_dict



def pos_out_vocab(xml_file, out_file):
    """
        create a Part Of Speech vocabulary with all the different POS available
        in the xml_file
        :input xml_file: xml file in the Raganato's format
        :input out_file: file where we are gonna write all the POS found
    """
    pos_vocab = []
    sentences = get_xmlsentences(xml_file)
    for sentence in sentences:
        for i in range(len(sentences[sentence])):
            if i%3 != 0:
                continue
            if sentences[sentence][i+2] not in pos_vocab:
                pos_vocab.append(sentences[sentence][i+2])
    
    with open(out_file,"w") as f:
        for elem in pos_vocab:
            f.write(str(elem) + "\n")

   
def fine_out_vocab(wn_file,xml_train_file,skipped_file,out_file):
    """
    create a vocabulary with fine-grained vocabulary with wn_ids and words
    :input wn_file: vocab file created by vocab_from_gold function
    :input xml_train_file: xml file in the Raganato's format
    :input skipped_file: file of the skipped words
    :input out_file: file where we will write the output vocab
    """
    print("creating vocabulary with fine grained terms\n")
    out_vocab = []  
    skipped = []
    with open(wn_file,"r") as f:
        for wn_id in f:
            wn_id = wn_id.strip()
            out_vocab.append(wn_id)
            
            
    with open(skipped_file,"r") as x:
        for word in x:
            word=word.strip()
            skipped.append(word)
            
            
    sentences = get_xmlsentences(xml_train_file)
    for sentence in sentences:

        for i in range(len(sentences[sentence])):
            if i % 3 != 0:
                continue
            lemma = sentences[sentence][i+1]
            #group all punctuations and numbers in two classes 
            if sentences[sentence][i+2] == "NUM":
                lemma = "<NUM>"
            elif sentences[sentence][i+2] == ".":
                lemma = "<PUNCT>"
            # if the word has no instance we must include it in the vocab (since 
            # we have no wn_id). The word is included only if it's not a skipped words
            if sentences[sentence][i] is None:
                if lemma not in skipped:
                    if lemma not in out_vocab:
                        out_vocab.append(lemma)
            else:
                pass
    
    first_written = False
    with open(out_file,"w") as o:
        for elem in out_vocab:
            if first_written:
                o.write("\n" + elem)
            else:
                first_written = True
                o.write(elem)
        
    print(out_file + " correctly created\n")

def coarse_grained_vocab(coarse_grained_file,out_file):
    """
        create a coarse_grained vocabulary (wordnetDomain or Lexname)
        :input coarse_grained_file: file .tsv
        :input out_file: file where we will write the output vocab
    """
    print("Creating coarse grained vocabulary ...\n")
    coarse_grained_vocab = []
    with open(coarse_grained_file,"r") as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
    
            if len(line) < 2:
                continue
            
            for elem in line[1:]:
                if elem not in coarse_grained_vocab:
                    coarse_grained_vocab.append(elem)
    
    first_written = False
    with open(out_file,"w") as out_f:
        for word in coarse_grained_vocab:
            if first_written:
                out_f.write("\n" + word)
            else:
                first_written = True
                out_f.write(word)
    
    print("Coarse grained vocabulary succesfully created\n")

   
def create_multiling_word2bn(inp_path,out_path,lang):
    """
        create a mapping between word and babelnet_id 
        :input inp_path: path to lemma2synsets4.0.xx.wn.ALL.txt file 
        :input out_path: path to the output vocab 
        :input lang: language ("IT","ES","DE","FR")
    """
    pos = []
    print("Creating " + lang + " word_to_bn vocabulary\n")
    word_to_bn = {}
    out_path = out_path + lang + ".txt"
    with open(inp_path,"r",encoding="utf8") as f_in:
        for line in f_in:
            line = line.strip()
            line = line.split("\t")
            actual_pos = "#".join(line[1].split("#")[-1])
            if line[0] != lang:
                continue
            else:
                word_to_bn[line[1]] = line[2:]
            if actual_pos not in pos:
                pos.append(actual_pos)
    
    # just for info 
    print( lang + " pos are: ")
    for p in pos:
        print(p + " " )
    
    first_written = False
    with open(out_path, "w",encoding="utf8") as f_out:
        for word in word_to_bn:
            if not first_written:
                f_out.write( word + " ")
                for subword in word_to_bn[word]:
                    f_out.write(subword + ",")
                first_written = True
            else:
                f_out.write("\n" + word + " ")
                for subword in word_to_bn[word]:
                    f_out.write(subword + ",")    
    print("word_to_bn vocabulary succesfully created\n")
                    
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
    multilingual_word2bn_file = Path.Paths.multilingual_word2bn_file
    multilingual_word2bn_vocab_file = Path.Paths.multilingual_word2bn_vocab_file
# =============================================================================
#     create vocabularies
# =============================================================================
#    vocab_from_xml(xml_train_file,word_vocab_file,skipped_file)
#    wordid_to_wnid_map(gold_train_file,wordid_to_wnsynset_file)
#    vocab_from_gold(gold_train_file,wnid_vocab_file)
#    fine_out_vocab(wnid_vocab_file,xml_train_file,skipped_file,fine_out_vocab_file)
#    pos_out_vocab(xml_train_file,pos_out_vocab_file)
#    coarse_grained_vocab(bn_to_wnd_file,wnd_out_vocab_file)
#    coarse_grained_vocab(bn_to_lex_file,lex_out_vocab_file)
#    fine_out_vocab(wnid_vocab_file,xml_train_file,skipped_file,fine_out_vocab_file)
#    bn_to_wn,wn_to_bn = read_tsv_file(bn_to_wn_file)
#    bn_to_wnd, _ = read_tsv_file(bn_to_wnd_file)
#    bn_to_lex, _ = read_tsv_file(bn_to_lex_file)
#    pos =  create_multiling_word2bn(multilingual_word2bn_file,multilingual_word2bn_vocab_file,"FR")
