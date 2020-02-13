import network
import vocab_parser
import Path
import tensorflow as tf
import tensorflow_hub as hub
import os 
import utilities
from tqdm import tqdm
import numpy as np


# =============================================================================
# PARAMETERS
# =============================================================================
BATCH_SIZE = 10
LEARNING_RATE = 0.01
HIDDEN_SIZE = 64

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Babelnet_domains labels prediction started\n")
    predictions = {}
# =============================================================================
#   VOCAB FILES
# =============================================================================
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
    wnd_out_vocab_file = os.path.join(resources_path,"Vocabs","wnd_out_vocab.txt")
    pos_out_vocab_file = os.path.join(resources_path,"Vocabs","pos_out_vocab.txt")
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
    word_vocab_file = os.path.join(resources_path,"Vocabs","semcor_vocab.txt")
#    wordid_to_wnsynset_file = os.path.join(resources_path,"Vocabs","wordid_to_wnsynset.txt")
#    bn_to_wnd_file = os.path.join(resources_path,"babelnet2wndomains.tsv")
    bn_to_wn_file = os.path.join(resources_path,"babelnet2wordnet.tsv")
#    skipped_file = os.path.join(resources_path,"Vocabs", "semcor_excluded.txt")
    weights_dir = os.path.join(resources_path,"Model","Wnd_model","Checkpoint")
    word_vocab = vocab_parser.read_vocab(word_vocab_file)
    _,bn_to_wn = vocab_parser.read_tsv_file(bn_to_wn_file)
    wnd_vocab = vocab_parser.read_vocab(wnd_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    output_vocab,inv_output_vocab = vocab_parser.read_semcor_vocab(fine_out_vocab_file)

    pos_dim = len(pos_vocab)
    wnd_dim = len(wnd_vocab)    
    output_dim = len(output_vocab)
    tf.reset_default_graph()
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) 
    MultiTaskBiLSTM = network.MultiTask_BiLSTM(output_dim,pos_dim,wnd_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
            print("\nModel restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            
        for sentences, lengths,words_id,pos in tqdm(MultiTaskBiLSTM.load_sentences_val(BATCH_SIZE,input_path)):
            fine_labels = sess.run(MultiTaskBiLSTM.Dense,\
                    feed_dict = {MultiTaskBiLSTM.input: sentences,MultiTaskBiLSTM.nopad_len: lengths})
            
            
            
            
            for n_sents in range(len(lengths)):
                
                for sen_len in range(lengths[n_sents]):
                    word_id = words_id[n_sents][sen_len]
                    word = sentences[n_sents][sen_len]
                    actual_pos = pos[n_sents][sen_len]
                    if word_id == None:
                        continue
                    else:

                        # =============================================================================
                        # MFS Strategy
                        # =============================================================================
                         if word not in word_vocab:
                            
                            wn_predict = utilities.wnid_from_lemmapos(word,actual_pos)[0]
                            if wn_predict == word:
                                wn_predict = utilities.wnid_from_lemma(word)[0]                            
                            predictions[word_id] = wn_predict
    
                         else:     
                            predicted_label = np.argmax(fine_labels[n_sents,sen_len,5:])
                            wn_predict = inv_output_vocab[predicted_label + 5]
                            if wn_predict[:3] != "wn:":
                                wn_predict = utilities.wnid_from_lemmapos(word,actual_pos)[0]
                                
                            predictions[word_id] = wn_predict 
                    
        for pred in predictions:
            predictions[pred] = bn_to_wn[predictions[pred]][0]
        
        first_elem = True
        with open(output_path,"w") as out:
            for elem in predictions:
                if first_elem:
                    out.write(elem  + " " + predictions[elem])
                    first_elem = False
                else:
                    out.write("\n" + elem + " " + predictions[elem])
            
                
        print("Prediction ended\n")
    


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Wordnet_domains labels prediction started\n")
    predictions = {}
# =============================================================================
#   VOCAB FILES
# =============================================================================
    wnd_out_vocab_file = os.path.join(resources_path,"Vocabs","wnd_out_vocab.txt")
    pos_out_vocab_file = os.path.join(resources_path,"Vocabs","pos_out_vocab.txt")
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
#    word_vocab_file = os.path.join(resources_path,"Vocabs","semcor_vocab.txt")
#    wordid_to_wnsynset_file = os.path.join(resources_path,"Vocabs","wordid_to_wnsynset.txt")
#    bn_to_wnd_file = os.path.join(resources_path,"babelnet2wndomains.tsv")
#    bn_to_wn_file = os.path.join(resources_path,"babelnet2wordnet.tsv")
#    skipped_file = os.path.join(resources_path,"Vocabs", "semcor_excluded.txt")
    weights_dir = os.path.join(resources_path,"Model","Wnd_model","Checkpoint")
    wnd_vocab,id_to_wnd_vocab = vocab_parser.read_semcor_vocab(wnd_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    output_vocab,inv_output_vocab = vocab_parser.read_semcor_vocab(fine_out_vocab_file)
    pos_dim = len(pos_vocab)
    wnd_dim = len(wnd_vocab)    
    output_dim = len(output_vocab)
    tf.reset_default_graph()
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) 
    MultiTaskBiLSTM = network.MultiTask_BiLSTM(output_dim,pos_dim,wnd_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
            print("\nModel restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            
        for sentences, lengths,words_id,_ in tqdm(MultiTaskBiLSTM.load_sentences_val(BATCH_SIZE,input_path)):
            wnd_labels = sess.run(MultiTaskBiLSTM.Dense_coarse,\
                    feed_dict = {MultiTaskBiLSTM.input: sentences,MultiTaskBiLSTM.nopad_len: lengths})
            
            
            
            
            for n_sents in range(len(lengths)):
                
                for sen_len in range(lengths[n_sents]):
                    word_id = words_id[n_sents][sen_len]
                    if word_id == None:
                        continue
                    else:
                        # if the network predicts one of the 5 tokens(<s>,</s>,<unk>,<skp>,<pad>)
                        
                        predicted_label = np.argmax(wnd_labels[n_sents,sen_len,5:])
                        wnd_predict = id_to_wnd_vocab[predicted_label + 5]
                        predictions[word_id] = wnd_predict 
                    
        
        first_elem = True
        with open(output_path,"w") as out:
            for elem in predictions:
                if first_elem:
                    out.write(elem  + " " + predictions[elem])
                    first_elem = False
                else:
                    out.write("\n" + elem + " " + predictions[elem])
            
                
        print("Prediction ended\n")
    
            


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Lexicographer_domains labels prediction started\n")
    predictions = {}
# =============================================================================
#   VOCAB FILES
# =============================================================================
    lex_out_vocab_file = os.path.join(resources_path,"Vocabs","lex_out_vocab.txt")
    pos_out_vocab_file = os.path.join(resources_path,"Vocabs","pos_out_vocab.txt")
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
#    word_vocab_file = os.path.join(resources_path,"Vocabs","semcor_vocab.txt")
#    wordid_to_wnsynset_file = os.path.join(resources_path,"Vocabs","wordid_to_wnsynset.txt")
#    bn_to_wnd_file = os.path.join(resources_path,"babelnet2wndomains.tsv")
#    bn_to_wn_file = os.path.join(resources_path,"babelnet2wordnet.tsv")
#    skipped_file = os.path.join(resources_path,"Vocabs", "semcor_excluded.txt")
    weights_dir = os.path.join(resources_path,"Model","Lex_model","Checkpoint")
    lex_vocab,id_to_lex_vocab = vocab_parser.read_semcor_vocab(lex_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    output_vocab,inv_output_vocab = vocab_parser.read_semcor_vocab(fine_out_vocab_file)
    pos_dim = len(pos_vocab)
    lex_dim = len(lex_vocab)    
    output_dim = len(output_vocab)
    tf.reset_default_graph()
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) 
    MultiTaskBiLSTM = network.MultiTask_BiLSTM(output_dim,pos_dim,lex_dim,LEARNING_RATE,HIDDEN_SIZE,elmo)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
            print("\nModel restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            
        for sentences, lengths,words_id,_ in tqdm(MultiTaskBiLSTM.load_sentences_val(BATCH_SIZE,input_path)):
            lex_labels = sess.run(MultiTaskBiLSTM.Dense_coarse,\
                    feed_dict = {MultiTaskBiLSTM.input: sentences,MultiTaskBiLSTM.nopad_len: lengths})
            
            
            
            
            for n_sents in range(len(lengths)):
                
                for sen_len in range(lengths[n_sents]):
                    word_id = words_id[n_sents][sen_len]
                    if word_id == None:
                        continue
                    else:
                        # if the network predicts one of the 5 tokens(<s>,</s>,<unk>,<skp>,<pad>)
                        
                        predicted_label = np.argmax(lex_labels[n_sents,sen_len,5:])
                        lex_predict = id_to_lex_vocab[predicted_label + 5]
                        predictions[word_id] = lex_predict 
                    
        
        first_elem = True
        with open(output_path,"w") as out:
            for elem in predictions:
                if first_elem:
                    out.write(elem  + " " + predictions[elem])
                    first_elem = False
                else:
                    out.write("\n" + elem + " " + predictions[elem])
            
                
        print("Prediction ended\n")


if __name__ == "__main__":
    Path.Paths()
    input_path = os.path.join(Path.Paths.EVAL_DIR,"semeval2015","semeval2015.data.xml")
    output_path = os.path.join(Path.Paths.DATA_DIR,"Predictions","semeval2015","prediction_babelnet.txt")
    resources_path = Path.Paths.DATA_DIR
    predict_babelnet(input_path,output_path,resources_path)