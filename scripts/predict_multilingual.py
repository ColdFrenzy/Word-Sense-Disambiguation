import network
import vocab_parser
import Path
import tensorflow as tf
import os 
import utilities
from tqdm import tqdm
import numpy as np
from elmoformanylangs import Embedder
import torch
torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#torch.cuda.current_device("cpu")
#torch.cuda._initialized = True



# =============================================================================
# PARAMETERS
# =============================================================================
BATCH_SIZE = 10
LEARNING_RATE = 0.01
HIDDEN_SIZE = 64

def predict_multilingual(input_path : str, output_path : str, resources_path : str, lang : str) -> None:
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
    :param lang: the language of the dataset specified in input_path
    :return: None
    """
    predictions = {}
# =============================================================================
#   VOCAB FILES
# =============================================================================
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
    wnd_out_vocab_file = os.path.join(resources_path,"Vocabs","wnd_out_vocab.txt")
    pos_out_vocab_file = os.path.join(resources_path,"Vocabs","pos_out_vocab.txt")
    fine_out_vocab_file = os.path.join(resources_path,"Vocabs","fine_out_vocab.txt")
#    wordid_to_wnsynset_file = os.path.join(resources_path,"Vocabs","wordid_to_wnsynset.txt")
#    bn_to_wnd_file = os.path.join(resources_path,"babelnet2wndomains.tsv")
    bn_to_wn_file = os.path.join(resources_path,"babelnet2wordnet.tsv")
#    skipped_file = os.path.join(resources_path,"Vocabs", "semcor_excluded.txt")
    weights_dir = os.path.join(resources_path,"Model","Wnd_model","Checkpoint")
    bn_to_wn,wn_to_bn = vocab_parser.read_tsv_file(bn_to_wn_file)
    wnd_vocab = vocab_parser.read_vocab(wnd_out_vocab_file)
    pos_vocab = vocab_parser.read_vocab(pos_out_vocab_file)
    output_vocab,inv_output_vocab = vocab_parser.read_semcor_vocab(fine_out_vocab_file) 

# =============================================================================
#  MODEL PATH
# =============================================================================
    if lang == "it":
        model_dir = os.path.join(resources_path,"Multilingual","it")
        multiling_dir = os.path.join(resources_path,"Vocabs","word2babelnetIT.txt")
    elif lang == "de":
        model_dir = os.path.join(resources_path,"Multilingual","de")
        multiling_dir = os.path.join(resources_path,"Vocabs","word2babelnetDE.txt")
    elif lang == "es":
        model_dir = os.path.join(resources_path,"Multilingual","es")
        multiling_dir = os.path.join(resources_path,"Vocabs","word2babelnetES.txt")
    elif lang == "fr":
        model_dir = os.path.join(resources_path,"Multilingual","fr")
        multiling_dir = os.path.join(resources_path,"Vocabs","word2babelnetFR.txt")


    multi_map = vocab_parser.read_map(multiling_dir)
    pos_dim = len(pos_vocab)
    wnd_dim = len(wnd_vocab)    
    output_dim = len(output_vocab)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    tf.reset_default_graph()

    MultiLingBiLSTM = network.MultiLing_BiLSTM(output_dim,pos_dim,wnd_dim,LEARNING_RATE,HIDDEN_SIZE)
    saver = tf.train.Saver()
    elmo = Embedder(model_dir,batch_size= BATCH_SIZE)
    
    with tf.Session(config=config) as sess:
        try:
            vars_to_restore=[v for v in tf.global_variables() if "MultiTask_BiLSTM" in v.name or  "optimizer" in v.name]
            vars_to_restore_dict = {}
            for v in vars_to_restore:
                vars_to_restore_dict[v.name[:-2]] = v
            saver = tf.train.Saver(vars_to_restore_dict)
            saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
            print("\nModel succesfully restored \n")
        except:
            print("\nModel not found, initializing a new model\n")
            tf.global_variables_initializer().run()  
            
            
        for sentences, lengths,words_id,pos in tqdm(MultiLingBiLSTM.load_sentences_val(BATCH_SIZE,input_path)):
            emb = elmo.sents2elmo(sentences)
            fine_labels = sess.run(MultiLingBiLSTM.Dense,\
                    feed_dict = {MultiLingBiLSTM.elmo_emb: emb})
            
            
            
            
            for n_sents in range(len(lengths)):
                
                for sen_len in range(lengths[n_sents]):
                    
                    word = sentences[n_sents][sen_len]
                    word = word.lower()
                    actual_pos = utilities.get_wnpos(pos[n_sents][sen_len])
                    word_id = words_id[n_sents][sen_len]
                    if word_id == None:
                        continue
                    else:

                        word = word + "#" + actual_pos
                        predicted_label = np.argmax(fine_labels[n_sents,sen_len,5:])
                        wn_predict = inv_output_vocab[predicted_label + 5]
                        try:
                            # if the result is a word and not wnid, we take it's MFS babelnet_id
                            if wn_predict[:3] != "wn:":
                                for i in range(len(multi_map[word])):
                                    if multi_map[word][i] in bn_to_wn: 
                                        wn_predict = multi_map[word][i]  
                                
                           
                            predictions[word_id] = wn_predict 
                        except: 
                            print("\nword " + word + " not present in the vocabulary, hence cannot be mapped to babelnet\n")
                            predictions[word_id] = "<UNK>"
                     
        for pred in predictions:
            if predictions[pred][:3] == "bn:":
                continue
            else:
                try:
                    predictions[pred] = wn_to_bn[predictions[pred]][0]
                except:
                    print("\n exception: the wnid " + predictions[pred] + " was not available in the file babelnet2wordnet \n")
                    predictions[pred] = "<UNK>"
                    
#        return predictions          
        first_elem = True
        with open(output_path,"w",encoding="utf8") as out:
            for elem in predictions:
                if first_elem:
                    out.write(elem  + " " + predictions[elem])
                    first_elem = False
                else:
                    out.write("\n" + elem + " " + predictions[elem])
            
                
        print("Prediction ended\n")
        
        
if __name__ == "__main__":
    Path.Paths()
    input_path = os.path.join(Path.Paths.MULTILING_DIR,"eval_dataset","semeval2013.fr.data.xml")
    output_path = os.path.join(Path.Paths.DATA_DIR,"Predictions","semeval2013","fr_babelnet.txt")
    resources_path = Path.Paths.DATA_DIR
    predict_multilingual(input_path,output_path,resources_path,"fr")