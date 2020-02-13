# Natual Language Processing Project

The details about the project are described in the pdf file.<br>
This readme describes the steps needed to follow in order to succesfully test 
the various scripts.

# Requirements
This project requires tensorflow 1.13, python 3.6 <br> and other packages
that can be installed through the following commands:

*  pip install lxml
*  pip install nltk
*  pip install tensorflow_hub
*  pip install tqdm
*  pip install json

In order to run the predict_multilingual function, you need to install the ElmoForManyLangs package
that can be found at the following link:<br><br><br>
[ElmoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs)
<br><br><br>
The main ElmoForManyLangs repository should be placed in the *"Dataset/Multilingual/"* directory.<br>
In the same page you can find download the different model (it,de,es,fr).
Successively you should unzip these model in the respective *"Dataset/Multilingual/LANG"*
directory.<br>
You should also insert the file  *"lemma2synsets4.0.xx.wn.ALL.txt"* in the Multilingual directory,
since it was too big to be uploaded here.<br>
In the *"Dataset/Eval"* directory you should place the evaluation directories (semeval2007,semeval2013,semval2015,senseval2,senseval3) available in the Raganato's framework.<br>
In the *"Dataset/Train"* directory you should place the *semcor.data.xml* and the *semcor.gold.key.txt* files available in the Raganato's framework.<br>
In order to use the different vocabularies you should first run the vocab_parser script.<br>
In order to run the different Networks, you need to download the weights and place them in the *"Dataset/Model/NAME_model"*. You can find the weights at the following google drive link:
<br><br>
[MODEL WEIGHTS](https://drive.google.com/open?id=1zemBqm7YOZJWWW2A3oLOFdKugEw0SeMv)
<br><br><br>