# Subproject: Model Limitations & Bias

### Description of Each of the Documents

OccupationalSexSeg.ipynb: This juptyer notebook is used for the downstream bias  & occupational sex segregation  portion of the latex. It walks through how we pulled 100 examples for multiple male or female dominated examples and quantified them in order to analyze the model's potential downstream bias.

SVD Exploration.ipynb: This juptyer notebook is used for the PCA analysis and covariance plots of the gender words and careers.

Get_vec_ATS.py: This is a version of the get_vec.py that Adriana tailored to include a couple additional functions to create samples, compare gender cosine similarities, and get the list of tokens of the next predicted word by the model based off an inputted word. 

get_vec_bias.py: This is the version of the get_vec.py that Vitoria tailored to include a couple of additional functions to compare gender cosine similarities and run samples.

Model_outputs Folder: This holds all the text files of the outputs from the model.

	- baseline_test.txt : hold the outputs for the baseline testing
	- def_word.txt: holds the word outputs for the chosen definition back to word test for phase 2 of word to definition back to word
	- gender_ex.txt: holds the 100 examples for the 10 chosen industries & is used in the occupationalsexseg.ipynb
	- output.txt: example of outputs
	- output_limiations.txt: holds all the outcomes for Foreign words, words with double meanings, slang/fake words, and misspelled tests.
	- weird.txt: some outputs that were used to understand the model that we found to be weird
	- word_def.txt: holds the outputted definitions for the list of 5 words for phase 1 of word to definition back to word



benchmark.py, demoing.py, get_vec.py, main.py, preproccess.py, requirements.txt, sampler.py: Are all the same files seen in the other main tune file.



