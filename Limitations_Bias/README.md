# Subproject: Model Limitations & Bias

### Description of Each of the Documents

`OccupationalSexSeg.ipynb`: This juptyer notebook is used for the downstream bias  & occupational sex segregation  portion of the latex. It walks through how we pulled 100 examples for multiple male or female dominated examples and quantified them in order to analyze the model's potential downstream bias.

`Bias_quantification_WEFAT1.ipynb`: This juptyer notebook is used for: 1. cosine difference analysis, 2. the PCA EDA plots to find clusters, 3. SVD exploration for covariance plots of the gender words and careers. Here you will find the code and the graphs for the section `3.3.2 Bias` exploring these topics in our Latex Writeup.


`Model_outputs Folder`: This holds all the text files of the outputs from the model.

	- baseline_test.txt : hold the outputs for the baseline testing
	- def_word.txt: holds the word outputs for the chosen definition back to word test for phase 2 of word to definition back to word
	- gender_ex.txt: holds the 100 examples for the 10 chosen industries & is used in the occupationalsexseg.ipynb
	- output.txt: example of outputs
	- output_limiations.txt: holds all the outcomes for Foreign words, words with double meanings, slang/fake words, and misspelled tests.
	- weird.txt: some outputs that were used to understand the model that we found to be weird
	- word_def.txt: holds the outputted definitions for the list of 5 words for phase 1 of word to definition back to word

