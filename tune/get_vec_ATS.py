import torch
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

def top_p_sample(text, p, temp=1.0, maxlen=128, minlen=2, num_samples=10, verbose=True):
    prefix = torch.tensor(tok.encode(text)).unsqueeze(0)

    sample_outputs = model.generate(
        prefix,
        pad_token_id=50256,
        do_sample=True,
        temperature=temp,
        max_length=prefix.shape[1] + maxlen,
        min_length=prefix.shape[1] + minlen,
        top_p=p,
        num_return_sequences=num_samples,
    )

    samples = []
    for i, sample_output in enumerate(sample_outputs):
        ox = tok.decode(sample_output, skip_special_tokens=True)
        samples.append(ox)

        if verbose:
            out = "{}: {}".format(i, ox)
            print(out)

    return samples


def get_word_vec(text):
    model_inp = tok(text, return_tensors='pt')
    outputs = model(**model_inp, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    final_rep = hidden_states[-1]
    if final_rep.shape[1] == 1:
        return final_rep.squeeze()
    else:
        return final_rep.squeeze().mean(axis=1)
################################## Hunter's Functions Above #############################################

def normalize_vec(text):
    vec = get_word_vec(text)
    vec_array = vec.detach().numpy()
    norm = np.linalg.norm(vec_array)
    new_vec = vec_array/norm
    return new_vec

def get_next(test_txt, amt):
    # example of getting weights
    model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/')  # definition to word
    # model = GPT2LMHeadModel.from_pretrained('model_unique_best/') # word to definition
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})
    tokens = tok.tokenize(test_txt)
    model_inp = tok(test_txt, return_tensors='pt')
    outputs = model(**model_inp, output_hidden_states=True)
    logits = outputs.logits
    prob = logits[0,-1,:]
    most_prob = torch.argsort(prob)[amt:]
    return tok.decode(most_prob)


def compare_gender(new_norm):
    m_cosine = cosine(m_norm, new_norm)
    print("Male comparision:", m_cosine)
    w_cosine = cosine(w_norm, new_norm)
    print("Female comparision", w_cosine)
    list_gen = [m_cosine, w_cosine]
    gender = ["men", "women"]
    index = np.argmin(list_gen)
    print(f"{gender[index]} is closer to zero")
    #closer to zero the more similar they are

def return_sample(text):
    samp = top_p_sample(text, top_p, temp=temp, minlen=minlen, maxlen=maxlen, num_samples=num_samples, )
    return samp


if __name__ == '__main__':
    #model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/') # definition to word
    model = GPT2LMHeadModel.from_pretrained('model_unique_best/') # word to definition
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    top_p = 0.9  # restricts to the top p% of the cumulative distribution
    temp = 1.0  # higher temperature -> will sample more "surisingly"
    minlen = 2  # minimum number of symbols the model will add
    maxlen = 128  # maximum number of symbols the model will add
    num_samples = 100  # the number of samples that will be generated

    # need to pass a space
    m_norm = normalize_vec(' man')
    w_norm = normalize_vec(' woman')




    import pdb

    pdb.set_trace()




