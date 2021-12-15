from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


def get_word_vec(text):
    model_inp = tok(text, return_tensors='pt')
    outputs = model(**model_inp, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    final_rep = hidden_states[-1]
    return final_rep.squeeze().mean(axis=1)


if __name__ == '__main__':
    model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/')
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    vx = get_word_vec('test')


