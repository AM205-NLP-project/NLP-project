import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


def sample_from_model(text):
    # how to sample from the model
    # I recommend checking out this guide: https://huggingface.co/blog/how-to-generate
    # sampling is very, very tricky and is still very much unsolved
    # see this paper for discussion: https://arxiv.org/pdf/1904.09751.pdf
    generated = torch.tensor(tok.encode(text)).unsqueeze(0)
    # generated = generated.to(device)

    # sample model
    sample_outputs = model.generate(
        # input seed; any pre-generation text we want to start with
        generated,

        # just setting the pad token to the EOS token for sampling
        pad_token_id=50256,

        # sample text using probabilities as opposed to greedy sampling / beam search
        do_sample=True,

        # higher temperature = sampling rarer words/tokens; low temperature = more conservative sample
        # temperature=0.9,

        # cap on the number of tokens that will be generated
        # currently have this set so that the model can generate up to 10 more tokens
        max_length=generated.shape[1] + 10,

        # forcing the model to generate at least 2 more tokens
        min_length=generated.shape[1] + 2,

        # sampling modalities
        # top_k only retains the top k tokens and randomly samples from thsoe
        # top_k=200,

        # top p = nucleus sampling
        # defined in terms of probabilities...
        # only retains the top words until the specified probability region is reached
        top_p=0.95,

        # how many samples to generate
        num_return_sequences=10,

        # params for beam search
        # number of hypotheses to evaluate in parallel
        # num_beams=5,

        # penalty for repeating ngrams in beam search
        # no_repeat_ngram_size=2,

        # quits beam search early if all beams have hit an EOS token
        # early_stopping=True,
    )
    for i, sample_output in enumerate(sample_outputs):
        ox = tok.decode(sample_output, skip_special_tokens=True)
        out = "{}: {}".format(i, ox)
        print(out)


if __name__ == '__main__':
    model = GPT2LMHeadModel.from_pretrained('model_best/')
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    # example of getting weights
    test_txt = 'Definition: (chiefly law) Nearest in relationship. (See also next of kin.) Word: '
    tokens = tok.tokenize(test_txt)
    model_inp = tok(test_txt, return_tensors='pt')
    print('Num tokens:', len(tokens))

    outputs = model(**model_inp, output_hidden_states=True)
    logits = outputs.logits
    print('Logits (aka language modeling predictions) shape:', logits.shape)

    hidden_states = outputs.hidden_states
    print('Hidden states (aka latent/internal representation) tuple length:', len(hidden_states))

    initial_embed = hidden_states[0]
    print('Initial embedding representation (before contextualization) shape:', initial_embed.shape)

    final_rep = hidden_states[-1]
    print('Final representation from GPT-2 shape:', final_rep.shape)

    for ix, rep in enumerate(hidden_states):
        print('\tLayer', ix, 'representation shape:', rep.shape)

    sample_from_model(test_txt)
    sample_from_model('Example: I am running so fast')

    import pdb
    pdb.set_trace()
