import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


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


if __name__ == '__main__':
    model = GPT2LMHeadModel.from_pretrained('model_unique_best/')
    # model = GPT2LMHeadModel.from_pretrained('model_unique_best_rev/')
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    text = 'Word: temporal ; Definition:'
    top_p = 0.9  # restricts to the top p% of the cumulative distribution
    temp = 1.0  # higher temperature -> will sample more "surisingly"
    minlen = 2  # minimum number of symbols the model will add
    maxlen = 128  # maximum number of symbols the model will add
    num_samples = 10  # the number of samples that will be generated

    top_p_sample(
        text, top_p,
        temp=temp,
        minlen=minlen, maxlen=maxlen,
        num_samples=num_samples,
    )

    # type and enter `q` to quit out of interactive mode!
    import pdb
    pdb.set_trace()
