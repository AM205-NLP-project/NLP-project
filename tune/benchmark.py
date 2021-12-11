import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from tqdm import tqdm

from main import WikiData


if __name__ == '__main__':
    rev = False
    if not rev:
        model = GPT2LMHeadModel.from_pretrained('model_best/')
        dataset_name = 'words_dataset'
    else:
        model = GPT2LMHeadModel.from_pretrained('model_best_rev/')
        dataset_name = 'words_dataset_rev'

    # use GPU access if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    # create dataset
    data = {
        'train': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_train.txt'), shuffle=True, batch_size=1),
        'val': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_val.txt'), shuffle=True, batch_size=1),
        'test': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_test.txt'), shuffle=True, batch_size=1),
    }

    for k, vs in data.items():
        print(k)
        nlls = []
        for ex in tqdm(vs):
            # text -> input tensors
            batch = tok(ex, padding=True, return_tensors='pt', truncation=True, max_length=256)

            # place tensors on GPU if using GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            trg_len = batch['input_ids'].shape[1]
            with torch.no_grad():
                outputs = model(**batch)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / len(nlls))
        print('Average ppl:', ppl)