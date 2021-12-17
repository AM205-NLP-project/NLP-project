import json

import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from tqdm import tqdm

from main import WikiData


if __name__ == '__main__':
    rev = True
    top_k = 10  # top-k best & worst to be logged
    dataset_name = 'words_dataset_unique.csv'

    if not rev:
        model = GPT2LMHeadModel.from_pretrained('model_unique_best/')
    else:
        model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/')

    # use GPU access if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    # create dataset
    data = {
        'train': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_train.json', rev=rev), shuffle=True,
                                             batch_size=1),
        'val': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_val.json', rev=rev), shuffle=True, batch_size=1),
        'test': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_test.json', rev=rev), shuffle=True, batch_size=1),
    }

    for k, vs in data.items():
        print(k)
        tok_nlls = []
        nlls = []

        best = []
        worst = []

        best_tok = []
        worst_tok = []
        for ex in tqdm(vs):
            # text -> input tensors
            batch = tok(ex, return_tensors='pt', truncation=True, max_length=256)

            # place tensors on GPU if using GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            trg_len = batch['input_ids'].shape[1]
            with torch.no_grad():
                outputs = model(**batch, labels=batch['input_ids'])
                neg_log_likelihood = outputs[0].cpu().item() * trg_len
                tok_nll = outputs[0].cpu().item()

            x = (neg_log_likelihood, ex[0])
            best.append(x)
            worst.append(x)
            best = sorted(best, key=lambda k: k[0])[:top_k]
            worst = sorted(worst, key=lambda k: k[0], reverse=True)[:top_k]

            x = (tok_nll, ex[0])
            best_tok.append(x)
            worst_tok.append(x)
            best_tok = sorted(best_tok, key=lambda k: k[0])[:top_k]
            worst_tok = sorted(worst_tok, key=lambda k: k[0], reverse=True)[:top_k]

            nlls.append(neg_log_likelihood)
            tok_nlls.append(tok_nll)

        tx = torch.tensor(nlls)
        tok_tx = torch.tensor(tok_nlls)

        nll_avg = tx.mean()
        nll_med = tx.median()
        nll_std = tx.std()
        tok_nll_avg = tok_tx.mean()
        tok_nll_med = tok_tx.median()
        tok_nll_std = tok_tx.std()

        ppl_avg = torch.exp(nll_avg)
        ppl_med = torch.exp(nll_med)
        ppl_std = torch.exp(nll_std)
        tok_ppl_avg = torch.exp(tok_nll_avg)
        tok_ppl_med = torch.exp(tok_nll_med)
        tok_ppl_std = torch.exp(tok_nll_std)

        json.dump({
            'best': best,
            'worst': worst,
            'best_tok': best_tok,
            'worst_tok': worst_tok,
            'token': {
                'nll': {
                    'mean': tok_nll_avg.item(),
                    'median': tok_nll_med.item(),
                    'std': tok_nll_std.item()
                },
                'ppl': {
                    'mean': tok_ppl_avg.item(),
                    'median': tok_ppl_med.item(),
                    'std': tok_ppl_std.item()
                }
            },
            'sequence': {
                'nll': {
                    'mean': nll_avg.item(),
                    'median': nll_med.item(),
                    'std': nll_std.item()
                },
                'ppl': {
                    'mean': ppl_avg.item(),
                    'median': ppl_med.item(),
                    'std': ppl_std.item()
                }
            }
        }, open(f'bench_{rev}_{k}.json', 'w+'), indent=2, sort_keys=True)