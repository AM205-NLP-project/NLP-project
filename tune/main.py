import numpy as np
import torch

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from transformers import get_scheduler

from tqdm.auto import tqdm


class WikiData(torch.utils.data.Dataset):

    def __init__(self, filename):
        with open(filename) as fp:
            self._data = fp.readlines()

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)


def sample_model(md, tk, text):
    # use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # put model in eval only mode
    md.eval()

    # process input text
    input_seq = text
    generated = torch.tensor(tk.encode(input_seq)).unsqueeze(0)
    generated = generated.to(device)

    # sample model
    sample_outputs = md.generate(
        generated,
        do_sample=True,
        # temperature=0.9,
        max_length=generated.shape[1] + 5,
        # top_k=200,
        # top_p=0.95,
        # pad_token_id=50256,
        # num_beams=5,
        # no_repeat_ngram_size=2,
        num_return_sequences=3,
        # early_stopping=True,
    )
    outs = []
    for i, sample_output in enumerate(sample_outputs):
        ox = tk.decode(sample_output, skip_special_tokens=True)
        out = "{}: {}".format(i, ox)
        print(out)
        outs.append(ox)

    return outs


if __name__ == '__main__':
    # these are taken as the default
    # from what HuggingFace recommends
    # see:
    params = {
        'model_str': 'gpt2',  # name of pre-trained model; used to load weights and tokenizer
        'lr': 5e-5,  # peak learning rate for tuning
        'batch_size': 8,  # number of examples per batch
        'train_log': 50,  # num batches per train log
        'val_log': 2000,  # num train batches before validation step
    }

    # load the tokenizer
    tok = AutoTokenizer.from_pretrained(params['model_str'])
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})
    print('Loaded Tokenizer')

    # load the pre-trained model
    model = GPT2LMHeadModel.from_pretrained(params['model_str'])
    print('Loaded model')

    # init an optimizer
    optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=0.0)

    # create dataset
    dataset_name = 'words_dataset'
    batch_size = params['batch_size']
    data = {
        'train': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_train.txt'), shuffle=True, batch_size=batch_size),
        'val': torch.utils.data.DataLoader(WikiData(f'{dataset_name}_val.txt'), shuffle=True, batch_size=batch_size),
    }

    # create a basic linear decay lr schedule
    num_epochs = 3
    num_training_steps = num_epochs * len(data['train'])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # use GPU access if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # basic training loop
    progress_bar = tqdm(range(num_training_steps))
    best_val = np.inf
    for epoch in range(num_epochs):
        model.train()
        tloss = 0.0

        # begin training batch
        for ix, batch in enumerate(data['train']):
            # text -> input tensors
            batch = tok(batch, padding=True, return_tensors='pt', truncation=True, max_length=256)

            # place tensors on GPU if using GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # compute outputs
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            cur_loss = loss.detach().item()
            tloss += cur_loss

            # backpropagate
            loss.backward()

            # update optimizer lr
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if (ix + 1) % params['train_log'] == 0:
                out = f'{epoch}\t{ix}\t{cur_loss}\n'
                print(out, end='')
                with open(dataset_name + '_train.log', 'a+') as fp:
                    fp.write(out)

            if (ix + 1) % params['val_log'] == 0:
                # validation loop
                model.eval()
                eloss = 0.0
                for bx in tqdm(data['val'], desc='Validating'):
                    bx = tok(bx, padding=True, return_tensors='pt', truncation=True, max_length=256)
                    bx = {k: v.to(device) for k, v in bx.items()}
                    with torch.no_grad():
                        outputs = model(**bx, labels=bx['input_ids'])
                        eloss += outputs.loss.detach().item()

                print(f'Epoch {epoch} Avg. Eval Loss: {eloss / len(data["val"])}')

                if eloss < best_val:
                    # save model
                    model.save_pretrained('model_best/')
                    print('Saved new best model!')

                out = f'{epoch}\t{ix}\t{eloss}\n'
                with open(dataset_name + '_val.log', 'a+') as fp:
                    fp.write(out)

                model.train()

        print(f'Epoch {epoch} Avg. Train Loss: {tloss / len(data["train"])}')

    model.eval()
    eloss = 0.0
    for bx in tqdm(data['val'], desc='Validating'):
        bx = tok(bx, padding=True, return_tensors='pt', truncation=True, max_length=256)
        bx = {k: v.to(device) for k, v in bx.items()}
        with torch.no_grad():
            outputs = model(**bx, labels=bx['input_ids'])
            eloss += outputs.loss.detach().item()

    print(f'Final Avg. Eval Loss: {eloss / len(data["val"])}')

    if eloss < best_val:
        # save model
        model.save_pretrained('model_best/')
        print('Saved new best model!')
    else:
        model.save_pretrained('model_final/')
        print('Saved final model! (did not exceed previous best)')
