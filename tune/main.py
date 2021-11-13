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


if __name__ == '__main__':
    # name of the model
    # used to load tokenizer & pre-trained weights
    model_str = 'gpt2'

    # load the tokenizer
    tok = AutoTokenizer.from_pretrained(model_str)
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    # load the pre-trained model
    model = GPT2LMHeadModel.from_pretrained(model_str)

    # init an optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # create dataset
    data = {
        'train': torch.utils.data.DataLoader(WikiData('toy_train.txt'), shuffle=True, batch_size=2),
        'val': torch.utils.data.DataLoader(WikiData('toy_val.txt'), shuffle=True, batch_size=2),
    }

    # create a basic linear decay lr schedule
    num_epochs = 10
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

    # create very basic training loop
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        model.train()
        tloss = 0.0
        for batch in data['train']:
            batch = tok(batch, padding=True, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            tloss += loss.detach().item()

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        print(f'Epoch {epoch} Avg. Train Loss: {tloss / len(data["train"])}')

        # create evaluation / validation loop
        model.eval()
        eloss = 0.0
        for batch in data['val']:
            batch = tok(batch, padding=True, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, labels=batch['input_ids'])
                eloss += outputs.loss.detach().item()

        print(f'Epoch {epoch} Avg. Eval Loss: {eloss / len(data["val"])}')

    # just sampling some outputs to see what our model is generating
    model.eval()
    for keyword in ['Word: pizza' + '<|endoftext|>', 'Word: learn' + '<|endoftext|>']:
        input_seq = keyword
        generated = torch.tensor(tok.encode(input_seq)).unsqueeze(0)
        generated = generated.to(device)
        sample_outputs = model.generate(
            generated,
            do_sample=True,
            temperature=0.9,
            max_length=30,
            num_return_sequences=5,
        )
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tok.decode(sample_output, skip_special_tokens=True)))
