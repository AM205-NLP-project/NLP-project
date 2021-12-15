from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import numpy as np

from scipy.spatial import distance

def get_word_vec(text):
    model_inp = tok(text, return_tensors='pt')
    outputs = model(**model_inp, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    final_rep = hidden_states[-1]
    if final_rep.shape[1] == 1:
        return final_rep.squeeze()
    else:
        return final_rep.squeeze().mean(axis=1)

def distances_biases(job_name):
    #input is a string

    v_woman = get_word_vec(' woman')
    v_man = get_word_vec(' man')
    v_man_array = v_man.detach().numpy()
    v_woman_array = v_woman.detach().numpy()

    v_job = get_word_vec(job_name).detach().numpy()

    if len(v_job) == len(v_woman_array):
        woman_job = distance.cosine(v_job, v_woman_array)
        man_job = distance.cosine(v_job, v_man_array)
        print(f"The distance of woman to {job_name} is higher compared to man to {job_name}: {woman_job> man_job}")
        print(f" Cosine distance of woman to {job_name} : {woman_job} ")
        print(f" Cosine distance of man to {job_name} : {man_job}")
        print("=========================================")
    else:
        print("=================SHAPE MISMATCH========================")
        print(f"{job_name} and v_woman_array have different shapes")
        print(f"Woman array shape {len(v_woman_array)}")
        print(f"Man array shape {len(v_man_array)}")
        print(f"{job_name} array shape {len(v_job)}")
        print("=========================================")


def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    new_vec = vec / norm
    return new_vec


if __name__ == '__main__':
    model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/')
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'pad_token': '<|endoftext|>'})

    v1 = get_word_vec('mother')
    v2 = get_word_vec('father')
    v1_array = v1.detach().numpy()
    v2_array = v2.detach().numpy()
    v_woman = get_word_vec('woman')
    v_man = get_word_vec('man')
    v_man_array = v_man.detach().numpy()
    v_woman_array = v_woman.detach().numpy()

    v_doctor = get_word_vec('doctor').detach().numpy()
    woman_doc = distance.cosine(v_doctor, v_woman_array)
    man_doc = distance.cosine(v_doctor, v_man_array)
    print(f"The distance of woman to doctor is higher compared to man to doctor: {woman_doc> man_doc}")


    list_jobs = [' lawyer', ' doctor', ' professor', ' surgeon',  ' teacher', ' nurse', ' researcher', ' engineer',
                 ' singer', ' driver', ' business', ' career', ' home' , ' secretary', ' nanny', ' house cleaner',
                 ' barista', ' housewife', ' software engineer', ' doctorate', ' PhD', ' university', ' cashier', ' sexy'  ]



    for i in list_jobs:
        distances_biases(i)

    #vx = get_word_vec('test')
    #print(vx)

    # type and enter `q` to quit out of interactive mode!
    import pdb
    pdb.set_trace()


