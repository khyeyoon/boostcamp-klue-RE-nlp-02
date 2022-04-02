import json
from gevent import config
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

from functools import partial
from collate_fn import collate_fn
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import label_to_num

def inference(model, tokenizer, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False, collate_fn=partial(collate_fn, sep_token_id=tokenizer.sep_token_id))
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                    )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)
    
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
    
    return origin_label

def load_test_dataset(dataset_dir, tokenizer, token_type, sep_type):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir, token_type)
    try:
        test_label = list(map(int,test_dataset['label'].values))
    except:
        test_label = label_to_num(test_dataset['label'].values)
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, sep_type)
    return test_dataset['id'], tokenized_test, test_label

def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    if not os.path.exists("./prediction"):
        os.mkdir("./prediction")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = os.path.join(args.model_dir, "..")
    # Tokenizer_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    print('tokenizer', tokenizer)

    ## load my model
    MODEL_NAME = args.model_dir # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    #print(model)

    with open(os.path.join(args.model_dir, "..", "model_config_parameters.json"), 'r') as f:
        model_config_parameters = json.load(f)
        token_type = model_config_parameters['token_type']
        sep_type = model_config_parameters['sep_type']
    #print(model_config_parameters)
    
    ## load test datset
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, token_type, sep_type)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label)
    # get random number to choose example sentence
    data_idx = np.random.randint(0, 100)
    print("[dataset 예시]", tokenizer.decode(Re_test_dataset[data_idx]['input_ids']), sep='\n')

    ## predict answer
    pred_answer, output_prob = inference(model, tokenizer, Re_test_dataset, device) # model에서 class 추론
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

    output.to_csv('./prediction/' + args.submission_name + ".csv", index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--model_dir', type=str, default="./results/best_loss")
    parser.add_argument('--submission_name', type=str, default="submission")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    print(args)
    main(args)
    