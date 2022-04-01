import torch


def collate_fn(batch, sep_token_id=2):
    # batch가 1 sample씩 들어있음
    max_len=0
    SEP_TOKEN = sep_token_id
    new_batch={'input_ids':[],'attention_mask':[],'token_type_ids':[],'labels':[]}

    # max_len 구하기
    for element in batch:
        input_id = element['input_ids']
        cur_len = int((input_id==SEP_TOKEN).nonzero(as_tuple=True)[0][-1])
        max_len = max(max_len, cur_len)

    for element in batch:
        for key,value in element.items():
            if key!='labels':
                new_batch[key].append(value[:max_len+1].numpy().tolist())
            else:
                new_batch[key].append(value.numpy().tolist())

    # 파이썬 리스트 -> 텐서 변환
    for key,value in new_batch.items():
        new_batch[key]=torch.tensor(new_batch[key])

    return new_batch