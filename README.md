# [NLP] 문장 내 개체간 관계 추출

문장의 단어(Entity)에 대한 속성과 관계를 예측하는 인공지능 모델 만들기

본 프로젝트에 대한 세부적인 내용은 아래 링크를 통해 확인하실 수 있습니다.

https://plaid-raja-512.notion.site/RE-Wrap-Up-Report-7c91a7b4fd3f4917bf40e1be99cd2612

# 대회 개요
문장 속에서 단어간에 관계성을 파악하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다.

그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 모델이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

## 예시

    sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
    subject_entity: 썬 마이크로시스템즈
    object_entity: 오라클
    relation: 단체:별칭 (org:alternate_names)

## input, output

input: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.

output: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다! 클래스별 확률의 순서는 주어진 dictionary의 순서에 맞게 일치시켜 주시기 바랍니다.
(dict_info.txt 파일 참고)
![image](https://user-images.githubusercontent.com/76618935/160109034-b2793868-4e41-4003-96d2-ba02fb118856.png)

# Description
dict_label_to_num.pkl: Dict[str:int] -> 
dict_num_to_label.pkl: Dict[int:str]


# How to use

1. Install requirements

        pip install -r requirements.txt
        
2. Run

        python train.py
        
3. [Option]
```
    --seed              type=int        default=42
    --model             type=str        default="klue/bert-base"
    --epochs            type=int        default=5
    --logging_step      type=int        default=100
    --eval_step         type=int        default=100
    --checkpoint        type=bool       default=False
    --batch_size        type=int        default=64
    --valid_batch_size  type=int        default=64
    --optimizer         type=str        default="AdamW"
    --lr                type=float      default=5e-5
    --val_ratio         type=float      default=0.1
    --criterion         type=str        default="cross_entropy" # 'cross_entropy', 'focal', 'label_smoothing', 'f1'
    --save_dir          type=str        default="./results"
    --report_name       type=str        
    --project_name      type=str        default="salt_v2"
    --token_type        type=str        default="origin" # 'origin', 'entity', 'type_entity', 'sub_obj', 'special_entity', 'special_type_entity'
    --wandb             type=bool       default=True
    --dropout           type=float      default=0.1
    --sep_type          type=str        default='SEP'
```
    model은 --save_dir의 경로에 저장합니다.
    wandb를 통해 학습을 기록하고 --project_name과 --report_name를 통해 원하는 project에 원하는 이름으로 학습을 저장할 수 있습니다.
    --wandb를 통해 wandb 기록 여부를 정할 수 있습니다.
    
    --token_type 은 entity special token를 주는 옵션입니다.
        Should be one of
        - 'origin'              :   이순신은 조선 중기의 무신이다.
        - 'entity'              :   [ENT]이순신[/ENT]은 조선 중기의 [ENT]무신[/ENT]이다.
        - 'type_entity'         :   [PER]이순신[/PER]은 조선 중기의 [POH]무신[/POH]이다.
        - 'sub_obj'             :   [SUB]이순신[/SUB]은 조선 중기의 [OBJ]무신[/OBJ]이다.
        - 'special_entity'      :   @ 이순신 @ 은 조선 중기의 # 무신 # 이다.
        - 'special_type_entity' :   @ * 사람 * 이순신 @ 은 조선 중기의 # ^ 지위 ^ 무신 # 이다.
        
        
    --sep_type은 input으로 들어오는 entity를 [SEP] token으로 나눌지 entity token으로 나눌지에 대한 옵션입니다.
        Should be one of
        - 'SEP' : [CLS]이순신[SEP]무신[SEP]이순신은 조선 중기의 무신이다.
        - 'ENT' : [CLS][PER]이순신[/PER][POH]무신[/POH][SEP]이순신은 조선 중기의 무신이다.
