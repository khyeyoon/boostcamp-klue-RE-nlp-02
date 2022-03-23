

python inference.py --model_dir ./bests/klue_roberta_base_ent --submission_name ./prediction/klue_roberta_base_ent_bestloss.csv --tokenizer_dir ./tokenizer
python inference.py --model_dir ./results/roberta_base_ent/checkpoint-8000 --submission_name ./prediction/klue_roberta_base_ent_bestf1.csv --tokenizer_dir ./tokenizer
