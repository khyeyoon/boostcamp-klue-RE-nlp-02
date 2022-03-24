
python train.py --model klue/bert-base --seed 42 --best_save_dir ./bests/bert_base_ent --batch_size 64 --valid_batch_size 128 --save_dir ./results/bert_base_ent --save_total_limit 10 --wandb_name mj_bert_ent
python train.py --model klue/bert-base --seed 42 --best_save_dir ./bests/bert_base_ent_maxlen_100 --batch_size 64 --valid_batch_size 128 --save_dir ./results/bert_base_ent_maxlen_100 --save_total_limit 10 --max_length 100 --wandb_name mj_bert_ent_maxlen100
python train.py --model klue/bert-base --seed 42 --best_save_dir ./bests/bert_base_ent_maxlen_150 --batch_size 64 --valid_batch_size 128 --save_dir ./results/bert_base_ent_maxlen_150 --save_total_limit 10 --max_length 150 --wandb_name mj_bert_ent_maxlen150

