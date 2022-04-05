








def do_train(train_loader, model, otpim, device, epochs, epoch, total_idx, logging_step, eval_step):
    total_f1, total_loss, total_acc = 0, 0, 0
    average_loss, average_f1, average_acc = 0,0,0

    for idx, batch in enumerate(tqdm(train_loader)):
        model.train()
        total_idx += 1

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids =  batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
        pred = outputs[1]
        metric = compute_metrics(pred, labels)

        # loss = outputs[0]
        loss = criterion(pred, labels)

        loss.backward()
        optim.step()
        total_loss += loss
        total_f1 += metric['micro f1 score']
        # total_auprc += metric['auprc']
        total_acc += metric['accuracy']

        average_loss = total_loss/(idx+1)
        average_f1 = total_f1/(idx+1)
        average_acc = total_acc/(idx+1)

        if idx%logging_step == 0:
            print(f"[K_FOLD:({kfold_idx})][TRAIN][EPOCH:({epoch + 1}/{epochs}) | loss:{average_loss:4.2f} | ", end="")
            print(f"micro_f1_score:{average_f1:4.2f} | accuracy:{average_acc:4.2f}]")

    
        if total_idx%eval_step == 0:
            yield model, total_idx, 




def do_eval(valid_loader, model, device):
    eval_total_loss, eval_total_f1, eval_total_auprc, eval_total_acc = 0, 0, 0, 0
    label_list, pred_list = [], []
    with torch.no_grad():
        model.eval()
        print("--------------------------------------------------------------------------")
        print(f"[K_FOLD:({kfold_idx})][EVAL] STEP:{total_idx}, BATCH SIZE:{args.batch_size}")
        for idx, batch in enumerate(tqdm(valid_loader)):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids =  batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)
            pred = outputs[1]
            eval_metric = compute_metrics(pred, labels)

            # loss = outputs[0]
            loss = criterion(pred, labels)

            eval_total_loss += loss
            eval_total_f1 += eval_metric['micro f1 score']
            eval_total_auprc += eval_metric['auprc']
            eval_total_acc += eval_metric['accuracy']

            pred_list.extend(list(pred.detach().cpu().numpy().argmax(-1)))
            label_list.extend(list(labels.detach().cpu().numpy()))

        eval_average_loss = eval_total_loss/len(valid_loader)
        eval_average_f1 = eval_total_f1/len(valid_loader)
        eval_total_auprc = eval_total_auprc/len(valid_loader)
        eval_average_acc = eval_total_acc/len(valid_loader)

        cf = confusion_matrix(label_list, pred_list)
        cf = str_cf(cf)
        cr = classification_report(label_list, pred_list)
        with open(os.path.join(k_save_path, "report.txt"), "a") as f:
            f.write("#"*10 + f"  {total_idx}  " + "#"*100 + '\n')
            f.write(cf)
            f.write(cr)
            f.write('\n')

        if args.checkpoint:
            model.save_pretrained(os.path.join(save_path, f"checkpoint-{total_idx}"))
            # os.makedirs( os.path.join(k_save_path, f"checkpoint-{total_idx}") , exist_ok=True)
            # torch.save(model, os.path.join(k_save_path, f"checkpoint-{total_idx}", "model.bin"))

        if eval_average_loss < best_eval_loss:
            model.save_pretrained(os.path.join(save_path, "best_loss"))
            # os.makedirs( os.path.join(k_save_path, "best_loss") , exist_ok=True)
            # torch.save(model, os.path.join(k_save_path, "best_loss", "model.bin"))
            best_eval_loss = eval_average_loss

        if eval_average_f1 > best_eval_f1:
            model.save_pretrained(os.path.join(save_path, "best_f1"))
            # os.makedirs( os.path.join(k_save_path, "best_f1") , exist_ok=True)
            # torch.save(model, os.path.join(k_save_path, "best_f1", "model.bin"))
            best_eval_f1 = eval_average_f1

        if args.wandb == "True":
            wandb.log({
                "step":total_idx,
                "eval_loss":eval_average_loss,
                "eval_f1":eval_average_f1,
                "eval_acc":eval_average_acc,
                "lr":optim.param_groups[0]["lr"]
                })

        if args.lr_scheduler:
            scheduler.step(eval_average_loss) # ReduceLROnPlateau에는 loss 필요
            # scheduler.step()
            # if args.lr_scheduler == 'ReduceLROnPlateau':
            #     scheduler.step(eval_average_loss)
            # elif args.lr_scheduler == 'StepLR':
            #     scheduler.step()

        print(f"[K_FOLD:({kfold_idx})][EVAL][loss:{eval_average_loss:4.2f} | auprc:{eval_total_auprc:4.2f} | ", end="")
        print(f"micro_f1_score:{eval_average_f1:4.2f} | accuracy:{eval_average_acc:4.2f}]")

    print("--------------------------------------------------------------------------")
            