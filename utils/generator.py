import datetime

def experiment_name(args):
    if args.exp_name is None:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        return args.exp_name
    
def epoch_log(epoch, action, loss, aucs):
    auc_log = ""
    for key in aucs:
        auc_log += f"| AUC_{key}: {aucs[key]:.5f} "
    jsonl = {f"":aucs[k] for k in aucs }
    jsonl['loss'] = loss
    jsonl['epoch'] = epoch
    return  f"Epoch {epoch} [{action}] | Loss: {loss:.5f} {auc_log}"
    