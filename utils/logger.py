import pandas as pd

class Logger:
    def __init__(self, exp_name, path=None):
        if path:
            self.exp_dir = path
            self.aucs = {"Train": {}, "Valid": {}}
            self.losses = {"Train": [], "Valid": []}
            self.recover(path)
        else:
            self.exp_dir = f"runs/{exp_name}"
            self.aucs = {"Train": {}, "Valid": {}}
            self.losses = {"Train": [], "Valid": []}
            open(f"{self.exp_dir}/log.txt", "w").write(
                f"# Experiment: {exp_name}" + "\n"
            )

    def log(self, epoch, action, loss, aucs, better=False, show=False):
        auc_log = ""
        for key in aucs:
            auc_log += f"| AUC_{key}: {aucs[key]:.5f} "

            if key not in self.aucs[action]:
                self.aucs[action][key] = []
            self.aucs[action][key].append(aucs[key])
        self.losses[action].append(loss)
        better_mark = "*" if better else " "
        log = f"Epoch {epoch} [ {action}{better_mark}] | Loss: {loss:.5f} {auc_log}"
        if action == "Train":
            open(f"{self.exp_dir}/train_log.txt", "a").write(log + "\n")
        if action == "Valid":
            open(f"{self.exp_dir}/valid_log.txt", "a").write(log + "\n")
        if show:
            print(log)
        self.save_auc_csv(action)

    def save_auc_csv(self, action):
        dc = {
            "epoch": range(1, len(self.losses[action]) + 1),
            "loss": self.losses[action],
            **self.aucs[action],
        }
        df = pd.DataFrame(dc)
        df.to_csv(f"{self.exp_dir}/{action}.csv", index=False)

    def recover(self, path):
        csv = pd.read_csv(f"{path}/Valid.csv")
        self.losses = {"Train": [], "Valid": list(csv["loss"])}
        for key in csv.keys():
            if key not in ["epoch", "loss"]:
                _key = key
                if key != "all":
                    # str2int
                    _key = int(key)
                self.aucs["Valid"][_key] = list(csv[key])
