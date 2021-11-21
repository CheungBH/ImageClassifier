import os


class txtLogger:
    def __init__(self, folder, metrics):
        self.folder = folder
        self.metrics = metrics

    def update(self, epoch, phase, metrics):
        with open(os.path.join(self.folder, "log.txt"), "a+") as f:
            out = "{}: {} | ".format(phase, epoch)
            for metric, name in zip(metrics, self.metrics):
                out += "{}: {} | ".format(name, metric)
            f.write(out + "\n")
            if phase == "val":
                f.write("---------------------------------------------\n")


class BNLogger:
    def __init__(self, folder):
        self.file = os.path.join(folder, "bn.txt")

    def update(self, epoch, bn_ave):
        with open(self.file, "a+") as f:
            f.write("{} -> {}\n".format(epoch, bn_ave))
