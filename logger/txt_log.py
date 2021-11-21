import os


class txtLogger:
    def __init__(self, folder, metrics):
        self.folder = folder
        self.file = open(os.path.join(folder, "log.txt"), "w")
        self.metrics = metrics

    def update(self, epoch, phase, metrics):
        out = "{}: {} | ".format(phase, epoch)
        for metric, name in zip(metrics, self.metrics):
            out += "{}: {}".format(name, metric)
        self.file.write(out + "\n")
        if phase == "val":
            self.file.write("---------------------------------------------\n")


class BNLogger:
    def __init__(self, folder):
        self.file = open(os.path.join(folder, "bn.txt"), "w")

    def update(self, epoch, bn_ave):
        self.file.write("{} -> {}\n".format(epoch, bn_ave))
