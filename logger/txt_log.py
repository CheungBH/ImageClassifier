import os


class txtLogger:
    def __init__(self, folder, metrics, round_size=(8, 4)):
        self.folder = folder
        self.metrics = metrics
        self.round_size = round_size
        if self.round_size:
            assert len(round_size) == len(metrics)

    def update(self, epoch, phase, metrics):
        phase = "valid" if phase == "val" else phase
        with open(os.path.join(self.folder, "log.txt"), "a+") as f:
            out = "{}: {} | ".format(phase, epoch)
            for metric, name, size in zip(metrics, self.metrics, self.round_size):
                out += "{}: {} | ".format(name, round(metric, size))
            f.write(out + "\n")
            if phase == "valid":
                f.write("-------------------------------------------------------\n")


class BNLogger:
    def __init__(self, folder):
        self.file = os.path.join(folder, "bn.txt")

    def update(self, epoch, bn_ave):
        with open(self.file, "a+") as f:
            f.write("{} -> {}\n".format(epoch, bn_ave))
