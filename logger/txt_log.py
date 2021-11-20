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
