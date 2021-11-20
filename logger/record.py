from .model_storage import ModelSaver
from .txt_log import txtLogger


class TrainRecorder:
    def __init__(self, args, metrics=(), directions=()):
        self.save_dir = args.save_dir
        self.directions = directions
        self.metrics = metrics
        self.initial_best = [0 for _ in range(len(metrics))]
        for idx, direct in enumerate(self.directions):
            if direct == "down":
                self.initial_best[idx] = -float("inf")
        self.epochs = []
        assert len(metrics) == len(directions), "The number of metrics and comparision directions is not equal"
        self.metrics_record = {"train": [[] for _ in range(len(metrics))],
                               "val": [[] for _ in range(len(metrics))]}
        self.best_recorder = {"train": self.initial_best, "val": self.initial_best}
        self.MS = ModelSaver(self.save_dir)
        self.txt_log = txtLogger(self.save_dir, self.metrics)

    def record_args(self, args):
        self.epochs = args.epochs
        self.sparse = args.sparse
        self.save_interval = args.save_interval
        self.data_path = args.data_path
        self.label_path = args.label_path
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.iterations = args.iteration

    def update(self, model, metrics, epoch, phase):
        self.epochs.append(epoch)
        self.txt_log.update(epoch, phase, metrics)
        epoch = -1 if epoch % self.save_interval != 0 else epoch
        updated_metrics = []
        for idx, (metric, m_name, direction, record) \
                in enumerate(zip(metrics, self.metrics, self.directions, self.metrics_record[phase])):
            if self.compare(record, metric, direction):
                updated_metrics.append(m_name)
                self.best_recorder[idx] = metric
        self.MS.update(model, epoch, updated_metrics)

    @staticmethod
    def compare(before, after, direction):
        if direction == "up":
            return True if after > before else False
        elif direction == "down":
            return True if after < before else False
        else:
            raise ValueError("Please assign the direction correctly")
