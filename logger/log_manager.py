from .logger import CustomizedLogger


class LoggerManager:
    def __init__(self, args, metrics, cls_metrics, phases=("train", "val")):
        self.phases = phases
        self.record_args(args)
        self.metrics = metrics
        self.cls_metrics = cls_metrics
        if self.auto:
            self.summary_logger = CustomizedLogger("/".join(self.save_dir.split("/")[:-1]), self.summary_csv_title(),
                                                   "train_result")
        self.individual_logger = CustomizedLogger(self.save_dir, self.individual_csv_title(), self.model_idx)

    def record_args(self, args):
        self.save_dir = args.save_dir
        self.model_idx = self.save_dir.split("/")[-1]
        self.data_name = args.data_path.split("/")[-1]
        self.trainval_ratio = args.trainval_ratio
        self.labels = args.labels

        self.backbone = args.backbone
        self.freeze = args.freeze

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.sparse = args.sparse
        self.load_weight = args.load_weight

        self.optMethod = args.optMethod
        self.LR = args.LR
        self.momentum = args.momentum
        self.weightDecay = args.weightDecay
        self.schedule = args.schedule
        self.schedule_gamma = args.schedule_gamma
        self.crit = args.crit

        self.auto = args.auto
        self.flops = args.flops
        self.params = args.params
        self.inf_time = args.inf_time

    def summary_csv_title(self):
        string = "idx,dataset,trainval_ratio,backbone,freeze,batch_size,epochs,sparse,load_weight,optMethod,LR," \
                  "momentum,weight_decay,schedule,schedule_gamma,crit,,flops,params,inf_time"

        for phase in self.phases:
            for metric in self.metrics:
                string += "{}_{},".format(phase, metric)
        return string[:-1] + "\n"

    def individual_csv_title(self):
        string = "epoch, "
        for metric in self.metrics:
            for phase in self.phases:
                string += "{}_{},".format(phase, metric)
        string += ","
        for metric in self.cls_metrics:
            for phase in self.phases:
                for label in self.labels:
                    string += "{}_{}_{},".format(phase, metric, label)
                string += ","
        return string[:-1] + "\n"

    def release(self, best_recorder):
        if self.auto:
            summary_log_value = [self.model_idx, self.data_name, self.trainval_ratio, self.backbone, self.freeze,
                                 self.batch_size, self.epochs, self.sparse, self.load_weight, self.optMethod, self.LR,
                                 self.momentum, self.weightDecay, self.schedule, self.schedule_gamma, self.crit, "",
                                 self.flops, self.params, self.inf_time]
            for phase in ["train", "val"]:
                for idx in range(len(self.metrics)):
                    summary_log_value.append(best_recorder[phase][idx])
            self.summary_logger.write(summary_log_value)

    def update(self, epoch, metrics, cls_metrics):
        individual_log_value = [epoch]
        for idx in range(len(self.metrics)):
            for phase in self.phases:
                individual_log_value.append(metrics[phase][idx][-1])
        individual_log_value.append("")
        for i in range(len(self.cls_metrics)):
            for phase in self.phases:
                for j in range(len(self.labels)):
                    individual_log_value.append(cls_metrics[phase][i][j][-1])
                individual_log_value.append("")
        self.individual_logger.write(individual_log_value)
