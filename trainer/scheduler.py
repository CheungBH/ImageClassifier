from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR


class SchedulerInitializer:
    def __init__(self):
        pass

    def get(self, args, optimizer):
        self.schedule = args.schedule
        if self.schedule == "step":
            epochs = args.epochs
            gamma = 0.1 if not args.schedule_gamma else args.schedule_gamma
            self.scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=gamma)
        elif self.schedule == "exp":
            gamma = 0.9999 if not args.schedule_gamma else args.schedule_gamma
            self.scheduler = ExponentialLR(optimizer, gamma=gamma)
        elif self.schedule == "stable":
            self.scheduler = None
        else:
            raise NotImplementedError("The scheduler is not supported")

    def update(self, phase, step):
        assert step in ["epoch", "iter"]
        if phase == "train":
            if step == "epoch" and self.schedule == "step":
                self.scheduler.step()
            elif step == "iter" and self.scheduler == "exp":
                self.scheduler.step()







