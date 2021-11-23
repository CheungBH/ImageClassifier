from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR


class SchedulerInitializer:
    def __init__(self):
        pass

    def get(self, args, optimizer):
        self.schedule = args.schedule
        if self.schedule == "step":
            start_epoch = args.start_epoch if args.start_epoch else -1
            epochs = args.epochs
            gamma = 0.1 if not args.schedule_gamma else args.schedule_gamma
            self.scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=gamma,
                                         last_epoch=start_epoch)
        elif self.schedule == "exp":
            start_iter = args.iterations if args.iterations else -1
            gamma = 0.9999 if not args.schedule_gamma else args.schedule_gamma
            self.scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=start_iter)
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







