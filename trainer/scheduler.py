from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR


class SchedulerInitializer:
    def __init__(self):
        pass

    def get(self, args, optimizer):
        schedule = args.schedule
        if schedule == "step":
            epochs = args.epochs
            gamma = 0.1 if not args.schedule_gamma else args.schedule_gamma
            return MultiStepLR(optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=gamma)
        elif schedule == "exp":
            gamma = 0.9999 if not args.schedule_gamma else args.schedule_gamma
            return ExponentialLR(optimizer, gamma=gamma)
        elif schedule == "stable":
            return None
        else:
            raise NotImplementedError("The scheduler is not supported")






