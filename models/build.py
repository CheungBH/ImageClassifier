from .models import CNNModel


class ModelBuilder:
    def __init__(self, model_name, cls_name, load_weight="", pretrain=False, device="cuda:0"):
        self.model_name = model_name
        self.cls_name = cls_name
        self.load_weight = load_weight
        self.pretrain = pretrain
        self.device = device
        self.build()

    def build(self):
        self.model = CNNModel(self.cls_name, self.model_name, load_pretrain=self.pretrain)
        if self.device != "cpu":
            self.model.model.cuda()
        self.params_to_update = self.model.model.parameters()
        return self.model.model

    def load_weight(self, path):
        self.model.load(path)

    def freeze(self):
        pass
        # try:
        #     feature_layer_num = config.freeze_pretrain[opt.backbone][0]
        #     classifier_layer_name = config.freeze_pretrain[opt.backbone][1]
        #     feature_num = int(opt.freeze * feature_layer_num)
        #
        #     for idx, (n, p) in enumerate(model.named_parameters()):
        #         if len(p.shape) == 1 and opt.freeze_bn:
        #             p.requires_grad = False
        #         elif classifier_layer_name not in n and idx < feature_num:
        #             p.requires_grad = False
        #         else:
        #             p.requires_grad = True
        # except:
        #     raise ValueError("This model is not supported for freezing now")






