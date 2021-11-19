from .models import CNNModel
import torch
import torch.nn as nn


class ModelBuilder:
    def __init__(self, model_name, cls_name, pretrain=False, device="cuda:0"):
        self.model_name = model_name
        self.cls_name = cls_name
        self.pretrain = pretrain
        self.device = device
        self.build()

    def build(self):
        self.CNN = CNNModel(self.cls_name, self.model_name, load_pretrain=self.pretrain)
        if self.device != "cpu":
            self.CNN.model.cuda()
        self.params_to_update = self.CNN.model.parameters()
        return self.CNN.model

    def load_weight(self, path):
        self.CNN.load(path)

    def inference(self, img_tns):
        img_tensor_list = [torch.unsqueeze(img_tns, 0)]
        input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
        self.image_batch_tensor = input_tensor.cuda()
        outputs = self.CNN.model(self.image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return outputs_tensor

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






