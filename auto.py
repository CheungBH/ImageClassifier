#-*-coding:utf-8-*-

from utils.utils import init_model_list_with_kw, init_model_list, get_backbone
import shutil, os
from test import AutoTester
from demo import AutoDemo

process_mapping = {
    "test": AutoTester,
    "demo": AutoDemo,
}


class AutoProcessor:
    def __init__(self, model_folder,  process_args, output_folder="", kw=(), display_interval=5):
        self.model_folder = model_folder
        self.output_folder = output_folder
        if kw:
            self.model_ls, self.option_ls = init_model_list_with_kw(model_folder, kw)
        else:
            self.model_ls, self.option_ls = init_model_list(model_folder)
        self.display_interval = display_interval
        self.processor = {}
        for key, value in process_args.items():
            self.processor[key] = process_mapping[key](*value)

    def process(self):
        for process_type, processor in self.processor.items():
            total_num = len(self.model_ls)
            for idx, (model, option) in enumerate(zip(self.model_ls, self.option_ls)):
                if idx % self.display_interval == 0:
                    print("-------------------[{}/{}]: {} for {}--------------".format(
                        idx+1, total_num, process_type, model))
                processor.run(model, get_backbone(option))
            if process_type == "test":
                shutil.move(os.path.join(self.model_folder, "{}_result.csv".format(process_type)), out_folder)


if __name__ == '__main__':
    model_folder = "weights/cat_dog_selected"
    out_folder = "result/cat_dog"
    kws = ()
    args = {
        "test": ["/home/hkuit155/Desktop/CNN_classification/data/CatDog", "", "val"],
        "demo": ["data/cat_dog_test", out_folder, "config/labels/cat_dog.txt"],
    }
    AP = AutoProcessor(model_folder, args, out_folder, kws)
    AP.process()


