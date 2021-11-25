#-*-coding:utf-8-*-

from utils.utils import init_model_list_with_kw, init_model_list, get_runtime_params, convert_csv
import shutil, os
from test import AutoTester
from demo import AutoDemo
from convert import AutoConvert
from error_analysis import AutoErrorAnalyser

process_mapping = {
    "test": AutoTester,
    "demo": AutoDemo,
    "convert": AutoConvert,
    "error_analyse": AutoErrorAnalyser
}


class AutoProcessor:
    def __init__(self, model_folder,  process_args, output_folder="", kw=(), display_interval=5):
        self.model_folder = model_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.model_ls, self.option_ls = init_model_list_with_kw(model_folder, kw) if kw \
            else init_model_list(model_folder)
        self.display_interval = display_interval
        self.processor = {key: process_mapping[key](*value) for key, value in process_args.items()}

    def process(self):
        for process_type, processor in self.processor.items():
            total_num = len(self.model_ls)
            for idx, (model, option) in enumerate(zip(self.model_ls, self.option_ls)):
                if idx % self.display_interval == 0:
                    print("-------------------[{}/{}]: {} for {}--------------".format(
                        idx+1, total_num, process_type, model))
                processor.run(model, get_runtime_params(process_type, option))
            if process_type == "test" or process_type == "error_analyse":
                excel_paths = [file for file in os.listdir(self.model_folder) if "{}".format(process_type) in file]
                for excel_path in excel_paths:
                    shutil.move(os.path.join(self.model_folder, excel_path),
                                os.path.join(self.output_folder, excel_path))
                    if process_type == "error_analyse":
                        convert_csv(os.path.join(self.output_folder, excel_path))


if __name__ == '__main__':
    model_folder = "weights/cat_dog_selected"
    out_folder = "result/cat_dog"

    out_folder = model_folder + "_result" if not out_folder else out_folder
    kws = ()
    args = {
        "test": ["/home/hkuit155/Desktop/CNN_classification/data/CatDog", "", "val"],
        "demo": ["data/cat_dog_test", out_folder, "config/labels/cat_dog.txt"],
        "convert": [out_folder],
        "error_analyse": ["/home/hkuit155/Desktop/CNN_classification/data/CatDog", "", "val"],
    }
    AP = AutoProcessor(model_folder, args, out_folder, kws)
    AP.process()


