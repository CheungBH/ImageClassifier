import torch


def calculate_BN(model):
    bn_sum, bn_num = 0, 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            bn_num += mod.num_features
            bn_sum += torch.sum(abs(mod.weight))
            # self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), self.curr_epoch)
    bn_ave = bn_sum / bn_num
    return bn_ave.tolist()


def compare(before, after, direction):
    if direction == "up":
        return True if after > before else False
    elif direction == "down":
        return True if after < before else False
    else:
        raise ValueError("Please assign the direction correctly")


def print_final_result(best_recorder, metrics, phases=("train", "val")):
    print("-----Printing final result-----")
    for phase in phases:
        for idx, metric in enumerate(metrics):
            if isinstance(best_recorder[phase][idx], list):
                print("Best {} {}: {}".format(phase, metric, best_recorder[phase][idx][0]))
            else:
                print("Best {} {}: {}".format(phase, metric, best_recorder[phase][idx]))
    print("-------------------------------")


def list2str(ls):
    tmp = ""
    for item in ls:
        if isinstance(item, str):
            tmp += item
        else:
            tmp += str(round(item, 4))
        tmp += ","
    return tmp[:-1]


def convert_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    data = df.values  # data是数组，直接从文件读出来的数据格式是数组
    index1 = list(df.keys())  # 获取原有csv文件的标题，并形成列表
    data = list(map(list, zip(*data)))  # map()可以单独列出列表，将数组转换成列表
    data = pd.DataFrame(data, index=index1)  # 将data的行列转换
    data.to_csv(path, header=0)


def merge_csv(files, out_file):
    import pandas as pd
    content = [pd.read_csv(file) for file in files]
    train = pd.concat(content)
    train.to_csv(out_file, index=0, sep=',')


if __name__ == '__main__':
    logger_path = "../test_result.csv"
    convert_csv(logger_path)
