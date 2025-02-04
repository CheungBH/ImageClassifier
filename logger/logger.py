import os


class BaseLogger:
    def __init__(self, folder):
        self.folder = folder
        self.model_idx = self.folder.split("/")[-1]
        self.title = ""

    def init(self, kw):
        if not os.path.isdir(self.folder):
            excel_path = os.path.join("/".join(self.folder.split("/")[:-1]), "{}.csv".format(kw))
        else:
            excel_path = os.path.join(self.folder, "{}.csv".format(kw))

        if not os.path.exists(excel_path):
            self.file = open(excel_path, "w", encoding="utf-8")
            self.write_title()
        else:
            self.file = open(excel_path, "a+")

    @staticmethod
    def list2str(ls):
        tmp = ""
        for item in ls:
            if isinstance(item, str):
                tmp += item
            else:
                tmp += str(round(item, 4))
            tmp += ","
        return tmp[:-1]

    def write_title(self):
        self.file.write(self.title)

    def write_summarize(self, ls):
        for item in ls:
            self.file.write("{}\n".format(self.list2str(item)))
        self.file.write("\n")

    def write(self, ls):
        self.file.write("{}\n".format(self.list2str(ls)))


class CustomizedLogger(BaseLogger):
    def __init__(self, folder, title, excel_name):
        super().__init__(folder)
        self.title = title
        self.init(excel_name)


