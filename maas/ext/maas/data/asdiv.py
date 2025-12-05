import json

class ASDivDataset:
    def __init__(self, path):
        self.path = path

    def load(self):
        data = []
        with open(self.path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
