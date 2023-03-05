import os
import shutil

import pandas as pd

dataset = "./to_parse"
folders = ["train"]


def gather_classes(csv: pd.DataFrame):
    classes = csv["class"].unique()
    classes = list(classes)

    return classes


def create_folders(classes: list[str]):
    try:
        os.mkdir(os.path.join(dataset, "output"))
    except FileExistsError:
        pass

    for class_name in classes:
        path = os.path.join(dataset, "output", class_name)

        if not os.path.exists(path):
            os.mkdir(path)


def parse_record(record: dict, i: int):
    path = os.path.join(dataset, folders[i], record["filename"])
    destination = os.path.join(dataset, "output",
                               record["class"], record["filename"])

    shutil.copyfile(path, destination)


def parse_dataset():
    classes = []
    data = []

    for folder in folders:
        path = os.path.join(dataset, folder, "_annotations.csv")
        csv = pd.read_csv(path)

        classes += gather_classes(csv)
        data.append(csv.to_dict("records"))

    print(len(data), data[0][0])
    create_folders(classes)

    for i, records in enumerate(data):
        for record in records:
            parse_record(record, i)


if __name__ == "__main__":
    parse_dataset()
