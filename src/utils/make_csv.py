import os
import pandas as pd
import random
import numpy as np

SEED = 42
random.seed(SEED)

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

if not os.path.isdir(path + '/data'):
    os.mkdir(path + '/data/')
    print("Created data directory but no data will be found...")
else:
    path += "/data/"

    images_path = []
    for (root, dir, files) in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                images_path.append(root + "/" + file)

    num_images = len(images_path)
    print(f"Found {num_images} images...")

    random.shuffle(images_path)


    train_split = int(0.7 * num_images)
    valid_split = int(0.15 * num_images)
    test_split = num_images - train_split - valid_split 

    datasets = ["train" for _ in range(train_split)]
    datasets.extend(["valid" for _ in range(valid_split)])
    datasets.extend(["test" for _ in range(test_split)])

    labels = ["dog" for _ in range(num_images)]
    class_id = [0 for _ in range(num_images)]

    csv_df = pd.DataFrame({"images_path" : images_path,
                        "datasets" : datasets,
                        "labels" : labels,
                        "class_id" : class_id})

    csv_df.to_csv(path + "/dataset_file.csv")
