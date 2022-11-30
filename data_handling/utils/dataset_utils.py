def create_csv_files_for_datasets(test_split):
    pass


def get_class_distribution_for_batch(y_batch, count):
    for label in y_batch:
        if not count:
            count[int(label)] = 1
        else:
            labels = list(count.keys())
            if label not in labels:
                count[int(label)] = 1

            else:
                count[int(label)] += 1
    return count


def get_class_distribution(dataloader):
    count = {}
    for x_batch, y_batch in dataloader:
        count = get_class_distribution_for_batch(y_batch, count)
    return count

"""
def main():
    print("--- TRAIN ---")
    train_count = get_class_distribution(train)
    print(train_count)

    print("--- VALIDATION ---")
    validation_count = get_class_distribution(validation)
    print(validation_count)

    print("--- TEST ---")
    test_count = get_class_distribution(test)
    print(test_count)

    distribution_dict = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count
    }
    df = pd.DataFrame.from_dict(distribution_dict)
    print(df.head())
    df.to_csv("distribution_of_classes.csv")
"""
