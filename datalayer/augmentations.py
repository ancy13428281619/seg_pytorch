from datalayer import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5)
    ]
    return albu.Compose(train_transform)
