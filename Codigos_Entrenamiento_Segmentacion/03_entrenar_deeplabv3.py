"""Entrena DeepLabV3+ para una clase especifica."""

from src.train_single import main


if __name__ == "__main__":
    main(default_architecture="deeplabv3")
