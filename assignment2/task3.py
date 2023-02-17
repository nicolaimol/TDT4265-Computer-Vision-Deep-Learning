import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    """# hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    shuffle_data = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()"""

    num_epochs = 25
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Settings for task 2 and 3. Keep all to false for task 2.
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    # Hyperparameters

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    model_2 = SoftmaxModel(
        neurons_per_layer,
        not use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_2, val_history_2 = trainer_2.train(num_epochs)

    model_3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        not use_improved_weight_init,
        use_relu)
    trainer_3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_3, val_history_3 = trainer_3.train(num_epochs)

    model_4 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        not use_relu)

    trainer_4 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_4, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_4, val_history_4 = trainer_4.train(num_epochs)

    model_5 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)

    trainer_5 = SoftmaxTrainer(
        momentum_gamma, not use_momentum,
        model_5, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_5, val_history_5 = trainer_5.train(num_epochs)

    model_6 = SoftmaxModel(
        neurons_per_layer,
        not use_improved_sigmoid,
        not use_improved_weight_init,
        not use_relu)

    trainer_6 = SoftmaxTrainer(
        momentum_gamma, not use_momentum,
        model_6, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_6, val_history_6 = trainer_6.train(num_epochs)

    plt.figure(figsize=(20, 12))
    plt.subplot(3, 4, 1)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(3, 4, 2)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history_2["loss"],
                    "Training Loss 2", npoints_to_average=10)
    utils.plot_loss(val_history_2["loss"], "Validation Loss 2")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(3, 4, 4)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history_2["accuracy"], "Training Accuracy 2")
    utils.plot_loss(val_history_2["accuracy"], "Validation Accuracy 2")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history_3["loss"],
                    "Training Loss 3", npoints_to_average=10)
    utils.plot_loss(val_history_3["loss"], "Validation Loss 3")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(3, 4, 6)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history_3["accuracy"], "Training Accuracy 3")
    utils.plot_loss(val_history_3["accuracy"], "Validation Accuracy 3")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history_4["loss"],
                    "Training Loss 4", npoints_to_average=10)
    utils.plot_loss(val_history_4["loss"], "Validation Loss 4")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(3, 4, 8)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history_4["accuracy"], "Training Accuracy 4")
    utils.plot_loss(val_history_4["accuracy"], "Validation Accuracy 4")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history_5["loss"],
                    "Training Loss 5", npoints_to_average=10)
    utils.plot_loss(val_history_5["loss"], "Validation Loss 5")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(3, 4, 10)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history_5["accuracy"], "Training Accuracy 5")
    utils.plot_loss(val_history_5["accuracy"], "Validation Accuracy 5")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.ylim([0., 0.9])
    utils.plot_loss(train_history_6["loss"],
                    "Training Loss 6", npoints_to_average=10)
    utils.plot_loss(val_history_6["loss"], "Validation Loss 6")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(3, 4, 12)
    plt.ylim([0.8, .99])
    utils.plot_loss(train_history_6["accuracy"], "Training Accuracy 6")
    utils.plot_loss(val_history_6["accuracy"], "Validation Accuracy 6")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()


    plt.savefig("task3_train_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
