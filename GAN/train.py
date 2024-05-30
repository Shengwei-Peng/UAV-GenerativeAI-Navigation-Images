from tqdm import tqdm
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import set_seed


def log_metrics(log_path, epoch, losses, lr):
    message = f"[Epoch: {epoch + 1}] " + " ".join(f"{k}: {v:.4f}" for k, v in losses.items()) + f" lr: {lr:.7f}"
    with open(log_path, "a") as log_file:
        log_file.write(f"{message}\n")


def main():
    opt = TrainOptions().parse()
    set_seed(seed=opt.seed)
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    log_path = f"{opt.checkpoints_dir}/{opt.name}/loss_log.txt"
    total_epochs = opt.n_epochs + opt.n_epochs_decay

    with tqdm(total=total_epochs, desc="Training progress") as epoch_pbar:
        for epoch in range(total_epochs):
            for data in dataset:
                model.set_input(data)
                model.optimize_parameters()

            losses = model.get_current_losses()
            lr = model.update_learning_rate()
            log_metrics(log_path, epoch, losses, lr)
            epoch_pbar.update()

    model.save_networks('latest')


if __name__ == '__main__':
    main()
