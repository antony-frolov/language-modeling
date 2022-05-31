from tqdm.auto import tqdm
from language_model import LMAccuracy
import torch


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for data in tqdm(dataloader):
        tokens, tokens_lens = data['tokens'], data['tokens_lens']
        tokens = tokens.to(device)
        tokens_lens = tokens_lens.to(device)

        output = model(tokens, tokens_lens)

        loss = loss_fn(output, tokens, tokens_lens)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


def evaluate(dataloader, model, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    accuracy_fn = LMAccuracy()
    with torch.no_grad():
        for data in dataloader:
            tokens, tokens_lens = data['tokens'], data['tokens_lens']
            tokens = tokens.to(device)
            tokens_lens = tokens_lens.to(device)

            output = model(tokens, tokens_lens)

            loss = loss_fn(output, tokens, tokens_lens)
            total_loss += loss

            accuracy = accuracy_fn(output, tokens, tokens_lens)
            total_accuracy += accuracy

    return total_loss / len(dataloader), total_accuracy / len(dataloader)


def train(
    train_loader, test_loader, model, loss_fn, optimizer, device, num_epochs
):
    test_losses = []
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    for epoch in range(num_epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device)

        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        print(
            'Epoch: {0:d}/{1:d}. Loss (Train/Test): {2:.3f}/{3:.3f}. Accuracy (Train/Test): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]
            )
        )
    return train_losses, train_accuracies, test_losses, test_accuracies
