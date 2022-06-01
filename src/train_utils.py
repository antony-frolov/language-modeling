from tqdm.auto import tqdm
from language_model import LMAccuracy
import torch
from timeit import default_timer
from copy import deepcopy


def train_epoch(dataloader, model, loss_fn, optimizer, device, verbose=True):
    model.train()
    iterator = tqdm(dataloader) if verbose else dataloader
    for data in iterator:
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
    train_loader, test_loader, model, loss_fn,
    optimizer, device, num_epochs, verbose=True, **kwargs
):
    stat = {
        'test_loss': [], 'train_loss': [],
        'test_acc': [], 'train_acc': [],
        'epoch': [], 'time': []
    }
    start_time = default_timer()
    for epoch in range(num_epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device, verbose)

        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        stat['train_acc'].append(train_acc.item())
        stat['train_loss'].append(train_loss.item())

        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        stat['test_acc'].append(test_acc.item())
        stat['test_loss'].append(test_loss.item())

        stat['epoch'].append(epoch+1)
        stat['time'].append(default_timer() - start_time)

        if verbose:
            print(
                f"Epoch: {stat['epoch'][-1]}/{num_epochs}." +
                f" Loss (Train/Test): {stat['train_loss'][-1]:.3f}/{stat['test_loss'][-1]:.3f}." +
                f" Accuracy (Train/Test): {stat['train_acc'][-1]:.3f}/{stat['test_acc'][-1]:.3f}"
            )

    if not verbose:
        print(
            f"Loss (Train/Test): {stat['train_loss'][-1]:.3f}/{stat['test_loss'][-1]:.3f}." +
            f" Accuracy (Train/Test): {stat['train_acc'][-1]:.3f}/{stat['test_acc'][-1]:.3f}"
        )
    return stat


def train_and_save(
    model_class, configs, default_config,
    default_train_config, save_path, device
):
    for config_name in configs:
        config = default_config.copy()
        params = config_name.split('-')
        config['glove'] = 'g' in params[0]
        config['embedding_dim'] = int(''.join(filter(str.isdigit, params[0])))
        config['hidden_dim'] = int(''.join(filter(str.isdigit, params[1])))
        config['num_layers'] = int(''.join(filter(str.isdigit, params[2])))
        config['dropout'] = int(''.join(filter(str.isdigit, params[3]))) * 10 ** (-len(params[3])+2)
        print(config_name, ':', end=' ')
        model = model_class(**config).to(device=device)
        train_config = deepcopy(default_train_config)
        train_config['optimizer'] = train_config['optimizer_class'](
            model.parameters(), **train_config['optimizer_params']
        )
        stat = train(model=model, **train_config)
        save_dict = {'config': config, 'stat': stat,
                     'state_dict': model.state_dict()}
        torch.save(save_dict, f"{save_path}/{config_name}")
