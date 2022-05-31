from tqdm.auto import tqdm
from language_model import LMAccuracy
import torch
from timeit import default_timer


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


def decode(model, start_tokens, start_tokens_lens, max_generated_len=20, top_k=None):
    """
    :param RNNLM model: Model
    :param torch.tensor start_tokens: Batch of seed tokens. Shape: [T, B]
    :param torch.tensor start_tokens_lens: Length of each sequence in batch. Shape: [B]
    :return Tuple[torch.tensor, torch.tensor]. Newly predicted tokens and length of generated part. Shape [T*, B], [B]
    """
    # Get embedding for start_tokens
    embedding = model.word_embeddings(start_tokens)

    # Pass embedding through rnn and collect hidden states and cell states for each time moment
    all_h = []
    h = embedding.new_zeros([model.rnn.num_layers, start_tokens.shape[1], model.hidden_dim])
    for time_step in range(start_tokens.shape[0]):
        _, h = model.rnn(embedding[None, time_step], h)

        all_h.append(h)

    all_h = torch.stack(all_h, dim=1)

    # Take final hidden state and cell state for each start sequence in batch
    # We will use them as h_0, c_0 for generation new tokens
    h = all_h[:, start_tokens_lens - 1, torch.arange(start_tokens_lens.shape[0])]

    # List of predicted tokens for each time step
    predicted_tokens = []
    # Length of generated part for each object in the batch
    decoded_lens = torch.zeros_like(start_tokens_lens, dtype=torch.long)
    # Boolean mask where we store if the sequence has already generated
    # i.e. `<eos>` was selected on any step
    is_finished_decoding = torch.zeros_like(start_tokens_lens, dtype=torch.bool)

    # Stop when all sequences in the batch are finished
    while not torch.all(is_finished_decoding) and torch.max(decoded_lens) < max_generated_len:
        # Evaluate next token distribution using hidden state h.
        # Note. Over first dimension h has hidden states for each layer of LSTM.
        #     We must use hidden state from the last layer
        logits = model.output(h[-1])

        if top_k is not None:
            # Top-k sampling. Use only top-k most probable logits to sample next token
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            # Mask non top-k logits
            logits[indices_to_remove] = -1e10
            # Sample next_token.
            probas = torch.nn.functional.softmax(logits, dim=1)
            next_token = probas.multinomial(1).squeeze(1)
        else:
            # Select most probable token
            next_token = logits.argmax(dim=1)

        predicted_tokens.append(next_token)

        decoded_lens += (~is_finished_decoding)
        is_finished_decoding |= (next_token == torch.tensor(model.vocab['<eos>']))

        # Evaluate embedding for next token
        embedding = model.word_embeddings(next_token)

        # Update hidden and cell states
        _, h = model.rnn(embedding.unsqueeze(0), h)

    return torch.stack(predicted_tokens), decoded_lens
