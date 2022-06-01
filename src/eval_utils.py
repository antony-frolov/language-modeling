import matplotlib.pyplot as plt
import torch
from language_model import RNNLangModel
import numpy as np


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


def plot_stats(stats, labels, title=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey='row')
    if title is not None:
        fig.suptitle(title, fontsize=14)
    for i, (stat, label) in enumerate(zip(stats, labels)):
        linestyle = 'solid' if i < 4 else 'dashed'
        axes[0, 0].plot(stat['epoch'], stat['train_loss'], label=label, linestyle=linestyle)
        axes[0, 1].plot(stat['epoch'], stat['test_loss'], label=label, linestyle=linestyle)
        axes[1, 0].plot(stat['epoch'], stat['train_acc'], label=label, linestyle=linestyle)
        axes[1, 1].plot(stat['epoch'], stat['test_acc'], label=label, linestyle=linestyle)
    axes[0, 0].legend()
    axes[0, 0].set_title('Train CE Loss')
    axes[0, 1].set_title('Test CE Loss')
    axes[1, 0].set_title('Train Accuracy')
    axes[1, 1].set_title('Test Accuracy')
    for ax in axes.flatten():
        ax.grid()
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
    plt.show()


def load_stats(configs, save_path):
    stats = []
    for config in configs:
        save_dict = torch.load(f"{save_path}/{config}")
        stats.append(save_dict['stat'])
    return stats


def continue_sentences(
    config, start_tokens, start_tokens_lens, save_path, top_k=None
):
    save_dict = torch.load(f"{save_path}/{config}")
    model = RNNLangModel(**save_dict['config'])
    model.load_state_dict(save_dict['state_dict'])
    model = model.cpu()
    model.eval()

    start_tokens = torch.tensor([model.vocab.lookup_indices(tokens) for tokens in start_tokens]).T
    start_tokens_lens = torch.tensor(start_tokens_lens)

    decoded_tokens, decoded_lens = decode(model, start_tokens, start_tokens_lens, max_generated_len=20, top_k=top_k)

    sentences = []
    for text_idx in range(start_tokens.shape[1]):
        decoded_text_tokens = decoded_tokens[:decoded_lens[text_idx], text_idx]
        tokens = start_tokens[:start_tokens_lens[text_idx], text_idx].tolist() + decoded_text_tokens.tolist()
        words = np.array(model.vocab.get_itos())[np.array(tokens)]
        sentences.append(' '.join(words))

    return sentences
