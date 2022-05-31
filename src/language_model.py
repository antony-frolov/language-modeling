import torch
import torchtext


class RNNLangModel(torch.nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, vocab, rec_layer=torch.nn.GRU,
        dropout=0.5, num_layers=1, glove=False, freeze=False, **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.output_size = len(vocab)
        self.embedding_dim = embedding_dim

        if glove:
            glove = torchtext.vocab.GloVe(name='6B', dim=embedding_dim, unk_init=lambda t: t.normal_(0, 1))
            embeddings = glove.get_vecs_by_tokens(vocab.get_itos())
            embeddings[vocab.lookup_indices(['<unk>'])] = embeddings.mean(dim=0)
            embeddings[vocab.lookup_indices(['<sos>', '<eos>', '<pad>'])] = torch.empty(3, embedding_dim).normal_(0, 1)
            self.word_embeddings = torch.nn.Embedding.from_pretrained(
                embeddings, freeze=freeze, padding_idx=vocab['<pad>']
            )
        else:
            self.word_embeddings = torch.nn.Embedding(
                len(vocab), self.embedding_dim, padding_idx=vocab['<pad>']
            )

        if dropout is not None:
            self.rnn = rec_layer(
                self.embedding_dim, self.hidden_dim,
                dropout=self.dropout, num_layers=self.num_layers, **kwargs
            )
        else:
            self.rnn = rec_layer(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, **kwargs)

        self.output = torch.nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, tokens, tokens_lens):
        """
        :param torch.tensor(dtype=torch.long) tokens:
            Batch of texts represented with tokens. Shape: [T, B]
        :param torch.tensor(dtype=torch.long) tokens_lens:
            Number of non-padding tokens for each object in batch. Shape: [B]
        :return torch.tensor: Distribution of next token for each time step.
            Shape: [T, B, V], V -- size of vocabulary
        """
        # Make embeddings for all tokens
        embeddings = self.word_embeddings(tokens)

        # Forward pass embeddings through network
        output, _ = self.rnn(embeddings)

        # Take all hidden states from the last layer of LSTM for each step and perform linear transformation
        output = self.output(output)

        return output


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.tensor tokens_lens: Length of each sequence in batch
        :return torch.tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        tokens_lens = tokens_lens.to('cpu')
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, tokens_lens-1,
            batch_first=False, enforce_sorted=False
        ).data
        packed_tokens = torch.nn.utils.rnn.pack_padded_sequence(
            tokens[1:], tokens_lens-1,
            batch_first=False, enforce_sorted=False
        ).data

        loss = super().forward(packed_outputs, packed_tokens)

        return loss


class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.tensor tokens_lens: Length of each sequence in batch
        :return torch.tensor: Accuracy for given logits and tokens
        """
        tokens_lens = tokens_lens.to('cpu')
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(
            outputs, tokens_lens-1,
            batch_first=False, enforce_sorted=False
        ).data
        packed_tokens = torch.nn.utils.rnn.pack_padded_sequence(
            tokens[1:], tokens_lens-1,
            batch_first=False, enforce_sorted=False
        ).data

        accuracy = (packed_tokens == packed_outputs.argmax(dim=1)).float().mean()

        return accuracy
