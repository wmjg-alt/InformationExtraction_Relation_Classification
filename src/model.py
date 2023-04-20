import torch.nn as nn
import torch


def bigrams(some_list: list):
    return zip(some_list, some_list[1:])


class DANClassifier(nn.Module):
    ''' base DAN classifier of vocab_size/embedding_dim with EmbeddingBat in mean mode'''
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_layer_sizes: list[int],
    ):
        super(DANClassifier, self).__init__()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size, embedding_dim=self.embedding_dim, mode="mean"
        )

        self.fc = nn.Sequential()

        in_out_dims = bigrams([self.embedding_dim] + hidden_layer_sizes)

        for idx, (in_dim, out_dim) in enumerate(in_out_dims):
            self.fc.add_module(
                name=f"{idx}_in{in_dim}_out{out_dim}", module=nn.Linear(in_dim, out_dim)
            )

        self.proj = nn.Linear(hidden_layer_sizes[-1], self.num_classes)

    def forward(self, token_indices: torch.Tensor, *args, **kwargs):
        avg_emb = self.embedding(token_indices)
        out = self.fc(avg_emb)
        out = self.proj(out)

        return out


class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    vocab_size : int
        The dimensionality of the vocabulary (input to Embedding layer)
    embedding_dim : int
        The dimensionality of the embedding
    hidden_layer_sizes : int
        Dimension size for hidden states within the LSTM
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    """

    def __init__(self, 
                 vocab_size=0, 
                 embedding_dim=0, 
                 hidden_layer_sizes=[100], 
                 num_classes=2, 
                 dr=0.2, 
                 llm=None,):
        super(LSTMTextClassifier, self).__init__()
        self.num_layers = 2
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size,embedding_dim) # 64, 128, 100
        self.dropout = nn.Dropout(dr)

        self.LSTM = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_layer_sizes[0],
                            num_layers=self.num_layers,
                            bidirectional =True,
                            dropout=dr)                             # 128, 100

        self.pool = nn.AdaptiveMaxPool1d(hidden_layer_sizes[0])               # --128--

        self.fc = nn.Linear(hidden_layer_sizes[0], self.num_classes)          # 100, 4

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.num_layers):
                torch.nn.init.xavier_uniform_(module.all_weights[i][0])
                torch.nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):
        # forward pass (from the outputs of the embedding)
        
        dropouted = self.dropout(embedded)
        output, (hn, cn) = self.LSTM(dropouted)

        pooled = self.pool(output)          # max pool
        pooled = pooled.mean(dim=1)         #then average

        logits = self.fc(pooled)
        return logits

    def forward(self, data, mask=None):
        data = self.embedding((data * mask) if mask else data)
        return self.from_embedding(data)


class LLMLSTMTextClassifier(LSTMTextClassifier):
    ''' LLM implementation of LSTM, utilizing LLM embedding layer'''
    def __init__(self, llm, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.llm = llm

    def forward(self, data, mask= None):
        data = self.llm((data * mask) if mask else data).last_hidden_state
        return self.from_embedding(data)

#-----------------------------------------------------------------------------
class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    vocab_size : int
        The dimensionality of the vocabulary (input to Embedding layer)
    embedding_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    hidden_layer_sizes : list of int, default = [3,4] // renamed hidden_layer_sizes from FILTER_WIDTHS
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, vocab_size=0, 
                       embedding_dim=0,
                       num_classes=2,
                       dr=0.2,
                       hidden_layer_sizes=[3,4], #filter_widths / hidden_layer_sizes
                       num_filters=100, 
                       num_conv_layers=2,
                       intermediate_pool_size=3,
                       llm=None,
                       **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.filter_modules = nn.ModuleList()

        for w in hidden_layer_sizes:
            # Each filter width requires a sequential chain of CNN_BLOCKS
            cnn_blocks = nn.Sequential()
            for n in range(num_conv_layers):
                # A CNN_BLOCK IN EACH LAYER
                cnn_blocks.append(nn.Dropout(dr))
                #emb_dim first pass, num_filters subsequent
                in_dim = embedding_dim if n == 0 else num_filters

                cnn_blocks.append(nn.Conv1d(in_channels= in_dim, 
                                           out_channels=num_filters,
                                           kernel_size=w, ))            # (embed_dim, num_filters) -> (num_filters, num_filters)
                cnn_blocks.append(nn.ReLU())

                if n == num_conv_layers -1:
                    cnn_blocks.append(nn.AdaptiveMaxPool1d(output_size=1))                  # final layer output to 1
                else:
                    cnn_blocks.append(nn.MaxPool1d(kernel_size=(intermediate_pool_size,)))  # intermediate layers
                
            self.filter_modules.append(cnn_blocks)
        
        self.fc = nn.Linear(num_filters * len(hidden_layer_sizes) , out_features=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight.data)

    def from_embedding(self, embedded):   
        # forward pass (from the outputs of the embedding)
        chain_in = embedded.permute(0,2,1)

        all_filters = []
        for i,chain in enumerate(self.filter_modules):
            chain_out = chain(chain_in)
            all_filters.append(chain_out.squeeze())

        all_filters = torch.cat(all_filters,dim=1)

        logits = self.fc(all_filters)
        return logits
    
    def forward(self, data, mask=None):
        data = self.embedding((data * mask) if mask else data)
        return self.from_embedding(data)



class LLMCNNTextClassifier(CNNTextClassifier):
    ''' implementation of the CNN with llm embeddings layer '''
    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    def forward(self, data, mask= None):
        data = self.llm((data * mask) if mask else data).last_hidden_state
        return self.from_embedding(data)
