import torch
import torch.nn as nn
import pandas as pd

# List of moves retrieved from https://bulbapedia.bulbagarden.net/wiki/List_of_moves
moves_df = pd.read_csv('data/moves.csv')
# num represents move ID
MOVE_NAME_TO_NUM = dict(zip(moves_df['name'], moves_df['num']))

# Note that there are 901 total moves in MOVE_NAME_TO_NUM dictionary. However, the identifers 'num' do NOT
# range from 0 to 900 as some are missing. Therefore, vocab_size will be slightly higher than actual number
# of possible moves to allow lookup without out of index error in the Embedding.
VOCAB_SIZE = max(MOVE_NAME_TO_NUM.values()) + 1
EMBEDDING_DIM = 5


class MoveEmbedding(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.move_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, move_name):
        # if move_name does NOT exist, default to 0 since move numbering starts from 1.
        move_num = MOVE_NAME_TO_NUM.get(move_name, 0)
        move_embed = self.move_embed(torch.tensor(move_num))
        return move_embed


"""
Example Usage:

model = MoveEmbedding()
move_embed = model("upperhand")
print(move_embed)
"""
