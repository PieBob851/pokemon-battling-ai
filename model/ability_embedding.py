import torch
import torch.nn as nn
import pandas as pd

# List of abilities retrieved from https://bulbapedia.bulbagarden.net/wiki/Ability
ability_df = pd.read_csv('data/abilities.csv')
# num represents move ID
ABILITY_NAME_TO_NUM = dict(zip(ability_df['name'], ability_df['num']))

# Note that there are 306 total abilities in ABILITY_NAME_TO_NUM dictionary. However, the identifers 'num' do NOT
# range from 0 to 305 as some are missing. Therefore, vocab_size will be slightly higher than actual number
# of possible abilities to allow lookup without out of index error in the Embedding.
VOCAB_SIZE = max(ABILITY_NAME_TO_NUM.values()) + 1
EMBEDDING_DIM = 5


class AbilityEmbedding(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.ability_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, ability_name):
        # if ability_name does NOT exist, default to 0 which is no ability.
        ability_num = ABILITY_NAME_TO_NUM.get(ability_name, 0)
        ability_embed = self.ability_embed(torch.tensor(ability_num))
        return ability_embed


"""
Example Usage:

model = AbilityEmbedding()
ability_embed = model("electricsurge")
print(ability_embed)
"""
