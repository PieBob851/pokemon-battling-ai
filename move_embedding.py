import torch
import torch.nn as nn

TYPE_TO_INDEX = {
    "Normal": 0, 
    "Fire": 1, 
    "Water": 2, 
    "Electric": 3, 
    "Grass": 4, 
    "Ice": 5, 
    "Fighting": 6, 
    "Poison": 7, 
    "Ground": 8, 
    "Flying": 9, 
    "Psychic": 10, 
    "Bug": 11, 
    "Rock": 12, 
    "Ghost": 13, 
    "Dragon": 14, 
    "Dark": 15, 
    "Steel": 16, 
    "Fairy": 17
}

CATEGORY_TO_INDEX = {
    "Physical": 0, 
    "Special": 1,
    "Status": 2
}

TARGET_TO_INDEX = {
    "adjacentAlly": 0,
    "adjacentAllyOrSelf": 1,
    "adjacentFoe": 2,
    "all": 3,
    "allAdjacent": 4,
    "allAdjacentFoes": 5,
    "allies": 6,
    "allySide": 7,
    "allyTeam": 8,
    "any": 9,
    "foeSide": 10,
    "normal": 11,
    "randomNormal": 12,
    "scripted": 13,
    "self": 14
}

EMBEDDING_DIM = 5
NUMERICAL_INPUT_SIZE = 3 # number of numerical stats used in MoveEmbedding

class MoveEmbedding(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, 
                 numerical_input_size=NUMERICAL_INPUT_SIZE, 
                 type_vocab_size=len(TYPE_TO_INDEX), 
                 category_vocab_size=len(CATEGORY_TO_INDEX), 
                 target_vocab_size=len(TARGET_TO_INDEX)):
        super().__init__()
        # Categorical data
        self.type_embed = nn.Embedding(type_vocab_size, embedding_dim)
        self.category_embed = nn.Embedding(category_vocab_size, embedding_dim)
        self.target_embed = nn.Embedding(target_vocab_size, embedding_dim)
        # Numerical data
        self.numerical_linear = nn.Linear(numerical_input_size, embedding_dim)
    
    def forward(self, move_info):
        type_embed = self.type_embed(torch.tensor(TYPE_TO_INDEX.get(move_info["type"])))
        category_embed = self.category_embed(torch.tensor(CATEGORY_TO_INDEX.get(move_info["category"])))
        target_embed = self.target_embed(torch.tensor(TARGET_TO_INDEX.get(move_info["target"])))

        numerical_inputs = self.get_numerical_inputs(move_info)
        numerical_embed = self.numerical_linear(numerical_inputs)

        final_embed = torch.cat((type_embed, category_embed, target_embed, numerical_embed), dim=-1)
        return final_embed

    def get_numerical_inputs(self, move_info):
        accuracy = move_info["accuracy"]
        if (accuracy is True):
            accuracy = 100
        basePower = move_info["basePower"]
        pp = move_info["pp"]
        numerical_inputs = torch.tensor([accuracy, basePower, pp]).float()
        return numerical_inputs

"""
Example Usage:

move_info = {
	"num": 71,
    "accuracy": 100,
    "basePower": 20,
    "category": "Special",
    "name": "Absorb",
    "pp": 25,
    "priority": 0,
    "flags": {"protect": 1, "mirror": 1, "heal": 1, "metronome": 1},
    "drain": [1, 2],
    "secondary": None,
    "target": "normal",
    "type": "Grass",
    "contestType": "Clever",
}

model = MoveEmbedding()
final_embed = model(move_info) # final DIM is 4 * embedding_dim
"""

