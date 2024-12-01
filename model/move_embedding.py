import torch
import torch.nn as nn
import pandas as pd
import json

with open('data/moves.json', 'r') as file:
    moves_data = json.load(file)

MOVE_NAME_TO_ATTRIBUTES = {
    move["name"].lower().replace(' ', '').replace('-', ''): {
        "basePower": move.get("basePower", 0),
        "accuracy": 1 if move.get("accuracy", True) == True else move.get("accuracy", 0),
        "type": move.get("type", "None"),
        "category": move.get("category", "None"),
        "pp": move.get("pp", 0)
    }
    for move in moves_data
}

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
    "Fairy": 17,
    "None": 18
}

CATEGORY_TO_INDEX = {
    "Physical": 0,
    "Special": 1,
    "Status": 2,
    "None": 3
}

NUMERICAL_INPUT_SIZE = 3


class MoveEmbedding(nn.Module):
    def __init__(self, embedding_dim,
                 numerical_input_size=NUMERICAL_INPUT_SIZE,
                 type_vocab_size=len(TYPE_TO_INDEX),
                 category_vocab_size=len(CATEGORY_TO_INDEX), ):
        super().__init__()
        self.type_embed = nn.Embedding(type_vocab_size, embedding_dim)
        self.category_embed = nn.Embedding(category_vocab_size, embedding_dim)

        self.numerical_linear = nn.Linear(numerical_input_size, embedding_dim)
        self.combine_fc = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, move_name):
        move_info = MOVE_NAME_TO_ATTRIBUTES.get(move_name, {
            "basePower": 0,
            "accuracy": 0,
            "type": "None",
            "category": "None",
            "pp": 0
        })
        # print(move_info)
        type_embed = self.type_embed(torch.tensor(TYPE_TO_INDEX.get(move_info["type"])))
        category_embed = self.category_embed(torch.tensor(CATEGORY_TO_INDEX.get(move_info["category"])))

        numerical_inputs = self.get_numerical_inputs(move_info)
        numerical_embed = self.numerical_linear(numerical_inputs)

        final_embed = torch.cat((type_embed, category_embed, numerical_embed), dim=-1)
        return self.combine_fc(final_embed)

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

model = MoveEmbedding()
move_embed = model("upperhand")
print(move_embed)
"""
