from model.ability_embedding import AbilityEmbedding
from model.move_embedding import MoveEmbedding
import torch
import torch.nn as nn

EMBEDDING_DIM = 128
HIDDEN_DIM = 64


class CustomPokemonModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.ability_embedding = AbilityEmbedding(embedding_dim=EMBEDDING_DIM)
        self.move_embedding = MoveEmbedding(embedding_dim=EMBEDDING_DIM)

        self.numerical_fc = nn.Linear(7 * 12, EMBEDDING_DIM)  # 7 stats for each pokemon on both teams

        self.hidden_layer = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * (12 + 48 + 1), HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)
        )

        self.action_fc = nn.Linear(EMBEDDING_DIM, 9)

    def forward(self, team, opponent_team):
        abilities = [p.ability for p in team.pokemon] + [p.ability for p in opponent_team.pokemon]

        moves = []
        numerical_data = []

        for p in team.pokemon:
            moves += p.moves
            numerical_data += p.stats
            numerical_data.append(p.current_hp)

        for p in opponent_team.pokemon:
            moves += p.moves
            numerical_data += p.stats
            numerical_data.append(p.current_hp)

        ability_embedding_output = torch.stack([self.ability_embedding(ability) for ability in abilities])
        # 12 total abilities, 1 per pokemon
        ability_embedding_output = ability_embedding_output.view(-1, 12 * EMBEDDING_DIM)

        move_embedding_output = torch.stack([self.move_embedding(move) for move in moves])
        # 12 pokemon * 4 moves each
        move_embedding_output = move_embedding_output.view(-1, 48 * EMBEDDING_DIM)

        numerical_fc_output = self.numerical_fc(torch.FloatTensor(numerical_data)).view(-1, EMBEDDING_DIM)

        combined_input = torch.cat([ability_embedding_output, move_embedding_output, numerical_fc_output], dim=-1)
        processed_features = self.hidden_layer(combined_input)

        action_logits = self.action_fc(processed_features)
        action_probs = nn.Softmax(dim=-1)(action_logits)

        mask = torch.FloatTensor(team.invalid_mask)
        if mask.sum() > 0:
            masked_action_probs = action_probs * mask
            masked_action_probs = masked_action_probs / masked_action_probs.sum()
        else:
            # TODO This seems to happen due to a weird case where we get a request but it's not active, it's "waiting"
            # I'm thinking we need to handle this case in a better way and wait for the next request from showdown
            # print('all moves invalid')
            return action_probs

        return masked_action_probs
