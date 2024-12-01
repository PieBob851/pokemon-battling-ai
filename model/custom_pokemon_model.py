from model.ability_embedding import AbilityEmbedding
from model.move_embedding import MoveEmbedding
import torch
import torch.nn as nn
import json

EMBEDDING_DIM = 128
HIDDEN_DIM = 64
LSTM_HIDDEN_DIM = 128


class CustomPokemonModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.ability_embedding = AbilityEmbedding(embedding_dim=EMBEDDING_DIM)
        self.move_embedding = MoveEmbedding(embedding_dim=EMBEDDING_DIM)

        self.numerical_fc = nn.Linear(7 * 12, EMBEDDING_DIM)  # 7 stats for each pokemon on both teams

        with open('data/species.json', 'r', encoding='utf-8') as f:
            pokemon_data = json.load(f)

            self.pokemon_types = {pokemon['name']: pokemon['types'] for pokemon in pokemon_data}
            self.type_map = {"Bug": 0, "Dark": 1, "Dragon": 2, "Electric": 3, "Fairy": 4, "Fighting": 5, "Fire": 6, "Flying": 7, "Ghost": 8,
                             "Grass": 9, "Ground": 10, "Ice": 11, "Normal": 12, "Poison": 13, "Psychic": 14, "Rock": 15, "Steel": 16, "Water": 17}
        self.type_embedding = nn.Embedding(len(self.type_map), EMBEDDING_DIM)

        self.reduce_dim_fc = nn.Linear(EMBEDDING_DIM * (12 + 48 + 1 + 12), EMBEDDING_DIM)

        self.lstm = nn.LSTM(
            input_size=(EMBEDDING_DIM),
            hidden_size=LSTM_HIDDEN_DIM,
            batch_first=True,
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)
        )

        self.action_fc = nn.Linear(EMBEDDING_DIM, 9)

    def forward(self, team, opponent_team, hidden_states=None):
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

        pokemon_type_embeddings = []
        for p in team.pokemon + opponent_team.pokemon:
            type_embeds = torch.stack([self.type_embedding(torch.tensor(self.type_map[t]))
                                      for t in self.pokemon_types[p.name]])
            pokemon_type_embeddings.append(type_embeds.mean(dim=0))
        pokemon_type_embeddings = torch.stack(pokemon_type_embeddings).view(1, -1)

        ability_embedding_output = torch.stack([self.ability_embedding(ability) for ability in abilities])
        # 12 total abilities, 1 per pokemon
        ability_embedding_output = ability_embedding_output.view(-1, 12 * EMBEDDING_DIM)

        move_embedding_output = torch.stack([self.move_embedding(move) for move in moves])
        # 12 pokemon * 4 moves each
        move_embedding_output = move_embedding_output.view(-1, 48 * EMBEDDING_DIM)

        numerical_fc_output = self.numerical_fc(torch.FloatTensor(numerical_data)).view(-1, EMBEDDING_DIM)

        combined_input = torch.cat([ability_embedding_output, move_embedding_output,
                                   numerical_fc_output, pokemon_type_embeddings], dim=-1)
        combined_input = self.reduce_dim_fc(combined_input)

        if hidden_states is None:
            h_0 = torch.zeros(1, combined_input.size(0), LSTM_HIDDEN_DIM)
            c_0 = torch.zeros(1, combined_input.size(0), LSTM_HIDDEN_DIM)
            hidden_states = (h_0, c_0)
        else:
            hidden_states = tuple(h.detach() for h in hidden_states)

        lstm_output, hidden_states = self.lstm(combined_input.unsqueeze(1), hidden_states)
        lstm_output = lstm_output.squeeze(0)

        processed_features = self.hidden_layer(lstm_output)

        action_logits = self.action_fc(processed_features)
        action_probs = nn.Softmax(dim=-1)(action_logits)

        mask = torch.FloatTensor(team.invalid_mask)
        masked_action_probs = action_probs * mask
        if masked_action_probs.sum() > 0:
            masked_action_probs = masked_action_probs / masked_action_probs.sum()
        else:
            # print('all moves invalid')
            action_probs = action_probs / action_probs.sum()
            return action_probs, hidden_states

        return masked_action_probs, hidden_states
