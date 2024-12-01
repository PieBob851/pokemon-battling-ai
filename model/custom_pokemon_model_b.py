from model.ability_embedding import AbilityEmbedding
from model.move_embedding import MoveEmbedding
import torch
import torch.nn as nn
import json

EMBEDDING_DIM = 128
HIDDEN_DIM = 512
HIDDEN_DIM_FACTOR = 4

class CustomPokemonModelB(nn.Module):
    def __init__(self):
        super().__init__()

        self.ability_embedding = AbilityEmbedding(embedding_dim=EMBEDDING_DIM)
        self.move_embedding = MoveEmbedding(embedding_dim=EMBEDDING_DIM)
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * (12 + 48) + 84 + 18 * 12, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // HIDDEN_DIM_FACTOR),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // HIDDEN_DIM_FACTOR, HIDDEN_DIM // HIDDEN_DIM_FACTOR // HIDDEN_DIM_FACTOR),
            nn.ReLU()
        )
        
        self.action_fc = nn.Linear(HIDDEN_DIM // HIDDEN_DIM_FACTOR // HIDDEN_DIM_FACTOR, 9)
        
        with open('species.json', 'r', encoding='utf-8') as f:
            pokemon_data = json.load(f)
            
            self.pokemon_types = {pokemon['name']: pokemon['types'] for pokemon in pokemon_data}
            self.type_map = {"Bug": 0, "Dark": 1, "Dragon": 2, "Electric": 3, "Fairy": 4, "Fighting": 5, "Fire": 6, "Flying": 7, "Ghost": 8, "Grass": 9, "Ground": 10, "Ice": 11, "Normal": 12, "Poison": 13, "Psychic": 14, "Rock": 15, "Steel": 16, "Water": 17}
    
    def forward(self, team, opponent_team):
        abilities = [p.ability for p in team.pokemon] + [p.ability for p in opponent_team.pokemon]

        moves = []
        numerical_data = []
        types = torch.zeros((18 * 12))
        
        counter = 0
        for p in team.pokemon:
            moves += p.moves
            numerical_data += p.stats
            numerical_data.append(p.current_hp)
            for p_type in self.pokemon_types[p.name]:
                types[counter * 18 + self.type_map[p_type]] = 1
            counter += 1

        for p in opponent_team.pokemon:
            moves += p.moves
            numerical_data += p.stats
            numerical_data.append(p.current_hp)
            for p_type in self.pokemon_types[p.name]:
                types[counter * 18 + self.type_map[p_type]] = 1
            counter += 1
        
        ability_embedding_output = torch.stack([self.ability_embedding(ability) for ability in abilities])
        # 12 total abilities, 1 per pokemon
        ability_embedding_output = ability_embedding_output.view(-1, 12 * EMBEDDING_DIM)

        move_embedding_output = torch.stack([self.move_embedding(move) for move in moves])
        # 12 pokemon * 4 moves each
        move_embedding_output = move_embedding_output.view(-1, 48 * EMBEDDING_DIM)
        
        numerical_data = torch.tensor(numerical_data).view(-1, 84)
        types = types.view(-1, 18 * 12)
        
        combined_input = torch.cat([ability_embedding_output, move_embedding_output, numerical_data, types], dim=-1)
        processed_features = self.hidden_layer(combined_input)

        action_logits = self.action_fc(processed_features)
        action_probs = nn.Softmax(dim=-1)(action_logits)

        return action_probs