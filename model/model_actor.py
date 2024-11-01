from battler import Actor, Team
from model.ability_embedding import AbilityEmbedding
from model.move_embedding import MoveEmbedding
import torch
import torch.nn as nn

#
EMBEDDING_DIM = 5
HIDDEN_DIM = 64


class ModelActor(Actor):
    def __init__(self, team: Team):
        super().__init__(team)

        self.ability_embedding = AbilityEmbedding()
        self.move_embedding = MoveEmbedding()

        self.numerical_fc = nn.Linear(7 * 12, EMBEDDING_DIM)  # 7 stats for each pokemon on both teams

        self.hidden_layer = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * (12 + 48 + 1), HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)
        )

        self.action_fc = nn.Linear(EMBEDDING_DIM, 9)

        self.prev_probs = None
        self.errors = 0

    def pick_move(self, knowledge) -> str:
        # knowledge['opponent'].print_short_info()
        # self.team.print_short_info()

        # error handling - if we receive any error from our prev choice we move to the next best choice
        # this isn't ideal because we can run into certain errors that wouldn't require making a new choice, or making a good choice
        # ex. I've seen "invalid move - it's not your turn" in some of the logs, in which case we can keep the same move
        # or trying to make a move with a fainted pokemon - we should immediately choose the best "switch" action
        if knowledge['error']:
            self.errors += 1
            # using % 9 is a very hacky fix that needs to be removed.
            # self.errors should never exceed 8 if we handle the errors described above
            choice = self.prev_probs[self.errors % 9]

            if choice < 4:
                return f'move {choice + 1}'
            else:
                return f'switch {choice - 2}'
        # once we stop erroring we can reset the error index to 0
        self.errors = 0

        abilities = [p.ability for p in self.team.pokemon] + [p.ability for p in knowledge['opponent'].pokemon]

        moves = []
        numerical_data = []

        for p in self.team.pokemon:
            moves += p.moves
            numerical_data += p.stats
            numerical_data.append(p.current_hp)

        for p in knowledge['opponent'].pokemon:
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

        # The output is 9 numbers which we're taking to represent the probability of choosing an action
        # indices 0-3 represent moves 1-4 of the active pokemon
        # indices 4-8 represent switching to pokemon 2-6

        # Using topk basically just to sort the list, but we grab the original indices to store
        choices = torch.topk(action_probs, 9).indices[0]
        # prev_probs[0] is the highest prob choice
        choice = choices[0]
        # we store the entire distribution of probs in prev_probs, so that on error we can try again with the next best choice
        self.prev_probs = choices

        if choice < 4:
            return f'move {choice + 1}'
        else:
            return f'switch {choice - 2}'
