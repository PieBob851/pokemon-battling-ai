from battler import Actor, Team
from model.custom_pokemon_model import CustomPokemonModel
from model.custom_pokemon_model_b import CustomPokemonModelB
import torch

class ModelActor(Actor):
    def __init__(self, team: Team, model="CustomPokemonModel"):
        super().__init__(team)
        
        self.model = model
        if self.model == "CustomPokemonModel":
            self.custom_pokemon_model = CustomPokemonModel()
        else:
            self.custom_pokemon_model = CustomPokemonModelB()
        self.prev_probs = None
        self.prev_probs_sorted = None
        self.errors = 0
        self.hidden_states = None

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
            choice = self.prev_probs_sorted[self.errors % 9]

            if choice < 4:
                return f'move {choice + 1}'
            else:
                return f'switch {choice - 2}'
        # once we stop erroring we can reset the error index to 0
        self.errors = 0
        
        if self.model == "CustomPokemonModel":
            action_probs, self.hidden_states = self.custom_pokemon_model.forward(
                self.team, knowledge['opponent'], self.hidden_states)
        else:
            action_probs = self.custom_pokemon_model.forward(self.team, knowledge['opponent'])

        # The output is 9 numbers which we're taking to represent the probability of choosing an action
        # indices 0-3 represent moves 1-4 of the active pokemon
        # indices 4-8 represent switching to pokemon 2-6

        # Using topk basically just to sort the list, but we grab the original indices to store
        choices = torch.topk(action_probs, 9).indices[0]
        # prev_probs[0] is the highest prob choice
        choice = choices[0]
        # we store the entire distribution of probs in prev_probs, so that on error we can try again with the next best choice
        self.prev_probs_sorted = choices
        self.prev_probs = action_probs

        if choice < 4:
            return f'move {choice + 1}'
        else:
            return f'switch {choice - 2}'
