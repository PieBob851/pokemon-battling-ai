from battler import Team, Actor
import random

possible_actions = [f"move {i}" for i in range(1, 5)] + [f"switch {i}" for i in range(2, 7)]


class PlayerActor(Actor):
    def __init__(self, team: Team):
        super().__init__(team)

    def pick_move(self, knowledge) -> str:
        knowledge['opponent'].print_short_info()
        self.team.print_short_info()
        move_choice = input()
        return move_choice


class RandomActor(Actor):
    # Random moves picked; useful for quick testing
    def __init__(self, team: Team):
        super().__init__(team)

    def pick_move(self, knowledge) -> str:
        return random.choice(possible_actions)

class DefaultActor(Actor):
    def __init__(self, team: Team):
        super().__init__(team)

    def pick_move(self, knowledge) -> str:
        return "default"