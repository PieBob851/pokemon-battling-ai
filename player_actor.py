from battler import Pokemon, Team, Actor, Battler
import random
from model.model_actor import ModelActor

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


score = {'BOT_1': 0, 'BOT_2': 0}

for i in range(100):
    actor1 = ModelActor(None)
    actor2 = RandomActor(None)

    battler = Battler(actor1, actor2)

    iteration = 0
    while battler.current_state != 'end':
        battler.make_moves()
        # print("iteration:", iteration, battler.current_state)
        iteration += 1
    score[battler.winner] += 1
    print(f'Game {i} finished')

print(score)
