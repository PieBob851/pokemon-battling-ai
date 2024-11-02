from battler import Battler
from model.model_actor import ModelActor
from player_actor import RandomActor

def simulate_games(n=100):
    score = {'BOT_1': 0, 'BOT_2': 0}

    for i in range(n):
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

simulate_games(10)