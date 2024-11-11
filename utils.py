from battler import Battler
from model.model_actor import ModelActor
from player_actor import RandomActor

def simulate_games(actor1, actor2, n=100):
    score = {'BOT_1': 0, 'BOT_2': 0}

    for i in range(n):
        battler = Battler(actor1, actor2)

        iteration = 0
        while battler.current_state != 'end':
            battler.make_moves()
            # print("iteration:", iteration, battler.current_state)
            iteration += 1
        score[battler.winner] += 1
        if (i % 10 == 0):
            print(f'Game {i} finished')

    print(score)
    print(f"Win rate for BOT_1: {(score['BOT_1'] / n * 100):.2f}")
    print(f"Win rate for BOT_2: {(score['BOT_2'] / n * 100):.2f}")

# simulate_games(ModelActor(None), RandomActor(None), 100)