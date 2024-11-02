from battler import Battler
from model.model_actor import ModelActor
from player_actor import RandomActor
from utils import simulate_games
import random
import torch.optim as optim

def train(model_actor, random_actor, gamma, epsilon, epsilon_decay, min_epsilon, num_episodes, optimizer):
    score = {'BOT_1': 0, 'BOT_2': 0}
    possible_actions = [f"move {i}" for i in range(1, 5)] + [f"switch {i}" for i in range(2, 7)]
    
    # Training Loop
    for episode in range(num_episodes):
        battler = Battler(model_actor, random_actor)
        battler.make_moves() # Initial game setup hack so p1move and p2move are not both False

        avg_loss = 0
        iterations = 0

        while battler.current_state != 'end':
            original_total_hp_actor_1 = battler.actor1.team.calculate_total_HP()
            original_total_hp_actor_2 = battler.actor2.team.calculate_total_HP()

            action_probs = model_actor.custom_pokemon_model.forward(battler.actor1.team, battler.actor2.team)
            current_q_value = action_probs.max()
            
            if battler.p1move:
                knowledge = {"field": None, "opponent": battler.actor2.team, "error": battler.error}
                # Epsilon-greedy policy for exploration
                # In the beginning, model actor is essentially picking random moves. Over time, it uses its learned
                # strategy more and more as epsilon decays. 
                if random.random() < epsilon:
                    move = f">p1 {random.choice(possible_actions)}"
                else:
                    move = f">p1 {battler.actor1.pick_move(knowledge)}"
                # print(move)
                battler.send_command(move)
                battler.p1move = False

            if battler.p2move:
                knowledge = {"field": None, "opponent": battler.actor1.team, "error": battler.error}
                move = f">p2 {battler.actor2.pick_move(knowledge)}"
                # print(move)
                battler.send_command(move)
                battler.p2move = False
            
            battler.receive_output()

            # Calculate reward based on damage exchanged (considering full teams in case a switch move occurred)
            damage_dealt = battler.actor2.team.calculate_total_HP() - original_total_hp_actor_2
            damage_recieved = battler.actor1.team.calculate_total_HP() - original_total_hp_actor_1
            reward = damage_dealt - damage_recieved
            # print(reward)

            next_probs = model_actor.custom_pokemon_model.forward(battler.actor1.team, battler.actor2.team)
            # TODO: figure out better / more robust way to calculate Q value instead of relying on action probs only
            target_q_value = reward + (gamma * next_probs.max().detach())
            
            # TODO: currently using MSE loss but we should probably refine this
            loss = (current_q_value - target_q_value) ** 2
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterations += 1
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay) # Decay epsilon
        if episode % 10 == 0:
            print(f"Episode {episode}, Epsilon {epsilon:.3f}")
            print(f"Game {episode} finished with avg. loss of {avg_loss / iterations}")
        
        score[battler.winner] += 1

    print("Score during traning: ")
    print(score)
    print(f"Win rate for BOT_1: {(score['BOT_1'] / num_episodes * 100):.2f}")
    print(f"Win rate for BOT_2: {(score['BOT_2'] / num_episodes * 100):.2f}")

############## Training ##############

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01
num_episodes = 100

model_actor = ModelActor(None)
random_actor = RandomActor(None)
optimizer = optim.Adam(model_actor.custom_pokemon_model.parameters(), lr=learning_rate)

train(model_actor, random_actor, gamma, epsilon, epsilon_decay, min_epsilon, num_episodes, optimizer)

############## Validation ##############

print("\nScore during validation: ")
simulate_games(model_actor, random_actor, 100)