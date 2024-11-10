from battler import Battler
from model.model_actor import ModelActor
from player_actor import RandomActor
from utils import simulate_games
import torch
import torch.optim as optim

def train(model_actor, random_actor, gamma, num_episodes, optimizer):
    score = {'BOT_1': 0, 'BOT_2': 0}
    
    ############## Implementing Policy Gradients as decribed below ##############
    # https://towardsdatascience.com/reinforcement-learning-explained-visually-part-6-policy-gradients-step-by-step-f9f448e73754

    # Training Loop
    for episode in range(num_episodes):
        battler = Battler(model_actor, random_actor)
        battler.make_moves() # Initial game setup hack so p1move and p2move are not both False

        gamma_factor = 1
        total_discounted_reward = 0
        training_samples = [] # store all game states and moves experienced by model in a single episode / battle

        ############## PLAY AN ENTIRE EPISODE / BATTLE AND RECORD EACH INTERACTION ##############
        while battler.current_state != 'end':
            model_team = None # model's team in current game state
            opponent_team = None # opponent's team in current game state
            model_move = None # action chosen by model in current game state
            discounted_reward = None # reward received after executing actions in current game state

            original_total_hp_actor_1 = model_actor.team.calculate_total_HP()
            original_total_hp_actor_2 = battler.actor2.team.calculate_total_HP()
            
            if battler.p1move:
                knowledge = {"field": None, "opponent": battler.actor2.team, "error": battler.error}
                model_team = model_actor.team
                model_move = model_actor.pick_move(knowledge)
                move = f">p1 {model_move}"
                # print(move)
                battler.send_command(move)
                battler.p1move = False

            if battler.p2move:
                knowledge = {"field": None, "opponent": model_actor.team, "error": battler.error}
                opponent_team = battler.actor2.team
                move = f">p2 {battler.actor2.pick_move(knowledge)}"
                # print(move)
                battler.send_command(move)
                battler.p2move = False
            
            battler.receive_output()

            # Calculate reward based on damage exchanged (considering full teams in case a switch move occurred)
            damage_dealt = battler.actor2.team.calculate_total_HP() - original_total_hp_actor_2
            damage_recieved = model_actor.team.calculate_total_HP() - original_total_hp_actor_1
            discounted_reward = damage_dealt - damage_recieved
            discounted_reward *= gamma_factor

            total_discounted_reward += discounted_reward
            gamma_factor *= gamma
            # print(discounted_reward)
            
            # Save each sample of observed experience as training data
            state = (model_team, opponent_team)
            training_samples.append((state, model_move, discounted_reward))
        
        ############## USE PREVIOUS BATTLE EXPERIENCES TO TUNE NETWORK WEIGHTS ##############
        total_loss = 0
        invalid_samples = 0

        for sample in training_samples:
            state, action, reward = sample
            team, opponent = state
            if (team is None or opponent is None or action is None):
                invalid_samples += 1
                total_discounted_reward -= reward
                continue

            model_actor.team = team
            knowledge = {"field": None, "opponent": opponent, "error": None}
            model_actor.pick_move(knowledge)

            action_type, action_num = action.split(" ")
            action_index = None
            if (action_type == "move"):
                action_index = int(action_num) - 1
            else:
                action_index = int(action_num) + 2

            prob_action = model_actor.prev_probs[0][action_index]
            if (prob_action > 0):
                loss = -torch.log(prob_action) * total_discounted_reward
            else: 
                # prev_probs is nan during some runs causing prob_action to be 0 / nan
                # a little debugging shows that model is probably updating weights in strange ways
                # so we need to refine the back prop step.
                print(model_actor.prev_probs)
                loss = 0
            total_loss += loss
            total_discounted_reward -= reward

        # The check below is done as sometimes: 
        # (1) training samples are too few and all of them are invalid, OR
        # (2) losses are 0 due to prev_probs being nan (see else block above)
        if (total_loss != 0):
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if episode % 10 == 0:
            print(f"Game {episode} finished with total loss = {total_loss}, % of invalid samples = {(invalid_samples / len(training_samples) * 100):.2f}")
        
        score[battler.winner] += 1

    print("Score during traning: ")
    print(score)
    print(f"Win rate for BOT_1: {(score['BOT_1'] / num_episodes * 100):.2f}")
    print(f"Win rate for BOT_2: {(score['BOT_2'] / num_episodes * 100):.2f}")

############## Training ##############

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
num_episodes = 100

model_actor = ModelActor(None)
random_actor = RandomActor(None)
optimizer = optim.Adam(model_actor.custom_pokemon_model.parameters(), lr=learning_rate)

train(model_actor, random_actor, gamma, num_episodes, optimizer)

############## Validation ##############

print("\nScore during validation: ")
simulate_games(model_actor, random_actor, 100)