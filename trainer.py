from battler import Battler
from model.model_actor import ModelActor
from player_actor import RandomActor, DefaultActor, PlayerActor
from utils import simulate_games
import torch
import torch.optim as optim

def train(model_actor, random_actor, gamma, num_episodes, optimizer, seed=None):
    score = {'BOT_1': 0, 'BOT_2': 0}
    # max_possible_reward = 600  # sort of a loose estimate, can adjust if we want to try normalizing rewards
    ############## Implementing Policy Gradients (REINFORCE ALGORITHM) below ##############

    # Training Loop
    for episode in range(num_episodes):
        battler = Battler(model_actor, random_actor, seed)
        battler.make_moves() # Initial game setup hack so p1move and p2move are not both False

        training_samples = []  # store relevant metadata from a single episode / battle

        ############## PLAY AN ENTIRE EPISODE / BATTLE AND RECORD EACH INTERACTION ##############
        while battler.current_state != 'end':
            prob_action = None  # probability of action chosen by model in current game state
            reward = None  # reward received after executing actions in current game state

            original_total_hp_model_actor = model_actor.team.calculate_total_HP()
            original_total_hp_actor_2 = battler.actor2.team.calculate_total_HP()

            if battler.p1move:
                knowledge = {"field": None, "opponent": battler.actor2.team, "error": battler.error}
                action = model_actor.pick_move(knowledge)

                action_type, action_num = action.split(" ")
                action_index = None
                if (action_type == "move"):
                    action_index = int(action_num) - 1
                else:
                    action_index = int(action_num) + 2

                prob_action = model_actor.prev_probs[0][action_index]
                # Earlier, I noticed that model's prev_probs is NAN during some runs (model updating weights in strange ways perhaps?)
                # Anyway, this issue appears to be mitigated for now, but leaving this check here for the future!
                if (torch.isnan(model_actor.prev_probs).any()):
                    print("Model's prev_probs contains NAN values")

                move = f">p1 {action}"
                # print(move)
                battler.send_command(move)
                battler.p1move = False

            if battler.p2move:
                knowledge = {"field": None, "opponent": model_actor.team, "error": battler.error}
                move = f">p2 {battler.actor2.pick_move(knowledge)}"
                # print(move)
                battler.send_command(move)
                battler.p2move = False

            battler.receive_output()

            # TODO: Improve reward heuristic
            # Calculate reward based on damage exchanged (considering full teams in case a switch move occurred)
            win = 0
            if battler.current_state == 'end':
                win = 100 if battler.winner == 'BOT_1' else -100
            damage_dealt = (original_total_hp_actor_2 -
                            battler.actor2.team.calculate_total_HP())
            damage_received = (original_total_hp_model_actor -
                               model_actor.team.calculate_total_HP())

            opponent_ko = original_total_hp_actor_2 > 0 and battler.actor2.team.active_fainted()
            self_ko = original_total_hp_model_actor > 0 and model_actor.team.active_fainted()

            win_weight = min(1.0, episode / 500)  # Scale win reward over first 500 episodes
            damage_weight = max(.01, 1.0 - win_weight)

            reward = (
                damage_weight * ((damage_dealt - damage_received)  # Immediate reward
                                 + (50 if opponent_ko else 0)  # Reward for KO
                                 - (50 if self_ko else 0))  # Penalty for losing a Pokémon
                + win_weight * win
            )
            # Normalize reward? In practice it doesn't seem to improve results or help convergence
            # reward = max(-1, min(1, reward / max_possible_reward))
            # print(reward)

            # Save each sample of observed experience as training data
            training_samples.append((prob_action, reward))

        ############## USE PREVIOUS BATTLE EXPERIENCES TO TUNE NETWORK WEIGHTS ##############

        # Compute discounted returns using gamma
        # TODO: Cross-check discounted returns formula since it directly impacts loss calculation. Normalize rewards?
        discounted_returns = []
        discounted_reward = 0
        for sample in training_samples[::-1]:
            _, reward = sample
            discounted_reward = reward + discounted_reward * gamma
            discounted_returns.append(discounted_reward)
        discounted_returns = discounted_returns[::-1]

        total_loss = 0
        invalid_samples = 0

        for i, sample in enumerate(training_samples):
            prob_action, _ = sample
            current_discounted_return = discounted_returns[i]
            # TODO: Can we somehow reduce # of invalid samples generated during battle?
            # prob_action is None when it is not p1's move so maybe it is okay to ignore these cases?
            # It is interesting that this number fluctuates greatly though!
            if (prob_action is None):
                invalid_samples += 1
                continue

            # Important Observations:
            # (1) High loss values could mean prob_action is close to 0 often (poor model convergence?)
            # (2) Hacky fix for now is to add 1e-10 to prob_action to avoid taking log of 0.
            loss = -torch.log(prob_action + 1e-10) * current_discounted_return

            total_loss = total_loss + loss  # negative loss values are due to negative rewards

        valid_samples = len(training_samples) - invalid_samples

        # This check is done as sometimes training samples are too few / invalid causing total_loss to be 0
        if (total_loss != 0 and not torch.isnan(total_loss)):
            mean_loss = total_loss / valid_samples
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
        else:
            print("not updating loss due to invalid loss")

        if episode % 10 == 0:
            print(f"Game {episode} finished with mean loss = {(mean_loss):.2f}, % of invalid samples = {(invalid_samples / len(training_samples) * 100):.2f}, score{score}")
        score[battler.winner] += 1

    print("Score during traning: ")
    print(score)
    print(f"Win rate for BOT_1: {(score['BOT_1'] / num_episodes * 100):.2f}")
    print(f"Win rate for BOT_2: {(score['BOT_2'] / num_episodes * 100):.2f}")

############## Training ##############


# Hyperparameters
learning_rate = 0.001
gamma = 0.99
num_episodes = 1000

model_actor = ModelActor(None, model="CustomPokemonModelB")
random_actor = RandomActor(None)
default_actor = DefaultActor(None)
player_actor = PlayerActor(None)
comments = "random, lstm, masking used"

optimizer = optim.Adam(model_actor.custom_pokemon_model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model_actor.custom_pokemon_model.parameters() if p.requires_grad)
print(f"Total trainable params in custom pokemon model: {total_params}")

train(model_actor, default_actor, gamma, num_episodes, optimizer, seed=[5,5,5,5])

############## Validation ##############

print("\nScore during validation: ")

simulate_games(model_actor, default_actor, 100)
# useful for comparing when running simultaneous training sims
print(f"learning rate: {learning_rate} \n gamma: {gamma} \n num episodes: {num_episodes} \n comments: {comments}")