import torch
import wandb
import numpy as np
import random
import time
import torch.nn as nn
import torch.optim as optim

from game.wrapped_flappy_bird import GameState
from src.tools import DEVICE
from src.preprocess import resize_and_bgr2gray, image_to_tensor


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = GameState(is_model=True)

    replay_memory = []
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(DEVICE)

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        action_index.to(DEVICE)

        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(DEVICE)
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(DEVICE)
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(DEVICE)
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).to(DEVICE)

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 100000 == 0:
            torch.save(model, "current_model_" + str(iteration+3) + ".pth")

        if iteration % 100 == 0:
            logs = {'MSE_loss': loss.item(), 'elapsed time': time.time() - start, 'epsilon': epsilon,
                    "action": action_index.cpu().detach().numpy(), "reward": reward.numpy()[0][0],
                    'Q max': np.max(output.cpu().detach().numpy())
                    }
            wandb.log(logs, step=iteration//100)

