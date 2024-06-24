import torch

from game.wrapped_flappy_bird import GameState
from src.preprocess import image_to_tensor, resize_and_bgr2gray
from src.tools import DEVICE


def test(model):
    game_state = GameState(is_model=True)
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32).to(DEVICE)
        action_index = torch.argmax(output).to(DEVICE)
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        state = state_1
