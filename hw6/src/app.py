import pygame
from pygame.locals import *
import sys
import wandb
import torch
import time

from game.wrapped_flappy_bird import GameState
from src.model import NeuralNetwork
from src.preprocess import init_weights
from src.tools import DEVICE
from src.test import test
from src.train import train


def main(mode):
    if mode == 'test':
        if DEVICE == torch.device('cuda'):
            model = torch.load('model.pth').eval()
            model.to(DEVICE)
        else:
            model = torch.load('model.pth', map_location='cpu').eval()
        test(model)
    elif mode == 'train':
        model = NeuralNetwork()
        model.to(DEVICE)
        model.apply(init_weights)

        wandb.init(project=f"hw6", name="flappy_bird")

        start = time.time()
        train(model, start)


def play_self():
    action_terminal = GameState(is_model=False)
    while True:
        input_actions = [1, 0]
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                input_actions = [0, 1]
            else:
                input_actions = [1, 0]
        action_terminal.frame_step(input_actions)


if __name__ == "__main__":
    main('test')
