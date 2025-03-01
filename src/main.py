"""
temp for debug
"""

import pygame
import sys

from const import *
from field import Field

pygame.init()
field = Field(15, 15, 30)
screen = pygame.display.set_mode((field.width, field.height))
pygame.display.set_caption("Hex Grid")


def draw_field(field, surface):
    font = pygame.font.Font(None, 20)
    for row in field.grid:
        for tile in row:
            pygame.draw.polygon(surface, tile.color, tile.points, 0)
            text = font.render(f"{tile.i}-{tile.j}", True, tile.text_color)
            text_rect = text.get_rect(center=tile.center)
            surface.blit(text, text_rect)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            # field.handleClick(*pos)

    screen.fill(WHITE)
    draw_field(field, screen)
    pygame.display.flip()

pygame.quit()
sys.exit()
