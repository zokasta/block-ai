import pygame
import sys
import numpy as np
import random
import time
import pickle
import os
import multiprocessing

# --- Main Configuration ---
GRID_LAYOUT = (3, 3)
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
INFO_PANEL_HEIGHT = 220

# --- Sub-Game Configuration ---
GRID_SIZE = 9
SUB_GAME_WIDTH = WINDOW_WIDTH // GRID_LAYOUT[1]
SUB_GAME_HEIGHT = (WINDOW_HEIGHT - INFO_PANEL_HEIGHT) // GRID_LAYOUT[0]
CELL_SIZE = min(SUB_GAME_WIDTH // (GRID_SIZE + 2), SUB_GAME_HEIGHT // (GRID_SIZE + 4))
BOARD_MARGIN_X = (SUB_GAME_WIDTH - GRID_SIZE * CELL_SIZE) // 2
BOARD_MARGIN_Y = 20

# --- Colors ---
BACKGROUND = (28, 33, 40)
GRID_COLOR = (50, 58, 69)
EMPTY_CELL_COLOR = (40, 46, 56)
TEXT_COLOR = (230, 230, 230)
PIECE_COLORS = [(52, 152, 219), (231, 76, 60), (114, 227, 137), (241, 196, 15), (155, 89, 182)]
GRAPH_LINE_COLOR = (52, 152, 219)
INFO_PANEL_BG = (40, 46, 56)
INPUT_BOX_ACTIVE_COLOR = (230, 230, 230)
INPUT_BOX_INACTIVE_COLOR = (100, 100, 100)
POSITIVE_WEIGHT_COLOR = (52, 152, 219)
NEGATIVE_WEIGHT_COLOR = (231, 76, 60)
NEURON_BORDER_COLOR = (150, 150, 150)

# --- Piece Definitions ---
PIECES = {
    'I2': [(0, 0), (1, 0)], 'I3': [(0, 0), (1, 0), (2, 0)], 'I4': [(0, 0), (1, 0), (2, 0), (3, 0)],
    'L': [(0, 0), (1, 0), (2, 0), (2, 1)], 'Dot': [(0, 0)], 'Box': [(0, 0), (0, 1), (1, 0), (1, 1)],
    'Corner': [(0, 0), (0, 1), (1, 0)],
}

# --- Genetic Algorithm Configuration ---
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.5

# --- Neural Network Configuration ---
INPUT_NEURONS = GRID_SIZE * GRID_SIZE # 81 inputs, one for each cell
HIDDEN_NEURONS = 24
OUTPUT_NEURONS = 1 # A single "goodness" score for the board

def lerp_color(c1, c2, t):
    t = max(0, min(1, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )

class NeuralNetwork:
    """A simple feed-forward neural network for evaluating board states."""
    def __init__(self, weights=None):
        if weights:
            self.weights_ih = weights["w_ih"].copy()
            self.weights_ho = weights["w_ho"].copy()
            self.bias_h = weights["b_h"].copy()
            self.bias_o = weights["b_o"].copy()
        else:
            self.weights_ih = np.random.randn(HIDDEN_NEURONS, INPUT_NEURONS)
            self.weights_ho = np.random.randn(OUTPUT_NEURONS, HIDDEN_NEURONS)
            self.bias_h = np.random.randn(HIDDEN_NEURONS, 1)
            self.bias_o = np.random.randn(OUTPUT_NEURONS, 1)
        self.activations = {}

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))

    def predict(self, input_array):
        inputs = np.array(input_array, ndmin=2).T
        hidden = self.sigmoid(np.dot(self.weights_ih, inputs) + self.bias_h)
        output = self.sigmoid(np.dot(self.weights_ho, hidden) + self.bias_o)
        self.activations = {'input': inputs, 'hidden': hidden, 'output': output}
        return output[0, 0]

    def mutate(self):
        def _mutate(matrix):
            mask = np.random.random(matrix.shape) < MUTATION_RATE
            mutation = np.random.randn(*matrix.shape) * MUTATION_STRENGTH
            return matrix + (mutation * mask)
        self.weights_ih, self.weights_ho, self.bias_h, self.bias_o = map(_mutate, [self.weights_ih, self.weights_ho, self.bias_h, self.bias_o])

    def get_weights_for_saving(self):
        return {"w_ih": self.weights_ih, "w_ho": self.weights_ho, "b_h": self.bias_h, "b_o": self.bias_o}

class Piece:
    """Represents a single block piece."""
    def __init__(self, shape_name):
        self.shape_name = shape_name
        self.shape = PIECES[shape_name]
        self.color_index = random.randint(0, len(PIECE_COLORS) - 1)
        self.color = PIECE_COLORS[self.color_index]

class Board:
    """Manages the 9x9 grid state."""
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    def is_valid_placement(self, piece, row, col):
        for r_offset, c_offset in piece.shape:
            r, c = row + r_offset, col + c_offset
            if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and self.grid[r, c] == 0):
                return False
        return True

    def place_piece(self, piece, row, col):
        for r_offset, c_offset in piece.shape:
            self.grid[row + r_offset, col + c_offset] = piece.color_index + 1

    def check_and_clear(self):
        score = 0
        to_clear = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        lines, cols, squares = 0, 0, 0
        for i in range(GRID_SIZE):
            if np.all(self.grid[i, :]): to_clear[i, :] = True; lines += 1
            if np.all(self.grid[:, i]): to_clear[:, i] = True; cols += 1
        for r in range(0, GRID_SIZE, 3):
            for c in range(0, GRID_SIZE, 3):
                if np.all(self.grid[r:r+3, c:c+3]): to_clear[r:r+3, c:c+3] = True; squares += 1
        cleared_count = np.sum(to_clear)
        if cleared_count > 0:
            base_score = (lines + cols) * 10 + squares * 20
            combo_multiplier = 1 + (lines + cols + squares - 1) * 0.5
            score = base_score * combo_multiplier + cleared_count
            self.grid[to_clear] = 0
        return score

class AI:
    """Neuroevolution AI. Its brain is a neural network that scores board states."""
    def __init__(self, brain=None):
        self.brain = brain or NeuralNetwork()

    def mutate(self):
        self.brain.mutate()

    def get_best_move(self, board, pieces):
        best_move_info, best_score = None, -float('inf')
        for i, piece in enumerate(pieces):
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if board.is_valid_placement(piece, r, c):
                        temp_board = Board()
                        temp_board.grid = board.grid.copy()
                        temp_board.place_piece(piece, r, c)
                        board_state_flat = temp_board.grid.flatten() / (len(PIECE_COLORS) + 1) # Normalize
                        score = self.brain.predict(board_state_flat)
                        if score > best_score:
                            best_score = score
                            best_move_info = (i, (r, c))
        return best_move_info

class GameInstance:
    """Manages the state of a single puzzle game."""
    def __init__(self, ai):
        self.ai = ai
        self.board = Board()
        self.score = 0
        self.moves_made = 0
        self.game_over = False
        self.generate_new_pieces()

    def generate_new_pieces(self):
        self.available_pieces = [Piece(random.choice(list(PIECES.keys()))) for _ in range(3)]

    def apply_move(self, move_info):
        if move_info:
            piece_index, (r, c) = move_info
            if piece_index < len(self.available_pieces):
                piece = self.available_pieces.pop(piece_index)
                self.board.place_piece(piece, r, c)
                self.score += self.board.check_and_clear()
                self.moves_made += 1
                if not self.available_pieces:
                    self.generate_new_pieces()
            else: self.game_over = True
        else: self.game_over = True

def run_ai_turn(args):
    """A top-level function for multiprocessing to call."""
    ai, board, pieces = args
    return ai.get_best_move(board, pieces)

class MainGame:
    """Manages the overall window, multiple environments, and UI."""
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.big_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.running = True
        self.game_speed = 60
        self.speed_input_text = str(self.game_speed)
        self.speed_input_active = False
        self.speed_input_rect = pygame.Rect(0, 0, 80, 28)
        self.mode = "train"
        self.champion_ai = None
        self.generation = 1
        self.generation_history = []
        self.population = [AI() for _ in range(GRID_LAYOUT[0] * GRID_LAYOUT[1])]
        self.games = [GameInstance(ai) for ai in self.population]
        self.vis_mode = False # New visualization mode flag

    def run(self):
        with multiprocessing.Pool() as pool:
            while self.running:
                self.handle_events()
                self.update(pool)
                self.draw()
                self.clock.tick(self.game_speed)
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.speed_input_active = self.speed_input_rect.collidepoint(event.pos)
            if event.type == pygame.KEYDOWN:
                if self.speed_input_active:
                    if event.key == pygame.K_RETURN:
                        try: self.game_speed = max(1, int(self.speed_input_text))
                        except ValueError: self.speed_input_text = str(self.game_speed)
                        self.speed_input_active = False
                    elif event.key == pygame.K_BACKSPACE: self.speed_input_text = self.speed_input_text[:-1]
                    elif event.unicode.isdigit(): self.speed_input_text += event.unicode
                else:
                    if event.key == pygame.K_s: self.save_champion()
                    elif event.key == pygame.K_l: self.load_champion()
                    elif event.key == pygame.K_v: self.vis_mode = not self.vis_mode # Toggle visualization
                    elif event.key == pygame.K_ESCAPE: self.running = False

    def update(self, pool):
        if self.mode == "train":
            tasks = [(game.ai, game.board, game.available_pieces) for game in self.games if not game.game_over]
            results = pool.map(run_ai_turn, tasks)
            alive_games = [game for game in self.games if not game.game_over]
            for game, move_info in zip(alive_games, results):
                game.apply_move(move_info)
            if all(game.game_over for game in self.games):
                self.next_generation()
        elif self.mode == "play":
            game = self.games[0]
            if not game.game_over:
                move_info = game.ai.get_best_move(game.board, game.available_pieces)
                game.apply_move(move_info)
            else:
                self.games = [GameInstance(self.champion_ai)]

    def next_generation(self):
        champion_game = max(self.games, key=lambda g: g.score)
        champion_ai = champion_game.ai
        champion_score = champion_game.score
        self.champion_ai = champion_ai
        self.generation_history.append((self.generation, champion_score))
        self.population = []
        for _ in range(GRID_LAYOUT[0] * GRID_LAYOUT[1]):
            new_ai = AI(brain=NeuralNetwork(weights=champion_ai.brain.get_weights_for_saving()))
            new_ai.mutate()
            self.population.append(new_ai)
        self.games = [GameInstance(ai) for ai in self.population]
        self.generation += 1

    def save_champion(self, filename="best_block_ai.pkl"):
        if not self.champion_ai: return
        try:
            with open(filename, "wb") as f: pickle.dump(self.champion_ai.brain.get_weights_for_saving(), f)
            print(f"Saved champion AI to {filename}")
        except Exception as e: print(f"Error saving AI: {e}")

    def load_champion(self, filename="best_block_ai.pkl"):
        if not os.path.exists(filename): return
        try:
            with open(filename, "rb") as f: weights = pickle.load(f)
            loaded_brain = NeuralNetwork(weights=weights)
            self.champion_ai = AI(brain=loaded_brain)
            self.games = [GameInstance(self.champion_ai)]
            self.mode = "play"
            self.vis_mode = True # Automatically switch to vis mode when loading
            print(f"Loaded champion AI. Switched to Play Mode.")
        except Exception as e: print(f"Error loading AI: {e}")

    def draw(self):
        self.screen.fill(BACKGROUND)
        if self.vis_mode:
            self.draw_large_visualization()
        elif self.mode == "train":
            for i, game in enumerate(self.games):
                row, col = divmod(i, GRID_LAYOUT[1])
                self.draw_game_instance(game, col * SUB_GAME_WIDTH, row * SUB_GAME_HEIGHT, is_preview=True)
        else:
            self.draw_game_instance(self.games[0], 0, 0, is_preview=False)
        self.draw_info_panel()
        pygame.display.flip()

    def draw_game_instance(self, game, x_offset, y_offset, is_preview):
        cell_size = CELL_SIZE if is_preview else 60
        board_margin_x = BOARD_MARGIN_X if is_preview else (WINDOW_WIDTH - GRID_SIZE * cell_size) // 2
        board_margin_y = BOARD_MARGIN_Y if is_preview else 50
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(x_offset + board_margin_x + c * cell_size, y_offset + board_margin_y + r * cell_size, cell_size, cell_size)
                color_index = game.board.grid[r, c]
                color = PIECE_COLORS[color_index - 1] if color_index > 0 else EMPTY_CELL_COLOR
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)
        piece_y = y_offset + board_margin_y + GRID_SIZE * cell_size + 10
        for i, piece in enumerate(game.available_pieces):
            self.draw_piece_preview(piece, x_offset + board_margin_x + i * (cell_size * 2.5), piece_y, cell_size * 0.5)
        score_text = self.font.render(f"Score: {game.score}", True, TEXT_COLOR)
        self.screen.blit(score_text, (x_offset + board_margin_x, y_offset + 5))
        moves_text = self.font.render(f"Moves: {game.moves_made}", True, TEXT_COLOR)
        self.screen.blit(moves_text, (x_offset + board_margin_x + 150, y_offset + 5))

    def draw_piece_preview(self, piece, x, y, preview_cell_size):
        for r_off, c_off in piece.shape:
            rect = pygame.Rect(x + c_off * preview_cell_size, y + r_off * preview_cell_size, preview_cell_size, preview_cell_size)
            pygame.draw.rect(self.screen, piece.color, rect, border_radius=2)

    def draw_info_panel(self):
        panel_y = WINDOW_HEIGHT - INFO_PANEL_HEIGHT
        pygame.draw.rect(self.screen, INFO_PANEL_BG, (0, panel_y, WINDOW_WIDTH, INFO_PANEL_HEIGHT))
        y_pos = panel_y + 10
        if self.mode == "train":
            gen_text = self.font.render(f"Generation: {self.generation}", True, TEXT_COLOR)
            self.screen.blit(gen_text, (20, y_pos))
            high_score = max(h[1] for h in self.generation_history) if self.generation_history else 0
            score_text = self.font.render(f"All-Time High Score: {high_score}", True, TEXT_COLOR)
            self.screen.blit(score_text, (250, y_pos))
            self.draw_graph(panel_y + 50)
        else:
            title_text = self.big_font.render("Champion AI Playing", True, TEXT_COLOR)
            self.screen.blit(title_text, (20, y_pos))
            score_text = self.big_font.render(f"Current Score: {self.games[0].score}", True, TEXT_COLOR)
            self.screen.blit(score_text, (20, y_pos + 40))
            controls_text = self.font.render("Press 'L' to return to training mode.", True, TEXT_COLOR)
            self.screen.blit(controls_text, (20, y_pos + 80))
        
        self.speed_input_rect.topleft = (WINDOW_WIDTH - 200, panel_y + 10)
        speed_label = self.font.render("FPS:", True, TEXT_COLOR)
        self.screen.blit(speed_label, (self.speed_input_rect.x - 50, panel_y + 15))
        border_color = INPUT_BOX_ACTIVE_COLOR if self.speed_input_active else INPUT_BOX_INACTIVE_COLOR
        pygame.draw.rect(self.screen, border_color, self.speed_input_rect, 2, border_radius=3)
        input_surf = self.font.render(self.speed_input_text, True, TEXT_COLOR)
        self.screen.blit(input_surf, (self.speed_input_rect.x + 5, self.speed_input_rect.y + 5))

    def draw_graph(self, y_pos):
        graph_x, graph_width = 20, WINDOW_WIDTH - 40
        graph_height = INFO_PANEL_HEIGHT - 60
        if len(self.generation_history) > 1:
            points = []
            max_score = max(h[1] for h in self.generation_history) if self.generation_history else 1
            max_gen = self.generation_history[-1][0]
            for gen, score in self.generation_history:
                x = graph_x + ((gen - 1) / max(1, max_gen - 1)) * graph_width
                y = y_pos + graph_height - (score / max(1, max_score)) * graph_height
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(self.screen, GRAPH_LINE_COLOR, False, points, 2)

    def draw_large_visualization(self):
        if not self.champion_ai:
            title_text = self.big_font.render("No Champion AI yet. Let one generation finish.", True, TEXT_COLOR)
            self.screen.blit(title_text, title_text.get_rect(center=(WINDOW_WIDTH/2, (WINDOW_HEIGHT-INFO_PANEL_HEIGHT)/2)))
            return
        
        title_text = self.big_font.render("Champion AI Brain (Press 'V' to close)", True, TEXT_COLOR)
        self.screen.blit(title_text, title_text.get_rect(center=(WINDOW_WIDTH/2, 50)))
        
        brain = self.champion_ai.brain
        x_offset, y_offset = 0, 100
        vis_height = WINDOW_HEIGHT - INFO_PANEL_HEIGHT - y_offset
        
        input_y_step, hidden_y_step, output_y_step = [vis_height / (n + 1) for n in [INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS]]
        input_x, hidden_x, output_x = x_offset + 150, x_offset + WINDOW_WIDTH / 2, x_offset + WINDOW_WIDTH - 150
        neuron_radius = 4

        input_act, hidden_act, output_act = [brain.activations.get(k, np.zeros((s, 1))) for k, s in [('input', INPUT_NEURONS), ('hidden', HIDDEN_NEURONS), ('output', OUTPUT_NEURONS)]]
        
        for i, w_row in enumerate(brain.weights_ih):
            for j, weight in enumerate(w_row):
                if abs(weight) > 0.1: # Only draw significant connections
                    start_pos, end_pos = (input_x, y_offset + (j + 1) * input_y_step), (hidden_x, y_offset + (i + 1) * hidden_y_step)
                    color = POSITIVE_WEIGHT_COLOR if weight > 0 else NEGATIVE_WEIGHT_COLOR
                    pygame.draw.aaline(self.screen, color, start_pos, end_pos)
        for i, w_row in enumerate(brain.weights_ho):
            for j, weight in enumerate(w_row):
                start_pos, end_pos = (hidden_x, y_offset + (j + 1) * hidden_y_step), (output_x, y_offset + (i + 1) * output_y_step)
                color = POSITIVE_WEIGHT_COLOR if weight > 0 else NEGATIVE_WEIGHT_COLOR
                width = int(min(4, 1 + abs(weight)))
                pygame.draw.line(self.screen, color, start_pos, end_pos, width)
        
        for i, act in enumerate(input_act):
            pos = (input_x, y_offset + (i + 1) * input_y_step)
            color = lerp_color(BACKGROUND, (255, 255, 255), act[0])
            pygame.draw.circle(self.screen, color, pos, neuron_radius)
        for i, act in enumerate(hidden_act):
            pos = (hidden_x, y_offset + (i + 1) * hidden_y_step)
            color = lerp_color(BACKGROUND, (255, 255, 255), act[0])
            pygame.draw.circle(self.screen, color, pos, neuron_radius * 2)
            pygame.draw.circle(self.screen, NEURON_BORDER_COLOR, pos, neuron_radius * 2, 1)
        for i, act in enumerate(output_act):
            pos = (output_x, y_offset + (i + 1) * output_y_step)
            color = lerp_color(BACKGROUND, (255, 255, 255), act[0])
            pygame.draw.circle(self.screen, color, pos, neuron_radius * 4)
            pygame.draw.circle(self.screen, NEURON_BORDER_COLOR, pos, neuron_radius * 4, 2)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Multi-Grid AI Block Puzzle Evolution")
    game = MainGame(screen)
    game.run()
