import pymem
import logging
import neat
import numpy as np
import pyautogui  # Ensure you have this library for mouse and keyboard controls
import time
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Global variables to track previous score and resources
previous_score = None
previous_resources = None

# Define base address and offsets for score and resources
BASE_ADDRESS = 0x400000  # Base address of the game module
SCORE_OFFSETS = [0x4C, 0x4, 0x84, 0x1C]  # Score offsets
RESOURCE_OFFSETS = {
    'wood': 0x104,   # Wood offset
    'food': 0x100,   # Food offset
    'gold': 0x10C,   # Gold offset
    'stone': 0x108   # Stone offset
}
HOTKEYS = ['q', 'w', 'e', 'r', 'h', 'a', '.', ',']  # Game hotkeys

def get_score(pm):
    try:
        # Get the base pointer for score
        base_pointer = pm.read_int(BASE_ADDRESS + 0x28E8F8)  # Adjust as necessary
        logging.debug(f"Base Pointer: {hex(base_pointer)}")

        # Follow the pointer chain to get the score
        score_address = base_pointer
        for offset in SCORE_OFFSETS:
            score_address = pm.read_int(score_address + offset)
            logging.debug(f"Score Address after offset {hex(offset)}: {hex(score_address)}, or {score_address}")
            if score_address is None:  # If reading fails
                logging.error("Failed to read the score address.")
                return None
        
        # Return the final score
        return score_address
    
    except pymem.exception.MemoryReadError as e:
        logging.error(f"Memory read error: {e}")
        return None

def get_resources(pm):
    resources = {}
    try:
        # Get the base pointer for resources
        base_pointer = pm.read_int(BASE_ADDRESS + 0x37EBF4)  # Adjust as necessary
        logging.debug(f"Base Pointer: {hex(base_pointer)}")

        # Read each resource using its offset
        for resource, offset in RESOURCE_OFFSETS.items():
            resource_address = base_pointer + offset
            resource_value = pm.read_float(resource_address)  # Read as an integer
            resources[resource] = resource_value
            logging.debug(f"{resource.capitalize()} Address: {hex(resource_address)}, Value: {resource_value}")

        return resources

    except pymem.exception.MemoryReadError as e:
        logging.error(f"Memory read error: {e}")
        return None

# Restart game function
def restart_game():
    logging.info("Restarting game...")
    time.sleep(2)
    pyautogui.click(x=765, y=12)  # Click menu button 
    time.sleep(1)
    pyautogui.click(x=765, y=12)  # Click menu button again
    time.sleep(3)
    pyautogui.click(x=401, y=282)  # Click Restart button 
    time.sleep(3)
    pyautogui.click(x=302, y=327)  # Confirm Yes 
    time.sleep(1)
    pyautogui.click(x=302, y=327)  # Confirm Yes again
    time.sleep(3)  # Allow some time for restart


def evaluate_genome(genomes, config):

    total_genomes = len(genomes)  # Get the total number of genomes
    for genome_id, genome in genomes:  # Iterate over the list of genomes
        # Create the neural network for the current genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        logging.info(f"Evaluating Genome: {genome_id}/{total_genomes}")
        
        # Initialize fitness score for the current genome
        fitness_score = 0

        # Start the timer for 80 seconds
        start_time = time.time()
        elapsed_time = 0

        try:
            pm = pymem.Pymem("age2_x1.exe")  # Attach to the game process

            while elapsed_time < 30:  # Run for 80 seconds
                # Get game data
                score = get_score(pm)
                resources = get_resources(pm)
                logging.info (f"score = {score}")
                logging.info(f"Resources = {resources}")

                # Check if resources and score are valid
                if resources is None or score is None:
                    genome.fitness = 0  # No fitness if we can't read the game data
                    logging.warning(f"Genome ID: {genome_id}, No valid game data")
                    break  # Exit the while loop if data is invalid

                # Prepare inputs for the neural network
                inputs = np.array([score] + list(resources.values()), dtype=float)

                # Normalize inputs if necessary
                if np.max(inputs) > 0:  # Prevent division by zero
                    inputs = inputs / np.max(inputs)  # Normalize inputs

                # Perform a forward pass through the network
                output = net.activate(inputs)

                # Simulate actions based on the output
                perform_actions(output, fitness_score)

                # Update elapsed time
                elapsed_time = time.time() - start_time

            # Calculate fitness after 30 seconds of actions
            fitness_score = calculate_fitness(output, resources)

        except pymem.exception.PymemError as e:
            logging.error(f"Error attaching to process: {e}")
            fitness_score = 0  # No fitness if there is an error

        # Assign the calculated fitness score to the genome
        genome.fitness = fitness_score
        # Log the genome's fitness score for debugging
        logging.info(f"Genome ID: {genome_id}, Fitness: {genome.fitness}")

        restart_game()

    # Return None or aggregate fitness values if needed
    return None

def perform_actions(output, fitness):
    logging.info(f"Output values: {output}")
    # Define safe boundaries to avoid corners
    safe_margin_x = 100  # Margin from the left and right
    safe_margin_y = 100  # Margin from the top and bottom
    
    # Map the normalized output to screen dimensions, excluding safe margins
    mouse_x = int((output[0] * (1920 - 2 * safe_margin_x)) + safe_margin_x)  # Map to screen width
    mouse_y = int((output[1] * (1080 - 2 * safe_margin_y)) + safe_margin_y)  # Map to screen height
    
    hotkey_index = np.argmax(output[2:])  # Get the index of the hotkey to press
    click_type = output[-1]  # Use the last output for deciding left/right click
    
    # Move mouse to (mouse_x, mouse_y) with a smooth motion
    pyautogui.moveTo(mouse_x, mouse_y, duration=0.5)
    logging.info(f"Moved mouse to: ({mouse_x}, {mouse_y})")
    logging.info(f"Output values: {output}") 
    # Press the corresponding hotkey
    if hotkey_index < len(HOTKEYS):
        hotkey = HOTKEYS[hotkey_index]
        pyautogui.press(hotkey)  # Press the chosen hotkey
        logging.info(f"Pressed hotkey: {hotkey}")
        fitness += 0.1  # Minor fitness reward for pressing a hotkey
    
    # Perform mouse click based on output[-1] (threshold at 0.5 for left/right click)
    if click_type > 0.5:
        pyautogui.rightClick()  # Perform a right-click
        logging.info("Performed a right-click")
        fitness += 0.2  # Adjust fitness reward for right-click
    else:
        pyautogui.click()  # Perform a left-click
        logging.info("Performed a left-click")
        fitness += 0.1  # Adjust fitness reward for left-click

    return fitness



def calculate_fitness(output, resources):
    global previous_score, previous_resources  # Use the global variables

    # Get current score and resources
    current_score = int(output[0])  # Assuming output[0] is the score as an integer
    current_resources = sum(resources.values())  # Sum of all resource values (float)

    # Initialize fitness score
    fitness = 0.0

    # Reward for score increase
    if previous_score is not None:
        if current_score > previous_score:
            fitness += (current_score - previous_score)  # Reward for score increase
        elif current_score < previous_score:
            fitness -= (previous_score - current_score)  # Penalty for score decrease

    # Reward for resource increase
    if previous_resources is not None:
        if current_resources > previous_resources:
            fitness += (current_resources - previous_resources)  # Reward for resource increase
        elif current_resources < previous_resources:
            fitness -= (previous_resources - current_resources)  # Penalty for resource decrease

    # Penalty for stagnation
    if previous_score is not None and previous_resources is not None:
        if current_score == previous_score and current_resources == previous_resources:
            fitness -= 10  # Arbitrary penalty for stagnation, adjust as needed

    # Update previous values
    previous_score = current_score
    previous_resources = current_resources

    # Log the fitness
    logging.info(f"Fitness calculated: {fitness}")

    return fitness


# Load NEAT state from checkpoint if exists
def load_checkpoint(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Save NEAT state to checkpoint
def save_checkpoint(population, filename):
    with open(filename, 'wb') as f:
        pickle.dump(population, f)

def main():
    # Load the NEAT configuration
    config_path = 'neat_config.txt'  # Adjust the path to your NEAT config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population
    population = neat.Population(config)

    # Load the population
    # population = load_checkpoint(filename)

    # Add a stdout reporter to show progress
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Run the NEAT algorithm
    winner = population.run(evaluate_genome, 2)  # Run for 10 generations
    logging.info(f'Best genome:\n{winner}')

    save_checkpoint(population, "Alex sin genetiske algoritme.pkl")

if __name__ == "__main__":
    main()

