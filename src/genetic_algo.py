import numpy as np
import random
from collections import defaultdict
from dtw import dtw
from tqdm import tqdm # For a nice progress bar

# =============================================================================
# HELPER FUNCTION: N-GRAM MODEL FOR MUSICALITY
# =============================================================================

def build_ngram_model(contours):
    """
    Builds a 2-gram (bigram) probability model from a list of contours.
    Returns a dictionary where model[note1][note2] is the probability of note2 following note1.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for contour in contours:
        for i in range(len(contour) - 1):
            prev_note = contour[i]
            current_note = contour[i+1]
            counts[prev_note][current_note] += 1

    probabilities = defaultdict(lambda: defaultdict(float))
    for prev_note, next_notes in counts.items():
        total_transitions = sum(next_notes.values())
        for next_note, count in next_notes.items():
            probabilities[prev_note][next_note] = count / total_transitions
    
    return probabilities

# =============================================================================
# THE GENETIC ALGORITHM CLASS
# =============================================================================

class GeneticAlgorithm:
    def __init__(self, original_contours, vocabulary,
                 population_size=100,
                 generations=200,
                 mutation_rate=0.02,
                 tournament_size=5,
                 fitness_weights=(0.5, 0.3, 0.2)):
        """
        Initializes the Genetic Algorithm.
        
        :param original_contours: The initial list of 10 contours to evolve from.
        :param vocabulary: A sorted list of all possible cent values.
        :param population_size: The number of individuals in each generation.
        :param generations: The number of generations to run the evolution.
        :param mutation_rate: The probability of a single gene (pitch) mutating.
        :param tournament_size: The number of individuals selected for a tournament.
        :param fitness_weights: A tuple (w1, w2) for Musicality and Similarity scores.
        """
        self.original_contours = [np.array(c) for c in original_contours]
        self.vocabulary = sorted(vocabulary)
        self.vocab_map = {val: i for i, val in enumerate(self.vocabulary)}
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.w_musicality, self.w_similarity, self.w_continuity = fitness_weights
        
        # print("🧠 Building musicality model from original contours...")
        self.ngram_model = build_ngram_model(self.original_contours)
        
        # print("🌱 Initializing population...")
        self.population = self._initialize_population()

    def _initialize_population(self):
        """Creates the starting population from the original contours."""
        population = []
        # Add the original contours to ensure they are part of the gene pool
        for contour in self.original_contours:
            population.append(contour)
            
        # Fill the rest of the population with mutated versions of the originals
        while len(population) < self.population_size:
            random_original = random.choice(self.original_contours).copy()
            mutated = self._mutation(random_original)
            population.append(mutated)
            
        return population

    def _calculate_fitness(self, contour, previous_contour=None):
            """
            The core fitness function. Handles optional previous_contour.
            :param contour: The new contour being evaluated.
            :param previous_contour: (Optional) The contour of the preceding syllable.
            """
            # --- 1. Novelty Check ---
            # If the contour is an exact copy of an original, it's not novel. Fitness = 0.
            if any(np.array_equal(contour, original) for original in self.original_contours):
                return 0

            # --- 2. Musicality Score (based on N-grams) ---
            log_prob_sum = 0
            # A small probability for transitions not seen in the original set
            min_prob = 1e-4
            for i in range(len(contour) - 1):
                prev_note = contour[i]
                current_note = contour[i+1]
                prob = self.ngram_model.get(prev_note, {}).get(current_note, min_prob)
                log_prob_sum += np.log(max(prob, min_prob))
            musicality_score = np.exp(log_prob_sum / (len(contour) - 1)) # Geometric mean of probabilities

            # --- 3. Similarity Score (based on DTW) ---
            min_dist = float('inf')
            for original in self.original_contours:
                dist = dtw(contour, original).distance
                if dist < min_dist:
                    min_dist = dist
            # The score is higher for smaller distances (more similar)
            similarity_score = 1.0 / (1.0 + min_dist)

            # --- 4: Conditional Continuity Score ---
            continuity_score = 0.0 # Default to 0
            if previous_contour is not None:
                # If a previous contour is provided, calculate the score
                last_note_prev = previous_contour[-1]
                first_note_current = contour[0]
                jump_distance = abs(first_note_current - last_note_prev)
                continuity_score = 1.0 / (1.0 + jump_distance)

            # --- Final Weighted Fitness (this formula now works for both cases) ---
            fitness = (self.w_musicality * musicality_score) + \
                    (self.w_similarity * similarity_score) + \
                    (self.w_continuity * continuity_score)
            return fitness
    
    def _selection(self, fitnesses):
        """Performs tournament selection."""
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitnesses = [(fitnesses[i], i) for i in tournament_indices]
        
        # Return the index of the winner (the one with the highest fitness)
        winner_index = max(tournament_fitnesses, key=lambda item: item[0])[1]
        return self.population[winner_index]

    def _crossover(self, parent1, parent2):
        """Performs two-point crossover."""
        size = len(parent1)
        p1, p2 = sorted(random.sample(range(1, size), 2))
        
        child = np.concatenate([
            parent1[:p1],
            parent2[p1:p2],
            parent1[p2:]
        ])
        return child

    def _mutation(self, contour):
        """Creep mutation with adaptive weighted step sizes and directional momentum."""
        mutated_contour = contour.copy()
        last_dir = 0
        vocab_size = len(self.vocabulary)

        for i in range(len(mutated_contour)):
            prob = self.mutation_rate * (10 if last_dir else 1)
            if random.random() < min(1, prob):
                idx = self.vocab_map[mutated_contour[i]]

                # Base step size distribution
                step_options = [1, 2, 3]
                step_probs = [0.7, 0.2, 0.1]

                # Direction choice with momentum
                direction = last_dir if last_dir else random.choice([-1, 1])

                # 🔑 Adaptive probability adjustment near edges
                if idx < 2:  # near lower bound
                    direction = 1   # force upward
                    step_probs = [0.9, 0.1, 0.0]  
                elif idx > vocab_size - 3:  # near upper bound
                    direction = -1  # force downward
                    step_probs = [0.9, 0.1, 0.0]

                # Sample step size with adaptive probabilities
                step = random.choices(step_options, weights=step_probs, k=1)[0]

                # Apply mutation
                new_idx = max(0, min(vocab_size - 1, idx + direction * step))
                mutated_contour[i] = self.vocabulary[new_idx]

                last_dir = direction
            else:
                last_dir = 0  # reset momentum

        return mutated_contour



    def run(self, previous_contour=None):
        """Runs the main evolutionary loop."""
        # print(f"🧬 Running evolution for {self.generations} generations...")
        if previous_contour is not None:
            # print("🎵 Running with context from previous syllable.")
            previous_contour = np.array(previous_contour)
        else:
            pass
            # print("🎵 Running for the first syllable (no context).")
        
        for generation in tqdm(range(self.generations), desc="Evolving"):
            # Calculate fitness for the entire population once per generation
            fitnesses = [self._calculate_fitness(ind, previous_contour) for ind in self.population]

            new_population = []
            for _ in range(self.population_size // 2):
                # Select parents
                parent1 = self._selection(fitnesses)
                parent2 = self._selection(fitnesses)
                
                # Create and mutate children
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1) # Can create a second child
                
                new_population.append(self._mutation(child1))
                new_population.append(self._mutation(child2))
            
            self.population = new_population

        print("✅ Evolution complete!")
        # Return the final population, sorted by fitness
        final_fitnesses = [self._calculate_fitness(ind, previous_contour) for ind in self.population]
        sorted_population = sorted(zip(self.population, final_fitnesses), key=lambda x: x[1], reverse=True)
        return [ind for ind, fit in sorted_population]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

'''
if __name__ == "__main__":
    # --- 1. Define Your Data ---
    # This is a dummy vocabulary. Replace with your actual cent values.

    # --- 2. Configure and Run the GA ---
    ga = GeneticAlgorithm(
        original_contours=ORIGINAL_CONTOURS,
        vocabulary=VOCABULARY,
        population_size=100,
        generations=200,
        mutation_rate=0.03,      # 3% chance for a pitch to mutate
        tournament_size=5,
        fitness_weights=(0.7, 0.3) # Prioritize musicality a bit more
    )
    
    evolved_contours = ga.run()

    # --- 3. Display the Results ---
    print("\n--- Best 5 Evolved Contours ---")
    for i in range(5):
        # Converting numpy array to list for cleaner printing
        print(f"Contour {i+1}: {list(evolved_contours[i])}")
'''