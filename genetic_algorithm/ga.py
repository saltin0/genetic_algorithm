'''
    :project:       Genetic Algorithm Implementation
    
    :description:   This algorithm search for global maximum or global minimum.
                    If your objective function must be maximized, you can change 
                    the self.obj_array in line : {-64-}. If your obj_function is 
                    different, you can replace the function with yours.
    
    :author:        manyetar@gmail.com
                    github:saltin0
    
    :reference: Ayca Altay
'''

import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__(self,upper_bound,lower_bound,dimension,population_size,iteration_number,cross_over_prob = 0.95,mutation_prob = 0.01):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        # dimenion: how many parameters will be estimated
        self.dimension = dimension
        # Create the population with given size
        self.population_size = population_size
        # Create the population
        self.population = self.create_population()
        # Create the initial obj_array
        self.obj_array = np.zeros((self.population_size,1))
        # Given crossover and mutation probabilistic values
        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob
        # Determine how many times the algorithm faster
        self.iteration_number = iteration_number
        # Debug array
        self.best_obj_tracker = []

    def create_population(self):
        '''
            :function:  create_population-Create the population with randomly distributed values.

            :return:    population-Return created population array with the size of (population_size,dimension)
        '''
        population = np.random.uniform(self.lower_bound,self.upper_bound,(self.population_size,self.dimension))
        return population


    def objective_function(self):
        '''
            :function:  objective_function-Objective function may be change by usage area of 
                        this algorithm. You can use any optimization problem here.
                        In example code ob_func is obj = xi^2 + xii ^2 + ..... + xdimensioni^2
        '''
        obj_array = [np.sum (x**2) for x in self.population]
        self.obj_array = np.array(obj_array).reshape(self.population_size,1)

    def natural_selection(self):
        '''
            :function:  natural_selection-It takes the population and eliminate the weakest chromosomes
                        by the rule of Roulette Wheel. Roulette Wheel is maximisation algorithms thus 
                        obj_array must be taken as power of -1. Then if maximise the power of -1, in general
                        problem solution obj will be minimized.
        '''
        self.obj_array = self.obj_array**-1
        sum_obj = np.sum(self.obj_array)
        # Natural selection survival probability
        #   if the gene has high probability then survive
        probs =self.obj_array/sum_obj
        # Initiate the cumulative probability
        cum_probs = probs 
        # Now calculate the cumulative probability
        for i in range(1,self.population_size):
            cum_probs[i] =cum_probs[i-1] + probs[i]  
        # Now assign the random number to determine survivor genes
        random_number = np.random.uniform(0,1,(self.population_size,1))

        inter_population=self.create_inter_population(random_number,cum_probs)
        return inter_population
        

        #print(f"Poulation : \n{self.population}\n Inter_population : \n{inter_population}")

    def create_inter_population(self,random_number,cum_probs):
        '''
            :function:  create_inter_poulation- It execute roulette wheel's rules.
                        chexk roulett rules.

            :return:    inter_population-Returns inter_population with survivors.
        '''
        inter_population = np.zeros((self.population_size,self.dimension))
        for i in range(self.population_size):
            # Find the index which random number is greater than the cumulative probability
            #   first and assign the population member which has found index to the inter_population
            #   with this step other members will be eliminated 
            choice = random_number[i]<cum_probs
            indexes = np.where(choice==True)
            index = indexes[0][0]
            inter_population[i][:] = self.population[index][:]

            #print(f"Index : {index}")
            # print(f"Index : {np.where(index==True)[0][0]}")
        return inter_population



    def cross_over(self,inter_population):
        '''
            :function:  cross_over-Check the probability of crossover is lesser than 
                        pre determined crossover rules. If the crossover prob is lesser 
                        than pre determined one then do crossover.

            :return:    inter_population-Returns inter_population with crossover is made.
        '''
        # Determine the pairs. It determines randomly which genes make pairs and do crossover.
        pairs = np.random.permutation(self.population_size)

        for i in range(int(self.population_size/2)):
            parent1_idx = pairs[2*i]
            parent2_idx = pairs[2*i+1]
            parent1 = inter_population[parent1_idx][:]
            parent2 = inter_population[parent2_idx][:]

            crs_over_prob = np.random.uniform(0,1)
            # If crossover prob is lesser than determined crossover prob
            #   then do crossover
            if crs_over_prob<self.cross_over_prob:
                cpoint = np.random.randint(1,self.dimension-1)
                dummy = parent1[cpoint:]
                parent1[cpoint:] = parent2[cpoint:]
                parent2[cpoint:] = dummy
                inter_population[parent1_idx][:] = parent1
                inter_population[parent2_idx][:] = parent2
        
        return inter_population



    def mutation(self,inter_population,delta=0.05):
        '''
            :function:  mutation-Make mutation with mutation probability according to 
                        the pre determined mutation prob. If mutation prob is lesser than 
                        pre determined one do mutation with adding or substracting small number.
            :return:    inter_population-All steps are done with mutation step. Thus new inter_population 
                        array resembles the new population.
        '''
        mutation_probs = np.random.uniform(0,1,(self.population_size,self.dimension))
        for i in range(self.population_size):
            for j in range(self.dimension):
                if mutation_probs[i][j]<self.mutation_prob:
                    mut_way_det = np.random.random()*2-1
                    inter_population[i][j] += mut_way_det*delta*(self.upper_bound-self.lower_bound)
        return inter_population

        


    def main(self):
        '''
            :funtion:   main-It runs the algorithm recursively to find best solutions.
            
            :return:    best_solution- It returns the best solution the algorithm found.
        '''
        # Determine the initial objective
        best_obj = 10000000000
        # With the code below update the best_obj in every iteration
        for _ in range(self.iteration_number):
            self.objective_function()
            if min(self.obj_array)<best_obj:
                best_obj = min(self.obj_array)
                # Find the best_obj index
                idx = np.where(min(self.obj_array)==best_obj)
                # Update the best solution
                best_solution = self.population[idx][:]
            # Update the interpopulation
            inter_population = self.natural_selection()
            inter_population = self.cross_over(inter_population)
            # Update the population
            self.population = self.mutation(inter_population)
            # Keep the objective values for debugging
            self.best_obj_tracker.append(best_obj)
        #print(f"Inter population : {inter_population}")
        # Return the result of algorithm
        return best_solution

'''
    Run the algorithm.
'''
if __name__=='__main__':
    plt.figure()
    for _ in range(10):
        gen = GA(1,-1,10,200,100)
        best = gen.main()
        plt.plot(gen.best_obj_tracker)
    print(best)
    plt.show()
