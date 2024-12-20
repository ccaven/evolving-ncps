# evolving-ncps

Goal: implement NeuroEvolution of Augmenting Topologies (NEAT) to evolve Neural Circuit Policies (NCPs) and apply to electroencephalogram (EEG) signal analysis.

TODO list:
 - Neural Circuit Policies
   - [ ] Differential equation solver
   - [ ] Weight constraints
   - [ ] Wirings
 - NeuroEvolution of Augmenting Topologies
   - [ ] Genotype to phenotype mapping
   - [ ] Fitness function
   - [ ] Reproduction
     - [ ] Compute number of offscreen per individual
     - [ ] Crossover operator
     - [ ] Mutation operator
   - [ ] Speciation
     - [ ] Genetic distance
     - [ ] Greedy algorithm for constructing species
     - [ ] Species stagnation
 - EEG signal analysis
   - [ ] Identity and load appropriate dataset
   - [ ] Tune hyperparameters
   - [ ] Develop interpretability techniques