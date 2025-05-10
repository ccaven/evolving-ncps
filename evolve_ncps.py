"""
This file includes customizations to the NEAT algorithm and a fitness function to run the neuroevolution experiment.
"""

import math
import multiprocessing
import os
import pickle

from typing import Self

import random

import neat.population
import neat.species
import torch

import neat
import neat.genome
import neat.config
import neat.genes
import neat.attributes
import neat.reporting

import ncps.wirings
import ncps.torch

import os.path

EPOCHS_PER_INDIVIDUAL = 2

train_dataset = torch.load(os.path.join(os.path.dirname(__file__), "lds_train.pkl"), weights_only=True)
valid_dataset = torch.load(os.path.join(os.path.dirname(__file__), "lds_valid.pkl"), weights_only=True)
test_dataset = torch.load(os.path.join(os.path.dirname(__file__), "lds_test.pkl"), weights_only=True)

# NEAT customization

class NeuronGene(neat.genes.BaseGene):
    __gene_attributes__ = []
    def distance(self, other, config):
        return 0.0

class SynapseGene(neat.genes.BaseGene):
    __gene_attributes__ = [
        neat.attributes.BoolAttribute("polarity"), 
        neat.attributes.BoolAttribute("enabled")
    ]

    def distance(self, other: Self, config):
        d = 0
        if self.polarity != other.polarity:
            d += 1.0
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient

class NetworkGenomeConfig:
    __params = [
        neat.config.ConfigParameter("num_inputs", int),
        neat.config.ConfigParameter("num_outputs", int),
        neat.config.ConfigParameter("num_units", int),
        neat.config.ConfigParameter("default_sparsity", float),
        neat.config.ConfigParameter("compatibility_weight_coefficient", float),
        neat.config.ConfigParameter("compatibility_disjoint_coefficient", float),
        neat.config.ConfigParameter("conn_add_prob", float),
        neat.config.ConfigParameter("conn_delete_prob", float),
        neat.config.ConfigParameter("node_add_prob", float),
        neat.config.ConfigParameter("node_delete_prob", float),
        #neat.config.ConfigParameter("sparsity_cap", float),
        neat.config.ConfigParameter("max_conn_per_node", int)
    ]

    def __init__(self, params):
        self.__params += NeuronGene.get_config_params()
        self.__params += SynapseGene.get_config_params()

        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        self.input_keys = [-i-1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def save(self, f):
        neat.config.write_pretty_params(f, self, self.__params)

class NetworkGenome:
    @classmethod
    def parse_config(cls, param_dict):
        return NetworkGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config: NetworkGenomeConfig):
        config.save(f)
    
    def __init__(self, key):
        self.key = key
        self.connections: dict[tuple[int, int], SynapseGene] = {}
        self.nodes: dict[int, NeuronGene] = {}
        self.fitness = None
        self.train_loss_list = None
    
    def mutate(self, config: NetworkGenomeConfig):
        # Handle possibly adding or removing nodes or connections
        for (method_name, config_prob_key) in (
            ("mutate_add_node", "node_add_prob"),
            ("mutate_delete_node", "node_delete_prob"),
            ("mutate_add_connection", "conn_add_prob"),
            ("mutate_delete_connection", "conn_delete_prob")
        ):
            if random.random() < config.__getattribute__(config_prob_key):
                self.__getattribute__(method_name)(config)
        
        # Mutate leftover nodes or connections
        for conn_gene in self.connections.values():
            conn_gene.mutate(config)
        
        for node_gene in self.nodes.values():
            node_gene.mutate(config)
        
        self.enforce_sparsity(config)
    
    def get_new_node_key(self):
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    @staticmethod
    def create_node(config, id: int):
        node = NeuronGene(id)
        node.init_attributes(config)
        return node
    
    @staticmethod
    def create_connection(config, input_id: int, output_id: int):
        connection = SynapseGene((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def add_connection(self, config, input_key, output_key):
        key = (input_key, output_key)
        connection = SynapseGene(key)
        connection.init_attributes(config)
        self.connections[key] = connection

    def mutate_add_node(self, config: NetworkGenomeConfig):
        # If we have literally no connections just add one somewhere please
        while len(self.connections.values()) == 0:
            self.mutate_add_connection(config)

        # Pick a victim
        conn_to_split = random.choice(list(self.connections.values()))
        conn_to_split.enabled = False

        # Add a new node
        new_node_id = self.get_new_node_key()
        node_gene = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = node_gene

        # Connect up the new node
        input_id, output_id = conn_to_split.key

        self.add_connection(config, input_id, new_node_id)
        self.add_connection(config, new_node_id, output_id)

    def mutate_delete_node(self, config: NetworkGenomeConfig):
        available_victims = [k for k in self.nodes.keys() if k not in config.output_keys]
        
        if len(available_victims) == 0:
            return -1
        
        del_key = random.choice(available_victims)

        bad_conns = [conn.key for conn in self.connections.values() if del_key in conn.key]

        for key in bad_conns:
            del self.connections[key]

        del self.nodes[del_key]
        
        return del_key

    def mutate_add_connection(self, config: NetworkGenomeConfig):
        possible_output_keys = list(self.nodes.keys())
        possible_input_keys = possible_output_keys + config.input_keys

        input_key = random.choice(possible_input_keys)
        output_key = random.choice(possible_output_keys)

        if input_key == output_key:
            return
        
        self.add_connection(config, input_key, output_key)
    
    def mutate_delete_connection(self, config: NetworkGenomeConfig):
        if len(self.connections) == 0:
            return -1
        key = random.choice(list(self.connections.keys()))
        del self.connections[key]
        return key
    
    def configure_crossover(self, genome1: Self, genome2: Self, config: NetworkGenomeConfig):
        # Order by fitness
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        
        for key in parent1.connections:
            if key in parent2.connections:
                self.connections[key] = parent1.connections[key].crossover(parent2.connections[key])
            else:
                self.connections[key] = parent1.connections[key].copy()

        for key in parent1.nodes:
            if key in parent2.nodes:
                self.nodes[key] = parent1.nodes[key].crossover(parent2.nodes[key])
            else:
                self.nodes[key] = parent1.nodes[key].copy()
        
        self.enforce_sparsity(config)

    def distance(self, other: Self, config: NetworkGenomeConfig):
        self_node_keys = set(self.nodes.keys())
        other_node_keys = set(other.nodes.keys())

        shared_keys = self_node_keys.intersection(other_node_keys)
        disjoint_keys = self_node_keys.symmetric_difference(other_node_keys)

        max_nodes = max(
            len(self_node_keys), 
            len(other_node_keys)
        )

        if max_nodes == 0:
            node_distance = 0
        else:
            node_distance = (
                sum(
                    self.nodes[key].distance(other.nodes[key], config) 
                    for key in shared_keys
                ) + \
                len(disjoint_keys) * config.compatibility_disjoint_coefficient
            ) / max_nodes

        self_connection_keys = set(self.connections.keys())
        other_connection_keys = set(other.connections.keys())

        shared_keys = self_connection_keys.intersection(other_connection_keys)
        disjoint_keys = self_connection_keys.symmetric_difference(other_connection_keys)

        max_connections = max(
            len(self_connection_keys), 
            len(other_connection_keys)
        )

        if max_connections == 0:
            connection_distance = 0
        else:
            connection_distance = (
                sum(
                    self.connections[key].distance(other.connections[key], config) 
                    for key in shared_keys
                ) + \
                len(disjoint_keys) * config.compatibility_disjoint_coefficient
            ) / max_connections

        return node_distance + connection_distance

    def size(self):
        return len(self.nodes), sum(1 for k in self.connections.values() if k.enabled)

    def add_hidden_nodes(self, config: NetworkGenomeConfig):
        for i in range(config.num_hidden):
            node_key = self.get_new_node_key()
            assert node_key not in self.nodes
            node = self.__class__.create_node(config, node_key)
            self.nodes[node_key] = node

    def configure_new(self, config: NetworkGenomeConfig):

        wiring = ncps.wirings.AutoNCP(
            config.num_units, 
            config.num_outputs, 
            sparsity_level=0.5,
            seed=random.randint(1, 10000)
        )
        wiring.build(config.num_inputs)
        
        # first wire up sensory synapses
        for input_id in range(config.num_inputs):
            for output_id in range(config.num_units):
                if wiring.sensory_adjacency_matrix[input_id, output_id] != 0:
                    polarity = wiring.sensory_adjacency_matrix[input_id, output_id]
                    if output_id not in self.nodes:
                        self.nodes[output_id] = NeuronGene(output_id)
                    self.add_connection(config, -input_id-1, output_id)
                    self.connections[(-input_id-1, output_id)].polarity = polarity == 1

        # then wire up normal synapses
        for input_id in range(config.num_units):
            for output_id in range(config.num_units):
                if wiring.adjacency_matrix[input_id, output_id] != 0:
                    polarity = wiring.adjacency_matrix[input_id, output_id]
                    if input_id not in self.nodes:
                        self.nodes[input_id] = NeuronGene(input_id)
                    if output_id not in self.nodes:
                        self.nodes[output_id] = NeuronGene(output_id)
                    self.add_connection(config, input_id, output_id)
                    self.connections[(input_id, output_id)].polarity = polarity == 1

    def enforce_sparsity(self, config: NetworkGenomeConfig):
        if True:
            pass

        conns_per_node: dict[int, list[int]] = {}
        for input_id, output_id in self.connections:
            if self.connections[(input_id, output_id)].enabled:
                if input_id in conns_per_node:
                    conns_per_node[input_id].append(output_id)
                else:
                    conns_per_node[input_id] = [output_id]
        
        conns_to_remove = []
        for input_id in conns_per_node:
            if len(conns_per_node[input_id]) > config.max_conn_per_node:
                num_excess = len(conns_per_node[input_id]) - config.max_conn_per_node
                output_ids = random.sample(conns_per_node[input_id], num_excess)
                conns_to_remove += [(input_id, output_id) for output_id in output_ids]
        
        for key in conns_to_remove:
            del self.connections[key]

# Reporting

class CustomReporter(neat.reporting.BaseReporter):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def post_evaluate(
        self, 
        config: neat.config.Config, 
        population: neat.population.Population, 
        species: neat.species.DefaultSpeciesSet, 
        best_genome: NetworkGenome
    ):
        # Save current generation of genomes
        # for future analysis
        
        if not os.path.exists("./saved_gens"):
            os.mkdir("./saved_gens")

        with open(f"./saved_gens/gen_{self.counter}.pkl", "wb") as handle:
            obj = {}

            for species_id in species.species:
                obj[species_id] = []
                for genome in species.species[species_id].members.values():
                    obj[species_id].append({
                        "nodes": { 
                            key: {}
                            for key in genome.nodes 
                        },
                        "connections": {
                            key: {
                                "polarity": genome.connections[key].polarity,
                                "enabled": genome.connections[key].enabled
                            }
                            for key in genome.connections
                        },
                        "fitness": genome.fitness,
                        "train_loss_list": genome.train_loss_list
                    })


            pickle.dump(obj, handle)
        
        self.counter += 1

# Fitness function

evaluation_cache_lock = multiprocessing.Lock()
evaluation_cache = {}

def eval_genome(
    genome: NetworkGenome, 
    config: NetworkGenomeConfig
):

    genome_config: neat.genome.DefaultGenomeConfig = config.genome_config
    input_keys: list[int] = genome_config.input_keys
    output_keys: list[int] = [i for i in range(genome_config.num_outputs)]
    connections = genome.connections
    input_dim = len(input_keys)
    output_dim = len(output_keys)

    enabled_connections: list[SynapseGene] = [
        conn
        for conn in connections.values()
        if conn.enabled
    ]

    dest_nodes = sorted(list(set(output_keys + [c.key[1] for c in enabled_connections])))

    # 1. Generate the NCP wiring

    wiring = ncps.wirings.Wiring(len(dest_nodes))

    wiring.set_output_dim(output_dim)
    wiring.build(input_dim)

    """
    wiring_key = (
        wiring.sensory_adjacency_matrix.tobytes(),
        (*wiring.sensory_adjacency_matrix.shape,),
        wiring.adjacency_matrix.tobytes(),
        (*wiring.adjacency_matrix.shape,),
    )
    """

    #with evaluation_cache_lock:
    #    if wiring_key in evaluation_cache:
    #        return evaluation_cache[wiring_key]

    for conn in enabled_connections:
        input_id, output_id = conn.key
        if input_id in input_keys:
            wiring.add_sensory_synapse(
                -input_id-1,
                dest_nodes.index(output_id),
                1 if conn.polarity else -1
            )
        elif input_id in dest_nodes:
            wiring.add_synapse(
                dest_nodes.index(input_id),
                dest_nodes.index(output_id),
                1 if conn.polarity else -1
            )

    # 2. Initialize NCP parameters

    network = ncps.torch.LTC(input_dim, wiring).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    # 3. Train NCP for a few epochs on training set
    
    us: torch.Tensor = train_dataset["u"]
    xs: torch.Tensor = train_dataset["x"]
    ys: torch.Tensor = train_dataset["y"]

    batch_size = 64

    train_loss_list = []

    for epoch in range(EPOCHS_PER_INDIVIDUAL):

        for i in range(0, xs.size(0), batch_size):
            batch_x = xs[i : i + batch_size].cuda() # (B, L, D)
            batch_u = us[i : i + batch_size].cuda() # (B, L, D)
            batch_y = ys[i : i + batch_size].cuda()

            seq_input = torch.cat((
                batch_x[:, :-1, :],
                batch_u[:, :-1, :]
            ), dim=-1)

            seq_output = torch.cat((
                batch_x[:, 1:, :],
                batch_y[:, :-1, :]
            ), dim=-1)

            pred_output, _ = network.forward(seq_input)

            loss = torch.nn.functional.mse_loss(pred_output, seq_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())

            # pbar.set_postfix({"loss" : loss.item()})

    # 4. Compute validation accuracy / loss

    us: torch.Tensor = valid_dataset["u"]
    xs: torch.Tensor = valid_dataset["x"]
    ys: torch.Tensor = valid_dataset["y"]

    optimizer.zero_grad()

    total_valid_loss = 0

    with torch.no_grad():

        for i in range(0, xs.size(0), batch_size):
            batch_x = xs[i : i + batch_size].cuda() # (B, L, D)
            batch_u = us[i : i + batch_size].cuda() # (B, L, D)
            batch_y = ys[i : i + batch_size].cuda()

            seq_input = torch.cat((
                batch_x[:, :-1, :],
                batch_u[:, :-1, :]
            ), dim=-1)

            seq_output = torch.cat((
                batch_x[:, 1:, :],
                batch_y[:, :-1, :]
            ), dim=-1)

            pred_output, _ = network.forward(seq_input)

            loss = torch.nn.functional.mse_loss(pred_output, seq_output)

            total_valid_loss += loss.item()

    fitness = -math.log(total_valid_loss / (xs.size(0) // batch_size))

    genome.fitness = fitness
    genome.train_loss_list = train_loss_list

    # 5. Return validation accuracy / loss
    return fitness

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ncps')
    config = neat.Config(
        NetworkGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(CustomReporter())
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(2, eval_genome)
    winner = pop.run(pe.evaluate)
    
    print("\n")
    print("finished with best score of:", winner.fitness)

if __name__ == '__main__':
    run()
