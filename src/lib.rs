use std::collections::{HashMap, HashSet};

pub mod data;
pub mod traits;

use crate::traits::{Gene, Genome, Evaluated};


struct PopulationConfig {
    pub population_size: u32,
    pub fitness_criterion: f32,
    pub fitness_threshold: f32
}

struct NetworkGenomeConfig {
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_units: u32,
    pub default_sparsity: f32,

    pub compatibility_disjoint_coefficient: f32,
    pub compatibility_weight_coefficient: f32,

    pub conn_add_prob: f32,
    pub conn_delete_prob: f32,
 
    pub node_add_prob: f32,
    pub node_delete_prob: f32,
 
    pub enabled_mutate_rate: f32,
    pub polarity_mutate_rate: f32
}

#[derive(Clone)]
struct NeuronGene {}

impl Gene<NetworkGenomeConfig> for NeuronGene {
    fn mutate(self, _config: &NetworkGenomeConfig) -> Self {
        self
    }

    fn crossover(&self, _other: &Self, _config: &NetworkGenomeConfig) -> Self {
        NeuronGene {  }
    }

    fn distance(&self, _other: &Self, _config: &NetworkGenomeConfig) -> f32 {
        0.0
    }
}

#[derive(Clone)]
struct SynapseGene {
    pub polarity: bool,
    pub enabled: bool
}

impl Gene<NetworkGenomeConfig> for SynapseGene {
    fn mutate(self, config: &NetworkGenomeConfig) -> Self {
        Self { 
            polarity: rand::random_bool(config.polarity_mutate_rate.into()) ^ self.polarity,
            enabled: rand::random_bool(config.enabled_mutate_rate.into()) ^ self.enabled
        }
    }

    fn crossover(&self, other: &Self, _config: &NetworkGenomeConfig) -> Self {
        SynapseGene { 
            polarity: if rand::random_bool(0.5) { self.polarity } else { other.polarity }, 
            enabled: if rand::random_bool(0.5) { self.enabled } else { other.enabled } 
        }
    }

    fn distance(&self, other: &Self, config: &NetworkGenomeConfig) -> f32 {
        return (
            if self.polarity == other.polarity { 0.0 } else { 1.0 } +
            if self.enabled == other.enabled { 0.0 } else { 1.0 }
        ) * config.compatibility_weight_coefficient;
    }
}

struct NetworkGenome {
    pub connections: HashMap<(i32, i32), SynapseGene>,
    pub nodes: HashMap<i32, NeuronGene>
}

impl Genome<NetworkGenomeConfig> for NetworkGenome {
    fn crossover(a: Evaluated<Self>, b: Evaluated<Self>, config: &NetworkGenomeConfig) -> Self {
        // Destructure and order such that a > b
        let (
            Evaluated {
                genome: a, 
                ..
            },
            Evaluated {
                genome: b, 
                ..
            }
        ) = if a.fitness > b.fitness { (a, b) } else { (b, a) };

        let mut connections = HashMap::<(i32, i32), SynapseGene>::new();
        let mut nodes = HashMap::<i32, NeuronGene>::new();

        for key in a.connections.keys() {
            if b.connections.contains_key(key) {
                let new_gene = a.connections[key].crossover(&b.connections[key], config);
                connections.insert(*key, new_gene);
            } else {
                connections.insert(*key, a.connections[key].clone());
            }
        }

        for key in a.nodes.keys() {
            if b.nodes.contains_key(key) {
                let new_gene = a.nodes[key].crossover(&b.nodes[key], config);
                nodes.insert(*key, new_gene);
            } else {
                nodes.insert(*key, a.nodes[key].clone());
            }
        }

        Self { connections, nodes }
    }

    fn mutate(self, config: &NetworkGenomeConfig) -> Self {
        let NetworkGenome { mut connections, mut nodes } = self;

        // Handle possibly adding or removing nodes
        if rand::random::<f32>() < config.node_add_prob {
            let candidates: Vec<(i32, i32)> = connections.keys().map(|x| *x).collect();

            if candidates.len() > 0 {
                let (input_id, output_id) = candidates[rand::random_range(0..candidates.len())];

                let new_id = {
                    let mut candidate_id = rand::random::<u32>();
                    while candidate_id < config.num_outputs || nodes.contains_key(&(candidate_id as i32)) {
                        candidate_id = rand::random::<u32>();
                    }
                    candidate_id
                } as i32;

                let old_polarity = connections[&(input_id, output_id)].polarity;

                connections.entry((input_id, output_id)).and_modify(|x| x.enabled = false);

                nodes.insert(new_id, NeuronGene {});

                connections.insert((input_id, new_id), SynapseGene { 
                    polarity: old_polarity, 
                    enabled: true
                });

                connections.insert((new_id, output_id), SynapseGene { 
                    polarity: old_polarity, 
                    enabled: true
                });
            }
        }

        if rand::random::<f32>() < config.node_delete_prob {
            let candidates: Vec<i32> = nodes.keys().filter(|key| **key < config.num_outputs as i32).map(|x| *x).collect();

            if candidates.len() > 0 {
                let del_key = candidates[rand::random_range(0..candidates.len())];

                let mut bad_keys = Vec::new();
            
                for key in connections.keys() {
                    if key.0 == del_key || key.1 == del_key {
                        bad_keys.push(*key);
                    }
                }
                
                for key in bad_keys {
                    connections.remove(&key);
                }

                nodes.remove(&del_key);
            }
        }

        if rand::random::<f32>() < config.conn_add_prob {
            let output_candidates: Vec<i32> = nodes.keys().map(|x| *x as i32).collect();
            
            let mut input_candidates: Vec<i32> = output_candidates.clone();

            for i in 0..config.num_inputs {
                let input_id = -1 - (i as i32);
                input_candidates.push(input_id);
            }

            let input_key = input_candidates[rand::random_range(0..input_candidates.len())];
            let output_key = output_candidates[rand::random_range(0..output_candidates.len())];

            connections.insert((input_key, output_key), SynapseGene { 
                polarity: rand::random_bool(0.5), 
                enabled: true 
            });
        }

        if rand::random::<f32>() < config.conn_delete_prob {
            let candidates: Vec<(i32, i32)> = connections.keys().map(|x| *x).collect();
            
            if candidates.len() > 0 {
                let (input_id, output_id) = candidates[rand::random_range(0..candidates.len())];
                connections.remove(&(input_id, output_id));
            }
        }

        // Mutate inplace
        let connections = connections
            .into_iter()
            .map(|(key, val)| (key, val.mutate(config)))
            .collect();
        
        let nodes = nodes
            .into_iter()
            .map(|(key, val)| (key, val.mutate(config)))
            .collect();
        
        Self {
            connections,
            nodes
        }
    }

    fn distance(&self, other: Self, config: &NetworkGenomeConfig) -> f32 {
        let node_distance = {
            let self_node_keys = self.nodes.keys().map(|x| *x).collect::<HashSet<i32>>();
            let other_node_keys = other.nodes.keys().map(|x| *x).collect::<HashSet<i32>>();

            let shared_keys = self_node_keys.intersection(&other_node_keys);
            let disjoint_keys = self_node_keys.symmetric_difference(&other_node_keys);

            let max_nodes = usize::max(self_node_keys.len(), other_node_keys.len());

            if max_nodes == 0 {
                0.0
            } else {
                (
                    shared_keys
                        .map(|x| self.nodes[x].distance(&other.nodes[x], config))
                        .sum::<f32>() +
                    (disjoint_keys.count() as f32) * config.compatibility_disjoint_coefficient
                ) / (max_nodes as f32)
            }
        };

        let conn_distance = {
            let self_conn_keys = self.connections.keys().map(|x| *x).collect::<HashSet<(i32, i32)>>();
            let other_conn_keys = other.connections.keys().map(|x| *x).collect::<HashSet<(i32, i32)>>();
            
            let shared_keys = self_conn_keys.intersection(&other_conn_keys);
            let disjoint_keys = self_conn_keys.symmetric_difference(&other_conn_keys);

            let max_conns = usize::max(self_conn_keys.len(), other_conn_keys.len());

            if max_conns == 0 {
                0.0
            } else {
                (
                    shared_keys
                        .map(|x| self.connections[x].distance(&other.connections[x], config))
                        .sum::<f32>() +
                    (disjoint_keys.count() as f32) * config.compatibility_disjoint_coefficient
                ) / (max_conns as f32)
            }
        };

        return node_distance + conn_distance;
    }

    fn descriptor(&self, _config: &NetworkGenomeConfig) -> String {
        return format!("({}, {})", self.nodes.len(), self.connections.len());
    }
}

