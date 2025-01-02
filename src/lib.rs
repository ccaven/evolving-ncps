use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
};

#[derive(Debug, Clone, Copy)]
pub struct NeuronConfig {
    pub leakage_conductance: f32,
    pub membrane_capacitance: f32,
    pub resting_potential: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SynapseConfig {
    pub weight: f32,
    pub reversal_potential: f32,
    pub sigma: f32,
    pub mu: f32,
    // Enabled flag is for the genetic algorithm
    pub enabled: bool,
}

pub struct Neuron {
    pub state: f32,
    pub config: NeuronConfig,
}

pub struct NetworkConfig {
    pub neurons: BTreeMap<u32, NeuronConfig>,

    // First key is destination neuron, second key is source neuron
    pub synapses: BTreeMap<u32, BTreeMap<u32, SynapseConfig>>,
}

impl NetworkConfig {
    /// Compute the genetic distance between two individuals.
    /// This is used by the speciation mechanism of NEAT.
    pub fn genetic_distance(&self, other: &Self) -> f32 {
        // Match and accumulate distance between genes associated with neurons
        let neuron_distance = {
            let mut neuron_distance = 0f32;

            let mut other_remaining_ids: BTreeSet<u32> = other.neurons.keys().cloned().collect();

            for (self_neuron_id, self_neuron_config) in self.neurons.iter() {
                if let Some(other_neuron_config) = other.neurons.get(self_neuron_id) {
                    neuron_distance +=
                        distance_between_neurons(self_neuron_config, other_neuron_config);
                    other_remaining_ids.remove(self_neuron_id);
                } else {
                    neuron_distance += distance_between_neuron_and_none(self_neuron_config);
                }
            }

            for other_neuron_id in other_remaining_ids {
                if let Some(other_neuron_config) = other.neurons.get(&other_neuron_id) {
                    neuron_distance += distance_between_neuron_and_none(other_neuron_config);
                }
            }

            neuron_distance
        };

        // Match and accumulate distance between genes associated with synapses
        let synapse_distance = {
            let mut synapse_distance = 0f32;

            let mut remaining_src_ids: BTreeSet<u32> = other.synapses.keys().cloned().collect();

            for (self_src_id, self_dst_map) in self.synapses.iter() {
                if let Some(other_dst_map) = other.synapses.get(self_src_id) {
                    let mut remaining_dst_ids: BTreeSet<u32> =
                        other_dst_map.keys().cloned().collect();

                    for (self_dst_id, self_synapse_config) in self_dst_map {
                        if let Some(other_synapse_config) = other_dst_map.get(self_dst_id) {
                            // Case 1: the same synapse exists in both individuals
                            synapse_distance += distance_between_synapses(
                                self_synapse_config,
                                other_synapse_config,
                            );
                            remaining_dst_ids.remove(self_dst_id);
                        } else {
                            // Case 2: src is same, dst does not exist in other
                            synapse_distance +=
                                distance_between_synapse_and_none(self_synapse_config);
                        }
                    }

                    // Case 3: src is the same, dst does not exist in self
                    for other_dst_id in remaining_dst_ids {
                        if let Some(other_synapse_config) = other_dst_map.get(&other_dst_id) {
                            synapse_distance +=
                                distance_between_synapse_and_none(other_synapse_config)
                        }
                    }

                    remaining_src_ids.remove(self_src_id);
                } else {
                    for (_, self_synapse_config) in self_dst_map {
                        // Case 4: src does not exist in other
                        synapse_distance += distance_between_synapse_and_none(self_synapse_config);
                    }
                }
            }

            for other_src_id in remaining_src_ids {
                if let Some(other_dst_map) = other.synapses.get(&other_src_id) {
                    for (_, other_synapse_config) in other_dst_map {
                        // Case 5: src does not exist in self
                        synapse_distance += distance_between_synapse_and_none(other_synapse_config);
                    }
                }
            }

            synapse_distance
        };

        neuron_distance + synapse_distance
    }

    /// Perform the mutation mechanism of the genetic algorithm.
    /// There are a few cases to consider:
    /// 1.
    pub fn mutation(&self) -> Self {
        todo!()
    }

    pub fn crossover(&self, other: &Self) -> Self {
        todo!()
    }
}

pub struct Network {
    pub config: NetworkConfig,
    pub neurons: BTreeMap<u32, Neuron>,
}

impl Network {
    const ODE_UNFOLDS: usize = 16;

    pub fn new(config: NetworkConfig) -> Self {
        let neurons: BTreeMap<u32, Neuron> = config
            .neurons
            .iter()
            .map(|(id, config)| {
                (
                    // This will be the key in the collected map
                    *id,
                    // This will be the value in the collected map
                    // In "Neural circuit policies enabling auditable autonomy",
                    // neuron states are initialized with zeros.
                    Neuron {
                        state: 0f32,
                        config: *config,
                    },
                )
            })
            .collect();

        Network { config, neurons }
    }

    /// Evaluates one fractional timestep of size t.
    /// See equation (3) in https://www.nature.com/articles/s42256-020-00237-3
    fn one_step(&mut self, t: f32) {
        // Bucket to hold computed values
        let mut next_states = BTreeMap::new();

        for (id, neuron) in self.neurons.iter() {
            let state = neuron.state;
            let config = neuron.config;

            // Collect incoming synapses (if any)
            let Some(incoming_synapses) = self.config.synapses.get(id) else {
                continue;
            };

            // Compute effects of sensory neurons
            let (sensory_numerator, sensory_denominator) = incoming_synapses
                .iter()
                // There is a shared term
                .filter_map(|(src_id, synapse)| {
                    if !synapse.enabled {
                        None
                    } else {
                        // If src_id does not exist, return None
                        // otherwise, compute the incoming activation value
                        self.neurons.get(src_id).map(|n| {
                            (
                                synapse.weight * sigmoid(n.state, synapse.mu, synapse.sigma),
                                // Keep track of the reversal potentials for the next step
                                synapse.reversal_potential,
                            )
                        })
                    }
                })
                // Compute the numerator and denominator simultaneously
                .fold(
                    (0f32, 0f32),
                    |(running_numerator, running_denominator),
                     (synapse_value, reversal_potential)| {
                        (
                            running_numerator + synapse_value * reversal_potential,
                            running_denominator + synapse_value,
                        )
                    },
                );

            // Run single semi-implicit Euler step
            let numerator = state * config.membrane_capacitance / t
                + config.leakage_conductance * config.resting_potential * sensory_numerator;

            let denominator =
                config.membrane_capacitance / t + config.leakage_conductance + sensory_denominator;

            let next_state = numerator / denominator;

            next_states.insert(*id, next_state);
        }

        // Update all neurons with next state
        for (id, next_state) in next_states {
            self.neurons.entry(id).and_modify(|n| n.state = next_state);
        }
    }

    pub fn step(&mut self, activations: BTreeMap<u32, f32>) {
        let t: f32 = 1.0f32 / (Self::ODE_UNFOLDS as f32);

        // Run semi-implicit Euler to solve system of differential equations
        for _ in 0..Self::ODE_UNFOLDS {
            // Treat activation as a change over one full unit of time
            for (id, activation) in activations.iter() {
                self.neurons
                    .entry(*id)
                    .and_modify(|n| n.state += activation * t);
            }

            self.one_step(t);
        }
    }
}

/// Compute an offset and scaled sigmoid function.
fn sigmoid(x: f32, mu: f32, sigma: f32) -> f32 {
    let x = (mu - x) * sigma;
    let x = x.exp();
    x / (1f32 + x)
}

fn distance_between_neurons(a: &NeuronConfig, b: &NeuronConfig) -> f32 {
    todo!()
}

fn distance_between_neuron_and_none(a: &NeuronConfig) -> f32 {
    todo!()
}

fn distance_between_synapses(a: &SynapseConfig, b: &SynapseConfig) -> f32 {
    todo!()
}

fn distance_between_synapse_and_none(a: &SynapseConfig) -> f32 {
    todo!()
}

#[cfg(test)]
mod tests {
    use crate::{Network, NetworkConfig, NeuronConfig, SynapseConfig};

    #[test]
    fn test_network() {
        let config: NetworkConfig = NetworkConfig {
            neurons: [
                (
                    0u32,
                    NeuronConfig {
                        leakage_conductance: 1.0,
                        membrane_capacitance: 0.6,
                        resting_potential: 1.0,
                    },
                ),
                (
                    1u32,
                    NeuronConfig {
                        leakage_conductance: 1.0,
                        membrane_capacitance: 0.6,
                        resting_potential: 1.0,
                    },
                ),
                (
                    2u32,
                    NeuronConfig {
                        leakage_conductance: 1.0,
                        membrane_capacitance: 0.6,
                        resting_potential: 1.0,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            synapses: [
                (
                    0u32,
                    [(
                        2u32,
                        SynapseConfig {
                            weight: 1.0,
                            reversal_potential: 1.0,
                            sigma: 3.0,
                            mu: 0.3,
                            enabled: true,
                        },
                    )]
                    .into_iter()
                    .collect(),
                ),
                (
                    1u32,
                    [(
                        0u32,
                        SynapseConfig {
                            weight: 1.0,
                            reversal_potential: 2.0,
                            sigma: 3.0,
                            mu: 0.3,
                            enabled: true,
                        },
                    )]
                    .into_iter()
                    .collect(),
                ),
                (
                    2u32,
                    [(
                        1u32,
                        SynapseConfig {
                            weight: 1.0,
                            reversal_potential: -0.3,
                            sigma: 3.0,
                            mu: 0.3,
                            enabled: true,
                        },
                    )]
                    .into_iter()
                    .collect(),
                ),
            ]
            .into_iter()
            .collect(),
        };

        let mut network = Network::new(config);

        for i in 0..50 {
            network.step([(0u32, (i as f32 * 0.1).sin().abs())].into_iter().collect());

            println!(
                "{:.4} -> {:.4} -> {:.4}",
                network.neurons.get(&0).map(|n| n.state).unwrap(),
                network.neurons.get(&1).map(|n| n.state).unwrap(),
                network.neurons.get(&2).map(|n| n.state).unwrap(),
            );
        }
    }
}
