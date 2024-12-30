use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct NeuronConfig {
    pub leakage_conductance: f32,
    pub membrane_capacitance: f32,
    pub resting_potential: f32
}

#[derive(Debug, Clone, Copy)]
pub struct SynapseConfig {
    pub weight: f32,
    pub reversal_potential: f32,
    pub sigma: f32,
    pub mu: f32
}

pub struct Neuron {
    pub state: f32,
    pub config: NeuronConfig
}

pub struct NetworkConfig {
    pub neurons: BTreeMap<u32, NeuronConfig>,

    // First key is destination neuron, second key is source neuron
    pub synapses: BTreeMap<u32, BTreeMap<u32, SynapseConfig>>
}

pub struct Network {
    pub config: NetworkConfig,
    pub neurons: BTreeMap<u32, Neuron>
}

impl Network {
    const ODE_UNFOLDS: usize = 16;

    pub fn new(config: NetworkConfig) -> Self {
        let neurons: BTreeMap<u32, Neuron> = config.neurons
            .iter()
            .map(|(id, config)| (*id, Neuron { state: 0f32, config: *config }) )
            .collect();

        Network {
            config,
            neurons
        }
    }

    fn one_step(&mut self, t: f32) {
        // Bucket to hold computed values
        let mut next_states = BTreeMap::new();

        for (id, neuron) in self.neurons.iter() {
            let state = neuron.state;
            let config = neuron.config;

            // Collect incoming synapses (if any)
            let Some(incoming_synapses) = self.config.synapses.get(id) else { continue };

            // Precompute effects of sensory neurons
            let (sensory_numerator, sensory_denominator) = incoming_synapses
                .iter()
                .filter_map(|(src_id, synapse)| { 
                    self.neurons
                        .get(src_id)
                        .map(|n| (
                            synapse.weight * sigmoid(n.state, synapse.mu, synapse.sigma), 
                            synapse.reversal_potential
                        ))
                })
                .fold((0f32, 0f32), |(n, d), (x, e)| (n + x * e, d + x));
            
            // Run single semi-implicit Euler step
            let numerator = 
                state * config.membrane_capacitance / t + 
                config.leakage_conductance * config.resting_potential * sensory_numerator;
            
            let denominator = 
                config.membrane_capacitance / t +
                config.leakage_conductance +
                sensory_denominator;

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
            for (id, activation) in activations.iter() {
                self.neurons.entry(*id).and_modify(|n| n.state += activation * t);
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

#[cfg(test)]
mod tests {
    use crate::{Network, NetworkConfig, NeuronConfig, SynapseConfig};
    
    #[test]
    fn test_network() {

        let config: NetworkConfig = NetworkConfig {
            neurons: [
                (0u32, NeuronConfig { leakage_conductance: 1.0, membrane_capacitance: 0.6, resting_potential: 1.0 }),
                (1u32, NeuronConfig { leakage_conductance: 1.0, membrane_capacitance: 0.6, resting_potential: 1.0 }),
                (2u32, NeuronConfig { leakage_conductance: 1.0, membrane_capacitance: 0.6, resting_potential: 1.0 }),
            ].into_iter().collect(),
            synapses: [
                (
                    0u32, 
                    [
                        (2u32, SynapseConfig { weight: 1.0, reversal_potential: 1.0, sigma: 3.0, mu: 0.3 }),
                    ].into_iter().collect()
                ),
                (
                    1u32, 
                    [
                        (0u32, SynapseConfig { weight: 1.0, reversal_potential: 2.0, sigma: 3.0, mu: 0.3 }),
                    ].into_iter().collect()
                ),
                (
                    2u32, 
                    [
                        (1u32, SynapseConfig { weight: 1.0, reversal_potential: -0.3, sigma: 3.0, mu: 0.3 }),
                    ].into_iter().collect()
                ),
            ].into_iter().collect()
        };

        let mut network = Network::new(config);

        for i in 0..50 {
            network.step([
                (0u32, (i as f32 * 0.1).sin().abs())
            ].into_iter().collect());

            println!(
                "{:.4} -> {:.4} -> {:.4}",
                network.neurons.get(&0).map(|n| n.state).unwrap(),
                network.neurons.get(&1).map(|n| n.state).unwrap(),
                network.neurons.get(&2).map(|n| n.state).unwrap(),
            );
        }
    }
}

