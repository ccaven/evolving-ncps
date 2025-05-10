pub struct Evaluated<T> {
    pub genome: T,
    pub fitness: f32
}

pub trait Gene<C>: Clone {
    fn mutate(self, config: &C) -> Self;
    fn crossover(&self, other: &Self, config: &C) -> Self;
    fn distance(&self, other: &Self, config: &C) -> f32;
}

pub trait Genome<C> where Self: Sized {
    fn crossover(a: Evaluated<Self>, b: Evaluated<Self>, config: &C) -> Self;
    fn mutate(self, config: &C) -> Self;
    fn distance(&self, other: Self, config: &C) -> f32;
    fn descriptor(&self, config: &C) -> String;
}

pub trait Evaluate<T> {
    fn evaluate(&self, genome: T) -> Evaluated<T>;
}