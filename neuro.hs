import Numeric.AD

type Layers = [Layer]
type Layer = [Neuron]
type Neuron = [Double]

type Inputs = [Double]
type Output = Double
type LayerOutput = [Output]
type LayersOutput = [LayerOutput]

activation :: Neuron -> Inputs -> Output
activation weights inputs =
  max 0 $ sum $ zipWith (*) weights inputs

initialNetwork = [
  -- layer 1
  [ [0.1, 0, 0.2],
    [0.3, 0.1, 0.3],
    [0.7, -0.1, 0.1]],

  -- layer 2
  [ [0.5, -0.5, 0.3, -0.1]]]


propagate :: Layers -> Inputs -> LayersOutput
propagate [] inputs = []
propagate (layer:layers) inputs =
  let layerOutput = map (flip activation (1:inputs)) layer
  in layerOutput : propagate layers layerOutput

