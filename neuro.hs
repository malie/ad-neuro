import Numeric.AD

import qualified Data.List as L

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


-- http://343hz.com/general-guidelines-for-deep-neural-networks/
-- claims: only positive random initial weights
initialNetwork = [
  -- layer 2
  [ [0.1, 0, 0.2],
    [0.3, 0.1, 0.3],
    [0.7, 0.1, 0.1]],

  -- layer 3
  [ [0.5, 0.5, 0.3, 0.1]]]


trainingData =
  [ ([0,0], [0]),
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [0]) ]


propagate :: Layers -> Inputs -> LayersOutput
propagate [] inputs = []
propagate (layer:layers) inputs =
  let layerOutput = map (flip activation (1:inputs)) layer
  in layerOutput : propagate layers layerOutput



-- the deltas
type LayersD = [LayerD]
type LayerD = [NeuronD]
type NeuronD = [Double]



type Errors = [[Double]]

reluder :: Double -> Double
reluder act | act <= 0  = 0
            | otherwise = 1.0

backpropagate :: Layers -> LayersOutput -> LayerOutput -> Errors
backpropagate [] [outputs] target =
  let outputError o t = reluder o * (t - o)
  in [zipWith outputError outputs target]
backpropagate (layer:layers) (outputs:moreOutputs) target =
  let errorsAbove@(errorAbove:_) =
        backpropagate layers moreOutputs target
      e (weights, output) =
        reluder output * sum (zipWith (*) errorAbove weights)
      layerT = tail $ L.transpose layer
      errors = map e $ zip layerT outputs
  in  errors : errorsAbove
backpropagate ls os t =
  error (show ("backpropagate", ls, os, t))


test = mapM_ test trainingData
  where
    net = initialNetwork
    test (inputs, targetOutputs) =
      do print (inputs, targetOutputs)
         let lo = propagate net inputs
         print ("prop", lo)
         let errors = backpropagate net ([]:lo) targetOutputs
         print ("errors", errors)
