<div style="flex-layout: row;" >
  <img src="images/title.png"/>
  <img src="images/nn.png"/>
</div>

---

A NN library written with numpy no other dependencies, except Halo for the progress bar.

Currently supports dense networks and sequential models. 


## Installation
---
```bash
$ git clone https://github.com/JonasRSV/pynn
$ cd pynn
$ python3 setup.py install
```


Example

```python3
from nn.models import Sequential
from nn.layers import Dense, Input
from nn.optimisers import SGDOptimiser
from nn.losses import Msq
import numpy as np

model = Sequential()
# Always assumes 0th dimension is variable size
model.add(Input(shape=[5]))

# Takes generic functions as activation
model.add(Dense(256, np.tanh))
model.add(Dense(256, np.tanh))
model.add(Dense(5, lambda x: x))

model.compile(optimiser=SGDOptimiser(learning_rate=1e-3), loss=Msq())

X = np.random.rand(10000, 5)
Y = np.sin(X)

model.train(X, Y, batch_sz=64, epochs=10, shuffle=True)

# To predict using the model
# model.predict(X)
```

output

```bash
✔ epoch: 0 -- loss: 0.25688688639664803
✔ epoch: 1 -- loss: 0.15908453529774016
✔ epoch: 2 -- loss: 0.057677440343894805
✔ epoch: 3 -- loss: 0.048399488869382
✔ epoch: 4 -- loss: 0.048029080820167915
✔ epoch: 5 -- loss: 0.04776034483025997
✔ epoch: 6 -- loss: 0.04749981198670529
✔ epoch: 7 -- loss: 0.04718455336704899
✔ epoch: 8 -- loss: 0.0468329489105723
✔ epoch: 9 -- loss: 0.0464983623036258
```
