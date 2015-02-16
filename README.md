# subtype

Trajectory subtyping code.

## Example usage

```python
import logging
import random
import pandas as pd
import subtype

logging.basicConfig(level=logging.INFO)
random.seed('johnshopkins')

pfvc = pd.read_csv('data/pfvc_0-10.csv')
static_covariates = pd.read_csv('data/static_covariates.csv')

trajectories = subtype.make_trajectories(
    'ptid', 'years_seen', 'pfvc', pfvc)

trajectories = subtype.add_covariates(
    trajectories, 'ptid',
    formula = 'I(female == "yes") + I(race == "african_american")',
    dataframe = static_covariates
)

num_test = 10
random.shuffle(trajectories)
train_set, test_set = trajectories[num_test:], trajectories[:num_test]

mybasis = subtype.Basis(-1.0, 11.0, [5.0], 2)

mycov = subtype.CompositeCovariance(
    subtype.RandomInterceptCovariance(16.0),
    subtype.DiagonalCovariance(1.0)
)

model = subtype.SubtypeMixture(6, 3, mybasis, mycov).fit(train_set)
```
