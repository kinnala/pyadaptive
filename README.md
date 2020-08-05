# pyadaptive

```python
from pyadaptive import refine

p = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

t = [[0, 1, 2],
     [1, 2, 3]]

for _ in range(10):
    p, t = refine(p, t, [1])
```
