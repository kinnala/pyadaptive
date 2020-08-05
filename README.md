# pyadaptive

Red-green-blue refinement of a triangular mesh:

```python
from pyadaptive import refine

p = [[0., 0.],
     [0., 1.],
     [1., 0.],
     [1., 1.]]

t = [[0, 1, 2],
     [1, 2, 3]]

p, t = refine(p, t, [0])  # refine first element
```

![Refined mesh](https://user-images.githubusercontent.com/973268/89417284-290b4900-d737-11ea-93f6-612e09e3fb8c.png)
