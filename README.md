# pyadaptive

Refine a triangular mesh:

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

![Refined mesh](https://user-images.githubusercontent.com/973268/89409977-9add9580-d72b-11ea-87fc-de5f556eb008.png)
