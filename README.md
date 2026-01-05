# forest_gen

[![Documentation](https://img.shields.io/badge/github-Documentation-magenta?logo=github)](https://grafczterech.github.io/forest_gen/)
[![PyPi](https://img.shields.io/badge/PyPI-forest_gen-blue?logo=python)](https://pypi.org/project/forest_gen/)

Showcase forest scene generation module utilizing
[stripe_kit](https://github.com/GrafCzterech/STRIPE-kit). It's rather generic,
should allow you to generate all kinds of forests, but it is entirely focused
on generating forests.

The core functionality, to be used with `stripe_kit` is isolated as
`forest_gen`. Trying to import that module without running IsaacLab will result
in an error. The auxiliary logic for forest generation is isolated as
`forest_gen_utils`, and feel free to import it in non-IsaacLab environments.

Be sure to check out the
[documentation](https://grafczterech.github.io/forest_gen/) for more
information, specifically notes on the installation and usage.

If you want to quickly preview how the generated scene looks like, we have
a simple script `sim.py` at repo root that simply generates the scene and 
simulates it in IsaacLab. Running it is as simple as:

```bash
python sim.py
```

We also have included an example of a task definition utilizing `stripe_kit`
and `forest_gen` in `task/task.py`. You should be able to launch it as is
using `train.py` in a command like this:

```bash
python3 train.py --task=task --video --headless --num_envs=500
```
