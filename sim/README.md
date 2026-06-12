# Five-Bar SCARA Simulator

Interactive matplotlib model of the five-bar parallel linkage used by this plotter.

## Run

```bash
python simulator.py
# with initial values (all still adjustable in the window):
python simulator.py --mode ik --distance 8 --proximal 6 --distal 9
```

| CLI flag | meaning | default |
|----------|---------|---------|
| `--mode` | `fk` (drive angles) or `ik` (drive pen) | `fk` |
| `--distance` | distance between the two motors (`d`) | `8` |
| `--proximal` | proximal link length (`L1`) | `6` |
| `--distal` | distal link length (`L2`) | `9` |

## Controls

- **Mode** — switch between Forward Kinematics (drive `theta1`/`theta2`) and
  Inverse Kinematics (drive pen `X`/`Y`). The inactive mode's sliders show the
  computed read-back values.
- **d / L1 / L2** — change the geometry live.
- **Trace** — toggle on to record the pen path as you move it; **Clear** erases it.
- **Show workspace** — flood the reachable area to visualise the robot's range.

The status line reports the current angles, pen position, and geometry;
unreachable poses are flagged in red.
