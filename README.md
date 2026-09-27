# Anisotropic Green Coordinates

This repository contains reference implementations of **Anisotropic Green Coordinates (AGC)** for cage-based and variational shape deformation.

The repository provides:

- A 2D Python implementation for image deformation and 2D variational deformation.
- A 3D C++ implementation for cage-based and variational shape deformation.


---

# 2D Deformation
## Cage-based Deformation

Python 3.10 or newer is recommended.

```bash
pip install numpy scipy matplotlib imageio pillow
```

The default input layout is:

```text
data/
└── giraffe/
├── img.png
├── cage.png
└── cage_deformed.txt
```

Each consecutive pair of numbers in `cage.txt` defines one image-space vertex `(x, y)`. Vertices must form a simple polygon in boundary order. Do not repeat the first vertex at the end. A target cage must contain the same number of vertices in the same order.

Examples:
```bash
python .\AGC_2D.py --folder cirrec --A 2.0 0.0 0.0 1.0
python .\AGC_2D.py --folder giraffe --A 1.5 0.5 0.5 1.5
```

### GUI Controls

- Drag cage vertices in the left panel.
- Click **Apply deformation** to update both panels.
- Click **Save cage** to export the current target cage.
- Click **Load target cage** to load and apply a target cage file.

The left panel uses classical Green Coordinates. The right panel uses AGC with the selected anisotropy matrix.

## Variational Shape Deformation

```text
data/
└── tower/
├── img.png
├── cage.txt
└── constraints.txt
```


A constraint file contains three sections:

```text
user_constraints_origin = [x1,y1; x2,y2; ...];
user_constraints_deformed = [x1,y1; x2,y2; ...];
medial_points = [x1,y1; x2,y2; ...];
```

Examples:
```bash
python .\Variational_2D.py --folder tower --A 1.0 0.0 0.0 4.0 --check-derivatives
python .\Variational_2D.py --folder strip --A 1.0 0.0 0.0 4.0 --samples-per-edge 2
```
When the `--check-derivatives` flag is enabled, we compare the closed-form gradient and Hessian of the coordinates against the numerically computed results.

---

# 3D Deformation

The 3D implementation is written in C++ and supports 3D anisotropic Green Coordinates and variational shape deformation with OBJ input and OBJ output. Our code framework is mainly inherited from the excellent work [CageModeler](https://github.com/DanStroeter/CageModeler). We sincerely thank the authors for their contributions—our work truly stands on the shoulders of giants.

We only retain the parts required by this algorithm, so vcpkg is not needed. We only need to download Eigen3 to the external folder before compiling. 

Using CMake to generate the project. Here, we specifically introduce the experience of configuring with CMake-gui and Visual Studio in Windows.
In the root directory of the project, run:
```bash
cmake-gui
```

Where is the source code: Choose the root directory of this project (where `main.cpp` is located).
Where to build the binaries: Create a `build` folder in the root directory and select this folder. 
Then, click `Configure`, select `Visual Studio` and use default settings. After that, click `Generate` in cmake-gui, and the Visual Studio project `cageDeformation3D.sln` will be generated in the `build` directory. Select the `RelWithDebInfo` mode, build the solution, and the `cageDeformation3D.exe` will be generated. 


Examples (Note: change `/path/to` to the specific path in your device):
```bash
./cageDeformation3D.exe cage-based path/to/data/Bench -A 1 0 0 0 0.25 0 0 0 1
./cageDeformation3D.exe cage-based path/to/data/Ogre -A 2 0 0 0 1 0 0 0 1
./cageDeformation3D.exe cage-based path/to/data/Bar -A 1 0 0 0 1 0 0 0 0.111111
./cageDeformation3D.exe variational path/to/data/Bar_variational -A 1 0 0 0 1 0 0 0 0.25
./cageDeformation3D.exe variational path/to/data/Bar_variational2 -A 1 0 0 0 4 0 0 0 9
```
The results will be saved at `./data/example_name/Results_CageBased(Variational).obj`

The check of the closed-form and numerical results of the gradient and Hessian is implemented in the `debugCompareNumericalAndClosedForm` function. To test this function, comment out lines 538–542 in `GreenCoordinates.cpp`.