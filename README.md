# forest-gen

Our scene generation module.

## Installation

Run the ```install.sh``` script if you are on a Linux system. May Ritchie have
mercy on thy soul if you want to run this on Windows.

The automatic installation won't install Isaac Lab, a necessary element for
getting this running. However, you may well use the mesh generation and
noise function without installing Isaac Lab, for ease of development
there is no overarching facade module, as such Python won't mind as long as
you stray from using modules importing Isaac Lab stuff. More specifically,
the ```heightmap``` module is written with no Isaac Lab imports and may be
tested separately.

## Heightmap

The underlying terrain is generated using
[OpenSimplex](https://en.wikipedia.org/wiki/OpenSimplex_noise) noise. It
suposedly is a superior iteration on Perlin noise.

## Assets

Trees for rudimentary simulation can be found under this address:
[TreesPackage](https://drive.google.com/file/d/1LULIDdkIpjy51-J21E4exNFdomADHbqy/view?usp=sharing)
