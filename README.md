# HoloBundle - A GUI for fibre bundle inline holographic microscopy in Python

HoloBundle is developed at [Mike Hughes' lab'](https://research.kent.ac.uk/applied-optics/hughes) in the [Applied Optics Group](https://research.kent.ac.uk/applied-optics/), School of Physics & Astronomy, 
University of Kent.

HoloBundle is a graphical user interface for performing inline holographic
microscopy through fibre imaging bundles. You can read about the technology
in these papers:
* [Inline holographic microscopy through fiber imaging bundles](https://arxiv.org/pdf/2006.09296.pdf)
* [Improved resolution in fiber bundle inline holographic microscopy using multiple illumination sources](https://preprints.opticaopen.org/ndownloader/files/43619601)

HoloBundle is built on [CAS](https://www.github.com/mikehugheskent/cas),  [PyFibreBundle](https://www.github.com/mikehugheskent/pyfibrebundle) and [PyHoloscope](https://www.github.com/mikehugheskent/pyholoscope). These packages must be 
set up in a folder structure as follows unless you edit the relative paths
at the top of holobundle.py (or have all 
these packages in your system path):

      

```bash
├── Top Level Folder
│   ├── HoloBundle
│   │   ├── src
│   ├── pyfibrebundle
│   │   ├── src
│   ├── pyholoscope
│   │   ├── src
│   ├── cas
│   │   ├── src

```

You are welcome to make use of HoloBundle in your own applications, but please
be aware that it is only partially documented. Academic collaborations are 
welcomed, please get in touch.

If you are looking for a documented package for holographic microscopy 
processing to build you own processing pipeline/GUI around, take a look at 
[PyHoloscope](https://www.github.com/mikehugheskent/pyholoscope).  

Or for fibre bundle imaging systems in general [PyFibreBundle](https://www.github.com/mikehugheskent/pyfibrebundle).

There is also a version of the GUI for conventional inline holographic microscopy:
[HoloSnake](https://www.github.com/mikehugheskent/holosnake).

## Requirements

HoloBundle requires:
* PIL (Python Image Library)
* OpenCV
* PyQt5
* Numpy
* Matplotlib
* pyserial
* scikit-image

In addition. CAS requirements depend on the specific camera used ([CAS](https://www.github.com/mikehugheskent/cas)).

