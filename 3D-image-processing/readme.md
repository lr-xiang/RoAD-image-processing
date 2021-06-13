# Project Title

RoAD 3D image processing

## Description

This is a program developed for processing 3D images collected by a laser sacnner in RoAD system. 

## Getting Started

### Dependencies

* OpenCV
* Point Cloud Library
* CMake

### Installing

* Clone the repo
```
git clone https://github.com/lr-xiang/RoAD-image-processing.git
```

* Go to project folder
```
cd RoAD-image-processing/3D-image-processing
```
* Build the programs
```
cd build
cmake ..
make
```

### Executing program

* Multiview point clouds registration
```
usage: ./merge [experiment folder] [experiment number]
```

* Merged point cloud traits extraction
```
usage: ./measure [experiment folder] [experiment number]
```

* Leaf segmentation
```
usage: ./leaf [experiment folder] [experiment number]
```

## Authors

[@XiangLirong](https://twitter.com/xianglirong)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

