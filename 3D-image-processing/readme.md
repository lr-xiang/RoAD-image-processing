# Project Title

RoAD 3D image processing

## Description

This is a program developed for processing 3D images collected by a laser sacnner in RoAD system. 

## Getting Started

### Dependencies

* OpenCV
* Point Cloud Library
* CMAKE

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

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
