# Stanford CS231n Convolutional Neural Networks for Visual Recognition

* Course website: <https://cs231n.github.io/>
* Assignment 1 (Spring 2018): <https://cs231n.github.io/assignments2018/assignment1/>
* Assignment 2 (Spring 2018): <https://cs231n.github.io/assignments2018/assignment2/>

This repository contains my notes & solutions to the assignments.

## Docker

* I run the assignments using a Docker container.
* Use the setup instructions for the miniconda-based container [here](https://github.com/agisga/coding_notes/blob/master/docker.md).
* To run the container:

    ```
    docker run -p 9999:8888 --name CS231n -v ~/github/my_CS231n/:/app/data cs231n
    ```

    where `cs231n` is the name of my Docker image.

* To restart the container after it has shut down:

    ```
    docker start -ia CS231n
    ```

    where `CS231n` is the name of my Docker container.
