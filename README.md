# Stanford CS231n Convolutional Neural Networks for Visual Recognition

* Course website: <https://cs231n.github.io/>
* Assignment 1 (Spring 2018): <https://cs231n.github.io/assignments2018/assignment1/>
* Assignment 2 (Spring 2018): <https://cs231n.github.io/assignments2018/assignment2/>
* Assignment 3 (Spring 2018): <https://cs231n.github.io/assignments2018/assignment3/>

This repository contains my notes & solutions to the assignments.

*Please note that I was not enrolled for the course and my solution was not submitted, checked or graded.*

## Docker

* I run the assignments using a Docker container.
* For problems that don't require TensorFlow or PyTorch I use a basic miniconda-based container (based on the [`continuumio/miniconda3` Docker container](https://hub.docker.com/r/continuumio/miniconda3/)), which is set up in a way similar to what I have described [here](https://github.com/agisga/coding_notes/blob/master/docker.md).
    - To run the container:

        ```
        docker run -p 9999:8888 --name CS231n -v ~/github/my_CS231n/:/app/data cs231n
        ```

        where `cs231n` is the name of my Docker image.

    - To restart the container after it has shut down:

        ```
        docker start -ia CS231n
        ```

        where `CS231n` is the name of my Docker container.
* For problems that use PyTorch I either use [this Dockerfile](https://github.com/agisga/dockerfiles/blob/master/PyTorch-jupyter/Dockerfile) locally, or work on AWS without Docker (see below).

## AWS

* To run the more computationally heavy stuff that uses TensorFlow or PyTorch, I use AWS spot instances initialized with Amazon's "Deep Learning AMI (Ubuntu)" image. [Here is a description of my workflow](https://github.com/agisga/coding_notes/blob/master/AWS.md) (under the section "AWS Deep Learning AMI").
