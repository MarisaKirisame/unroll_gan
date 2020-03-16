# pytorch unrolled gan
This repository implements UNROLLED GENERATIVE ADVERSARIAL NETWORKS (https://arxiv.org/pdf/1611.02163.pdf).
This implementation is based on another implementation of unrolled gan https://github.com/andrewliao11/unrolled-gans.
The reason for the reimplementation is that the previous implementation did not unroll Discriminator's gradient,
but only used the simulated Discriminator as the new discriminator to update Generator. 

In this implementation, I used facebook's higher order gradient library higher https://github.com/facebookresearch/higher.


# Run
running script provided in `run.sh`
```sh
run.sh
```