The purpose of this Repo is to recreate an entire FrankenMoE using only raw PyTorch which will then be eventually ran locally on a compute cluster with a combined VRAM of 17gb as well as 32gb of CPU

What is a frankenMoe:
They are not true MoE's as they are not trained together with the router, traditionally experts are simple FFN. In the context of this, we are sort of creating a small system that mimicks how MoEs would be like. 

Take note that in a real setting , this is what a true MoE would look like (it is part of the transformer block specifically in the FFN area)


```

                        x (Input to Transformer Block)
                              |
        +---------------------+---------------------+
        |                     |                     |
        V                     V                     V
    +-----------------+ +-----------------+ +-----------------+
    |   Layer Norm    | |    Attention    | |     (Input x)   |
    +-----------------+ +-----------------+ +-----------------+
              |                     |                     |
              +---------------------+                     |
                        |                                 |
                        V                                 |
           (Output of Attention + Residual)               |
                        |                                 |
                        +---------------------------------+
                        |        (1st Residual Connection)
                        V
           +-------------------------+
           |       Layer Norm        | (Pre-MoE Normalization)
           +-------------------------+
                        |
                        V
           +-------------------------+
           |         Router          |
           |   (Gating Network)      |
           +-------------------------+
                        |
            +-----------+-----------+-----------------------+
            |           |           |                       |
            V           V           V                       V
    +-------------+-------------+-------------+ ... +-------------+
    |   FFN 1     |   FFN 2     |   FFN 3     |     |   FFN       |          
    +-------------+-------------+-------------+ ... +-------------+
            |           |           |           |           |
            +-----------+-----------+-----------+-----------+
                        |
                        V
           +-------------------------+
           |      Weighted Sum       | (Combines Expert Outputs)
           +-------------------------+
                        |
          +-------------+---------------------+
          |             |                     |
          V             |                     V
    (Output of MoE)     |                     | (Input to 2nd Residual)
          +-------------+---------------------+
          |        + (2nd Residual Connection)
          +-------------------------+
                        |
                        V
             Output of Transformer Block
```


This is what a FrankenMoE looks like


```
                     Input Sequence
                           |
                           V
              +---------------------------+
              |          Router           |
              |     (Gating Network)      |
              +---------------------------+
                           |
                           V
              +---------------------------+
              |        [Decision]         |
              |  (e.g., based on input    |
              |   tokens or sequence)      |
              +---------------------------+
                           |
              +---------------------------+
              |          (Weights)          |
              | (How much to use each expert)|
              +---------------------------+
                           |
          +-------+-------+-------+-------+
          |       |       |       |       |
          V       V       V       V       V
  +----------+ +----------+ +----------+ ... +----------+
  | Expert A | | Expert B | | Expert N |     |  ...     |
  |Llama 3.2 | |Llama 3.2 | |Llama 3.2 |     |          |
  +----------+ +----------+ +----------+ ... +----------+
  ```



Current Work:
- Working on the Llama 3.2 individual models

Next work: 
- Init the llama3.2 models 
- Selecting which finetuned models to work on 
- Constructing the FrankenMoE architecture 
- Knowing which gating network to choose simple MLP with TopK approach then we train it using QA pairs?
- Knowing what kind of routing architecture and decision & weights we have to do 
- Knowing how to do load balancing (suggestion is to stick to an Auxillary Load Balancing Loss)


VRAM Allocation:
- 2gb / expert 
- 8gb in Total for experts (Assuming we are using 4 experts)
- 500mb - 2gb for Routing & Gating network (Assuming we are using a simple 2 layer MLP)

Challenges:
- FrankenMoE initialisation
(we might need to use TopK approach) 

- Managing the loadbalancing from the router itself (ensure that the expert-specific parameters do not get under-utilized)



