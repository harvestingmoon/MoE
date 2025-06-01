The purpose of this Repo is to recreate an entire FrankenMoE using only raw Torch

we will develop by first making LLama 3.2 in Torch => then we will pull download various torch weights and models (about 3~4) => then we will merge them via FrankenMoE with a TopK approach routing algorithm or another algorithm we can find

Challenges:
- FrankenMoE initialisation
(we might need to use TopK approach) 
- Managing the loadbalancing from the router itself (ensure that the expert-specific parameters do not get under-utilized)

