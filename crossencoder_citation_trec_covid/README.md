---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:18849
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['are cardiac complications likely in patients with covid - 19?', '[ cit = 0. 381 ] epidemiologie et impact clinique des nouveaux virus respiratoires. nan'],
    ['are cardiac complications likely in patients with covid - 19?', '[ cit = 0. 461 ] elad, a deubiquitinating protease expressed by e. coli. background : ubiquitin and ubiquitin - like proteins ( ubl ) are designed to modify polypeptides in eukaryotes. covalent binding of ubiquitin or ubls to substrate proteins can be reversed by specific hydrolases. one particular set of cysteine proteases, the ce clan, which targets ubiquitin and ubls, has homologs in eukaryotes, prokaryotes, and viruses. findings : we have cloned and analyzed the e. coli protein elad, which is distantly related to eukaryotic ce clan members of the ulp / senp protease family that are specific for sumo and nedd8. previously misannotated as a putative sulfatase / phosphatase, elad is an efficient and specific deubiquitinating enzyme in vitro. interestingly, elad is present in all intestinal pathogenic e. coli strains, but conspicuously absent from extraintestinal pathogenic strains ( expecs ). further homologs of this protease can be found in acanthamoeba polyphaga mimivirus, and in alpha -, beta - and gammaproteobacteria. conclusion : the expression of ulp / senp - related hydrolases in bacteria therefore extends to plant pathogens and medically relevant strains of escherichia coli, legionella pneumophila, rickettsiae, chlamydiae, and salmonellae, in which the elad ortholog ssel has recently been identified as a virulence factor with deubiquitinating activity. as a counterpoint, our phylogenetic and functional examination reveals that ancient eukaryotic ulp / senp proteases also have the potential of ubiquitin - specific hydrolysis, suggesting an early common origin of this peptidase clan.'],
    ['how much impact do masks have on preventing the spread of the covid - 19?', '[ cit = 0. 996 ] covid 19 : impacts and implications for pediatric practice. since the rapid emergence of the novel coronavirus in december of 2019 and subsequent development of a global pandemic, clinicians around the world have struggled to understand and respond effectively and efficiently. with global response encompassing social, political, organizational, and economic realms, world leaders are struggling to keep pace with the rapid changes. challenges within global healthcare system and the healthcare profession itself include rationing supplies and services within health care systems, many of which were stretched to the brink before this latest viral outbreak ( american hospital association, 2020 ). leaders are making policy decisions while balancing the slow and precise nature of science with the rapid and pressing need for life - saving information ( altmann, douek, & boyton, 2020 ). shortcuts on research are occurring, including publishing papers with lack of peer review. social media and lurid reporting bolster feelings of mistrust and panic - buying while burgeoning conspiracy theories commandeer national dialogue. this is a time in history to prioritize global health and thoughtful pandemic preparedness ( lancet, 2020 ). pediatric nurse practitioners ( pnps ) are ideally situated to be a trusted source of accurate health information for children. this continuing education article summarizes the latest evidence - based information on the rapidly developing coronavirus pandemic ; equipping pnps for clinical preparation and response. 1.. distinguish risk factors for covid - 19 - related morbidity and mortality and identify modes of transmission. 2.. appraise appropriate covid - 19 testing parameters and procedures for children. 3.. compare pediatric clinical presentation to adults with covid - 19 infection and recommend appropriate treatment measures. 4.. state appropriate infection - control measures to reduce transmission. 5.. describe measures to reduce the risk of infection spread, mitigate adverse health effects in high - risk children, and to promote general health through preventive care.'],
    ['are there any clinical trials available for the coronavirus', '[ cit = 0. 307 ] the evolving world of protein - g - quadruplex recognition : a medicinal chemist ’ s perspective. abstract the physiological and pharmacological role of nucleic acids structures folded into the non canonical g - quadruplex conformation have recently emerged. their activities are targeted at vital cellular processes including telomere maintenance, regulation of transcription and processing of the pre - messenger or telomeric rna. in addition, severe conditions like cancer, fragile x syndrome, bloom syndrome, werner syndrome and fanconi anemia j are related to genomic defects that involve g - quadruplex forming sequences. in this connection g - quadruplex recognition and processing by nucleic acid directed proteins and enzymes represents a key event to activate or deactivate physiological or pathological pathways. in this review we examine protein - g - quadruplex recognition in physiologically significant conditions and discuss how to possibly exploit the interactions ’ selectivity for targeted therapeutic intervention.'],
    ['are patients taking angiotensin - converting enzyme inhibitors ( ace ) at increased risk for covid - 19?', '[ cit = 0. 524 ] diabetes and covid - 19 : evidence, current status and unanswered research questions.. patients with diabetes who get coronavirus disease 2019 ( covid - 19 ) are at risk of a severe disease course and mortality. several factors especially the impaired immune response, heightened inflammatory response and hypercoagulable state contribute to the increased disease severity. however, there are many contentious issues about which the evidence is rather limited. there are some theoretical concerns about the effects of different anti - hyperglycaemic drugs. similarly, despite the recognition of angiotensin converting enzyme 2 ( ace2 ) as the receptor for severe acute respiratory syndrome coronavirus 2 ( sars cov - 2 ), and the role of ace2 in lung injury ; there are conflicting results with the use of angiotensin converting enzyme ( ace ) inhibitors and angiotensin receptor blockers ( arb ) in these patients. management of patients with diabetes in times of restrictions on mobility poses some challenges and novel approaches like telemedicine can be useful. there is a need to further study the natural course of covid - 19 in patients with diabetes and to understand the individual, regional and ethnic variations in disease prevalence and course.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'are cardiac complications likely in patients with covid - 19?',
    [
        '[ cit = 0. 381 ] epidemiologie et impact clinique des nouveaux virus respiratoires. nan',
        '[ cit = 0. 461 ] elad, a deubiquitinating protease expressed by e. coli. background : ubiquitin and ubiquitin - like proteins ( ubl ) are designed to modify polypeptides in eukaryotes. covalent binding of ubiquitin or ubls to substrate proteins can be reversed by specific hydrolases. one particular set of cysteine proteases, the ce clan, which targets ubiquitin and ubls, has homologs in eukaryotes, prokaryotes, and viruses. findings : we have cloned and analyzed the e. coli protein elad, which is distantly related to eukaryotic ce clan members of the ulp / senp protease family that are specific for sumo and nedd8. previously misannotated as a putative sulfatase / phosphatase, elad is an efficient and specific deubiquitinating enzyme in vitro. interestingly, elad is present in all intestinal pathogenic e. coli strains, but conspicuously absent from extraintestinal pathogenic strains ( expecs ). further homologs of this protease can be found in acanthamoeba polyphaga mimivirus, and in alpha -, beta - and gammaproteobacteria. conclusion : the expression of ulp / senp - related hydrolases in bacteria therefore extends to plant pathogens and medically relevant strains of escherichia coli, legionella pneumophila, rickettsiae, chlamydiae, and salmonellae, in which the elad ortholog ssel has recently been identified as a virulence factor with deubiquitinating activity. as a counterpoint, our phylogenetic and functional examination reveals that ancient eukaryotic ulp / senp proteases also have the potential of ubiquitin - specific hydrolysis, suggesting an early common origin of this peptidase clan.',
        '[ cit = 0. 996 ] covid 19 : impacts and implications for pediatric practice. since the rapid emergence of the novel coronavirus in december of 2019 and subsequent development of a global pandemic, clinicians around the world have struggled to understand and respond effectively and efficiently. with global response encompassing social, political, organizational, and economic realms, world leaders are struggling to keep pace with the rapid changes. challenges within global healthcare system and the healthcare profession itself include rationing supplies and services within health care systems, many of which were stretched to the brink before this latest viral outbreak ( american hospital association, 2020 ). leaders are making policy decisions while balancing the slow and precise nature of science with the rapid and pressing need for life - saving information ( altmann, douek, & boyton, 2020 ). shortcuts on research are occurring, including publishing papers with lack of peer review. social media and lurid reporting bolster feelings of mistrust and panic - buying while burgeoning conspiracy theories commandeer national dialogue. this is a time in history to prioritize global health and thoughtful pandemic preparedness ( lancet, 2020 ). pediatric nurse practitioners ( pnps ) are ideally situated to be a trusted source of accurate health information for children. this continuing education article summarizes the latest evidence - based information on the rapidly developing coronavirus pandemic ; equipping pnps for clinical preparation and response. 1.. distinguish risk factors for covid - 19 - related morbidity and mortality and identify modes of transmission. 2.. appraise appropriate covid - 19 testing parameters and procedures for children. 3.. compare pediatric clinical presentation to adults with covid - 19 infection and recommend appropriate treatment measures. 4.. state appropriate infection - control measures to reduce transmission. 5.. describe measures to reduce the risk of infection spread, mitigate adverse health effects in high - risk children, and to promote general health through preventive care.',
        '[ cit = 0. 307 ] the evolving world of protein - g - quadruplex recognition : a medicinal chemist ’ s perspective. abstract the physiological and pharmacological role of nucleic acids structures folded into the non canonical g - quadruplex conformation have recently emerged. their activities are targeted at vital cellular processes including telomere maintenance, regulation of transcription and processing of the pre - messenger or telomeric rna. in addition, severe conditions like cancer, fragile x syndrome, bloom syndrome, werner syndrome and fanconi anemia j are related to genomic defects that involve g - quadruplex forming sequences. in this connection g - quadruplex recognition and processing by nucleic acid directed proteins and enzymes represents a key event to activate or deactivate physiological or pathological pathways. in this review we examine protein - g - quadruplex recognition in physiologically significant conditions and discuss how to possibly exploit the interactions ’ selectivity for targeted therapeutic intervention.',
        '[ cit = 0. 524 ] diabetes and covid - 19 : evidence, current status and unanswered research questions.. patients with diabetes who get coronavirus disease 2019 ( covid - 19 ) are at risk of a severe disease course and mortality. several factors especially the impaired immune response, heightened inflammatory response and hypercoagulable state contribute to the increased disease severity. however, there are many contentious issues about which the evidence is rather limited. there are some theoretical concerns about the effects of different anti - hyperglycaemic drugs. similarly, despite the recognition of angiotensin converting enzyme 2 ( ace2 ) as the receptor for severe acute respiratory syndrome coronavirus 2 ( sars cov - 2 ), and the role of ace2 in lung injury ; there are conflicting results with the use of angiotensin converting enzyme ( ace ) inhibitors and angiotensin receptor blockers ( arb ) in these patients. management of patients with diabetes in times of restrictions on mobility poses some challenges and novel approaches like telemedicine can be useful. there is a need to further study the natural course of covid - 19 in patients with diabetes and to understand the individual, regional and ethnic variations in disease prevalence and course.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 18,849 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                         | label                                                         |
  |:--------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                             | float                                                         |
  | details | <ul><li>min: 13 characters</li><li>mean: 68.78 characters</li><li>max: 171 characters</li></ul> | <ul><li>min: 26 characters</li><li>mean: 1340.27 characters</li><li>max: 2867 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.6</li><li>max: 2.0</li></ul> |
* Samples:
  | sentence_0                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>are cardiac complications likely in patients with covid - 19?</code>             | <code>[ cit = 0. 381 ] epidemiologie et impact clinique des nouveaux virus respiratoires. nan</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>0.0</code> |
  | <code>are cardiac complications likely in patients with covid - 19?</code>             | <code>[ cit = 0. 461 ] elad, a deubiquitinating protease expressed by e. coli. background : ubiquitin and ubiquitin - like proteins ( ubl ) are designed to modify polypeptides in eukaryotes. covalent binding of ubiquitin or ubls to substrate proteins can be reversed by specific hydrolases. one particular set of cysteine proteases, the ce clan, which targets ubiquitin and ubls, has homologs in eukaryotes, prokaryotes, and viruses. findings : we have cloned and analyzed the e. coli protein elad, which is distantly related to eukaryotic ce clan members of the ulp / senp protease family that are specific for sumo and nedd8. previously misannotated as a putative sulfatase / phosphatase, elad is an efficient and specific deubiquitinating enzyme in vitro. interestingly, elad is present in all intestinal pathogenic e. coli strains, but conspicuously absent from extraintestinal pathogenic strains ( expecs ). further homologs of this protease can be found in acanthamoeba polyphaga mimivirus, and in al...</code> | <code>0.0</code> |
  | <code>how much impact do masks have on preventing the spread of the covid - 19?</code> | <code>[ cit = 0. 996 ] covid 19 : impacts and implications for pediatric practice. since the rapid emergence of the novel coronavirus in december of 2019 and subsequent development of a global pandemic, clinicians around the world have struggled to understand and respond effectively and efficiently. with global response encompassing social, political, organizational, and economic realms, world leaders are struggling to keep pace with the rapid changes. challenges within global healthcare system and the healthcare profession itself include rationing supplies and services within health care systems, many of which were stretched to the brink before this latest viral outbreak ( american hospital association, 2020 ). leaders are making policy decisions while balancing the slow and precise nature of science with the rapid and pressing need for life - saving information ( altmann, douek, & boyton, 2020 ). shortcuts on research are occurring, including publishing papers with lack of peer review. soc...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 6
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 6
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.8475 | 500  | 0.0774        |
| 1.6949 | 1000 | -1.0548       |
| 2.5424 | 1500 | -1.5367       |
| 3.3898 | 2000 | -2.0061       |
| 4.2373 | 2500 | -2.379        |
| 5.0847 | 3000 | -2.6143       |
| 5.9322 | 3500 | -2.7649       |


### Framework Versions
- Python: 3.12.11
- Sentence Transformers: 5.1.0
- Transformers: 4.56.1
- PyTorch: 2.8.0+cu126
- Accelerate: 1.10.1
- Datasets: 4.0.0
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->