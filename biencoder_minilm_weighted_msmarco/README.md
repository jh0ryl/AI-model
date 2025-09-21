---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4871
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: how to remove infopath form data to sharepoint list
  sentences:
  - Go to List Settings. In the General Settings section, pick Form Settings. You
    can then pick to use either the InfoPath form or the default SharePoint form,
    and optionally delete the InfoPath form if so desired.o to List Settings. In the
    General Settings section, pick Form Settings. You can then pick to use either
    the InfoPath form or the default SharePoint form, and optionally delete the InfoPath
    form if so desired.
  - Latintimes · a month ago. Mexican model Karla Karina Garza Garcia is dead after
    being murdered in her own home in Monterrey, Mexico . According to local reports,
    the suspects entered the house asking for Karla, who was in the kitchen with some
    family members, including her mother and brother. According to Breit Bart , Karina
    was an exclusive Diamond Girls model, an agency that represents models for dances,
    social events and TV shows. Save.
  - If a potential energy can be associated with a force, we call the force conservative.
    Examples of conservative forces are the spring force and the gravitational force.
    If a potential energy can not be associated with a force, we call that force non-conservative.
    An example is the friction force.
- source_sentence: what county is spring grove il
  sentences:
  - The best color to wear if you want to make a good first impression (like at a
    job interview or meeting the new dude's parents) is more personal than a hard-and-fast
    equation. There's no magic bullet answer for this question, Leatrice explained.
    Wear a color that makes you feel more confident, no matter what it is, and the
    feeling will spill over to anyone meeting you for the first time..
  - 'Loma Linda University Medical Center and Loma Linda University Children''s Hospital
    for Hospitalizations & Outpatients Services Monday - Thursday: 8 a.m. - 5 p.m.
    Friday: 8 a.m. - 2 p.m. Phone: 909-651-4191 Fax : 909-651-4180. Loma Linda University
    Health Clinic (Clinic Visits) Monday-Thursday: 8 a.m. - 5 p.m. Friday: 8 a.m.
    - 3 p.m. Phone: 909-651-4191 Fax: 909-558-2454. Medical Records at LLU Behavioral
    Medicine Center'
  - Spring Grove is a village in McHenry County, Illinois, United States. The population
    was 3,880 at the 2000 census. The population has grown to 5,303 as of 2005. It
    is also home to Chain O'Lakes State Park. The current village president is Mark
    Eisenberg. Spring Grove is home to Richardson's corn maze, known as the largest
    corn maze in the world. The pattern of the maze changes annually, often featuring
    Chicago sports teams.
- source_sentence: how many miles is the channel tunnel
  sentences:
  - On the English side of the channel, the train departs from Folkestone in the county
    of Kent. It arrives into the city of Coquelles, near Calais, on the French side.
    The tunnel is 31 miles long, with 23 of those miles beneath the surface of the
    water, traveling at an average depth of 150 feet below the seabed.
  - (March 2012). Oral hygiene is the practice of keeping the mouth and teeth clean
    to prevent dental problems, most commonly, dental cavities, gingivitis, periodontal
    (gum) diseases and bad breath.There are also oral pathologic conditions in which
    good oral hygiene is required for healing and regeneration of the oral tissues.March
    2012). Oral hygiene is the practice of keeping the mouth and teeth clean to prevent
    dental problems, most commonly, dental cavities, gingivitis, periodontal (gum)
    diseases and bad breath.
  - 'While they are designed to work even in cold temperatures, it can take a little
    while for the engine to turnover in severely cold temperatures. DieHard: DieHard
    batteries are top of the line. They were originally the longest lasting car battery
    as they lasted longer than your car. Cars last longer today than when the battery
    was first produced, but it still offers some of the best performance, even in
    bitterly cold weather. Be aware that DieHard batteries are expensive and not considered
    a budget or affordable battery. However considering they last so long, you will
    easily get your money''s worth.'
- source_sentence: what is hormone replacement therapy biote
  sentences:
  - 'But this belief is belied by the facts, which show that homelessness is caused
    by a complex interplay between a person''s individual circumstances and adverse
    ''structural'' factors outside their direct control. These problems can build
    up over years until the final crisis moment when a person becomes homeless. Personal
    causes of homelessness. A number of different personal and social factors can
    contribute towards people becoming homeless. These may include one or more of
    the following:'
  - What is BioTE® Therapy? BioTE® therapy is a type of hormone replacement therapy
    designed to use pellets derived from natural plant sources. These pellets are
    implanted beneath the skin, which in turn allow for them to slowly release a desirable
    amount of hormones over a prolonged period of time.
  - Make Davidson Village Inn, one of the leading Davidson bed and breakfasts, your
    destination. The town of Davidson, NC, features quaint shops, great restaurants,
    and family-friendly events – all within walking distance of Davidson College.
- source_sentence: what county is larchmont in
  sentences:
  - According to how it is treated, cellulose can be used to make paper, film, explosives,
    and plastics, in addition to having many other industrial uses. The paper in this
    book contains cellulose, as do some of the clothes you are wearing. For humans,
    cellulose is also a major source of needed fiber in our diet.
  - 'Warn is defined as to tell or caution against danger. An example of warn is to
    tell someone that there is a slippery road ahead. to tell (a person) of a danger,
    coming evil, misfortune, etc.; put on guard; caution. to caution about certain
    acts; admonish: warned against smoking in the building.'
  - Westchester County is a county in the U.S. state of New York. Westchester covers
    an area of 450 square miles (1,200 km 2), consisting of 48 municipalities. According
    to the 2010 Census, the county had a population of 949,113, estimated to have
    increased by 2.5% to 972,634 by 2014. Established in 1683, Westchester was named
    after the city of Chester, England.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a None-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** None dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): WeightedPooling(
    (attention): Linear(in_features=384, out_features=1, bias=True)
  )
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'what county is larchmont in',
    'Westchester County is a county in the U.S. state of New York. Westchester covers an area of 450 square miles (1,200 km 2), consisting of 48 municipalities. According to the 2010 Census, the county had a population of 949,113, estimated to have increased by 2.5% to 972,634 by 2014. Established in 1683, Westchester was named after the city of Chester, England.',
    'According to how it is treated, cellulose can be used to make paper, film, explosives, and plastics, in addition to having many other industrial uses. The paper in this book contains cellulose, as do some of the clothes you are wearing. For humans, cellulose is also a major source of needed fiber in our diet.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9928, 0.9649],
#         [0.9928, 1.0000, 0.9677],
#         [0.9649, 0.9677, 1.0000]])
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

* Size: 4,871 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                         | label                                                         |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             | float                                                         |
  | details | <ul><li>min: 4 tokens</li><li>mean: 8.42 tokens</li><li>max: 20 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 79.2 tokens</li><li>max: 200 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                   | sentence_1                                                                                                                                                                                                                                                                                              | label            |
  |:-----------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>bob par meaning</code>                                                 | <code>Bob Parr is the first human protagonist in a Pixar feature film. Bob Parr is also the first Pixar hero to kill people. Mostly for self-defense. Mr. Incredible is one of the few Pixar characters whose blood is shown, along with Dory and the construction worker Carl attacked from Up.</code> | <code>1.0</code> |
  | <code>how to download multiple nitroflare files from download manager</code> | <code>To add multiple files to your IDM queue, drag and drop the download URLs into the Drop Target window. Alternatively, select multiple URLs in a document or application, and then use the right-click context menu to add the files to the queue.</code>                                           | <code>1.0</code> |
  | <code>cost to repair transmission pump seal</code>                           | <code>more then likely the torque converter/front pump seal is laking and is the most common leak in this trans, the cost to repair is about $500 as the transmission must come out. Oct 25, 2010 \| 2002 Hyundai Santa Fe.</code>                                                                      | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `num_train_epochs`: 4
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
- `fp16`: False
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
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.6393 | 500  | 0.0208        |
| 3.2787 | 1000 | 0.0002        |


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