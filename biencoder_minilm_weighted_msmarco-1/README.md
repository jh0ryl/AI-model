---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:49496
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: what are dandelion flowers good for
  sentences:
  - The definition of Celtic music must start with a definition of what a Celt is.
    The origin of the term goes back to the word keltoi, which was used by the ancient
    Greeks to refer to certain barbarian tribes. In modern terminology, Celtic is
    used to refer to a certain branch of the Indo-European language group. Some of
    the tribes called keltoi spoke Celtic languages. However, it is not certain that
    the Greeks applied keltoi only to Celtic-speaking tribes.
  - Indeed, the average person can’t hold their breath for more than 30 seconds or
    so — and even someone in excellent health is gasping by two minutes. Sietas pictured
    at the start (left) and finish (right) of the new breath-holding record of 18
    minutes and 16 seconds in a water tank. Most Japanese pearl divers, who dive deep
    without oxygen for their treasure, can’t manage more than seven minutes.
  - Dandelion root can by ground and used as a substitute for coffee, and dandelion
    flowers can be used in recipes and for garnish. The French have a well-known soup
    called creme de pissenlits (cream of dandelion soup), which is easy to make, as
    is Dandelion Syrup.andelion root can by ground and used as a substitute for coffee,
    and dandelion flowers can be used in recipes and for garnish. The French have
    a well-known soup called creme de pissenlits (cream of dandelion soup), which
    is easy to make, as is Dandelion Syrup.
- source_sentence: weight lifting bar dimensions
  sentences:
  - 'While 125 MB/s might not sound as impressive as the word gigabit, think about
    it: a network running at this speed should be able to theoretically transfer a
    gigabyte of data in a mere eight seconds. A 10 GB archive could be transferred
    in only a minute and 20 seconds. This speed is incredible, and if you need a reference
    point, just recall how long it took the last time you moved a gigabyte of data
    back before USB keys were as fast as they are today.'
  - These bars are used in Olympic weightlifting competitions because they hold so
    much weight. They are 7 feet long and weigh 45 lbs. The center of the bar is a
    little over 1 inch thick, but the sleeves, the end of the bar that hold the weight
    plates, are 2 inches in diameter. You can purchase Olympic bars that are shorter
    than the standard, 7 foot length.
  - Home / Products / Cable Hose Chokers. Cable Hose Chokers. The CABLE CHOKER has
    a specially designed nylon spool which allows the cable to tighten down on the
    hose during a failure. Unlike the WHIP CHECK or the steel Hobble Clamp, the CABLE
    CHOKER will continue tightening down sometimes even choking off the air for a
    more controlled release.
- source_sentence: what are nitrites in urine
  sentences:
  - Therefore the presence of nitrites in urine is generally an indication of the
    presence of a urinary tract infection. A urinary tract infection may occur when
    the bacteria enter the urinary system through the bloodstream or via skin contact
    of the urethra with bacteria during urination.he results of a urinalysis identify
    and verify a number of different parameters, and nitrites in urine is one of them.
    Nitrites in urine is a condition that is different from nitrates in urine. The
    content of nitrates in urine is quite low.
  - The definition of a throttle is a valve that controls the fluid flow. An example
    of a throttle is a car part that controls how fuel moves through the system. Throttle
    is defined as to control the speed of an engine or the flow of fuel in an engine,
    or to hold down or choke.
  - 1 The average golf cart-electric, with a top and windshield, goes for between
    $5,000 and $7,000. 2  This will get you a basic electric cart and some useful
    add-ons such as a cooler or club covers. 3  Buyers reported purchasing used vehicles
    for between $2,500 and $5,000.1hp gas golf cart with lift kit, flip-back seat,
    brush guard, headlights, tail/brake lights, turn signals, rear-view mirror, seat
    belts, windshield, towing package, and extended top.. -- Apalachicola, Florida.
    Yamaha 2006 g22, gasoline, with light kit, full side covers, and shipping to my
    door -- $4,645.
- source_sentence: ifb contract definition
  sentences:
  - In most years, Seattle averages a daily maximum temperature for September that's
    between 69 and 73 degrees Fahrenheit (20 to 23 degrees Celsius). The minimum temperature
    usually falls between 52 and 55 °F (11 to 13 °C). The days at Seattle begin to
    cool down quickly during September.
  - invitation for bid (IFB) A part of the sealed bid process. Companies or government
    organizations give in-depth specifications for projects and invite contractors
    to bid their proposals for various projects. The sealed bids are opened at the
    predetermined time and recorded. A contract is awarded to the best bid.
  - 'Among some of the current statistics: 1  Since the beginning of the epidemic,
    39 million have died of AIDS-related causes.  Of the 37 million people living
    with HIV today, an estimated 1.5 million died in 2013, a drop of 22% from 2009
    estimates and 35% from 2005.'
- source_sentence: composing numbers definition
  sentences:
  - Demonstration, with Cuisenaire rods, of the divisors of the composite number 10.
    A composite number is a positive integer that can be formed by multiplying together
    two smaller positive integers. Equivalently, it is a positive integer that has
    at least one divisor other than 1 and itself.
  - Is there a minimum age for riders? In order to use the Uber rider app to request
    a ride, you must be 18 years of age or older. Please visit the link below to learn
    more. TERMS OF SERVICE.
  - It is derived literally from the word kobe which is of the meaning 'tortoise'.
    Kobe is used chiefly in the Japanese language and its origin is also Japanese.
    Its meaning is from the element 'kobe'. The first name is derived from the name
    of the city of Kobe in Japan, noted for its quality Kobe beef.
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
    'composing numbers definition',
    'Demonstration, with Cuisenaire rods, of the divisors of the composite number 10. A composite number is a positive integer that can be formed by multiplying together two smaller positive integers. Equivalently, it is a positive integer that has at least one divisor other than 1 and itself.',
    "It is derived literally from the word kobe which is of the meaning 'tortoise'. Kobe is used chiefly in the Japanese language and its origin is also Japanese. Its meaning is from the element 'kobe'. The first name is derived from the name of the city of Kobe in Japan, noted for its quality Kobe beef.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.5421, -0.1189],
#         [ 0.5421,  1.0000,  0.0091],
#         [-0.1189,  0.0091,  1.0000]])
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

* Size: 49,496 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 4 tokens</li><li>mean: 8.56 tokens</li><li>max: 28 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 80.83 tokens</li><li>max: 202 tokens</li></ul> |
* Samples:
  | sentence_0                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  |:---------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>visual figure ground definition</code>       | <code>Figure-Ground Perception. Figure-ground perception is the ability to focus on one specific piece of information in a busy background. Visual figure-ground is the ability to see an object in a busy background; while auditory figure-ground helps a child pick out a voice or sound from a noisy environment. This article deals with the visual perceptual skill and offers some activity ideas to inspire parents and teachers!</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | <code>does mono cause fatigue?</code>              | <code>However, long-term fatigue and other CFS symptoms can be signs of more serious medical or psychological problems. It is important to rule out other conditions that can cause these symptoms by performing a careful evaluation and laboratory tests. Infectious Mononucleosis and Epstein-Barr Virus. Infectious mononucleosis causes fatigue and swollen glands. It primarily affects adolescents and young adults. Research finds that fatigue may last for a year or more in a small percentage of adolescents who have had mononucleosis. Females and people with more severe fatigue are more likely to develop chronic fatigue syndrome after mononucleosis. Blood tests can detect the Epstein-Barr virus (EBV), which causes mononucleosis. Autoimmune Diseases. Some diseases, including systemic lupus erythematosus, multiple sclerosis, and rheumatoid arthritis are caused by autoimmunity, a condition in which the person's immune system attacks the body's own tissues.</code> |
  | <code>does your heart beat faster when sick</code> | <code>Other Causes of Elevated Heart Rate. Elevated heart rate when sick is actually your heart’s aid in order to quell the sickness. However, there can be other causes for elevated heart rate as well. Electrical signals produced and sent to the heart tissues are responsible for controlling the heart rate.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `num_train_epochs`: 3
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
| 0.1616 | 500  | 0.0066        |
| 0.3232 | 1000 | 0.0081        |
| 0.4848 | 1500 | 0.0088        |
| 0.6464 | 2000 | 0.0055        |
| 0.8080 | 2500 | 0.0042        |
| 0.9696 | 3000 | 0.0079        |
| 1.1312 | 3500 | 0.0041        |
| 1.2928 | 4000 | 0.0025        |
| 1.4544 | 4500 | 0.0024        |
| 1.6160 | 5000 | 0.0027        |
| 1.7776 | 5500 | 0.003         |
| 1.9392 | 6000 | 0.0031        |
| 2.1008 | 6500 | 0.0028        |
| 2.2624 | 7000 | 0.0012        |
| 2.4240 | 7500 | 0.0016        |
| 2.5856 | 8000 | 0.0016        |
| 2.7473 | 8500 | 0.0023        |
| 2.9089 | 9000 | 0.0019        |


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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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