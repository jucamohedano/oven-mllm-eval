


### Scoring
bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -m 256G \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-2b-instruct/20260613_015038_566176/20260613_015038_566176_samples.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-max-num-seqs 2048 \
    --judge-mode free-form --judge-n 3 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0

bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -t 12:00:00 \
    --mem 256G \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-2b-instruct/20260613_015038_566176/20260613_015038_566176_samples.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-gpu-util 0.80 --judge-max-num-seqs 1024 \
    --judge-mode free-form --judge-n 3 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0



  # ── 2B ──
  bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -c 32 -m 256G -t 12:00:00 \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-2b-instruct/20260613_015038_566176/20260613_015038_566176_samples.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-gpu-util 0.80 --judge-max-num-seqs 1024 \
    --judge-mode free-form --judge-n 3 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0


  bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -c 32 -m 256G -t 12:00:00 \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-2b-instruct/20260613_015038_566176/20260613_015038_566176_samples.jsonl \
    --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
    --judge-gpu-util 0.90 --judge-max-num-seqs 2048 \
    --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0

  # ── 4B ──
  bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -m 256G -t 12:00:00 \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-4b-instruct/20260613_020525_655341/20260613_020525_655341_samples.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-gpu-util 0.80 --judge-max-num-seqs 1024 \
    --judge-mode free-form --judge-n 3 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0


  bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -c 32 -m 256G -t 12:00:00 \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-4b-instruct/20260613_020525_655341/20260613_020525_655341_samples.jsonl \
    --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
    --judge-gpu-util 0.90 --judge-max-num-seqs 2048 \
    --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0

  # ── 8B ──
  bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -m 256G -t 12:00:00 \
    --input logs/schedule/oven_naive-sampling_concise/qwen_qwen3-vl-8b-instruct/20260613_021340_914911/20260613_021340_914911_samples.jsonl \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-gpu-util 0.80 --judge-max-num-seqs 1024 \
    --judge-mode free-form --judge-n 3 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --num-workers 0


### inferece+scoring job:

bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.92 --mem 256G -t 12:00:00 \
    --model Qwen/Qwen3-VL-2B-Instruct --method naive-sampling \
    --samples-per-example 256 --prompt concise --max-tokens 16 \
    --max-model-len 1024 --max-num-seqs 2048 --chunk-size 128 \
    --max-examples 4 \
    --input data/processed/vlm_compatible_val_aligned.jsonl \
    --image-root /leonardo_work/EUHPC_D33_243/oven/ \
    --judge-model Qwen/Qwen3-8B --judge-gpus 4 \
    --judge-max-num-seqs 2048 \
    --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20




 bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-2B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 8192 --chunk-size 512 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20




bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-2B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 10240 --chunk-size 128 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 10240 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20




bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-4B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 8192 --chunk-size 512 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20


bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-8B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 4096 --chunk-size 128 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 4096 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20



bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 1 --dp 4 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-4B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 4096 --chunk-size 256 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 4096 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20 \
      --resume


bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 1 -c 32 --tp 1 --dp 1 --gpu-util 0.95 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-4B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 4096 --chunk-size 256 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20 \
      --resume


### Judge on the 3 models using with description enriched judge prompts
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=( [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
            [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
            [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630 )
for m in 2b 4b 8b; do
      d=$B/${R[$m]}; id=$(basename ${R[$m]})
      seqs=8192; [ "$m" = "8b" ] && seqs=4096
      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 128G -t 24:00:00 \
      --input  $d/${id}_samples.jsonl \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 --judge-mode free-form \
      --judge-with-desc --desc-chains data/raw/oven_wikidata_chains_cleaned_descs.jsonl \
      --judge-temperature 0.7 --judge-top-p 0.8 --judge-top-k 20 --judge-n 1 \
      --judge-max-num-seqs $seqs \
      --judge-output $d/${id}_samples_judged_qwen_qwen3-4b_with_desc_rich.jsonl \
      --measure "exact_match cascade" \
      --output  $d/${id}_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl \
      --summary $d/${id}_results_qwen_qwen3-4b_with_desc_rich.json
done




### same as above but without the judge --- Recomputing only the taxonomy-aware metrics
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=(
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

for m in 4b 8b; do


B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
      [32b]=qwen_qwen3-vl-32b-instruct/20260614_201554_466144
)

for m in 2b 4b 8b 32b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      judged="$d/${id}_samples_judged_qwen_qwen3-4b_with_desc_rich.jsonl"

      bash scripts/schedule_scoring.sh \
      -A EUHPC_D33_243 \
      -p boost_usr_prod \
      -c 32 \
      -m 256G \
      -t 24:00:00 \
      --gpus 1 \
      --input "$judged" \
      --measure "exact_match rouge cascade" \
      --output "$d/${id}_samples_scored_qwen_qwen3-4b_with_desc_rich_recomputed.jsonl" \
      --summary "$d/${id}_results_qwen_qwen3-4b_with_desc_rich_recomputed.json" \
      --num-workers 0
done

#### Judging and scoring with gemma4 model
[2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
[4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
[8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
[32b]=qwen_qwen3-vl-32b-instruct/20260614_201554_466144


## let's do it for these subset
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=(
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
      [32b]=qwen_qwen3-vl-32b-instruct/20260614_201554_466144
)

for m in 4b 8b 32b; do
      d=$B/${R[$m]}
      id=$(basename "${R[$m]}")
      samples="$d/${id}_samples.jsonl"
      lbl="google_gemma-4-e4b-it_with_desc_rich"

      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 128G -t 24:00:00 \
      --input "$samples" \
      --judge-model google/gemma-4-E4B-it \
      --judge-mode free-form \
      --judge-with-desc \
      --judge-gpus 4 \
      --judge-max-num-seqs 512 \
      --judge-temperature 1.0 --judge-top-p 0.95 --judge-top-k 64 \
      --measure "exact_match cascade" \
      --output  "$d/${id}_samples_scored_${lbl}.jsonl" \
      --summary "$d/${id}_results_${lbl}.json" \
      --judge-output "$d/${id}_samples_judged_${lbl}.jsonl" \
      --num-workers 0
done

### on rsa with qwen3
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

for m in 2b 4b 8b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      samples="$d/${id}_samples_rsa_solution_n16_k4_t5.jsonl"
      lbl="qwen_qwen3-4b_with_desc_rich"

      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 16 -m 128G -t 24:00:00 \
      --input "$samples" \
      --judge-only \
      --judge-model Qwen/Qwen3-4B \
      --judge-mode free-form \
      --judge-with-desc \
      --judge-gpus 2 \
      --judge-max-num-seqs 512 \
      --judge-temperature 0.7 --judge-top-p 0.8 --judge-top-k 20 \
      --judge-output "$d/${id}_samples_judged_rsa_solution_n16_k4_t5_${lbl}.jsonl" \
      --output "$d/${id}_samples_scored_rsa_solution_n16_k4_t5_with_desc_rich.jsonl" \
      --summary "$d/${id}_results_rsa_solution_n16_k4_t5_${lbl}.json"
done

### on rsa with gemma4
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

for m in 2b 4b 8b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      samples="$d/${id}_samples_rsa_solution_n16_k4_t5.jsonl"
      lbl="google_gemma-4-e4b-it_with_desc_rich"

      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 16 -m 128G -t 24:00:00 \
      --input "$samples" \
      --judge-only \
      --judge-model google/gemma-4-E4B-it \
      --judge-mode free-form \
      --judge-with-desc \
      --judge-gpus 2 \
      --judge-max-num-seqs 256 \
      --judge-temperature 1.0 --judge-top-p 0.95 --judge-top-k 64 \
      --judge-output "$d/${id}_samples_judged_rsa_solution_n16_k4_t5_${lbl}.jsonl" \
      --output "$d/${id}_samples_scored_rsa_solution_n16_k4_t5_with_desc_rich.jsonl" \
      --summary "$d/${id}_results_rsa_solution_n16_k4_t5_${lbl}.json"
done
      
-----


I'm still running some experiments and the later chapters are not ready to be written yet. Now, we are monving on to writing chapter2.tex. This chapter is very important because it presents the foundations that form the background of my thesis research.  The most important works that I checked most frequently for my project are in  @../../../thesis_supporting_material/reference_works.md . However, this is a very small subset of models and the thesis needs more research priors to be high quality. So before we focus on writing chapter2.tex, I want to perform a careful and complete literature research step to populate the list of related works. Also, we can speed up a part of that literature research because we should re-use the same works used in ttw_research_project.tex. Another important source that will contain important background work for my thesis is Garosi's thesis: read oven-mllm-eval/docs/thesis_supporting_material/README.md . Another source for finding good related works are the related works discussed in the papers in oven-mllm-eval/docs/thesis_supporting_material/reference_works.md . Especially these two papers:
```
- On Large Multimodal Models as Open-World Image Classifiers (10.48550/arXiv.2503.21851)
- Large Multimodal Models as General In-Context Classifiers(10.48550/arXiv.2602.23229)
```
and `Large Multimodal Models as General In-Context` is Marco Garosi's latest research which contains related/background work not cited in his thesis. Those two works come from the same research lab too. Additionally, the work `Specificity-aware reinforcement learning for fine-grained open-world classification (10.48550/arXiv.2603.03197)` also comes from the same lab and they are applying an RL post-training solution to solve for specificity in open world classification. Other RL works that I provided are more towards understanding the sampling mechanism vs RL training. Maybe we can also discuss self-distillation which is an alternative technique used to RL.

You have skills available to read papers, but if you don't get all of the details from only using those skills, you can also download and parse them with the llamaparse skill. Most importantly, you have the skill thesis-background-research-gate that targets how to perform the reasearch for chapter2. For each paper that passes the gate, write it to a markdown file and put it in the thesis_suporting_material following a well organized structure, feel free to create a literature-review subdirectory to store all of the research there. We have to be very organized. Then update the list of related works as you discover the best and strongest candidates.
Important note: when using subagents, don't instruct them to provide you with summaries for this task. We are doing a very very important labour here, and we must be sure that the background research work that we'll add on the thesis is accurate and well documented, and that's why the skill thesis-background-research-gate provides the necessary instructions.

This is an extensive and meticulous task. Plan how to do with a high level of quality.



bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 256G -t 12:00:00 \
      --input logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples.jsonl \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20 \
      --num-workers 0



bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod       -g 4 -c 32 --tp 2 --dp 2 --gpu-util 0.90 --mem 256G -t 24:00:00       --model Qwen/Qwen3-VL-32B-Instruct --method naive-sampling       --samples-per-example 256 --prompt concise_no_idk --max-tokens 16       --max-model-len 1024 --max-num-seqs 2048 --chunk-size 512       --input data/processed/vlm_compatible_val_aligned.jsonl   --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144    --image-root /leonardo_work/EUHPC_D33_243/oven/       --judge-model Qwen/Qwen3-4B --judge-gpus 4       --judge-max-num-seqs 8192      --judge-mode free-form --judge-n 1 --judge-temperature 0.7       --judge-top-p 0.8 --judge-top-k 20 --resume


bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 2 --dp 2 --enforce-eager --gpu-util 0.90 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-32B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 4096 --chunk-size 512 \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-with-desc --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20 \
      --resume


### Running new judge prommpt with description of the labels
RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_samples.jsonl
bash scripts/schedule_scoring.sh  -A EUHPC_D33_243 -p boost_usr_prod \
    -c 32 -m 256G -t 12:00:00 \
    --input $RUN_DIR \
    --judge-model Qwen/Qwen3-4B \
    --judge-gpus 4 \
    --judge-mode free-form \
    --judge-n 1 \
    --judge-temperature 0.7 \
    --judge-top-p 0.8 \
    --judge-top-k 20 \
    --judge-max-num-seqs 8192 \
    --judge-with-desc \
    --num-workers 0

### Running without description
RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples.jsonl
bash scripts/schedule_scoring.sh  -A EUHPC_D33_243 -p boost_usr_prod \
    -c 32 -m 256G -t 12:00:00 \
    --input $RUN_DIR \
    --judge-model Qwen/Qwen3-4B \
    --judge-gpus 4 \
    --judge-mode free-form \
    --judge-n 1 \
    --judge-temperature 0.7 \
    --judge-top-p 0.8 \
    --judge-top-k 20 \
    --judge-max-num-seqs 8192 \
    --num-workers 0

### Generation of metrics alone

RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260615_184120_262016
RUN_ID=20260615_184120_262016

  # Previous/base judge metrics
  python -m scripts.score_predictions \
    --input $RUN_DIR/${RUN_ID}_samples_judged.jsonl \
    --output $RUN_DIR/${RUN_ID}_samples_scored_base_judge.jsonl \
    --summary $RUN_DIR/${RUN_ID}_results_base_judge.json \
    --taxonomy-index data/processed/oven_taxonomy_index.json \
    --measure exact_match \
    --num-workers 0



### Compute the aggregated metrics from the existing scored jsonl file
RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630
  RUN_ID=20260614_123530_550630

  python -m scripts.score_predictions \
    --input "$RUN_DIR/${RUN_ID}_scored.jsonl" \
    --summary "$RUN_DIR/generations_results.json" \
    --measure exact_match \
    --aggregate


logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_scored.jsonl

RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810
RUN_ID=20260614_121741_936810

python -m scripts.score_predictions \
--input "$RUN_DIR/${RUN_ID}_scored.jsonl" \
--summary "$RUN_DIR/generations_results.json" \
--measure exact_match \
--aggregate


RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972
RUN_ID=20260614_123428_725972

python -m scripts.score_predictions \
--input "$RUN_DIR/${RUN_ID}_samples_scored.jsonl" \
--summary "$RUN_DIR/generations_results.json" \
--measure exact_match \
--aggregate


bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
    -g 4 -c 32 --tp 2 --dp 2 --enforce-eager --gpu-util 0.85 --mem 256G -t 24:00:00 \
    --model Qwen/Qwen3-VL-32B-Instruct --method naive-sampling \
    --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
    --max-model-len 1024 --max-num-seqs 2048 --chunk-size 128 \
    --input data/processed/vlm_compatible_val_aligned.jsonl \
    --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144 \
    --image-root /leonardo_work/EUHPC_D33_243/oven/ \
    --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
    --judge-max-num-seqs 8192 \
    --judge-mode free-form --judge-with-desc --judge-n 1 --judge-temperature 0.7 \
    --judge-top-p 0.8 --judge-top-k 20 \
    --resume

### Running audit

python scripts/audit_judge_false_positives.py \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-
    instruct/20260614_121741_936810/20260614_121741_936810_scored.jsonl \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-
    instruct/20260614_123428_725972/20260614_123428_725972_samples_scored.jsonl \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-
    instruct/20260614_123530_550630/20260614_123530_550630_scored.jsonl \
    --taxonomy-index data/processed/oven_taxonomy_index.json


  - exact: normalized prediction exactly matches the normalized ground-truth answer.
  - alias: normalized prediction matches a known alias of the ground-truth entity from
    oven_taxonomy_index.json.

  - contains_answer: normalized prediction contains the normalized ground-truth answer.
  - answer_contains_prediction: normalized ground-truth answer contains the normalized prediction.


   Model       Rows     Judge    Supported      Judge     Supported          Judge     Supported    Supported
                          Hit      Hit Ex.     Pass@1        Pass@1      Positives     Positives       / Judge
                          Ex.
  ━━━━━━━  ━━━━━━━━━  ━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━
   2B       115,552    99,060       53,305      0.265         0.153      7,851,659     4,537,154         57.8%
  ───────  ─────────  ────────  ───────────  ─────────  ────────────  ─────────────  ────────────  ────────────
   4B       115,552    90,519       49,205      0.360         0.199     10,651,461     5,891,716         55.3%
  ───────  ─────────  ────────  ───────────  ─────────  ────────────  ─────────────  ────────────  ────────────
   8B       115,552    86,410       51,531      0.373         0.216     11,027,479     6,381,830         57.9%




### generate pass@k
python scripts/plot_pass_at_k.py \
    --run-dirs \
      logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810 \
      logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972 \
      logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630 \
    --output viz/pass_at_k_aligned_concise_no_idk_with_desc.png \
    --title "pass@k — aligned concise_no_idk, Qwen3-4B judge with descriptions"


### plotting ci distribution
python scripts/plot_ci_distribution.py \
    --scored-2b logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-
    instruct/20260614_121741_936810/20260614_121741_936810_scored.jsonl \
    --scored-4b logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-
    instruct/20260614_123428_725972/20260614_123428_725972_samples_scored.jsonl \
    --scored-8b logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-
    instruct/20260614_123530_550630/20260614_123530_550630_scored.jsonl \
    --output viz/ci_distribution_judge_aligned_concise_no_idk.png



### Explore models
uv run streamlit run scripts/explore_judgments.py -- \
    --scored logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_scored.jsonl \
    --max-samples 100


### self-agg

#### individually running within job
RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples.jsonl
python scripts/run_recursive_self_agg.py \
    --input $RUN_DIR \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --population 16 --k 4 --steps 2 --max-examples 100 \
    --image-root /leonardo_work/EUHPC_D33_243/oven/ \
    --max-tokens 16 --chunk-size 8 --resume

Okay, so now to run the judge:
RSA_SAMPLES=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_rsa_n16_k4_t2.jsonl

python -m scripts.run_judge \
--input "$RSA_SAMPLES" \
--output "${RSA_SAMPLES%.jsonl}_judged_qwen_qwen3-4b_with_desc.jsonl" \
--judge-model Qwen/Qwen3-4B \
--judge-mode free-form \
--judge-n 1 \
--judge-temperature 0.7 \
--judge-top-p 0.8 \
--judge-top-k 20 \
--judge-with-desc \
--taxonomy-index data/processed/oven_taxonomy_index.json \
--desc-chains data/raw/oven_wikidata_chains_cleaned_descs.jsonl \
--max-num-seqs 1024


python -m scripts.score_predictions \
    --input "${RSA_SAMPLES%.jsonl}_judged_qwen_qwen3-4b_with_desc.jsonl" \
    --output "${RSA_SAMPLES%.jsonl}_scored_qwen_qwen3-4b_with_desc.jsonl" \
    --summary "${RSA_SAMPLES%.jsonl}_results_qwen_qwen3-4b_with_desc.json" \
    --taxonomy-index data/processed/oven_taxonomy_index.json \
    --measure exact_match \
    --num-workers 0



BASE_SAMPLES=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples.jsonl

python -m scripts.run_judge \
    --input "$BASE_100" \
    --output "${BASE_100%.jsonl}_judged_qwen_qwen3-4b_with_desc.jsonl" \
    --judge-model Qwen/Qwen3-4B \
    --judge-mode free-form \
    --judge-n 1 \
    --judge-temperature 0.7 \
    --judge-top-p 0.8 \
    --judge-top-k 20 \
    --judge-with-desc \
    --taxonomy-index data/processed/oven_taxonomy_index.json \
    --desc-chains data/raw/oven_wikidata_chains_cleaned_descs.jsonl \
    --max-num-seqs 1024


#### run RSA job
RSA_INPUT=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples.jsonl

bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
-g 1 -c 8 -m 128G -t 24:00:00 \
--input "$RSA_INPUT" \
--model Qwen/Qwen3-VL-8B-Instruct \
--population 16 \
--k 4 \
--steps 2 \
--image-root /leonardo_work/EUHPC_D33_243/oven/ \
--max-tokens 16 \
--chunk-size 64 \
--max-num-seqs 4096 \
--resume


B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

declare -A M=(
      [2b]=Qwen/Qwen3-VL-2B-Instruct
      [4b]=Qwen/Qwen3-VL-4B-Instruct
      [8b]=Qwen/Qwen3-VL-8B-Instruct
)

for m in 2b 4b 8b; do
      d=$B/${R[$m]}; id=$(basename "${R[$m]}"); input=$d/${id}_samples.jsonl
      bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 --dp 4 -c 32 -m 128G -t 24:00:00 \
      --input "$input" \
      --model "${M[$m]}" \
      --population 16 --k 4 --steps 10 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --max-tokens 16 --chunk-size 64 --max-num-seqs 4096 \
      --resume
done


#### RSA evaluation on the --candidate-format solution which also works to get the dataset from later for RL'ing
RSA_CACHE_ROOT=/leonardo_scratch/fast/EUHPC_D33_243/rsa_compile_cache
B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

declare -A M=(
      [2b]=Qwen/Qwen3-VL-2B-Instruct
      [4b]=Qwen/Qwen3-VL-4B-Instruct
      [8b]=Qwen/Qwen3-VL-8B-Instruct
)

for m in 2b 4b 8b; do
      d=$B/${R[$m]}; id=$(basename "${R[$m]}"); input=$d/${id}_samples.jsonl
      bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 --dp 4 -c 32 -m 256G -t 24:00:00 \
      --input "$input" \
      --model "${M[$m]}" \
      --candidate-format solution \
      --population 16 --k 4 --steps 5 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --temperature 1.0 --top-p 1.0 --top-k -1 \
      --max-tokens 256 --max-model-len 4096 \
      --chunk-size 128 --max-num-seqs 4096 \
      --resume
done


RSA_CACHE_ROOT=/leonardo_scratch/fast/EUHPC_D33_243/rsa_compile_cache
B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [32b]=qwen_qwen3-vl-32b-instruct/20260614_201554_466144
)

declare -A M=(
      [32b]=Qwen/Qwen3-VL-32B-Instruct
)

for m in 32b; do
      d=$B/${R[$m]}; id=$(basename "${R[$m]}"); input=$d/${id}_samples.jsonl
      bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 --dp 2 --tp 2 -c 32 -m 256G -t 24:00:00 \
      --input "$input" \
      --model "${M[$m]}" \
      --candidate-format solution \
      --population 16 --k 4 --steps 5 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --temperature 1.0 --top-p 1.0 --top-k -1 \
      --max-tokens 256 --max-model-len 4096 \
      --chunk-size 16 --max-num-seqs 2048 \
      --resume
done





### Checking rsa results
for RUN in \
  logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_rsa_n16_k4_t10.jsonl \
  logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples_rsa_n16_k4_t10.jsonl \
  logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples_rsa_n16_k4_t10.jsonl
  do
    echo "$RUN"
    wc -l "$RUN"
  done



### RSA Judge + Rescore

export OVEN_NODE_EMB_DIR=/leonardo_work/EUHPC_D33_243/oven_node_emb

B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=( [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
            [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
            [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630 )

for m in 2b 4b 8b; do
d=$B/${R[$m]}
id=$(basename ${R[$m]})
rsa=$d/${id}_samples_rsa_n16_k4_t10.jsonl

bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
-c 32 -m 256G -t 24:00:00 \
--input "$rsa" \
--judge-model Qwen/Qwen3-4B \
--judge-gpus 4 \
--judge-mode free-form \
--judge-n 1 \
--judge-temperature 0.7 \
--judge-top-p 0.8 \
--judge-top-k 20 \
--judge-max-num-seqs 8192 \
--judge-with-desc \
--measure "exact_match cascade" \
--output "${rsa%.jsonl}_scored_qwen_qwen3-4b_with_desc.jsonl" \
--summary "${rsa%.jsonl}_results_qwen_qwen3-4b_with_desc.json" \
--num-workers 0
done



bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 -c 32 --tp 2 --dp 2 --enforce-eager --gpu-util 0.85 --mem 256G -t 24:00:00 \
      --model Qwen/Qwen3-VL-32B-Instruct --method naive-sampling \
      --samples-per-example 256 --prompt concise_no_idk --max-tokens 16 \
      --max-model-len 1024 --max-num-seqs 2048 --chunk-size 128 \
      --image-workers 16 --prefetch-images \
      --input data/processed/vlm_compatible_val_aligned.jsonl \
      --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 \
      --judge-max-num-seqs 8192 \
      --judge-mode free-form --judge-with-desc --judge-n 1 --judge-temperature 0.7 \
      --judge-top-p 0.8 --judge-top-k 20 \
      --resume







(base) [jcamacho@login05 oven-mllm-eval]$ bash scripts/schedule_sbatch.sh -A EUHPC_D33_243 -p boost_usr_prod       -g 4 -c 32 --tp 2 --dp 2 --enforce-eager --gpu-util 0.85 --mem 256G -t 24:00:00       --model Qwen/Qwen3-VL-32B-Instruct --method naive-sampling       --samples-per-example 256 --prompt concise_no_idk --max-tokens 16       --max-model-len 1024 --max-num-seqs 1024 --chunk-size 64       --image-workers 16 --prefetch-images       --input data/processed/vlm_compatible_val_aligned.jsonl       --output-dir logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144       --image-root /leonardo_work/EUHPC_D33_243/oven/       --judge-model Qwen/Qwen3-4B --judge-gpus 4       --judge-max-num-seqs 8192       --judge-mode free-form --judge-with-desc --judge-n 1 --judge-temperature 0.7       --judge-top-p 0.8 --judge-top-k 20       --resume


bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod -c 16 -m 64G -t 01:00:00 \
    --input  $D/20260614_121741_936810_samples_judged_qwen_qwen3-4b_with_desc.jsonl \
    --measure "exact_match cascade" --max-examples 200 \
    --output  $D/20260614_121741_936810_test_scored_with_desc.jsonl \
    --summary $D/20260614_121741_936810_test_results_with_desc.json


D=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810
bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod -c 16 -m 64G -t 01:00:00 \
    --input  $D/20260614_121741_936810_samples_judged_qwen_qwen3-4b_with_desc.jsonl \
    --measure "exact_match cascade" --max-examples 200 \
    --output  $D/20260614_121741_936810_test_scored_with_desc.jsonl \
    --summary $D/20260614_121741_936810_test_results_with_desc.json


### Rerun metrics on baseline for the 3 models --- this new metrics are representative of hF and judge
export OVEN_NODE_EMB_DIR=/leonardo_work/EUHPC_D33_243/oven_node_emb
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=( [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
            [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
            [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630 )
for m in 2b 4b 8b; do
      d=$B/${R[$m]}; id=$(basename ${R[$m]})
      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 128G -t 24:00:00 --gpus 1 \
      --input  $d/${id}_samples_judged_qwen_qwen3-4b_with_desc.jsonl \
      --measure "exact_match cascade" \
      --output  $d/${id}_samples_scored_qwen_qwen3-4b_with_desc.jsonl \
      --summary $d/${id}_results_qwen_qwen3-4b_with_desc.json
done


export OVEN_NODE_EMB_DIR=/leonardo_work/EUHPC_D33_243/oven_node_emb
B=logs/schedule/oven_naive-sampling_concise_no_idk
declare -A R=( [32b]=qwen_qwen3-vl-32b-instruct/20260614_201554_466144 )
for m in 32b; do
      d=$B/${R[$m]}; id=$(basename ${R[$m]})
      bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 128G -t 24:00:00 --gpus 1 \
      --input  $d/${id}_samples_judged_qwen_qwen3-4b_with_desc.jsonl \
      --measure "exact_match cascade" \
      --output  $d/${id}_samples_scored_qwen_qwen3-4b_with_desc.jsonl \
      --summary $d/${id}_results_qwen_qwen3-4b_with_desc.json
done




Read the pdf from which we got the RSA implementation in "resources/Venkatraman et al. - 2026 - Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models.pdf"  to find out what's the most optimal recipe for their downstream task, so that we can evaluate if we have to try different n,k,t




export OVEN_NODE_EMB_DIR=/leonardo_work/EUHPC_D33_243/oven_node_emb

RUN_DIR=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810
ID=20260614_121741_936810
rsa=$RUN_DIR/${ID}_samples_rsa_n16_k4_t2.jsonl

bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
-c 32 -m 256G -t 24:00:00 \
--input "$rsa" \
--judge-model Qwen/Qwen3-4B \
--judge-gpus 1 \
--judge-mode free-form \
--judge-n 1 \
--judge-temperature 0.7 \
--judge-top-p 0.8 \
--judge-top-k 20 \
--judge-max-num-seqs 8192 \
--judge-with-desc \
--measure "exact_match cascade" \
--output "${rsa%.jsonl}_scored_qwen_qwen3-4b_with_desc.jsonl" \
--summary "${rsa%.jsonl}_results_qwen_qwen3-4b_with_desc.json" \
--num-workers 0




export OVEN_NODE_EMB_DIR=/leonardo_work/EUHPC_D33_243/oven_node_emb
d=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144
id=20260614_201554_466144

bash scripts/schedule_scoring.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -c 32 -m 128G -t 24:00:00 \
      --input  $d/${id}_samples.jsonl \
      --judge-model Qwen/Qwen3-4B --judge-gpus 4 --judge-mode free-form \
      --judge-with-desc --desc-chains data/raw/oven_wikidata_chains_cleaned_descs.jsonl \
      --judge-temperature 0.7 --judge-top-p 0.8 --judge-top-k 20 --judge-n 1 \
      --judge-max-num-seqs 8192 \
      --measure "exact_match cascade" \
      --output  $d/${id}_samples_scored_qwen_qwen3-4b_with_desc.jsonl \
      --summary $d/${id}_results_qwen_qwen3-4b_with_desc.json






So why did the 2B model have larger pass@k than the 8B model for large k values?
I tried to enrich the context of the judge model by providing it with the parent and grandparent of the ground truth label, and also with descriptions of the ground-truth as well as for the parent and grandparent if available. And with all of this info, the trend was still repeating. 
Then I checked for false positives in the judgements. Three ways to check:
- exact matching
- alias matching (the dataset provides aliases for some of the labels)
- ground-truth contained in prediction
- alias contained in prediction

I gathered those predictions that pass these checks into a set called support set. The result is shown in the table below:

| Run | Rows    | k   | JudgeHit | SuppHit | J p@1 | S p@1 | J p@k | S p@k | J Pos     | S Pos     | JPos/Hit |
|------|---------|-----|----------|---------|-------|-------|-------|-------|-----------|-----------|----------|
| 2B   | 115,552 | 256 | 99,060   | 36,609  | 0.265 | 0.094 | 0.857 | 0.317 | 7,851,659 | 2,766,679 | 79.3     |
| 4B   | 115,552 | 256 | 90,519   | 37,932  | 0.360 | 0.136 | 0.783 | 0.328 | 10,651,461| 4,020,067 | 117.7    |
| 8B   | 115,552 | 256 | 86,410   | 39,563  | 0.373 | 0.149 | 0.748 | 0.342 | 11,027,479| 4,408,778 | 127.6    |


Each row in the table is an example, in total 115k examples (question+image).
J p@1 is pass@1 with the judge 
S p@1 is pass@1 with the filtered support set
JPos is the total number of individual rollouts the judge marked correct (total is 115,552 x 256 = 29.6M)
JudgeHit is the number of examples that have at least one accepted rollout (cᵢ ≥ 1). Example-level.
JPos/Hit is the average of accepted rollouts by the judge


Details on the rollout counts for the filtering rules:

| Run | Exact | Alias (=) | Pred⊃Ans (answer⊆pred) | Pred⊃Alias (alias⊆pred) |
|-----|-------|-----------|------------------------|-------------------------|
| 2B  | 2,466,457 | 83,905  | 206,839 | 9,478  |
| 4B  | 3,348,036 | 140,413 | 513,432 | 18,186 |
| 8B  | 3,703,092 | 137,153 | 546,436 | 22,097 |


After filtering pass@256: 8B 0.342 > 4B 0.328 > 2B 0.317 (and S p@1 same order: 0.149 > 0.136 > 0.094)
Concentration median JPos/Hit: 110 > 82 > 29 — when 8B hits it commits (~110/256 rollouts); 2B spreads thin (median 29).

This behavior is also seen in the c_i distribution plot, where c_i is the number of the 256 rollouts that have correct predictions. The 2B model has the highest % of correctness when the tail is fragile (c_i<128) and the smallest committed bin (c_i>128).

So the conclusion is that the judge is turning predictions into false positives, because after inspecting the coverage pass@k under the support set, we see the results that we expected.


More so, in the report from Qwen3-VL we see that the models were applied strong-to-weak distillation (off-policy + on-policy KL). The 2B,4B,8B are distilled frmo the 235B teacher. Since smaller models have less capacity, they cannot reproduce the teacher's sharp distribution, and so the student's output probability distribution comes out flatter (higher entropy).


Also, note that the 8B model has a larger SigLIP2-400M encoder, that translates into better visual graunding on the visual entity recognition task, hence higher pass@1. The capacity limitation of models is consistent with the support set results on pass@k.



when did i fix the issue with the judge model where it wasn't always getting evidence, i.e., sometimes not even the taxonomy chain (parent, grandparent) was added to the evidence when it's always available. At what time did I commit that change?

[2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
[4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
[8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630


python scripts/plot_metrics_from_results.py \
    --results \
      2B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_results_qwen_qwen3-4b_with_desc_rich.json \
      4B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_results_qwen_qwen3-4b_with_desc_rich.json \
      8B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_results_qwen_qwen3-4b_with_desc_rich.json \
    --measures exact_match cascade \
    --out-prefix viz/hierarchical/aligned_concise_no_idk_with_desc_rich_2b_4b_8b \
    --title "aligned · concise_no_idk · Qwen3-4B judge with taxonomy evidence"



python3 -c "
import json, sys
from pathlib import Path
from PIL import Image

# pull image paths from the val JSONL
paths = set()
for line in open('data/processed/vlm_compatible_val_aligned.jsonl'):
      paths.add(json.loads(line)['image_path'])
      if len(paths) >= 500: break

ws, hs = [], []
for p in paths:
      im = Image.open(p)
      ws.append(im.width); hs.append(im.height)
      im.close()

ws.sort(); hs.sort()
n = len(ws)
print(f'n={n}  median W={ws[n//2]}  median H={hs[n//2]}')
print(f'range W=[{ws[0]}, {ws[-1]}]  H=[{hs[0]}, {hs[-1]}]')
"


python scripts/build_verl_oven_parquet.py \
    --dataset-mode rsa_trace \
    --input data/processed/vlm_compatible_train_aligned.jsonl \
    --val-input data/processed/vlm_compatible_val_aligned.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --candidate-solutions /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_50k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --output-dir data/processed/verl_oven_rsa_trace_aligned \
    --image-root /leonardo_work/EUHPC_D33_243/oven \
    --aggregation-fraction 0.5 \
    --aggregation-k 4 \
    --question-policy aligned \
    --overwrite


## prepare balanced dataset QIDs to generate RL dataset
TRAIN_250K=data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl
python scripts/sample_jsonl_balanced_by_key.py \
    --input data/processed/vlm_compatible_train_aligned.jsonl \
    --output "$TRAIN_250K" \
    --manifest data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42_manifest.json \
    --max-rows 250000 \
    --key entity_id \
    --seed 42 \
    --shuffle-output \
    --overwrite

Then generate candidates from that balanced file:

<!-- TRAIN_250K=data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl -->

CAND=data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl

bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 --dp 4 -c 32 -m 128G -t 24:00:00 \
      --input "$TRAIN_250K" \
      --output "$CAND" \
      --model Qwen/Qwen3-VL-4B-Instruct \
      --candidate-format solution \
      --population 16 --k 4 --steps 1 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --temperature 1.0 --top-p 1.0 --top-k -1 \
      --max-tokens 256 --max-model-len 4096 \
      --chunk-size 8 --max-num-seqs 128 \
      --resume



### Generate the parquet dataset for GRPO
TRAIN_250K=data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl
CAND=data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl
OUT=data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42

python scripts/build_verl_oven_parquet.py \
    --dataset-mode rsa_trace \
    --input "$TRAIN_250K" \
    --val-input data/processed/vlm_compatible_val_aligned.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --candidate-solutions "$CAND" \
    --output-dir "$OUT" \
    --image-root /leonardo_work/EUHPC_D33_243/oven \
    --aggregation-fraction 0.5 \
    --aggregation-k 4 \
    --question-policy aligned \
    --overwrite


### 
OUT=data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42

python - <<'PY'
import pandas as pd
from pathlib import Path

out = Path("data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42")
df = pd.read_parquet(out / "train.parquet")

for kind in ["standard", "aggregation"]:
      row = df[df["extra_info"].map(lambda x: x["prompt_type"]) == kind].iloc[0]
      extra = row["extra_info"]

      print("\n" + "=" * 80)
      print(f"PROMPT TYPE: {kind}")
      print("=" * 80)
      print("data_id:", extra["data_id"])
      print("qid:", extra["entity_id"])
      print("answer:", extra["answer"])
      print("question:", extra["question"])
      print("answer_format:", extra["answer_format"])
      print("candidate finals:", list(extra["candidate_final_answers"]))
      print("\n--- prompt ---")
      for msg in row["prompt"]:
            print(f"\n[{msg['role']}]\n{msg['content']}")
PY



CUDA_VISIBLE_DEVICES=0 python -m scripts.run_judge \
      --input logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples.jsonl \
      --output /tmp/smoke_gemma_image_judged.jsonl \
      --judge-model google/gemma-4-E4B-it \
      --judge-mode free-form \
      --judge-image \
      --max-examples 8 \
      --max-model-len 4096 --max-num-seqs 256 --gpu-util 0.92



Okay, I think it looks better. I will need to move onto the next chapeter.
  Important notes:
  - If you fan out subagents, DON'T ASK FOR SUMMARIES ONLY, ASK FOR CITATIONS IN THE TEXT. This is a thesis 
  and we have to be accurate and precise in our claims.





python - <<'PY'
import pandas as pd
from pathlib import Path

out = Path("data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42")
df = pd.read_parquet(out / "train.parquet")

for i in [0, len(df)//2, len(df)-1]:
      uri = df.iloc[i]["images"][0]["image"]
      path = uri.removeprefix("file://")
      print(i, uri, "exists=", Path(path).exists())
PY




TRAIN_250K=data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl
CAND=data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl
OUT=data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42
python scripts/build_verl_oven_parquet.py \
    --dataset-mode rsa_trace \
    --input "$TRAIN_250K" \
    --val-input data/processed/vlm_compatible_val_aligned.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --candidate-solutions "$CAND" \
    --output-dir "$OUT" \
    --image-root /leonardo_work/EUHPC_D33_243/oven \
    --aggregation-fraction 0.5 \
    --aggregation-k 4 \
    --question-policy aligned \
    --overwrite


We have to run a second round of revision. Your plan was very good, but we can do better. The reference
thesis from Garosi has ~90 citations in chapter 2 and we fall quite below. You failed several times to use
web search, try again I think it's fixed, otherwise I have added more tools to the tool set /skills. Let's
focus again on reviewing the chapter 2 from Marco and spotting those citations that can we can re-use in
our work. Careful! I'm not saying to copy or paraphrase what he's saying. Stick to the same plan we
executed before, but this time be more rigorous, collect the research literature into the clusters and
reflect on it. We have to fix the gaps and improve the literature that we have at the moment. Respect my
previous decisions. Remember to follow /thesis-background-research-gate





Warning: Permanently added 'login05-ext.leonardo.cineca.it,131.175.44.5' (RSA) to the list of known hosts.
receiving incremental file list
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_results_google_gemma-4-e4b-it_with_desc_rich.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard0_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard1_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard2_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard3_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_results_qwen_qwen3-4b_with_desc_rich_recomputed.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard0_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard1_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard2_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard3_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_results_google_gemma-4-e4b-it_with_desc_rich.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard0_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard1_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard2_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard3_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_results_google_gemma-4-e4b-it_with_desc_rich.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard0_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard1_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard2_metadata.json
schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_samples_judged_google_gemma-4-e4b-it_with_desc_rich.jsonl_shard3_metadata.json

sent 572 bytes  received 23.62K bytes  6.91K bytes/sec
total size is 180.19K  speedup is 7.45


python scripts/plot_pass_at_k.py \
    --results-pattern "*results_google_gemma-4-e4b-it_with_desc_rich.json" \
    --run-dirs \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810 \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972 \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630 \
    logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144 \
    --output viz/pass_at_k/pass_at_k_aligned_concise_no_idk_gemma_with_desc_rich_2b_4b_8b_32b.png \
    --title "pass@k — aligned concise_no_idk, Gemma-4-E4B judge with rich descriptions"

logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144

python scripts/plot_hierarchical_metrics.py \
    --results \
      Qwen3-VL-2B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-2b-instruct/20260614_121741_936810/20260614_121741_936810_results_qwen_qwen3-4b_with_desc_rich_recomputed.json \
      Qwen3-VL-4B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-4b-instruct/20260614_123428_725972/20260614_123428_725972_results_qwen_qwen3-4b_with_desc_rich_recomputed.json \
      Qwen3-VL-8B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-8b-instruct/20260614_123530_550630/20260614_123530_550630_results_qwen_qwen3-4b_with_desc_rich_recomputed.json \
      Qwen3-VL-32B=logs/schedule/oven_naive-sampling_concise_no_idk/qwen_qwen3-vl-32b-instruct/20260614_201554_466144/20260614_201554_466144_results_qwen_qwen3-4b_with_desc_rich_recomputed.json \
    --variants leaf \
    --views all mapped \
    --out-prefix viz/taxonomy/aligned_concise_no_idk_qwen3-4b_with_desc_rich_recomputed \
    --title "aligned concise_no_idk · Qwen3-4B judge with rich descriptions"


python - <<'PY'
from verl.utils.reward_score.oven_boxed import compute_score
print(compute_score("oven", r"\boxed{Air gun}", "Air gun"))
print(compute_score("oven", r"\boxed{bolt-action rifle}", "Air gun"))
print(compute_score("oven", "Air gun", "Air gun"))
PY


python scripts/build_verl_oven_parquet.py \
    --dataset-mode rsa_trace \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --val-input data/processed/vlm_compatible_val_aligned.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --output-dir data/processed/verl_oven_rsa_trace_smoke_512 \
    --image-root /leonardo_work/EUHPC_D33_243/oven \
    --aggregation-fraction 0.5 \
    --aggregation-k 4 \
    --question-policy aligned \
    --max-train-rows 512 \
    --max-val-rows 128 \
    --overwrite

bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    -A EUHPC_D33_243 -p boost_usr_prod \
    -g 4 -c 32 -m 256G -t 10:00:00 \
    --mode smoke --steps 2 --save-freq -1 \
    --conda-env verl \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    --wandb

cd /leonardo_scratch/fast/EUHPC_D33_243/verl
VERL_IMPORT_PROBE=1 \
ROLLOUT_TP=1 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=1 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_N=1 \
ROLLOUT_GPU_UTIL=0.20 \
MAX_RESPONSE_LENGTH=128 \
RAY_OBJECT_STORE_MEMORY=4294967296 \
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
-A EUHPC_D33_243 -p boost_usr_prod \
-g 1 -c 16 -m 128G -t 04:00:00 \
--mode smoke \
--conda-env verl \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
--logger '["console"]'
--wandb







LORA_MERGE=True \
ROLLOUT_TP=1 TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=1 \
MAX_RESPONSE_LENGTH=128 ROLLOUT_MAX_MODEL_LEN=4224 \
RAY_OBJECT_STORE_MEMORY=4294967296 \
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 1 -c 8 -m 128G -t 04:00:00 \
--mode smoke --conda-env verl \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
--logger '["console"]'


MODEL_USE_REMOVE_PADDING=True MODEL_USE_FUSED_KERNELS=True MODEL_ATTN_IMPLEMENTATION=flash_attention_2 \
ROLLOUT_MIN_MODEL_LEN=4608 \
DATASET_DIR=/leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42 \
VAL_FILE=/leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
LORA_MERGE=True \
ROLLOUT_TP=1 TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=1 \
MAX_RESPONSE_LENGTH=128 \
RAY_OBJECT_STORE_MEMORY=4294967296 \
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 1 -c 16 -m 128G -t 04:00:00 \
--mode smoke --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh


ls -la /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/envs/verl-v080/lib/python3.12/site-packages/tensordict/__init__.py



RSA_CACHE_ROOT=/leonardo_scratch/fast/EUHPC_D33_243/rsa_compile_cache
B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
)

declare -A M=(
      [2b]=Qwen/Qwen3-VL-2B-Instruct
)

for m in 2b; do
      d=$B/${R[$m]}; id=$(basename "${R[$m]}"); input=$d/${id}_samples.jsonl
      bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 4 --dp 4 -c 32 -m 256G -t 24:00:00 \
      --input "$input" \
      --model "${M[$m]}" \
      --candidate-format solution \
      --population 16 --k 4 --steps 5 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --temperature 1.0 --top-p 1.0 --top-k -1 \
      --max-tokens 256 --max-model-len 4096 \
      --chunk-size 32 --max-num-seqs 4096 \
      --resume
done

RSA_CACHE_ROOT=/leonardo_scratch/fast/EUHPC_D33_243/rsa_compile_cache
B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

declare -A M=(
      [8b]=Qwen/Qwen3-VL-8B-Instruct
)

for m in 8b; do
      d=$B/${R[$m]}; id=$(basename "${R[$m]}"); input=$d/${id}_samples.jsonl
      bash scripts/schedule_rsa.sh -A EUHPC_D33_243 -p boost_usr_prod \
      -g 2 --dp 2 -c 16 -m 128G -t 24:00:00 \
      --input "$input" \
      --model "${M[$m]}" \
      --candidate-format solution \
      --population 16 --k 4 --steps 5 \
      --image-root /leonardo_work/EUHPC_D33_243/oven/ \
      --temperature 1.0 --top-p 1.0 --top-k -1 \
      --max-tokens 256 --max-model-len 4096 \
      --chunk-size 16 --max-num-seqs 4096 \
      --resume
done



B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [8b]=qwen_qwen3-vl-8b-instruct/20260614_123530_550630
)

for m in 8b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      judged="$d/${id}_samples_judged_qwen_qwen3-4b_with_desc_rich.jsonl"

      bash scripts/schedule_scoring.sh \
      -A EUHPC_D33_243 \
      -p boost_usr_prod \
      -c 16 \
      -m 128G \
      -t 24:00:00 \
      --gpus 2 \
      --input "$judged" \
      --measure "exact_match cascade" \
      --output "$d/${id}_samples_scored_qwen_qwen3-4b_with_desc_rich_recomputed.jsonl" \
      --summary "$d/${id}_results_qwen_qwen3-4b_with_desc_rich_recomputed.json" \
      --num-workers 0
done




B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
)

for m in 4b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      judged="$d/${id}_samples_judged_qwen_qwen3-4b_with_desc_rich.jsonl"

      bash scripts/schedule_scoring.sh \
      -A EUHPC_D33_243 \
      -p boost_usr_prod \
      -c 16 \
      -m 128G \
      -t 24:00:00 \
      --gpus 2 \
      --input "$judged" \
      --measure "exact_match cascade" \
      --output "$d/${id}_samples_scored_qwen_qwen3-4b_with_desc_rich_recomputed.jsonl" \
      --summary "$d/${id}_results_qwen_qwen3-4b_with_desc_rich_recomputed.json" \
      --num-workers 0
done



B=logs/schedule/oven_naive-sampling_concise_no_idk

declare -A R=(
      [2b]=qwen_qwen3-vl-2b-instruct/20260614_121741_936810
      [4b]=qwen_qwen3-vl-4b-instruct/20260614_123428_725972
)

suffix=samples_rsa_solution_n16_k4_t5.jsonl

export OVEN_EMBED_SEARCH_CHUNK_SIZE=1024
for m in 2b 4b; do
      d="$B/${R[$m]}"
      id="$(basename "${R[$m]}")"
      judged="$d/${id}_samples_rsa_solution_n16_k4_t5.jsonl"

      bash scripts/schedule_scoring.sh \
      -A EUHPC_D33_243 \
      -p boost_usr_prod \
      -c 16 \
      -m 128G \
      -t 24:00:00 \
      --gpus 1 \
      --input "$judged" \
      --measure "exact_match cascade" \
      --output "$d/${id}_samples_scored_rsa_solution_n16_k4_t5_with_desc_rich.jsonl" \
      --summary "$d/${id}_results_rsa_solution_n16_k4_t5_with_desc_rich.json" \
      --num-workers 16
done


# Pattern to generate 2k datasets and how to mine unlockable entities

# ── v2 exact (standard-only, reasoning, no 1-shot) ──
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.0 \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # ── v3 cb (standard-only, compute_buffer, no 1-shot) ──
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.0 \
    --standard-prompt-variant compute_buffer \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # ── 50/50 standard/aggregation (RSA-style, no 1-shot in either) ──
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v2_exact_agg05_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.5 \
    --traversal-fraction 0.0 \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # ── v4 traversal structured (no 1-shot) ──
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v4_traversal_structured_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 1.0 \
    --traversal-variant structured \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # ── v4 traversal wikidata (no 1-shot) ──
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v4_traversal_wikidata_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 1.0 \
    --traversal-variant wikidata \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # ── Re-mine unlockable subsets ──
  python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
      --seed 42

  python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42 \
      --seed 42

  python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_v2_exact_agg05_aligned_balanced_qid_2k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_v2_exact_agg05_aligned_balanced_qid_2k_seed42 \
      --seed 42




# GRPO-exact v2 (with 1-shot reasoning + filtered aggregation)
python scripts/build_verl_oven_parquet.py \
--input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
--labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
--descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
--output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
--dataset-mode rsa_trace \
--candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
--aggregation-fraction 0.5 --aggregation-k 4 \
--question-policy aligned \
--max-train-rows 2000 --seed 42 --overwrite \
--image-root /leonardo_work/EUHPC_D33_243/oven

## grpo-exact v3 (with 1-shot compute_buffer + filtered aggregation)
python scripts/build_verl_oven_parquet.py \
--input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
--labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
--descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
--output-dir data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42 \
--dataset-mode rsa_trace \
--candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
--aggregation-fraction 0.5 --aggregation-k 4 \
--question-policy aligned \
--max-train-rows 2000 --seed 42 --overwrite \
--image-root /leonardo_work/EUHPC_D33_243/oven \
--standard-prompt-variant compute_buffer

# GRPO-traversal v2 (with 1-shot + filtered aggregation + traversal prompts)
python scripts/build_verl_oven_parquet.py \
--input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
--labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
--descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
--output-dir data/processed/verl_oven_rsa_v2_traversal_aligned_balanced_qid_2k_seed42 \
--dataset-mode rsa_trace \
--candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
--aggregation-fraction 0.5 --aggregation-k 4 \
--traversal-fraction 0.33 \
--question-policy aligned \
--max-train-rows 2000 --seed 42 --overwrite \
--image-root /leonardo_work/EUHPC_D33_243/oven


# === 3. Mine unlockable from v2 exact ===
python scripts/mine_unlockable_examples.py \
--rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
--train-parquet data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet \
--output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
--taxonomy-index data/processed/oven_taxonomy_index.json



echo "=== GRPO-exact v2 ===" && python3 -c "
import pyarrow.parquet as pq
from collections import Counter
tbl = pq.read_table('data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet')
extras = tbl['extra_info'].to_pylist()
pts = Counter(e.get('prompt_type','?') for e in extras)
print(f'{len(tbl)} rows, {len(set(e.get(\"entity_id\",\"\") for e in extras))} QIDs')
print(f'Prompt types: {dict(pts)}')
" && echo "=== GRPO-traversal v2 ===" && python3 -c "
import pyarrow.parquet as pq
from collections import Counter
tbl = pq.read_table('data/processed/verl_oven_rsa_v2_traversal_aligned_balanced_qid_2k_seed42/train.parquet')
extras = tbl['extra_info'].to_pylist()
pts = Counter(e.get('prompt_type','?') for e in extras)
print(f'{len(tbl)} rows, {len(set(e.get(\"entity_id\",\"\") for e in extras))} QIDs')
print(f'Prompt types: {dict(pts)}')
"



## Regenerate the datasets for the v2 and v3 to use the same type of prompt strategy
# v2 exact → standard-only (reasoning), no agg, no traversal
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.0 \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

  # v3 cb → standard-only (compute_buffer), no agg, no traversal
  python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.0 \
    --standard-prompt-variant compute_buffer \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

### dataset generation for rsa+one-shot prompt and training command
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_agg08_one-shot_aligned_balanced_qid_3k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.8 \
    --traversal-fraction 0.0 \
    --question-policy aligned \
    --max-train-rows 3072 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_agg08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_agg08_one-shot_aligned_balanced_qid_3k_seed42 \
      --seed 42


bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_agg08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 500 --test-freq 150 --save-freq 192 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5376 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_agg08_one-shot_3k_shaped_n16 \
    --exp-name qwen3_vl_4b_oven_grpo_agg08_one-shot_3k_shaped_bs16_300steps_n16 \
    --wandb-run-id qwen3-vl-4b-oven-grpo-agg08-one-shot-3k-shaped-n16-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32




# let's also do 50/50:
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_agg05_one-shot_aligned_balanced_qid_3k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.8 \
    --traversal-fraction 0.0 \
    --question-policy aligned \
    --max-train-rows 3072 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_agg05_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_agg05_one-shot_aligned_balanced_qid_3k_seed42 \
      --seed 42

bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_agg05_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 500 --test-freq 150 --save-freq 192 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5376 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_agg08_one-shot_3k_shaped_n16 \
    --exp-name qwen3_vl_4b_oven_grpo_agg05_one-shot_3k_shaped_bs16_300steps_n16 \
    --wandb-run-id qwen3-vl-4b-oven-grpo-agg05-one-shot-3k-shaped-n16-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32

## traversal only
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.8 \
    --question-policy aligned \
    --max-train-rows 3072 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42 \
      --seed 42

bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 500 --test-freq 150 --save-freq 192 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5376 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_agg08_one-shot_3k_shaped_n16 \
    --exp-name qwen3_vl_4b_oven_grpo_trav08_one-shot_3k_shaped_bs16_300steps_n16 \
    --wandb-run-id qwen3-vl-4b-oven-grpo-trav08-one-shot-3k-shaped-n16-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32  


# ── GRPO-exact v2 (clean: standard-only, reasoning prompt) ──
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 128 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_exact_v2_clean \
--exp-name qwen3_vl_4b_oven_grpo_exact_v2_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-v2-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False


 # ── GRPO-exact v3 (clean: standard-only, compute_buffer prompt) ──
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_exact_v3_cb_clean \
--exp-name qwen3_vl_4b_oven_grpo_exact_v3_cb_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-v3-cb-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False


------ debugging ----------
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 01:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
    --steps 6 --test-freq 5 --save-freq -1 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 8 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 4608 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 1024 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/debug_val_test \
    --exp-name debug_val_test \
    -- actor_rollout_ref.rollout.free_cache_engine=False \
    data.val_max_samples=512 \
    data.dataloader_num_workers=0


# ── GRPO-unlockable v2 (clean: standard-only, unlockable subset) ──

### this job submission is from the old one-shot prompts for cb
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v2_clean_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_unlockable_v2_clean_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-unlockable-v2-clean-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32


#### standard + rsa
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v2_exact_agg05_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v2_rsa_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_unlockable_v2_rsa_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-unlockable-v2-rsa-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32


# standard (no one-shot)
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v2_standard_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_unlockable_v2_standard_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-unlockable-v2-standard-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32

# compute_buffer (no one-shot)
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v3_cb_exact_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v3_cb_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_unlockable_v3_cb_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-unlockable-cb-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32

# traversal structured (no one-shot)
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_structured_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v4_traversal_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_v4_traversal_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-v4-traversal-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32


# traversal structured wikidata (no one-shot)
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_wikidata_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 4608 \
--max-response-length 1024 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_unlockable_v4_traversal_wikidata_fuzzy_lora32 \
--exp-name qwen3_vl_4b_oven_grpo_exact_v4_traversal_wikidata_fuzzy_n16_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-exact-v4-traversal-wikidata-fuzzy-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False \
--gpu-util 0.6 \
-- trainer.log_val_generations=20 \
actor_rollout_ref.model.lora_rank=32


  # ── GRPO-traversal structured (exact reward) ── 
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_structured_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 5376 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_traversal_structured_exact_clean \
--exp-name qwen3_vl_4b_oven_grpo_traversal_structured_exact_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-traversal-structured-exact-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False

  # ── GRPO-traversal structured (shaped reward) ── 
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_structured_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 5376 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_traversal_structured_shaped_clean \
--exp-name qwen3_vl_4b_oven_grpo_traversal_structured_shaped_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-traversal-structured-shaped-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False


# ── GRPO-traversal wikidata (exact reward) ── 
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_wikidata_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 5376 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_traversal_wikidata_exact_clean \
--exp-name qwen3_vl_4b_oven_grpo_traversal_wikidata_exact_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-traversal-wikidata-exact-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False

  # ── GRPO-traversal wikidata (shaped reward) ──
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
--mode full --wandb --conda-env verl-v080 \
--conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
-A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
--train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v4_traversal_wikidata_aligned_balanced_qid_2k_seed42/train.parquet \
--val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
--reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
--steps 300 --test-freq 150 --save-freq 125 \
--train-batch-size 16 --ppo-mini-batch-size 4 \
--rollout-tp 1 --rollout-agents 2 \
--rollout-min-model-len 5376 \
--max-response-length 256 --max-prompt-length 5120 \
--total-epochs 10 --val-batch-size 256 --val-max-samples 16384 \
--ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_traversal_wikidata_shaped_clean \
--exp-name qwen3_vl_4b_oven_grpo_traversal_wikidata_shaped_clean_n8_bs16_300steps \
--wandb-run-id qwen3-vl-4b-oven-grpo-traversal-wikidata-shaped-clean-seed42 \
--wandb-resume allow \
--resume-mode auto \
--val-before-train False


### regenerate dataset for traversal with wikidata in prompt strategy
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_v4_traversal_wikidata_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 1.0 \
    --traversal-variant wikidata \
    --question-policy aligned \
    --max-train-rows 2000 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven


### test command to check that validation works
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 01:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_v2_exact_aligned_balanced_qid_2k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed_exact_metrics.py \
    --steps 6 --test-freq 5 --save-freq -1 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 8 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 4608 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 1024 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/debug_val_test \
    --exp-name debug_val_test \
    -- actor_rollout_ref.rollout.free_cache_engine=False \
    data.val_max_samples=512 \
    data.dataloader_num_workers=0




python - <<'PY'
import wandb

run = wandb.Api().run(
      "jucamohedano/oven_rsa_trace_grpo/qwen3-vl-4b-oven-grpo-exact-v2-clean-seed42"
)

wanted = [
      "val-core",
      "val-aux",
      "exact_match",
      "boxed_parse",
      "specific_hF",
      "raw_hF",
      "path_match",
      "wrong_final",
      "critic/score/mean",
      "response_length/clip_ratio",
]

rows = []
keys = set()
for row in run.scan_history(page_size=1000):
      selected = {k: v for k, v in row.items() if any(w in k for w in wanted)}
      if selected:
            selected["_step"] = row.get("_step")
            rows.append(selected)
            keys.update(selected)

print("matched keys:")
for k in sorted(keys):
      print(k)

print("\nvalidation rows:")
for r in rows:
      if any(k.startswith("val-") for k in r):
            print(r)
PY



> --exp-name qwen3_vl_4b_oven_grpo_exact_v2_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-exact-v2-clean-seed42 \
> --exp-name qwen3_vl_4b_oven_grpo_exact_v3_cb_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-exact-v3-cb-clean-seed42 \
> --exp-name qwen3_vl_4b_oven_grpo_unlockable_v2_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-unlockable-v2-clean-seed42 \
> --exp-name qwen3_vl_4b_oven_grpo_traversal_structured_exact_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-traversal-structured-exact-clean-seed42 \
> --exp-name qwen3_vl_4b_oven_grpo_traversal_structured_shaped_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-traversal-structured-shaped-clean-seed42 \
> --exp-name qwen3_vl_4b_oven_grpo_traversal_wikidata_exact_clean_n8_bs16_300steps \
> --wandb-run-id qwen3-vl-4b-oven-grpo-traversal-wikidata-exact-clean-seed42 \



python scripts/select_examples.py \
  --standard \
    2B=$R2/20260614_121741_936810_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl \
    4B=$R4/20260614_123428_725972_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl \
    8B=$R8/20260614_123530_550630_samples_scored_qwen_qwen3-4b_with_desc_rich.jsonl \
  --rsa \
    2B=$R2/20260614_121741_936810_samples_judged_rsa_solution_n16_k4_t5_qwen_qwen3-4b_with_desc_rich.jsonl \
    4B=$R4/20260614_123428_725972_samples_judged_rsa_solution_n16_k4_t5_qwen_qwen3-4b_with_desc_rich.jsonl \
    8B=$R8/20260614_123530_550630_samples_judged_rsa_solution_n16_k4_t5_qwen_qwen3-4b_with_desc_rich.jsonl \
  --criterion rsa-beats-standard-all --num 5 --seed 0 \
  --judge-label "Qwen3-4B" \
  --output viz/examples/selected_examples_qwen.json




(base) [jcamacho@login07 verl]$ cat logs/slurm/48784041.err | grep "wandb: Run data is saved locally"
(TaskRunner pid=538306) wandb: Run data is saved locally in /leonardo_scratch/fast/EUHPC_D33_243/wandb_runs/wandb/offline-run-20260707_094923-qwen3-vl-4b-oven-grpo-agg05-one-shot-3k-shaped-n16-seed42
(base) [jcamacho@login07 verl]$ cat logs/slurm/48784075.err | grep "wandb: Run data is saved locally"
(TaskRunner pid=579414) wandb: Run data is saved locally in /leonardo_scratch/fast/EUHPC_D33_243/wandb_runs/wandb/offline-run-20260707_094908-qwen3-vl-4b-oven-grpo-trav08-one-shot-3k-shaped-n16-seed42
(base) [jcamacho@login07 verl]$ cat logs/slurm/48784028.err | grep "wandb: Run data is saved locally"
(TaskRunner pid=928028) wandb: Run data is saved locally in /leonardo_scratch/fast/EUHPC_D33_243/wandb_runs/wandb/offline-run-20260707_094828-qwen3-vl-4b-oven-grpo-agg08-one-shot-3k-shaped-n16-seed42






### last few experiments before submission

## traversal only


# Example

python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42 \
    --dataset-mode rsa_trace \
    --candidate-solutions data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
    --aggregation-fraction 0.0 \
    --traversal-fraction 0.8 \
    --question-policy aligned \
    --max-train-rows 3072 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42 \
      --seed 42

bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav08_one-shot_aligned_balanced_qid_3k_seed42/train.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 500 --test-freq 150 --save-freq 192 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5376 \
    --max-response-length 256 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 256 --val-max-samples 8192 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_agg08_one-shot_3k_shaped_n16 \
    --exp-name qwen3_vl_4b_oven_grpo_trav08_one-shot_3k_shaped_bs16_300steps_n16 \
    --wandb-run-id qwen3-vl-4b-oven-grpo-trav08-one-shot-3k-shaped-n16-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32 


### my run

# for unlockable
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42 \
    --dataset-mode rsa_trace \
    --aggregation-fraction 0.0 --traversal-fraction 1.0 \
    --question-policy aligned \
    --max-train-rows 8192 --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven

python scripts/mine_unlockable_examples.py \
      --rsa-file data/rl_candidates/qwen_qwen3-vl-4b-instruct/train_aligned_balanced_qid_250k_rsa_solution_n16_k4_t1_seed42.jsonl \
      --train-parquet data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42/train.parquet \
      --output-dir data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42 \
      --seed 42

# 2048 for val
python scripts/build_verl_oven_parquet.py \
    --input data/processed/vlm_compatible_train_aligned_balanced_qid_250k_seed42.jsonl \
    --labels data/raw/oven_wikidata_chains_cleaned_labels.jsonl \
    --descs data/raw/oven_wikidata_chains_cleaned_descs_train_val.jsonl \
    --output-dir data/processed/verl_oven_rsa_val_traversal_2k_seed42 \
    --dataset-mode rsa_trace \
    --aggregation-fraction 0.0 \
    --val-prompt-type traversal \
    --question-policy aligned \
    --max-train-rows 1 --max-val-rows 2048 \
    --seed 42 --overwrite \
    --image-root /leonardo_work/EUHPC_D33_243/oven


# train traversal with non-gated shape reward
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 24:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_val_traversal_2k_seed42/val.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 250 --test-freq 50 --save-freq 100 \
    --train-batch-size 16 --ppo-mini-batch-size 4 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5888 \
    --max-response-length 768 --max-prompt-length 5120 \
    --total-epochs 10 --val-batch-size 256 --val-max-samples 2048 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_trav10_768_unlockable_shaped_n16 \
    --exp-name qwen3_vl_4b_oven_grpo_trav10_768_unlockable_shaped_n16 \
    --wandb-run-id qwen3-vl-4b-oven-grpo-trav10-768-unlockable-shaped-n16-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.log_val_generations=20 \
      actor_rollout_ref.actor.optim.lr=1e-5 \
      actor_rollout_ref.model.lora_rank=32


# elicitation battery on the base model
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 02:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_val_traversal_2k_seed42/val.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 1 --test-freq 1 --save-freq 1000 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5888 \
    --max-response-length 768 --max-prompt-length 5120 \
    --val-batch-size 256 --val-max-samples 2048 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/elicit_scratch \
    --exp-name qwen3_vl_4b_oven_elicit_trav_n8 \
    --wandb-run-id qwen3-vl-4b-oven-elicit-trav-n8-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.val_only=True \
      trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32 \
      actor_rollout_ref.rollout.val_kwargs.n=8 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_k=-1

# (a)-standard — the OFF arm of the elicitation pair
bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 02:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trace_aligned_balanced_qid_250k_seed42/val_unseen.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 1 --test-freq 1 --save-freq 1000 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5888 \
    --max-response-length 768 --max-prompt-length 5120 \
    --val-batch-size 256 --val-max-samples 2048 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/elicit_scratch \
    --exp-name qwen3_vl_4b_oven_elicit_std_n8 \
    --wandb-run-id qwen3-vl-4b-oven-elicit-std-n8-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.val_only=True \
      trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32 \
      actor_rollout_ref.rollout.val_kwargs.n=8 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_k=-1


bash examples/grpo_trainer/schedule_qwen3_vl_oven_rsa_trace_grpo.sh \
    --mode full --wandb --conda-env verl-v080 \
    --conda-sh /leonardo_scratch/fast/EUHPC_D33_243/miniconda3/etc/profile.d/conda.sh \
    -A EUHPC_D33_243 -p boost_usr_prod -g 2 -c 16 -m 128G -t 02:00:00 \
    --train-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_trav10_768_aligned_balanced_qid_2k_seed42/train_unlockable.parquet \
    --val-file /leonardo_scratch/fast/EUHPC_D33_243/oven-mllm-eval/data/processed/verl_oven_rsa_val_traversal_2k_seed42/val.parquet \
    --reward-fn /leonardo_scratch/fast/EUHPC_D33_243/verl/verl/utils/reward_score/oven_boxed.py \
    --steps 1 --test-freq 1 --save-freq 1000 \
    --rollout-n 16 --rollout-tp 1 --rollout-agents 2 \
    --rollout-min-model-len 5888 \
    --max-response-length 768 --max-prompt-length 5120 \
    --val-batch-size 256 --val-max-samples 2048 \
    --ckpts-dir /leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/elicit_scratch \
    --exp-name qwen3_vl_4b_oven_elicit_trav08ckpt_trav_n8 \
    --wandb-run-id qwen3-vl-4b-oven-elicit-trav08ckpt-trav-n8-seed42 \
    --wandb-resume allow \
    --gpu-util 0.6 \
      -- trainer.val_only=True \
      trainer.resume_mode=resume_path \
      trainer.resume_from_path=/leonardo_work/EUHPC_D33_243/oven_grpo_checkpoints/grpo_agg08_one-shot_3k_shaped_n16/global_step_500 \
      trainer.log_val_generations=20 \
      actor_rollout_ref.model.lora_rank=32 \
      actor_rollout_ref.rollout.val_kwargs.n=8 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
      actor_rollout_ref.rollout.val_kwargs.top_k=-1


import shutil, os
OUT='/leonardo_scratch/fast/EUHPC_D33_243/palace_examples'
os.makedirs(OUT, exist_ok=True)
for r in rows:
    ei=r['extra_info']
    if ei['question'].strip().lower()=='what is this palace?':
        src=r['images'][0]['image'].replace('file://','')
        name = ('Mateus' if ei['entity_id']=='Q1410441' else 'Dukes') + '_' + ei['image_id'] + '.jpg'
        shutil.copy(src, os.path.join(OUT, name))
print('copied to', OUT)