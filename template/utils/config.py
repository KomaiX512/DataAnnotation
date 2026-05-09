# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import subprocess
import argparse
from pathlib import Path
import bittensor as bt
from .logging import setup_events_logger


def is_cuda_available():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-L"], stderr=subprocess.STDOUT
        )
        if "NVIDIA" in output.decode("utf-8"):
            return "cuda"
    except Exception:
        pass
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in output:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        events_logger = setup_events_logger(
            config.neuron.full_path, config.neuron.events_retention_size
        )
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=is_cuda_available(),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        help="Turn off wandb.",
        default=False,
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--miner.training_workspace",
        type=str,
        help="Writable directory where miner training artifacts are produced.",
        default=str(Path.cwd() / "artifacts" / "miner_training"),
    )

    parser.add_argument(
        "--miner.private_dataset_root",
        type=str,
        help="Optional miner-owned private images directory to mix into training.",
        default="",
    )

    parser.add_argument(
        "--miner.enable_auto_hpo",
        action="store_true",
        help="Enable miner-side auto-HPO loop for training tasks.",
        default=False,
    )

    parser.add_argument(
        "--miner.autoresearch",
        action="store_true",
        help="Enable Karpathy-style autoresearch loop before YOLO training.",
        default=False,
    )

    parser.add_argument(
        "--miner.autoresearch_max_iters",
        type=int,
        help="Maximum iterations for autoresearch loop when enabled.",
        default=4,
    )

    parser.add_argument(
        "--miner.autoresearch_experiment_minutes",
        type=int,
        help="Per-iteration budget in minutes for autoresearch experiments.",
        default=5,
    )

    parser.add_argument(
        "--miner.autoresearch_log_level",
        type=str,
        help="Autoresearch log level.",
        default="INFO",
    )
    parser.add_argument(
        "--miner.response_mode",
        type=str,
        choices=["standard", "replay_nonce", "malformed_manifest", "wrong_model_hash"],
        help="Stress-test mode for miner response shaping.",
        default="standard",
    )

    parser.add_argument(
        "--miner.annotation_backend",
        type=str,
        choices=["deterministic", "yolo"],
        help="Annotation backend for dual-flywheel tasks (deterministic for CI; yolo uses fine-tuned weights).",
        default="deterministic",
    )

    parser.add_argument(
        "--miner.dual_flywheel_r2_prefix",
        type=str,
        help="R2 key prefix for dual-flywheel artifacts (per-task subfolders are appended).",
        default="miners/dual_flywheel",
    )

    parser.add_argument(
        "--miner.random_hpo_draw",
        action="store_true",
        help=(
            "Dual-flywheel / training: pick one random hyperparameter bundle from the autoresearch "
            "grid (use with --miner.hpo_seed so miners diverge). Mutually exclusive with autoresearch "
            "when autoresearch is off."
        ),
        default=False,
    )

    parser.add_argument(
        "--miner.hpo_seed",
        type=int,
        help="Seed for --miner.random_hpo_draw (different seeds => different hyperparameter draws).",
        default=0,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="template-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.forward_step_sleep_seconds",
        type=float,
        help=(
            "Wall-clock seconds to sleep after each successful validation step (operator pacing; "
            "e.g. 300 for five-minute rounds)."
        ),
        default=0.0,
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
    )

    parser.add_argument(
        "--neuron.training_timeout",
        type=float,
        help="Dendrite timeout for miner training tasks. Must cover the full train/upload response path.",
        default=0.0,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=50,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--neuron.dataset_root",
        type=str,
        help="Path to validator-owned hazard dataset partitions.",
        default=str(
            Path(__file__).resolve().parents[2] / "data" / "hazard"
        ),
    )

    parser.add_argument(
        "--neuron.scheduler_seed",
        type=int,
        help="Deterministic seed for cohort scheduler and dataset sampling.",
        default=13,
    )

    parser.add_argument(
        "--neuron.promotion_threshold",
        type=float,
        help="Minimum final score required for model promotion.",
        default=0.75,
    )
    parser.add_argument(
        "--neuron.serving_recency_decay",
        type=float,
        help="Linear decay factor for promoted model serving priority by age-in-steps.",
        default=0.003,
    )
    parser.add_argument(
        "--neuron.serving_min_live_multiplier",
        type=float,
        help="Minimum recency multiplier retained for older promoted models.",
        default=0.35,
    )

    parser.add_argument(
        "--neuron.baseline_checkpoint_uri",
        type=str,
        help="URI to the current global baseline checkpoint miners fine-tune.",
        default="yolov8s.pt",
    )

    parser.add_argument(
        "--neuron.baseline_checkpoint_hash",
        type=str,
        help="Expected SHA256 hash of the current global baseline checkpoint.",
        default="",
    )

    parser.add_argument(
        "--neuron.max_training_seconds",
        type=int,
        help="Training budget for smoke or production TrainingTask synapses.",
        default=60,
    )

    parser.add_argument(
        "--neuron.incentive_temperature",
        type=float,
        help="Temperature for broad softmax incentive shaping.",
        default=0.25,
    )

    parser.add_argument(
        "--neuron.incentive_floor",
        type=float,
        help="Minimum nonzero share for eligible value-adding miners.",
        default=0.002,
    )

    parser.add_argument(
        "--neuron.incentive_min_score",
        type=float,
        help="Minimum EMA score required before a miner receives broad-softmax share.",
        default=0.05,
    )

    # ---------- Dual-flywheel (annotation + training) configuration ----------
    parser.add_argument(
        "--neuron.task_mode",
        type=str,
        choices=["legacy_hazard_detection", "dual_flywheel"],
        help=(
            "Validator orchestration mode. 'dual_flywheel' dispatches "
            "AnnotationAndTrainingTask synapses, scores annotations against the "
            "Golden Set + consensus, evaluates miner checkpoints, and assembles "
            "a per-image_id commercial dataset."
        ),
        default="dual_flywheel",
    )
    parser.add_argument(
        "--neuron.flywheel_image_cache_root",
        type=str,
        help=(
            "Local filesystem root where the validator materializes images for "
            "miner consumption. Each image is keyed by its content-hash image_id."
        ),
        default=str(Path(__file__).resolve().parents[2] / "data" / "flywheel" / "image_cache"),
    )
    parser.add_argument(
        "--neuron.flywheel_image_serving_base_url",
        type=str,
        help=(
            "Optional public base URL prefix for serving images to miners. When "
            "empty the validator emits file:// URIs (single-host / localnet)."
        ),
        default="",
    )
    parser.add_argument(
        "--neuron.flywheel_golden_dataset_id",
        type=str,
        help="Hugging Face dataset id for the golden labeled construction-safety dataset.",
        default="keremberke/construction-safety-object-detection",
    )
    parser.add_argument(
        "--neuron.flywheel_golden_split",
        type=str,
        help="Split within the golden dataset to load before applying the 30/70 partition.",
        default="train",
    )
    parser.add_argument(
        "--neuron.flywheel_golden_ratio",
        type=float,
        help="Fraction of the labeled construction-safety dataset reserved as the validator-only Golden Set.",
        default=0.3,
    )
    parser.add_argument(
        "--neuron.flywheel_golden_split_seed",
        type=int,
        help="Seed used to deterministically split the labeled dataset into Golden vs Training pools.",
        default=20260509,
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_dataset_ids",
        type=str,
        help=(
            "Comma-separated Hugging Face dataset ids for the unlabeled annotation pool. "
            "Use hub_id@split per entry (e.g. org/ds@test); otherwise flywheel_annotation_split applies."
        ),
        default=(
            "keremberke/construction-safety-object-detection@test,"
            "keremberke/construction-safety-object-detection@validation"
        ),
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_split",
        type=str,
        help="Split name to load from each annotation dataset (use 'train' if unsure).",
        default="train",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_max_per_dataset",
        type=int,
        help="Per-dataset cap for the unlabeled annotation pool. Use 0 for no cap.",
        default=512,
    )
    parser.add_argument(
        "--neuron.flywheel_benchmark_dataset_id",
        type=str,
        help=(
            "Cross-domain Hugging Face dataset id used by the validator to detect "
            "miner overfitting (never shown to miners)."
        ),
        default="rishitdagli/cppe-5",
    )
    parser.add_argument(
        "--neuron.flywheel_benchmark_split",
        type=str,
        help="Split name to load from the cross-domain benchmark dataset.",
        default="test",
    )
    parser.add_argument(
        "--neuron.flywheel_benchmark_max_samples",
        type=int,
        help="Maximum benchmark samples loaded per round. Use 0 for no cap.",
        default=64,
    )
    parser.add_argument(
        "--neuron.flywheel_hf_revision",
        type=str,
        help=(
            "Hugging Face git revision for Hub datasets (use refs/convert/parquet when the "
            "repo still ships a legacy dataset script). Empty = Hub default branch loader."
        ),
        default="refs/convert/parquet",
    )
    parser.add_argument(
        "--neuron.flywheel_annotation_request_size",
        type=int,
        help="Total images per AnnotationAndTrainingTask request (golden + non-golden).",
        default=10,
    )
    parser.add_argument(
        "--neuron.flywheel_golden_injection_per_request",
        type=int,
        help="Number of Golden images injected (unlabeled to the miner) into each annotation request.",
        default=2,
    )
    parser.add_argument(
        "--neuron.flywheel_training_images_per_request",
        type=int,
        help="Number of labeled training images surfaced to miners for fine-tuning per round.",
        default=16,
    )
    parser.add_argument(
        "--neuron.flywheel_alpha_annotation",
        type=float,
        help="Annotation-fidelity weight in the final on-chain weight formula.",
        default=0.4,
    )
    parser.add_argument(
        "--neuron.flywheel_beta_model",
        type=float,
        help="Model-accuracy weight in the final on-chain weight formula.",
        default=0.4,
    )
    parser.add_argument(
        "--neuron.flywheel_gamma_adoption",
        type=float,
        help="Adoption-bonus weight in the final on-chain weight formula.",
        default=0.2,
    )
    parser.add_argument(
        "--neuron.flywheel_hallucination_penalty",
        type=float,
        help="Multiplicative penalty applied per hallucinated annotation on a Golden image.",
        default=0.5,
    )
    parser.add_argument(
        "--neuron.flywheel_commercial_dataset_prefix",
        type=str,
        help=(
            "Object-storage prefix (file://, r2://, or s3://) where the assembled per-image_id "
            "winning annotations are exported as the subnet's commercial dataset."
        ),
        default=str(
            (Path(__file__).resolve().parents[2] / "artifacts" / "commercial_dataset").as_uri()
        ),
    )
    parser.add_argument(
        "--neuron.flywheel_commercial_export_every",
        type=int,
        help="Number of validator forward steps between commercial dataset exports.",
        default=10,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="template-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="opentensor-dev",
    )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
