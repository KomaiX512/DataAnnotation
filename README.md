# Decentralized Data Annotation Subnet

Welcome to the Decentralized Data Annotation Subnet! This project leverages the power of the Bittensor network to create high-quality, commercial-grade datasets through a decentralized, trustless pipeline of human-in-the-loop and model-assisted labeling.

---

## What is this subnet?
This subnet is a decentralized data annotation network where miners compete to label images, identify objects, and detect hazards. Validators distribute annotation tasks, secretively injecting a hidden **Golden Set** (a small subset of pre-labeled images with verified ground-truth annotations) to evaluate performance. The highest-quality annotations from honest miners are then mathematically fused into a robust, auditable commercial JSONL dataset.

## How does it work?
The subnet operates in a continuous, linear miner-validator cycle:
1. **Task Distribution**: The validator packages a set of images—combining unlabeled images that need annotation with secret **Golden Set** images—and distributes them to miners.
2. **Annotation and Upload**: Miners run object detection algorithms (such as YOLO or vision-language models) to locate and classify hazards. They upload their completed annotations to Cloudflare R2 and return the references to the validator.
3. **Fidelity Scoring**: The validator checks the miner's annotations against the hidden Golden Set. Miners receive a **Fidelity Score** based on how accurately they identified and matched the ground-truth classes and bounding boxes.
4. **Bayesian Fusion**: Using the fidelity scores as weightings, the validator merges miner annotations on the unlabeled images to form a single, high-confidence consensus annotation.
5. **Weight Settlement**: The validator translates these evaluation scores into on-chain weights, directing rewards to the best-performing miners on the Bittensor network.

## How are incentives distributed?
Incentives are distributed via a **dual-flywheel reward system** designed to reward only accurate, honest work:
* **Fidelity Reward**: Miners are rewarded for high-accuracy annotations on the hidden Golden Set. Failed detections or incorrect labels reduce this score, and any hallucinated labels (detecting objects that aren't there) trigger a strict penalty.
* **Adoption Bonus**: Miners earn additional rewards when their unlabeled annotations are selected and adopted into the final, consensus-fused commercial dataset. 

If a miner attempts to game the system by only labeling Golden Set images or submitting low-quality annotations, their overall reliability weight drops to near-zero, neutralizing their influence and eliminating their rewards.

## How do I get started?
To get started as a miner or validator, clone the repository, set up a virtual environment, install the dependencies, and copy the template configuration file. 
```bash
git clone https://github.com/KomaiX512/JHA_subnet.git
cd JHA_subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Once installed, open `.env` and fill in your Bittensor wallet names, hotkeys, Cloudflare R2 storage credentials, and desired miner backend or validator settings. 

Ready to dive deeper? Choose your role:

# [👉 MINER SETUP GUIDE (MINER.md)](file:///home/komail/bittensor-subnet-template-1/MINER.md)

# [👉 VALIDATOR SETUP GUIDE (VALIDATOR.md)](file:///home/komail/bittensor-subnet-template-1/VALIDATOR.md)
