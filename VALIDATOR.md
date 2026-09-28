# Validator Overview

The validator coordinates annotation rounds, keeps the labeled golden set private, evaluates miner fidelity, and curates useful public-image contributions for export. Rewards reflect both fidelity and selected contributions from active miners across rounds.

## Round lifecycle

The validator plans durable epoch tasks containing up to 30 distinct images. The saved epoch seed and cursor allow a restart to continue the same plan without repeating or skipping public images. Golden membership and labels remain validator-only.

All sampled miners receive the same canonical task images through miner-specific opaque IDs. A task's 600-second monotonic response window begins at dispatch and includes inference, artifact retrieval, and schema validation. Timeout, invalid payloads, and responses received after the deadline are abstentions. Closing a task is final; later tasks have new task IDs.

After the miner window closes, the validator scores golden annotations with the existing fidelity scorer and sends public submissions through its adjudication boundary. Adjudication time is measured separately from the response window. Each public image contributes at most one complete submitted annotation record or is left unassigned; selected records are not marked as human ground-truth verified.

## Scoring and operations

The existing golden scoring and alpha configuration remain in use. Selected-image contribution is recorded separately and feeds the existing reward path. The on-chain weight calculation is unchanged.

For a local verification run, keep the chain, wallet, state, image cache, miner artifacts, and exports inside fresh run-specific resources. Use a loopback endpoint proven to belong to that run. Read the dataset manifest and images without writing to the source bucket; write artifacts and exports to isolated local storage. Never reuse or clean up resources owned by another run.

The selection evaluator requires an operator-approved local visual checkpoint and pinned revision. It must fail closed if that checkpoint is missing or its processor cannot accept images. The evaluator's private implementation and deployment instructions are not part of this guide.
