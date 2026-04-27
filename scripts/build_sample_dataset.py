"""Builder for the bundled ResearchGraph sample dataset.

Reads ``data/sample_papers.json`` (the curated set), appends any seed
entries listed below that aren't already present (matched by title),
attaches deterministic citation references, and writes the merged
file back. Idempotent: running again with no seed-list changes
produces an identical JSON.

Run from the repo root:

    python scripts/build_sample_dataset.py

The seed list is intentionally small per batch so the dataset can be
grown in reviewable chunks. Each new paper:

  * gets the next free ``p<N>`` paperId
  * picks references from a topic-tagged pool of existing papers
  * also references two of its sibling new papers (so the new batch
    is well-connected to itself, not just to the legacy core)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_papers.json"

# ---------------------------------------------------------------------------
# Topic-tagged pools of existing paperIds. New seeds reference from these
# pools according to their ``topic`` field, giving the resulting graph
# meaningful cluster structure rather than uniform-random edges.
# ---------------------------------------------------------------------------
TOPIC_REFS: dict[str, list[str]] = {
    "nlp": [
        "p001", "p002", "p003", "p010", "p011", "p020", "p025",
        "p026", "p032", "p041", "p042", "p045", "p046", "p049",
    ],
    "vision": [
        "p009", "p016", "p017", "p018", "p030", "p043", "p044",
    ],
    "diffusion": [
        "p004", "p005", "p013", "p021", "p022", "p027",
        "p028", "p031", "p039", "p047",
    ],
    "gnn": [
        "p006", "p007", "p014", "p015", "p023", "p024", "p029",
        "p034", "p038", "p048", "p050",
    ],
    "rl": [
        "p008", "p016", "p042",
    ],
    "foundation": [
        "p001", "p010", "p011", "p016", "p035", "p036", "p037",
    ],
}

# ---------------------------------------------------------------------------
# Seed entries. Add new ones at the end; do not reorder, so paperIds stay
# stable across runs.
# ---------------------------------------------------------------------------
SEEDS: list[dict] = [
    # ----- NLP / language models -----
    {
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "year": 2019, "venue": "arXiv", "citation_count": 21000,
        "authors": ["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du",
                    "Mandar Joshi", "Danqi Chen", "Omer Levy",
                    "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"],
        "topic": "nlp",
        "abstract": ("A replication study of BERT pretraining that carefully "
                     "measures the impact of key hyperparameters and training "
                     "data size, finding BERT was significantly undertrained."),
    },
    {
        "title": ("ELECTRA: Pre-training Text Encoders as Discriminators "
                  "Rather Than Generators"),
        "year": 2020, "venue": "ICLR", "citation_count": 5300,
        "authors": ["Kevin Clark", "Minh-Thang Luong",
                    "Quoc V. Le", "Christopher D. Manning"],
        "topic": "nlp",
        "abstract": ("Replaces masked language modeling with a discriminative "
                     "task that detects replaced tokens, training the encoder "
                     "more sample-efficiently than BERT."),
    },
    {
        "title": ("BART: Denoising Sequence-to-Sequence Pre-training for "
                  "Natural Language Generation, Translation, and "
                  "Comprehension"),
        "year": 2019, "venue": "ACL", "citation_count": 9100,
        "authors": ["Mike Lewis", "Yinhan Liu", "Naman Goyal",
                    "Marjan Ghazvininejad", "Abdelrahman Mohamed",
                    "Omer Levy", "Veselin Stoyanov", "Luke Zettlemoyer"],
        "topic": "nlp",
        "abstract": ("A denoising autoencoder pretrained by corrupting text "
                     "with an arbitrary noising function and learning to "
                     "reconstruct it; effective for both NLG and NLU."),
    },
    {
        "title": ("XLNet: Generalized Autoregressive Pretraining for "
                  "Language Understanding"),
        "year": 2019, "venue": "NeurIPS", "citation_count": 9000,
        "authors": ["Zhilin Yang", "Zihang Dai", "Yiming Yang",
                    "Jaime Carbonell", "Ruslan Salakhutdinov",
                    "Quoc V. Le"],
        "topic": "nlp",
        "abstract": ("Combines the bidirectionality of BERT with the "
                     "autoregressive formulation of Transformer-XL via a "
                     "permutation language modeling objective."),
    },
    {
        "title": ("PaLM: Scaling Language Modeling with Pathways"),
        "year": 2022, "venue": "JMLR", "citation_count": 3500,
        "authors": ["Aakanksha Chowdhery", "Sharan Narang", "Jacob Devlin",
                    "Maarten Bosma", "Gaurav Mishra", "Adam Roberts",
                    "Paul Barham", "Hyung Won Chung", "Charles Sutton",
                    "Sebastian Gehrmann"],
        "topic": "nlp",
        "abstract": ("A 540B-parameter dense decoder-only Transformer trained "
                     "with the Pathways system, achieving breakthrough "
                     "few-shot performance across hundreds of NLP tasks."),
    },

    # ----- Vision transformers / self-supervised vision -----
    {
        "title": ("Swin Transformer: Hierarchical Vision Transformer using "
                  "Shifted Windows"),
        "year": 2021, "venue": "ICCV", "citation_count": 14000,
        "authors": ["Ze Liu", "Yutong Lin", "Yue Cao", "Han Hu",
                    "Yixuan Wei", "Zheng Zhang", "Stephen Lin",
                    "Baining Guo"],
        "topic": "vision",
        "abstract": ("A general-purpose vision Transformer whose "
                     "hierarchical feature maps and shifted-window "
                     "self-attention give linear complexity to image size."),
    },
    {
        "title": "A ConvNet for the 2020s",
        "year": 2022, "venue": "CVPR", "citation_count": 4200,
        "authors": ["Zhuang Liu", "Hanzi Mao", "Chao-Yuan Wu",
                    "Christoph Feichtenhofer", "Trevor Darrell",
                    "Saining Xie"],
        "topic": "vision",
        "abstract": ("ConvNeXt modernizes a standard ResNet toward the "
                     "design of a vision Transformer, recovering "
                     "competitive accuracy with pure convolutions."),
    },
    {
        "title": ("A Simple Framework for Contrastive Learning of Visual "
                  "Representations"),
        "year": 2020, "venue": "ICML", "citation_count": 18000,
        "authors": ["Ting Chen", "Simon Kornblith", "Mohammad Norouzi",
                    "Geoffrey Hinton"],
        "topic": "vision",
        "abstract": ("SimCLR shows that contrastive self-supervised learning "
                     "with strong augmentation and a learned projection head "
                     "matches supervised ImageNet performance."),
    },
    {
        "title": "Momentum Contrast for Unsupervised Visual Representation Learning",
        "year": 2020, "venue": "CVPR", "citation_count": 12000,
        "authors": ["Kaiming He", "Haoqi Fan", "Yuxin Wu", "Saining Xie",
                    "Ross Girshick"],
        "topic": "vision",
        "abstract": ("MoCo builds a dynamic dictionary with a queue and a "
                     "momentum-updated encoder, enabling large and consistent "
                     "negative sets for contrastive pretraining."),
    },
    {
        "title": ("Bootstrap Your Own Latent: A New Approach to "
                  "Self-Supervised Learning"),
        "year": 2020, "venue": "NeurIPS", "citation_count": 7800,
        "authors": ["Jean-Bastien Grill", "Florian Strub",
                    "Florent Altché", "Corentin Tallec",
                    "Pierre H. Richemond", "Elena Buchatskaya",
                    "Carl Doersch", "Bernardo Avila Pires",
                    "Zhaohan Daniel Guo", "Mohammad Gheshlaghi Azar",
                    "Bilal Piot", "Koray Kavukcuoglu", "Rémi Munos",
                    "Michal Valko"],
        "topic": "vision",
        "abstract": ("BYOL learns visual representations without negative "
                     "pairs by predicting a target network's projection "
                     "from an online network and bootstrapping."),
    },

    # ----- Diffusion models -----
    {
        "title": ("GLIDE: Towards Photorealistic Image Generation and "
                  "Editing with Text-Guided Diffusion Models"),
        "year": 2021, "venue": "ICML", "citation_count": 2900,
        "authors": ["Alex Nichol", "Prafulla Dhariwal", "Aditya Ramesh",
                    "Pranav Shyam", "Pamela Mishkin", "Bob McGrew",
                    "Ilya Sutskever", "Mark Chen"],
        "topic": "diffusion",
        "abstract": ("Compares CLIP-guided and classifier-free-guided "
                     "diffusion for text-to-image, finding classifier-free "
                     "guidance produces more photorealistic outputs."),
    },
    {
        "title": "Adding Conditional Control to Text-to-Image Diffusion Models",
        "year": 2023, "venue": "ICCV", "citation_count": 4100,
        "authors": ["Lvmin Zhang", "Anyi Rao", "Maneesh Agrawala"],
        "topic": "diffusion",
        "abstract": ("ControlNet adds spatial conditioning (edges, depth, "
                     "pose, …) to a frozen pretrained diffusion model via a "
                     "trainable copy connected by zero-initialized layers."),
    },
    {
        "title": "Elucidating the Design Space of Diffusion-Based Generative Models",
        "year": 2022, "venue": "NeurIPS", "citation_count": 2500,
        "authors": ["Tero Karras", "Miika Aittala", "Timo Aila",
                    "Samuli Laine"],
        "topic": "diffusion",
        "abstract": ("Reframes diffusion training and sampling in a unified "
                     "design space and proposes EDM, a sampler and training "
                     "recipe that materially advances state-of-the-art FID."),
    },
    {
        "title": "Scalable Diffusion Models with Transformers",
        "year": 2023, "venue": "ICCV", "citation_count": 1300,
        "authors": ["William Peebles", "Saining Xie"],
        "topic": "diffusion",
        "abstract": ("DiT replaces the U-Net backbone of latent diffusion "
                     "with a Transformer, demonstrating that diffusion "
                     "models scale with compute like other Transformers."),
    },
    {
        "title": "Imagen Video: High Definition Video Generation with Diffusion Models",
        "year": 2022, "venue": "arXiv", "citation_count": 900,
        "authors": ["Jonathan Ho", "William Chan", "Chitwan Saharia",
                    "Jay Whang", "Ruiqi Gao", "Alexey Gritsenko",
                    "Diederik P. Kingma", "Ben Poole", "Mohammad Norouzi",
                    "David J. Fleet", "Tim Salimans"],
        "topic": "diffusion",
        "abstract": ("A cascade of video diffusion models that generate "
                     "high-definition video from text, combining v-prediction "
                     "training with progressive distillation for fast sampling."),
    },

    # ----- GNNs -----
    {
        "title": "Do Transformers Really Perform Bad for Graph Representation?",
        "year": 2021, "venue": "NeurIPS", "citation_count": 1700,
        "authors": ["Chengxuan Ying", "Tianle Cai", "Shengjie Luo",
                    "Shuxin Zheng", "Guolin Ke", "Di He",
                    "Yanming Shen", "Tie-Yan Liu"],
        "topic": "gnn",
        "abstract": ("Graphormer adapts the Transformer to graphs through "
                     "centrality, spatial, and edge encodings, winning the "
                     "OGB-LSC quantum chemistry track."),
    },
    {
        "title": "Graph Convolutional Neural Networks for Web-Scale Recommender Systems",
        "year": 2018, "venue": "KDD", "citation_count": 3800,
        "authors": ["Rex Ying", "Ruining He", "Kaifeng Chen",
                    "Pong Eksombatchai", "William L. Hamilton",
                    "Jure Leskovec"],
        "topic": "gnn",
        "abstract": ("PinSage scales GCNs to billions of nodes via random "
                     "walks for neighborhood sampling and on-the-fly "
                     "convolutions, deployed at Pinterest."),
    },

    # ----- RL / decision-making -----
    {
        "title": ("Mastering the Game of Go with Deep Neural Networks and "
                  "Tree Search"),
        "year": 2016, "venue": "Nature", "citation_count": 18000,
        "authors": ["David Silver", "Aja Huang", "Chris J. Maddison",
                    "Arthur Guez", "Laurent Sifre",
                    "George van den Driessche", "Julian Schrittwieser",
                    "Ioannis Antonoglou", "Veda Panneershelvam",
                    "Marc Lanctot"],
        "topic": "rl",
        "abstract": ("AlphaGo combines value and policy networks with "
                     "Monte Carlo Tree Search, defeating top human "
                     "professionals at the game of Go."),
    },
    {
        "title": "Proximal Policy Optimization Algorithms",
        "year": 2017, "venue": "arXiv", "citation_count": 18500,
        "authors": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal",
                    "Alec Radford", "Oleg Klimov"],
        "topic": "rl",
        "abstract": ("PPO is a family of policy-gradient methods that uses a "
                     "clipped surrogate objective to balance trust-region "
                     "stability with the simplicity of first-order updates."),
    },

    # ----- Foundational embeddings -----
    {
        "title": "Distributed Representations of Words and Phrases and their Compositionality",
        "year": 2013, "venue": "NeurIPS", "citation_count": 41000,
        "authors": ["Tomas Mikolov", "Ilya Sutskever", "Kai Chen",
                    "Greg Corrado", "Jeffrey Dean"],
        "topic": "foundation",
        "abstract": ("Skip-gram with negative sampling and the subsampling "
                     "of frequent words yield word vectors that capture "
                     "syntactic and semantic regularities."),
    },
]


def _next_free_pid(existing_ids: set[str]) -> int:
    """Return the smallest N such that ``p<N>`` is not in ``existing_ids``."""
    n = 1
    while f"p{n:03d}" in existing_ids:
        n += 1
    return n


def main() -> None:
    rng = random.Random(0)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing_titles = {r["title"] for r in existing}
    existing_ids = {r["paperId"] for r in existing}

    # Skip seeds whose title already lives in the dataset (idempotency).
    pending = [s for s in SEEDS if s["title"] not in existing_titles]
    skipped = len(SEEDS) - len(pending)

    next_n = _next_free_pid(existing_ids)
    new_records: list[dict] = []
    for i, seed in enumerate(pending):
        pid = f"p{next_n + i:03d}"
        if pid in existing_ids:
            raise RuntimeError(f"id collision while assigning {pid}")
        topic = seed.get("topic", "foundation")
        pool = TOPIC_REFS.get(topic, TOPIC_REFS["foundation"])
        n_topic = rng.randint(3, min(6, len(pool)))
        chosen = rng.sample(pool, k=n_topic)
        # Add a couple of cross-cluster references for richer structure.
        cross_pool = sorted(existing_ids - set(chosen))
        chosen.extend(rng.sample(cross_pool, k=min(2, len(cross_pool))))
        new_records.append({
            "paperId": pid,
            "title": seed["title"],
            "abstract": seed["abstract"],
            "year": seed["year"],
            "authors": [{"name": a} for a in seed["authors"]],
            "venue": seed["venue"],
            "citationCount": seed["citation_count"],
            "references": [{"paperId": r} for r in chosen],
            "url": f"https://arxiv.org/abs/{pid}",
        })

    # Stitch the new papers to each other so the new batch isn't an
    # isolated star around the legacy core.
    new_ids = [r["paperId"] for r in new_records]
    for r in new_records:
        peers = [pid for pid in new_ids if pid != r["paperId"]]
        for ref in rng.sample(peers, k=min(2, len(peers))):
            r["references"].append({"paperId": ref})

    merged = existing + new_records
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} papers to {DATA_PATH}")
    print(f"  - {len(existing)} previously curated (kept verbatim)")
    print(f"  - {len(new_records)} newly added")
    if skipped:
        print(f"  - {skipped} seed(s) skipped (titles already present)")


if __name__ == "__main__":
    main()
