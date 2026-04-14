# LLM Wiki Schema - Brain Emulation Q&A System

This document defines the wiki structure and conventions for the Brain Emulation Q&A System project. It follows the LLM Wiki pattern where the LLM owns all wiki maintenance while humans curate sources and ask questions.

## Core Principles

### 1. Persistent Knowledge Accumulation

The wiki is not just retrieval at query time—it's a **compounding artifact** that grows richer with every source ingested and question answered. Cross-references are pre-built, contradictions are flagged, and synthesis reflects all accumulated knowledge.

### 2. LLM Writes, Human Reads

The LLM creates and maintains all wiki pages. Humans provide sources, ask questions, and explore the knowledge graph. This division of labor ensures maintenance burden stays near zero.

### 3. Questions Become Wiki Pages

Good answers, analyses, and comparisons should be saved as new wiki pages rather than disappearing into chat history. Every valuable insight compounds the knowledge base.

---

## Directory Structure

```
brain_coding_qa/
├── README.md                    # Project documentation (root level)
├── requirements.txt             # Dependencies
├── src/                         # Source code
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── training/
│   └── main.py
├── wiki/                        # LLM-maintained knowledge base
│   ├── AGENTS.md                # This schema file
│   ├── index.md                 # Catalog of all wiki pages
│   ├── log.md                   # Chronological activity log
│   ├── concepts/                # Concept/theory pages
│   │   ├── colocation.md
│   │   ├── sparse-spiking.md
│   │   ├── event-driven-processing.md
│   │   ├── leaky-integrate-and-fire.md
│   │   ├── surrogate-gradient.md
│   │   ├── associative-memory.md
│   │   └── energy-efficiency.md
│   ├── entities/                # Implementation entity pages
│   │   ├── brain-coding-model.md
│   │   ├── spiking-encoder.md
│   │   ├── associative-memory-module.md
│   │   ├── spiking-decoder.md
│   │   └── brain-trainer.md
│   ├── sources/                 # Source document summaries
│   │   └── brain-emulation-paper.md
│   ├── syntheses/               # Integrated analysis pages
│   │   ├── architecture-overview.md
│   │   └── energy-analysis.md
│   └── comparisons/             # Comparative analysis (future)
└── raw/                         # Raw source materials (future)
    ├── sources/                 # Original documents (immutable)
    └── wiki-assets/             # Images and attachments
```

---

## Operations

### Ingest (Add New Source)

When a new source is added:

1. Read source and extract key information
2. Create summary page → `wiki/sources/`
3. Update relevant concept/entity pages
4. Update `wiki/index.md` with new page
5. Log entry in `wiki/log.md`
6. Check for contradictions with existing content

**Example flow:**
```
New paper on Hopfield Networks →
  - Create wiki/sources/hopfield-networks-2024.md
  - Update wiki/concepts/associative-memory.md
  - Update wiki/syntheses/architecture-overview.md
  - Add entry to wiki/log.md
  - Update wiki/index.md
```

### Query (Ask Questions)

1. Search `wiki/index.md` for relevant pages
2. Read related concepts, entities, and syntheses
3. Synthesize answer with citations
4. If answer is valuable, save as new wiki page
5. Log query in `wiki/log.md`

**Example queries:**
- "How does colocation improve energy efficiency?"
- "Compare LIF neurons with traditional MLP layers"
- "What's the state of neuromorphic hardware in 2024?"

### Lint (Health Check)

Periodically run maintenance:

- Check for contradictions between pages
- Update stale claims with new information
- Find orphan pages (no inbound links)
- Add missing cross-references
- Identify gaps needing new sources
- Suggest new investigation directions

---

## Page Format Rules

### YAML Frontmatter (All Pages)

```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2, tag3]
sources: [source1, source2]  # References to raw sources
---
```

### Naming Conventions

- **File names**: kebab-case (e.g., `sparse-spiking.md`)
- **Titles**: Title Case (e.g., "Sparse Spiking")
- **Internal links**: `[[path/to/page|display text]]`
- **Cross-references**: Use full wiki path format

### Content Structure

**Concept pages** (`wiki/concepts/`):
- Definition and explanation
- Mathematical formulation (if applicable)
- Implementation examples
- Related concepts and entities
- References to sources

**Entity pages** (`wiki/entities/`):
- Component description
- Code implementation
- Key parameters
- Usage examples
- Related components

**Source pages** (`wiki/sources/`):
- Summary of main points
- Key findings
- Relevance to project
- Citations to original work

**Synthesis pages** (`wiki/syntheses/`):
- Integrated analysis
- Architecture overviews
- Comparative studies
- Tradeoff analysis
- Future directions

---

## Index and Log

### index.md

Content-oriented catalog of all wiki pages:

```markdown
## Concepts
- [[wiki/concepts/colocation|Colocation]] - Memory and computation in same space
- [[wiki/concepts/sparse-spiking|Sparse Spiking]] - Threshold-based activation

## Entities
- [[wiki/entities/brain-coding-model|BrainCodingModel]] - Main integrated model

## Syntheses
- [[wiki/syntheses/architecture-overview|System Architecture]]
```

**Purpose**: Quick navigation, query starting point, page discovery

### log.md

Chronological append-only record:

```markdown
## [2026-04-15] ingest | Brain Emulation Research Paper
- Summary: Documented core research foundations
- Updated pages: concepts/associative-memory.md, sources/brain-emulation-paper.md

## [2026-04-15] query | How does colocation work?
- Answer format: Concept page with implementation example
- New page created: None (existing content sufficient)
```

**Purpose**: Timeline of wiki evolution, parseable with Unix tools

**Query example**: `grep "^## \[" wiki/log.md | tail -10`

---

## Tools and Workflow

### Recommended Tools

- **Obsidian**: IDE for viewing and browsing wiki
- **Obsidian Graph View**: Visualize wiki structure and connections
- **Git**: Version control for wiki (free branching, history)
- **Dataview plugin**: Query YAML frontmatter for dynamic tables
- **Marp plugin**: Generate slide decks from wiki content

### Workflow Tips

1. **One source at a time**: Prefer sequential ingestion with human oversight
2. **Read summaries**: Review LLM-generated summaries before accepting
3. **Follow links**: Explore graph view to discover connections
4. **Guide emphasis**: Direct LLM on what aspects to highlight
5. **Regular lint**: Monthly health checks keep wiki consistent

### Future Enhancements

- **Local search engine**: qmd for BM25/vector search + LLM re-ranking
- **CLI utilities**: Scripts for batch processing, validation
- **Automated linting**: Scheduled health checks
- **Web interface**: Browser-based wiki exploration

---

## This Project's Wiki

The Brain Emulation Q&A wiki documents:

- **Core principles**: Colocation, sparse spiking, event-driven processing
- **Architecture**: SNN components, memory systems, training pipeline
- **Research**: Papers and sources on neuromorphic computing
- **Implementation**: Code structure, parameter choices, design decisions
- **Analysis**: Energy efficiency tradeoffs, performance metrics

**Current state**: 18 pages covering 7 concepts, 5 entities, 2 syntheses, 1 source

**Next steps**:
- Add more research sources
- Create comparison pages (vs. Transformers, vs. traditional RNNs)
- Document training experiments and results
- Expand hardware implementation guides

---

## Evolution

This schema evolves as the project grows. Add new directories, page types, or workflows as needed. The key is maintaining consistency while allowing flexibility for domain-specific requirements.

**Last updated**: 2026-04-15