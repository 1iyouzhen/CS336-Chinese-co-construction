# Data Engineering

<div align="center">
<img width="980" height="470" alt="673cb3c261b9da565f5b896b453808c7" src="https://github.com/user-attachments/assets/b1a0f623-9aad-497d-a804-d1b5212fa5c6" />
   <p>Figure 11.1 Data Engineering and LLM Training</p>
</div>

In previous chapters, we discussed how to train stronger models through architecture design, optimization methods, tokenization, and scaling—assuming the training data was already given. From this chapter on, we turn to a more fundamental question: what data should language models actually be trained on? Real-world LLM development shows that **data is often more critical than model structure itself**—mainstream foundation models almost always openly publish their complete architecture and training processes, yet remain highly vague about the specific composition of training data. This precisely indicates that data is the hardest to replicate and the most competitively valuable part. With the long-tail nature of data, the capability boundaries of models in the real world are ultimately defined by the coverage scope of the training data.

## 11.1 Data Acquisition

Whether Llama 3 or DeepSeek, they not only open-source weights but even disclose architectural details—**yet they remain silent on data**. Beyond trade secrets and legal risks, this is because **data cleaning and formulation** are the true core of modern LLMs.

### 11.1.1 Training Data

Modern LLM training is divided into three interconnected stages:

- **Pre-training**: Data mainly from large-scale raw corpora—web crawls (Common Crawl), books, Wikipedia—typically at the **trillion-token scale (3T–15T)**. The goal is for the model to systematically learn natural language statistics, grammar, and broad world knowledge.
- **Mid-training**: Data from rigorously filtered high-quality text, emphasizing **STEM data** (math, code) and long-context documents, typically at the **10B–100B token scale**. Used to reinforce reasoning, math, code generation, and long-text understanding while maintaining general ability.
- **Post-training**: Data primarily **human-constructed or annotated**, including instruction data (SFT), multi-turn dialogues, and human preference feedback data (RLHF). The goal is to guide the model to follow instructions, engage in safe and helpful interactions, and align behaviorally with human values.

**Historical evolution of training data:**

**BERT**: Training corpus came from BooksCorpus (~800M words) and English Wikipedia (~2.5B words). Both contain long, continuous, naturally-formed document-level text. BERT required document-level corpora rather than randomly shuffled sentence-level samples.

**GPT-2: Mining Gold from Web Pages**: OpenAI proposed a clever heuristic filtering method—WebText dataset built from Reddit community-curated external links, only including pages from posts with at least **3 upvotes**, effectively filtering out junk ads and low-quality content.

**GPT-3: Scale and Diversity**: Introduced **Common Crawl** as a data source, with systematic cleaning including deduplication, HTML tag removal, and non-text content cleaning. The Pile open-source diverse corpus (22 high-quality domains including ArXiv, GitHub, StackExchange) further enhanced data diversity.

**Recent LLM training data sources:**

**OLMo 2 Pre-training**: ~3.9T tokens, >95% from web text. Sources include DCLM-Baseline (foundational web text), StarCoder (high-quality code), academic papers, arXiv, OpenWebMath, and Wikipedia. Mid-training (5-10% of compute) injects domain-specific knowledge using top-7% DCLM data, FLAN instruction data, Stack Exchange, and ~10.7B tokens of math enhancement data. Post-training uses the Tülu 3 framework with SFT (866K PersonaHub synthetic instructions + WildChat), DPO (UltraFeedback preference data from 20 model families judged by GPT-4o), and RLVR (GSM8K, MATH for reinforcement training on verifiable tasks).

**Qwen-3 Training Data**: Pre-training reaches ~**36T tokens**, almost double Qwen2.5, covering **119 languages and dialects**. Uses fine-tuned Qwen2.5-VL to extract text from PDFs. Mid-training adds ~5T high-quality tokens focusing on STEM, reasoning, code, using expert models (Qwen2.5-Math, Qwen2.5-Coder) to synthesize training data. Post-training focuses on long-context capability, extending max context from 4K to 32K tokens.

**The "real data lays the foundation, synthetic data refines" paradigm has become a key component of current LLM training workflows.**

### 11.1.2 Specialized Domain Data

| Data Type | Source | Core Value | Notes |
| ----- | ------------------- | --------------------- | --------------------- |
| Code | GitHub | Improves logical reasoning, multi-step problem handling, code generation | Deduplication, license parsing, low-quality repo removal |
| Books | Gutenberg, public domain | Long-text understanding, story logic, coherent narrative | Avoid copyright risk; prioritize authorized/public domain |
| Math & Science | ArXiv, StackExchange | Domain expertise, scientific reasoning, instruction following | Parse formulas, clean text, align Q&A |

### 11.1.3 Data Security Issues

**Copyright**: Nearly all internet content is protected by default. AI companies claim fair use—models learn statistical patterns rather than copying content. However, lawsuits (e.g., NYT vs. OpenAI) could force expensive licensing.

**Data Poisoning**: Attackers inject "malicious trigger patterns" into public data sources. When crawled and used for training, models may learn incorrect behaviors. Mitigation: data cleaning and filtering, data validation, continuous monitoring.

### 11.1.4 Internet Data Cleaning

Common methods include **heuristic (rule-based)** filtering (e.g., C4 dataset cleaning), **deduplication** (removing duplicate content), and **quality filtering** using model-based scoring. A typical cleaning pipeline:

```python
import re
from bs4 import BeautifulSoup

def clean_web_text_strict(html_text):
    # Strip HTML tags
    soup = BeautifulSoup(html_text, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

## 11.2 Data Processing Pipelines

Modern LLM data processing typically follows these stages:

1. **Collection**: Crawling web pages, collecting books, papers, code repositories
2. **Filtering**: Removing low-quality, duplicate, or harmful content
3. **Deduplication**: Exact and fuzzy matching to remove redundant data
4. **Quality Scoring**: Using trained classifiers or heuristic rules to score text quality
5. **Mixing**: Combining different data sources in proportion (data recipe/formulation)

**Data formulation (配方)** is one of the most closely guarded secrets—determining the optimal ratio of web text, code, books, math, and multilingual data for training.

## 11.3 Summary

In LLM development, **data engineering is the hardest-to-replicate and most competitively valuable part**. The key principles are:

- **Quality over quantity**: Clean, well-structured data beats noisy data at scale
- **Diversity matters**: Covering multiple domains, styles, and languages improves generalization
- **Stage-appropriate data**: Pre-training needs large-scale diverse data; post-training needs high-quality instructional data
- **Data security is non-negotiable**: Copyright compliance, data poisoning prevention, and content safety must be addressed
- **Synthetic data complements real data**: Using expert models to generate training data is now standard practice
