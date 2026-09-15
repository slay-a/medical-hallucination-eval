"""ch1_4.py — Chapters 1 to 4 of the thesis (Introduction, Literature Review, Data and Preprocessing, Methodology).

Every number comes from the Results object (thesis/results_loader.py); nothing is typed in by hand.
Citations use [[cite:key]] (thesis/references.py); cross-references use [[tab:label]] / [[fig:label]].
"""
from results_loader import Results, f3, f2, pct, pct0, fp, fpn, COND_NAME

FIG = "results"


def P(t):
    return ("p", t)


def H1(t):
    return ("h1", t)


def H2(t):
    return ("h2", t)


def H3(t):
    return ("h3", t)


def blocks(R: Results) -> list:
    b = []
    hdr0, hdr1 = R.headers_by_cond.get("E0", 0), R.headers_by_cond.get("E1", 0)
    e0_claims_before = int(R.hdr("E0", "CR", "n_claims_before")); e1_claims_before = int(R.hdr("E1", "CR", "n_claims_before"))
    hc0 = int(R.hdr("E0", "CR", "headers_contradicted")); hc1 = int(R.hdr("E1", "CR", "headers_contradicted"))
    t_cr_before = R.hdr_test("before_filter", "CR"); t_cr_after = R.hdr_test("after_filter", "CR")

    # ══════════════════════════════════ CHAPTER 1 ══════════════════════════════════
    b += [H1("Chapter 1 Introduction"), H2("1.1 Motivation")]
    b += [P("Clinical care generates large volumes of free-text documentation: admission notes, consultation reports, "
            "operative notes, and discharge summaries. Much of this text is written by clinicians for clinicians, and "
            "patients frequently leave a hospital or clinic with instructions that they do not fully understand. Large "
            "language models (LLMs) have recently demonstrated the ability to condense such documentation into fluent "
            "summaries. In a large reader study across four clinical summarization tasks, adapted LLMs produced summaries "
            "that physicians judged to be equivalent to or better than expert-written summaries on completeness, "
            "correctness and conciseness [[cite:vanveen2024]]. LLMs also encode a substantial amount of clinical knowledge "
            "[[cite:singhal2023]], and they can rewrite technical clinical language into plain patient-facing prose, which "
            "makes automatically generated patient summaries an attractive application [[cite:hegselmann2024,adams2021]]."),
          P("The same models, however, are prone to *hallucination*: the generation of fluent text that is not supported by, "
            "or that contradicts, the input they were asked to summarize [[cite:ji2023,huang2025,maynez2020]]. In general-"
            "domain summarization this failure mode has been documented repeatedly, including for the current generation of "
            "LLMs [[cite:kryscinski2019,tam2023]]. In medicine the consequences are more serious than an embarrassing "
            "error. A patient-facing summary that names a medication that was never prescribed, invents a follow-up "
            "appointment, reverses a negation (\"no evidence of infection\" becoming \"evidence of infection\"), or states a "
            "diagnosis that the clinician only suspected can directly influence what a patient does after leaving care. "
            "Asgari et al. therefore argue that hallucination rates must be measured with clinically grounded safety "
            "frameworks before LLM summarization is deployed [[cite:asgari2025]], and dedicated medical hallucination "
            "benchmarks have begun to appear [[cite:pal2023]]. Notably, unsupported statements are not unique to machines: "
            "when Hegselmann et al. asked medical experts to annotate doctor-written discharge instructions from MIMIC-IV, "
            "a substantial share of sentences contained facts that the preceding hospital course did not support "
            "[[cite:hegselmann2024]]."),
          P("Measuring hallucination is harder than it appears. The metrics most often reported for summarization, ROUGE, "
            "BLEU and BERTScore, quantify lexical or embedding overlap with a reference summary [[cite:lin2004,papineni2002,"
            "zhang2020]]. They were never designed to detect factual errors, they correlate weakly with human judgments of "
            "consistency [[cite:fabbri2021,kryscinski2020]], and in clinical note generation they correlate only weakly to "
            "moderately with clinicians' assessments [[cite:moramarco2022]]. Moreover, a reference summary is usually not "
            "available for a patient-facing summary of an individual note. What is needed instead is a *reference-free* "
            "procedure that checks each statement of a generated summary against the source document itself.")]
    b += [H2("1.2 Problem Statement")]
    b += [P("This thesis addresses two linked problems. The first is a *measurement* problem: how can the unsupported and "
            "contradicted content of an LLM-generated medical summary be quantified automatically, at the level of "
            "individual claims, in a way that is reproducible and independent of any reference summary? Natural language "
            "inference (NLI) models, which classify whether a hypothesis is entailed by, contradicted by, or neutral with "
            "respect to a premise, have been proposed as the core of such procedures [[cite:falke2019,maynez2020,laban2022,"
            "honovich2022]], but their validity for patient-facing clinical summaries has not been established. An automatic "
            "judge that is itself unreliable would produce misleading conclusions about every mitigation strategy tested "
            "with it."),
          P("The second is a *mitigation* problem: does grounding the generator in retrieved passages of the source note, "
            "the strategy known as retrieval-augmented generation (RAG) [[cite:lewis2020]], actually reduce unsupported and "
            "contradicted content, and at what cost? Retrieval focuses the model on selected passages, which may suppress "
            "fabrication but may also omit clinically relevant content that was not retrieved. A credible answer requires a "
            "controlled comparison on the same documents, an extractive reference point that bounds what a maximally "
            "faithful system can achieve, paired statistical tests, and an explicit measure of coverage."),
          P("Both problems must be answered together. This thesis therefore builds a claim-level NLI judge, applies it in a "
            "controlled comparison of three summarization approaches, and then validates the judge against sentence-level "
            "annotations produced by medical experts [[cite:hegselmann2025data]]. The validation step turns out to be "
            "decisive for how the comparison must be interpreted.")]
    b += [H2("1.3 Research Questions and Hypotheses")]
    b += [P("The study is organized around three research questions that were fixed in the thesis proposal and carried "
            "through the two progress reports of Spring 2026."),
          ("numbers", [
              "**RQ1.** How frequently do statements in patient-facing summaries generated by a modern LLM (GPT-4o-mini) lack "
              "support in, or contradict, the source clinical note, when measured by a claim-level NLI judge?",
              "**RQ2.** Which kinds of statements are unsupported, and how does the automatic judge itself err when compared "
              "with medical experts?",
              "**RQ3.** To what extent does document-grounded RAG reduce unsupported and contradicted statements relative to "
              "zero-context generation, how does it compare with a conservative extractive summarizer, and what does it cost "
              "in coverage of the source note?"]),
          P("Three hypotheses were stated in advance of the experiments."),
          ("numbers", [
              "**H1 (measurability and validity).** Zero-context LLM summarization exhibits a measurable rate of unsupported "
              "facts, and the claim-level NLI judge agrees with medical-expert annotations of unsupported facts substantially "
              "better than chance.",
              "**H2 (retrieval grounding).** Document-grounded RAG reduces the Unsupported Fact Rate and the Contradiction Rate "
              "relative to zero-context generation.",
              "**H3 (verification).** Adding a structured verification and revision step (Chain-of-Verification) on top of "
              "RAG further reduces unsupported and contradicted statements. This hypothesis is tested with condition E3."])]
    b += [H2("1.4 Overview of the Approach")]
    b += [P(f"Fifty de-identified clinical transcriptions (consultation histories and physicals, and discharge summaries) were "
            f"sampled from the public MTSamples corpus [[cite:mtsamples]] with a fixed random seed. Each note was summarized "
            f"under five conditions, illustrated in [[fig:pipeline]]:"),
          ("bullets", [
              "**E0, zero-context LLM summarization.** GPT-4o-mini receives the full note and writes a 150 to 250 word "
              "patient-facing summary with a fixed four-part structure.",
              "**E1, retrieval-augmented generation.** The note is split into five-sentence chunks, the three chunks most "
              "similar to a query built from the note's description are retrieved with a sentence-embedding model, and "
              "GPT-4o-mini writes the same kind of summary from those excerpts only.",
              "**E1b, retrieval-augmented generation with the full note.** The model receives the same excerpts together with the "
              "full note, which separates the effect of focusing attention from the effect of withholding content.",
              "**E3, retrieval-augmented generation with Chain-of-Verification.** Every claim of the E1 summary is verified by the "
              "model against the sentences retrieved for that claim from the full note, and the summary is rewritten to keep, "
              "correct or remove each claim.",
              "**E2, extractive summarization.** A centroid-based extractive summarizer selects the five most representative "
              "sentences of the note without any language model; because every sentence is copied verbatim, E2 bounds the "
              "faithfulness that any summarizer can reach and exposes the judge's own error rate."]),
          P("Every summary is then scored by the same claim-level NLI judge: the summary is segmented into sentence-level "
            "claims, the three source sentences most similar to each claim are retrieved, a cross-encoder NLI model labels "
            "each claim Supported, Not-Supported or Contradicted, and two rates are computed per summary, the Unsupported Fact "
            "Rate (UFR) and the Contradiction Rate (CR). Conditions are compared with paired non-parametric tests and "
            "bootstrap confidence intervals. The judge is then validated against 210 patient summaries annotated by two "
            "medical experts in the ann-pt-summ dataset [[cite:hegselmann2025data]], its robustness to its own design "
            "choices is examined in a series of ablations, and a coverage proxy quantifies the omission cost of retrieval.")]
    b += [H2("1.5 Contributions")]
    b += [("numbers", [
              "An open, fully reproducible claim-level evaluation pipeline for source-grounded summaries, built from "
              "open-weight components (spaCy, all-MiniLM-L6-v2, cross-encoder/nli-MiniLM2-L6-H768) with a fixed sampling seed "
              "and pinned dependencies.",
              f"A controlled comparison of five summarization conditions, spanning zero-context generation, two forms of retrieval "
              f"grounding, generation with verification and an extractive bound, on {R.n_docs} clinical notes with paired statistics, "
              "including the first correction of two evaluation artifacts, markdown section headers counted as claims and "
              "comma-encoded line breaks in MTSamples, that had inflated previously reported effects.",
              "A validation of the NLI judge against medical-expert annotations of unsupported facts, including sentence-level "
              "agreement, threshold sweeps, summary-level correlation and five evidence-aggregation variants, which shows how "
              "far an off-the-shelf NLI judge can and cannot be trusted for this task.",
              "A quantitative analysis of the coverage cost of excerpt-only RAG, and a keyword-assisted taxonomy of the "
              "statements that remain unsupported under every condition.",
              "Open, documented implementations of every condition, including RAG with the full note (E1b) and RAG with "
              "Chain-of-Verification (E3), so that each can be re-run or extended with a single command."])]
    b += [H2("1.6 Scope and Delimitations")]
    b += [P("The scope of the empirical work is deliberately narrow so that every comparison is controlled. A single "
            "generator, GPT-4o-mini [[cite:openai2024]], is used for all LLM conditions. Generation experiments use the public "
            "MTSamples corpus rather than protected health records, because the OpenAI API cannot receive credentialed "
            "MIMIC text without a zero-data-retention agreement; MIMIC-derived data are used only to validate the judge, "
            "with models that run locally. The thesis proposal also described a document-grounded question-answering task; "
            "that task was not implemented and is left to future work, and the title of this thesis has been narrowed "
            "accordingly. Every condition was generated once per document at a fixed temperature; sampling variance is not "
            "measured."),
          P("Within this scope the thesis makes no claim about the clinical acceptability of any summary. The metrics "
            "measure whether statements are supported by the note as judged by an automatic model, and Chapter 5 shows "
            "precisely how that judgment relates to the judgment of medical experts.")]
    b += [H2("1.7 Organization of the Thesis")]
    b += [P("Chapter 2 reviews the literature on hallucination, faithfulness evaluation, NLI models, LLM-based clinical "
            "summarization, retrieval-augmented generation and verification methods. Chapter 3 describes the three data "
            "sources, the preprocessing steps, the two artifacts that were discovered and corrected, and the ethical and "
            "governance measures under which the data were handled. Chapter 4 specifies the evaluation framework, the "
            "three approaches, the statistical analysis, the judge validation protocol and the robustness analyses. "
            "Chapter 5 reports the results, Chapter 6 discusses them in the light of the research questions and states the "
            "limitations, and Chapter 7 concludes and outlines future work. The appendices contain the exact prompts, the "
            "per-document results, worked examples with every claim label, and a description of the code repository.")]

    # ══════════════════════════════════ CHAPTER 2 ══════════════════════════════════
    b += [H1("Chapter 2 Literature Review"), H2("2.1 Hallucination in Natural Language Generation")]
    b += [P("The term hallucination entered the natural language generation literature to describe output that is fluent "
            "but unfaithful to its input. Maynez et al. distinguished *intrinsic* hallucinations, which misrepresent "
            "information that is present in the source, from *extrinsic* hallucinations, which introduce information that "
            "the source does not contain, and showed that both are frequent in abstractive news summarization even for "
            "models with strong ROUGE scores [[cite:maynez2020]]. Ji et al. organized the field in a survey covering "
            "definitions, causes (noisy training data, exposure bias, decoding strategies and parametric knowledge that "
            "overrides the input), metrics and mitigation methods across summarization, dialogue, question answering and "
            "translation [[cite:ji2023]]. For large language models specifically, Huang et al. separate *factuality* "
            "hallucinations, statements that conflict with world knowledge, from *faithfulness* hallucinations, statements "
            "that conflict with the user-provided context or instructions [[cite:huang2025]]. Source-grounded summarization "
            "of a clinical note is a faithfulness problem: the note is the only ground truth that matters, and a statement "
            "that is medically plausible but absent from the note is still an error."),
          P("Two further observations from this literature shape the present work. First, the rate of unfaithful content in "
            "abstractive summaries has historically been high; Kryściński et al. estimated that about thirty percent of "
            "summaries produced by then state-of-the-art models on news data contained factual inconsistencies "
            "[[cite:kryscinski2019]], and Tam et al. found that even instruction-tuned LLMs frequently produce factually "
            "inconsistent news summaries [[cite:tam2023]]. Second, extrinsic content is not always wrong; a patient "
            "summary that adds standard safety advice may be extrinsic yet appropriate. Whether such content should count as "
            "hallucination is a question of the intended use, and Section 6.2 returns to it.")]
    b += [H2("2.2 Limitations of Overlap-Based Evaluation")]
    b += [P("ROUGE measures n-gram recall or longest-common-subsequence overlap between a candidate and one or more "
            "reference summaries [[cite:lin2004]]; BLEU measures n-gram precision and was designed for machine translation "
            "[[cite:papineni2002]]; BERTScore replaces exact matches with contextual-embedding similarity [[cite:zhang2020]]. "
            "All three assume that a good reference exists and that similarity to it indicates quality. Neither assumption "
            "holds for faithfulness. A summary can reproduce the vocabulary of a note while inverting a negation, changing a "
            "dose or attributing a finding to the wrong body part, and it will still score well. In the SummEval "
            "meta-evaluation, automatic metrics correlated weakly with human ratings of consistency in particular "
            "[[cite:fabbri2021]], and Kryściński et al. showed that a dedicated factual-consistency classifier detected "
            "errors that overlap metrics missed [[cite:kryscinski2020]]."),
          P("In the clinical domain the mismatch is, if anything, larger. Moramarco et al. evaluated automatically generated "
            "consultation notes with practicing clinicians and found that common automatic metrics correlated only weakly "
            "to moderately with clinician judgments and with the post-editing effort required to make notes usable "
            "[[cite:moramarco2022]]. Asgari et al. argue that clinical deployments need hallucination rates expressed in "
            "clinically meaningful units and assessed within an explicit safety framework rather than overlap scores "
            "[[cite:asgari2025]]. These findings motivate reference-free, source-grounded evaluation.")]
    b += [H2("2.3 Automatic Faithfulness Evaluation")]
    b += [P("Three families of reference-free faithfulness metrics have emerged. *Entity-based* methods compare the "
            "entities, relations or numbers mentioned in the summary with those in the source [[cite:goodrich2019,"
            "nan2021]]; they are precise for what they cover but blind to errors that are not entity mentions. "
            "*Question-answering* methods generate questions from the summary, answer them from the source, and score the "
            "agreement; QAGS [[cite:wang2020qags]], FEQA [[cite:durmus2020]] and QuestEval [[cite:scialom2021]] follow this "
            "design and correlate better with human judgments than overlap metrics, at the price of two additional "
            "generation models whose own errors propagate into the score."),
          P("*Entailment-based* methods use an NLI classifier to decide whether the source entails the summary. Early "
            "attempts were sobering: Falke et al. found that NLI models of 2019 could not reliably re-rank summaries by "
            "correctness because they were trained on short, artificial sentence pairs [[cite:falke2019]]. Maynez et al. "
            "nonetheless observed that entailment probabilities correlated with human faithfulness judgments better than "
            "other automatic measures [[cite:maynez2020]]. Laban et al. identified *granularity* as the decisive design "
            "choice: applying NLI to whole documents fails because premises are far longer than anything the model saw in "
            "training, whereas splitting both document and summary into sentences and aggregating sentence-pair scores "
            "(SummaC) yields state-of-the-art inconsistency detection [[cite:laban2022]]. The TRUE meta-evaluation confirmed "
            "that large NLI models fine-tuned on diverse consistency data are among the strongest reference-free metrics "
            "available [[cite:honovich2022]], and AlignScore unified NLI, QA and paraphrase signals into a single alignment "
            "model [[cite:zha2023]]. FActScore pushed granularity further by decomposing long outputs into atomic facts and "
            "checking each against a knowledge source [[cite:min2023]]. SelfCheckGPT takes a different route, detecting "
            "hallucinations without a source by measuring the inconsistency of multiple sampled generations "
            "[[cite:manakul2023]]."),
          P("The judge built in this thesis belongs to the entailment family and follows SummaC's sentence-level design, "
            "but with two differences that turn out to matter. The premise for each claim is not the whole document but "
            "the three most similar source sentences retrieved by a bi-encoder, and the NLI classifier is a small "
            "distilled cross-encoder rather than a large fine-tuned model. Section 2.4 describes these components and "
            "Section 5.4 tests the consequences.")]
    b += [H2("2.4 Natural Language Inference Models")]
    b += [P("NLI is the task of deciding whether a hypothesis sentence is *entailed* by, *contradicts*, or is *neutral* with "
            "respect to a premise sentence. The Stanford NLI corpus [[cite:bowman2015]] and the Multi-Genre NLI corpus "
            "[[cite:williams2018]] provided hundreds of thousands of crowd-sourced sentence pairs and made NLI the standard "
            "benchmark for sentence understanding. Transformer encoders [[cite:vaswani2017,devlin2019]] fine-tuned on these "
            "corpora reach high accuracy on in-distribution test sets."),
          P("Two architectural choices govern how a transformer is used for pairwise classification. A *bi-encoder* embeds "
            "each sentence independently and compares the embeddings, which allows the embeddings of a large collection to "
            "be pre-computed and searched efficiently; Sentence-BERT is the canonical example [[cite:reimers2019]]. A "
            "*cross-encoder* concatenates the two sentences and processes them jointly with full attention, which is far "
            "more accurate for fine-grained decisions such as entailment but must be run once per pair [[cite:reimers2019,"
            "thakur2021]]. The pipeline in this thesis uses both in their natural roles: a bi-encoder (all-MiniLM-L6-v2) "
            "to retrieve candidate evidence sentences by cosine similarity, and a cross-encoder (nli-MiniLM2-L6-H768) to "
            "classify each claim against the retrieved sentences. Both models are MiniLM distillations, which transfer the "
            "self-attention behavior of a large teacher into a six-layer student that runs quickly on a CPU "
            "[[cite:wang2020minilm]]."),
          P("The clinical domain poses a known challenge for such models. Romanov and Shivade created MedNLI from MIMIC "
            "clinical notes and showed that models trained on general-domain NLI transfer poorly: clinical entailment "
            "depends on abbreviations, negation, temporal qualifiers and medical knowledge that crowd-sourced corpora do not "
            "contain [[cite:romanov2018]]. The judge used here was not trained on clinical data. Its agreement with medical "
            "experts is therefore an empirical question, which Chapter 5 answers.")]
    b += [H2("2.5 Large Language Models for Clinical Summarization")]
    b += [P("Large autoregressive language models acquire summarization ability without task-specific training and can be "
            "steered by natural-language instructions [[cite:brown2020]]. Singhal et al. showed that such models encode "
            "clinical knowledge sufficient to reach passing scores on medical licensing questions [[cite:singhal2023]]. "
            "Adams et al. laid the groundwork for hospital-course summarization from MIMIC-III and characterized the "
            "extractive and abstractive properties that such summaries require [[cite:adams2021]]. Van Veen et al. adapted "
            "eight LLMs to four clinical summarization tasks and found, in a blinded reader study with ten physicians, that "
            "the best adapted model was preferred to or judged equivalent to medical experts in most comparisons "
            "[[cite:vanveen2024]]. Tang et al. evaluated LLM summaries of medical evidence and documented factual "
            "inconsistency and over-generalization as the main failure modes, together with a poor correlation between "
            "automatic metrics and human evaluation [[cite:tang2023]]."),
          P("Most relevant to this thesis, Hegselmann et al. generated patient-facing summaries from the hospital course "
            "of MIMIC-IV discharge summaries, had two medical experts annotate unsupported facts in both doctor-written and "
            "LLM-generated summaries, and showed that cleaning the training data of unsupported facts yields models whose "
            "summaries contain fewer unsupported facts than the doctors' own instructions [[cite:hegselmann2024]]. The "
            "resulting annotation dataset, ann-pt-summ, is the reference standard against which this thesis validates its "
            "judge [[cite:hegselmann2025data]]. Complementary benchmarks such as Med-HALT probe medical hallucination with "
            "reasoning and memory tests rather than summaries [[cite:pal2023]].")]
    b += [H2("2.6 Retrieval-Augmented Generation")]
    b += [P("Retrieval-augmented generation combines a parametric model with a non-parametric memory: passages retrieved "
            "from an index are placed in the generator's context so that the output can be conditioned on explicit "
            "evidence [[cite:lewis2020,guu2020]]. Dense retrieval with learned passage encoders [[cite:karpukhin2020]] "
            "replaced sparse retrieval in many systems, and a growing body of engineering knowledge covers chunking, the "
            "number of passages retrieved and re-ranking [[cite:gao2023rag]]. In medicine, Xiong et al. benchmarked RAG on "
            "medical question answering and reported accuracy improvements of up to eighteen percent over the same LLMs "
            "without retrieval [[cite:xiong2024]], and Zakka et al. showed that a retrieval-grounded clinical assistant "
            "produced answers that clinicians judged more factual and safer than an unaugmented chat model "
            "[[cite:zakka2024]]."),
          P("The RAG studied here differs from these systems in one important respect: the retrieval corpus is the source "
            "note itself, not an external knowledge base. The purpose is not to add knowledge but to focus and constrain the "
            "generator on the passages most relevant to the requested summary sections, in the expectation that a model "
            "shown only relevant excerpts will fabricate less. The same design also creates an obvious risk. Whatever the "
            "retriever does not return, the generator cannot summarize, so document-grounded RAG may trade unsupported "
            "content for omissions. The thesis proposal identified this trade-off as a risk to be monitored, and Section 4.8 "
            "defines a coverage proxy for that purpose.")]
    b += [H2("2.7 Verification and Revision Methods")]
    b += [P("A complementary strategy checks the generator's own output. Chain-of-Verification (CoVe) has the model draft a "
            "response, plan verification questions, answer them independently of the draft so that it does not simply "
            "repeat its errors, and produce a revised response; factored variants that answer each question in a separate "
            "prompt reduced hallucination most in list-based, closed-book and long-form tasks [[cite:dhuliawala2024]]. RARR "
            "researches evidence for each claim of a generated text and revises the text to be attributable to that "
            "evidence while preserving its original intent [[cite:gao2023rarr]]. SelfCheckGPT detects unreliable sentences "
            "from the disagreement among sampled generations [[cite:manakul2023]]. These methods are natural complements "
            "to RAG for source-grounded summarization: retrieval constrains what the model sees, verification catches what "
            "it nevertheless invents. Condition E3 in this thesis implements a factored CoVe on top of RAG in which each "
            "draft claim is verified against evidence retrieved for that claim from the full note.")]
    b += [H2("2.8 Datasets for Clinical Summarization Research")]
    b += [P("MIMIC-IV is a de-identified electronic health record database from Beth Israel Deaconess Medical Center "
            "covering hospital admissions from 2008 to 2019 [[cite:johnson2023mimic]]; MIMIC-IV-Note adds the free-text "
            "discharge summaries and radiology reports [[cite:johnson2023note]]. Both are distributed through PhysioNet "
            "[[cite:goldberger2000]] under a credentialing process that requires human-subjects research training and a "
            "signed data use agreement. The ann-pt-summ dataset derived from MIMIC-IV-Note contains one hundred doctor-"
            "written and one hundred LLM-generated patient summaries with expert span annotations of unsupported facts, "
            "plus ten validation summaries [[cite:hegselmann2025data]]. MTSamples is a public collection of roughly five "
            "thousand sample medical transcriptions across forty specialties that was created for transcriptionist training "
            "and has been widely used for teaching and for natural language processing prototypes [[cite:mtsamples]]. Its "
            "documents are realistic in structure and vocabulary but are not real patient records, and because the corpus "
            "is public it may be present in the pre-training data of commercial LLMs. Chapter 3 explains how each dataset "
            "is used in this thesis and why.")]
    b += [H2("2.9 Summary of the Research Gap")]
    b += [P("The literature establishes that LLMs hallucinate, that overlap metrics cannot detect it, that sentence-level "
            "NLI can detect inconsistency when granularity and aggregation are handled carefully, that general-domain NLI "
            "models transfer imperfectly to clinical text, and that retrieval and verification are promising mitigations. "
            "What has been missing is a single study that (i) applies a claim-level NLI judge to patient-facing summaries "
            "of clinical notes in a controlled comparison of zero-context generation, document-grounded RAG and an "
            "extractive bound, (ii) quantifies the coverage cost of retrieval, and (iii) validates the judge itself against "
            "medical-expert annotations before drawing conclusions from it. This thesis fills that gap.")]

    # ══════════════════════════════════ CHAPTER 3 ══════════════════════════════════
    b += [H1("Chapter 3 Data and Preprocessing"), H2("3.1 Overview of the Data Sources")]
    b += [P("Three data sources play distinct roles ([[tab:datasources]]). MTSamples supplies the fifty notes on which the "
            "five summarization conditions are generated and compared. The ann-pt-summ expert annotations supply an "
            "independent reference standard for validating the judge. MIMIC-IV-Note was obtained under PhysioNet "
            "credentialing and is staged for the future migration of the generation experiments to real clinical notes, but "
            "it is not used to generate summaries in this thesis, for the governance reasons given in Section 3.7."),
          ("table", dict(label="datasources", caption="Data sources used in this thesis, their access conditions and their roles.",
                         columns=["Source", "Access", "Content", "Size", "Role in this thesis"],
                         widths=[1.1, 1.0, 2.1, 0.8, 1.5], font=9.5, align=["left", "left", "left", "center", "left"],
                         rows=[["MTSamples [[cite:mtsamples]]", "Public", f"{R.total_rows:,} sample medical transcriptions, "
                                f"{R.n_specialties} specialty categories", "17 MB", f"Generation and comparison of E0, E1, E1b, E2 and E3 on {R.n_docs} sampled notes"],
                               ["ann-pt-summ v1.0.1 [[cite:hegselmann2025data]]", "PhysioNet credentialed, DUA", "210 patient summaries with expert span "
                                "annotations of unsupported facts (100 doctor-written, 100 LLM-generated, 10 validation)", "2.3 MB used",
                                "Validation of the NLI judge with local models only"],
                               ["MIMIC-IV-Note v2.2 [[cite:johnson2023note]]", "PhysioNet credentialed, DUA", "331,794 de-identified discharge summaries and "
                                "2.3 million radiology reports", "1.8 GB", "Staged for future migration of the generation experiments; not used for generation here"]]))]
    b += [H2("3.2 MTSamples")]
    b += [P(f"MTSamples is distributed as a single comma-separated file with one row per transcription and the columns "
            f"description, medical specialty, sample name, transcription and keywords [[cite:mtsamples]]. The copy used "
            f"here contains {R.total_rows:,} rows in {R.n_specialties} specialty categories. Two categories were selected "
            f"because they resemble the documents from which patient-facing summaries are written in practice: Consult - "
            f"History and Physical notes and Discharge Summaries. After removing rows with an empty transcription, "
            f"{R.n_eligible} documents were eligible ({R.n_consult_eligible} consultations and {R.n_discharge_eligible} discharge "
            f"summaries). Fifty documents were drawn with pandas' sampling routine and random seed 42, giving "
            f"{R.n_consult} consultations and {R.n_discharge} discharge summaries. The sample is therefore weighted toward "
            f"consultation notes in the same proportion as the eligible pool. [[tab:mtstats]] summarizes the sample."),
          ("table", dict(label="mtstats", caption="Characteristics of the fifty sampled MTSamples documents.",
                         columns=["Characteristic", "Value"], widths=[3.4, 3.1], font=10, align=["left", "left"],
                         rows=[["Eligible documents (two note types, non-empty)", f"{R.n_eligible} ({R.n_consult_eligible} consultation, {R.n_discharge_eligible} discharge)"],
                               ["Sampled documents", f"{R.n_docs} ({R.n_consult} consultation, {R.n_discharge} discharge), seed 42"],
                               ["Source length, words: mean (SD)", f"{R.src_words['mean']:.0f} ({R.src_words['sd']:.0f})"],
                               ["Source length, words: minimum / median / maximum", f"{R.src_words['min']:.0f} / {R.src_words['median']:.0f} / {R.src_words['max']:.0f}"],
                               ["Characters passed to the E0 generator", "first 4,500 characters of the note"],
                               ["Characters of retrieved excerpts passed to the E1 generator", "at most 4,000"]])),
          P("Two properties of MTSamples matter for the experiments. First, the transcriptions are heavily structured: "
            "they contain upper-case section headings such as CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, PHYSICAL "
            "EXAMINATION and ASSESSMENT AND PLAN, and many findings are stated as terse negations (\"No nausea, vomiting, "
            "or diarrhea\"). Second, the public file has lost its original line breaks. Wherever the source document had a "
            "new line, the CSV contains a comma, so that text reads \"CHIEF COMPLAINT:,  Foul-smelling urine and stomach "
            "pain after meals.,HISTORY OF PRESENT ILLNESS:,  Stomach pain with most meals\". Section 3.5 describes how "
            "this artifact affects sentence segmentation and how it is handled.")]
    b += [H2("3.3 MIMIC-IV-Note")]
    b += [P("MIMIC-IV-Note version 2.2 contains 331,794 de-identified discharge summaries and about 2.3 million radiology "
            "reports for patients of the Beth Israel Deaconess Medical Center [[cite:johnson2023note,johnson2023mimic]]. "
            "Protected health information has been replaced by placeholders, so names, dates and locations appear as "
            "underscores, and the notes are therefore not suitable for direct display in a patient-facing summary without "
            "further processing. Credentialed access was approved on April 28, 2026, after completion of the required CITI "
            "training, and the files were downloaded and verified against the published checksums on May 14, 2026. Because "
            "the notes are real patient records, the data use agreement prohibits sharing them with third parties, which "
            "includes commercial API providers unless a compliant data-handling agreement is in place. The generation "
            "experiments were therefore run on MTSamples, and MIMIC-IV-Note is reserved for a future re-run with a locally "
            "hosted open-weight generator or a zero-data-retention API arrangement (Section 7.3).")]
    b += [H2("3.4 Expert Annotations: ann-pt-summ")]
    lab_rows = []
    lab_desc = {"word_unsupported": "A word or short phrase without support in the hospital course", "condition_unsupported": "A diagnosis, symptom or condition not supported by the source",
                "medication_unsupported": "A medication, dose or medication instruction not supported by the source", "location_unsupported": "An anatomical location or place not supported by the source",
                "time_unsupported": "A time, date, duration or temporal qualifier not supported by the source", "name_unsupported": "A name (person, department, device) not supported by the source",
                "contradicted_fact": "A statement that contradicts the source", "procedure_unsupported": "A procedure, test or intervention not supported by the source",
                "number_unsupported": "A numeric value not supported by the source", "other_unsupported": "Any other unsupported fact"}
    if R.cal_labcounts is not None:
        for _, r in R.cal_labcounts.iterrows():
            lab_rows.append([r.expert_label.replace("_", " "), str(int(r.n_spans)), lab_desc.get(r.expert_label, "")])
        n_spans = int(R.cal_labcounts.n_spans.sum())
    else:
        n_spans = 0
    b += [P("The ann-pt-summ dataset [[cite:hegselmann2025data]] was created for the study of Hegselmann et al. "
            "[[cite:hegselmann2024]]. Its source texts are the Brief Hospital Course sections of MIMIC-IV discharge "
            "summaries, restricted to at most 4,000 characters, and its summaries are the corresponding Discharge "
            "Instructions, either written by the discharging clinician or generated by one of five LLM configurations "
            "(Llama-2 70B before and after training-data cleaning, GPT-4 zero-shot, and GPT-4 with two prompting variants). "
            "Two medical experts independently marked every span of a summary that was not supported by the hospital "
            "course and then reconciled their annotations; each agreed span carries a start and end character offset and "
            f"one of the labels in [[tab:expertlabels]]. The portion of the archive used here comprises 100 doctor-written "
            f"summaries, 100 LLM-generated summaries (20 hospital courses times five systems) and 10 validation summaries, "
            f"with {n_spans} annotated spans in total. Every span offset was verified to match its quoted text exactly "
            "before use."),
          ("table", dict(label="expertlabels", caption="Expert annotation labels in ann-pt-summ and the number of agreed spans of each type in the 210 summaries used. Descriptions paraphrase the label names of the published annotation protocol.",
                         columns=["Label", "Spans", "Description"], widths=[1.7, 0.7, 4.1], font=9.5, align=["left", "center", "left"], rows=lab_rows)),
          P("The archive downloaded from PhysioNet in May 2026 was incomplete: the download stopped after 934 MB of an "
            "archive of more than 3.4 GB, so that the compressed index at the end of the file is missing. The fourteen "
            "annotation and derived files, which are stored at the beginning of the archive, were extracted with a "
            "sequential reader and every one of them matched the SHA-256 digest published in the archive's manifest. The "
            "missing portion consists of the 100,175-pair reconstruction of MIMIC-IV-Note used to train the models of the "
            "original study, which this thesis does not need.")]
    b += [H2("3.5 Preprocessing Pipeline")]
    b += [P("Preprocessing determines what the judge sees, and two of its steps changed the conclusions of the study. The "
            "steps are listed in the order in which they are applied."),
          ("numbers", [
              "**Filtering and sampling.** Rows of MTSamples are filtered to the two note types, rows with an empty "
              "transcription are dropped, and fifty rows are sampled with seed 42. The description field of each row, a "
              "one-sentence account of the visit, is retained because the E1 retrieval query is built from it.",
              "**Truncation.** The E0 prompt receives the first 4,500 characters of the transcription and the E1 prompt at most "
              "4,000 characters of retrieved excerpts, so that every request stays well within the model's context window "
              "and costs are predictable. Source sentences for the judge are drawn from the complete transcription.",
              "**Sentence segmentation.** Both the source note and every summary are segmented with spaCy's en_core_web_sm "
              "pipeline [[cite:honnibal2020]], whose sentence boundaries are derived from the dependency parse. Sentences "
              "shorter than ten characters are discarded because they are almost always residual punctuation or numbering.",
              "**Chunking for retrieval (E1 only).** Source sentences are grouped into consecutive, non-overlapping chunks of "
              "five sentences; the last chunk of a document may be shorter.",
              "**Removal of section headers from summaries.** GPT-4o-mini follows the requested four-part structure by "
              "emitting bold markdown headers such as \"**Key Findings:**\" on their own line. Such lines contain no "
              "verifiable proposition. They are removed from the claim set with a pattern that matches lines consisting only "
              "of bold text, a markdown heading, or a short title-case phrase ending in a colon. Section 3.6 quantifies why "
              "this step is necessary.",
              "**Removal of explicit abstentions.** Condition E3 writes \"Not stated in the note.\" for a section that has no "
              "supported content. Such a line is an abstention, not a claim about the patient. Abstention lines are excluded "
              "from the claim set and counted separately; Section 5.5.4 shows why this matters, since the judge labels almost all "
              "of them as contradictions.",
              "**Repair of comma-encoded line breaks (ablation only).** A cleaning function restores line breaks where the "
              "MTSamples file replaced them with commas (a period followed by a comma, a heading colon followed by a comma, "
              "or a comma preceding an upper-case heading), and heading-only lines are dropped before segmentation. Because "
              "the summaries were generated from the unrepaired text, the repair is applied only on the evidence side and "
              "is reported as an ablation (Section 4.8) rather than silently changed in the primary pipeline.",
              "**Character offsets for validation.** For the expert-annotated summaries, every sentence is stored with its "
              "start and end character offsets so that it can be matched against the experts' span annotations."])]
    b += [H2("3.6 Why Preprocessing Matters: Two Artifacts")]
    b += [P(f"The original April 2026 pipeline treated every spaCy sentence of a summary as a claim, including markdown "
            f"headers, although the accompanying draft stated that headers had been excluded. Re-examination of the stored "
            f"claims showed that {hdr0} of the {e0_claims_before} E0 claims and {hdr1} of the {e1_claims_before} E1 claims "
            f"were header lines, and that the NLI model had labeled {hc0} of the E0 headers and {hc1} of the E1 headers as "
            f"Contradicted. Because the zero-context condition emits about four times as many headers as the RAG condition, "
            f"part of the contradiction-rate difference that had been reported between E0 and E1 was an artifact of "
            f"formatting rather than of medical content. [[tab:headerfilter]] shows the effect of removing header lines. "
            f"The mean E0 contradiction rate falls from {f3(R.hdr('E0','CR','mean_before'))} to {f3(R.hdr('E0','CR','mean_after'))}, "
            f"the E1 rate from {f3(R.hdr('E1','CR','mean_before'))} to {f3(R.hdr('E1','CR','mean_after'))}, and the paired "
            f"E1 versus E0 difference in CR shrinks from {signed_(t_cr_before.delta_mean)} ({fp(t_cr_before.p)}) to "
            f"{signed_(t_cr_after.delta_mean)} ({fp(t_cr_after.p)}). All results in this thesis use the corrected claim set."),
          ("table", dict(label="headerfilter", caption="Effect of removing markdown header lines from the claim set. Means are per-document means over the fifty documents.",
                         columns=["Condition", "Claim rows before", "Header lines", "Claims after", "Headers labeled Contradicted", "Mean CR before", "Mean CR after", "Mean UFR before", "Mean UFR after"],
                         widths=[0.8, 0.75, 0.7, 0.7, 0.9, 0.7, 0.7, 0.7, 0.7], font=9,
                         rows=[[c, str(int(R.hdr(c, "CR", "n_claims_before"))), str(int(R.hdr(c, "CR", "n_headers"))), str(int(R.hdr(c, "CR", "n_claims_after"))),
                                str(int(R.hdr(c, "CR", "headers_contradicted"))), f3(R.hdr(c, "CR", "mean_before")), f3(R.hdr(c, "CR", "mean_after")),
                                f3(R.hdr(c, "UFR", "mean_before")), f3(R.hdr(c, "UFR", "mean_after"))] for c in R.conds])),
          P("The second artifact is the comma-encoded line break described in Section 3.2. When the unrepaired text is "
            "segmented, evidence sentences such as \",CARDIOVASCULAR: ,RESPIRATORY:\" or \"Renal insufficiency.,4.\" are "
            "produced, and the extractive baseline copies such fragments into its summaries. The cleaned-evidence ablation "
            "in Section 5.5 measures how much of the judge's error on verbatim sentences this artifact explains.")]
    b += [H2("3.7 Ethics, Data Governance and Reproducibility")]
    b += [P("The study involves no interaction with human subjects and no identifiable patient data. MTSamples is a public "
            "corpus of sample transcriptions that were never real patient records. MIMIC-IV-Note and ann-pt-summ are "
            "de-identified in accordance with the HIPAA Safe Harbor standard [[cite:hipaa]] and were obtained under "
            "PhysioNet's credentialed-access process: the author completed the CITI Program courses \"Data or Specimens Only "
            "Research\" and \"Conflicts of Interest\" on April 10, 2026, credentialing was approved on April 28, 2026, and the "
            "data use agreements for both datasets were accepted before download. Under those agreements the data may not be "
            "shared, redistributed or used in any attempt at re-identification, and they may not be transmitted to third "
            "parties. Consequently, no MIMIC-derived text was ever sent to the OpenAI API; the judge validation in this thesis "
            "runs entirely with models executed on the author's computer. Per-sentence outputs that contain MIMIC-derived "
            "text are written to a directory excluded from version control, and only aggregate statistics are published. "
            "The credentialed files are stored on a single encrypted personal computer and are not placed in shared cloud "
            "storage."),
          P("The summaries generated for the fifty MTSamples notes were produced with the OpenAI API. Because MTSamples is "
            "public and non-identifiable, this raises no privacy concern, but it does raise a validity concern that Section "
            "6.5 discusses: the corpus may be part of the model's pre-training data. Reproducibility measures include the "
            "fixed sampling seed, pinned package versions in the repository, storage of every generated summary and every "
            "claim label in comma-separated files, and a recomputation script that reproduces every number in this thesis "
            "from those stored files without any API or model call. The complete code is available in the public repository "
            "described in Appendix D.")]

    # ══════════════════════════════════ CHAPTER 4 ══════════════════════════════════
    b += [H1("Chapter 4 Methodology"), H2("4.1 Overview")]
    b += [P("The methodology has two layers ([[fig:pipeline]]). The *generation layer* produces one summary per document "
            "under each of five conditions. The *evaluation layer* applies an identical claim-level NLI judge to every "
            "summary, so that differences between conditions cannot be attributed to differences in measurement. A third "
            "component, the validation study, applies the same judge to expert-annotated summaries to establish how the "
            "judge's labels relate to expert judgment, and a fourth, the robustness analyses, varies the judge's own design "
            "choices."),
          ("figure", dict(label="pipeline", path=f"{FIG}/fig_pipeline.png", width=6.5,
                          caption="Overview of the experimental pipeline. Each source note is summarized under five conditions; every summary is scored by the same claim-level NLI judge, whose evidence sentences are retrieved from the same note."))]
    b += [H2("4.2 The Claim-Level Evaluation Framework"), H3("4.2.1 Claim Segmentation")]
    b += [P("A summary is decomposed into claims at sentence granularity. Sentence-level claims are coarser than the atomic "
            "facts of FActScore [[cite:min2023]] but avoid an additional generation step whose errors would propagate into "
            "the metric, and they match the granularity of the expert annotations used for validation, which are mapped to "
            "sentences in Section 4.7. Segmentation uses spaCy, sentences under ten characters are dropped, and section "
            "header lines and explicit abstention lines are removed as described in Section 3.5.")]
    b += [H3("4.2.2 Evidence Retrieval")]
    b += [P("For each claim c, the source note is segmented into sentences s(1), ..., s(n) and each sentence and the claim "
            "are embedded with the all-MiniLM-L6-v2 bi-encoder, which maps a sentence to a 384-dimensional vector "
            "[[cite:reimers2019,wang2020minilm]]. The evidence set for the claim is the set of the k = 3 source sentences with "
            "the highest cosine similarity to the claim:"),
          ("eq", "sim(c, s) = (e(c) · e(s)) / (‖e(c)‖ ‖e(s)‖),     E(c) = top-3 sentences by sim(c, s)"),
          P("Retrieval limits the premise presented to the NLI model to short, relevant text, following the granularity "
            "argument of Laban et al. [[cite:laban2022]], and it makes the judge's cost linear in the number of claims rather "
            "than in the product of claims and source sentences. Whether three sentences are enough, and whether they should "
            "be scored separately or together, is examined in Sections 5.4 and 5.5.")]
    b += [H3("4.2.3 Natural Language Inference Classification")]
    b += [P("Each (evidence sentence, claim) pair is passed to the cross-encoder/nli-MiniLM2-L6-H768 model, a six-layer "
            "MiniLM cross-encoder with hidden size 768 fine-tuned on SNLI and MultiNLI [[cite:bowman2015,williams2018,"
            "wang2020minilm]], with the evidence sentence as premise and the claim as hypothesis. The model outputs three "
            "logits, for contradiction, entailment and neutral, which are converted to probabilities with the softmax "
            "function. Because a claim may be supported by any one of its evidence sentences, the claim's entailment score "
            "p(e) is the maximum entailment probability over the three pairs and its contradiction score p(c) is the "
            "maximum contradiction probability over the three pairs. The label is then assigned by a symmetric rule with "
            "decision threshold τ = 0.5:"),
          ("eq", "Supported if p(e) ≥ τ and p(e) > p(c);   Contradicted if p(c) ≥ τ and p(c) > p(e);   Not-Supported otherwise"),
          P("Not-Supported therefore covers two situations: the model considers the claim neutral with respect to every "
            "evidence sentence, or its entailment and contradiction probabilities are both below the threshold. The "
            "threshold is varied in Section 5.5.")]
    b += [H3("4.2.4 Metrics")]
    b += [P("Two rates are computed for every summary from its N claims, of which N(c) are labeled Contradicted and N(n) "
            "Not-Supported:"),
          ("eq", "UFR = (N(c) + N(n)) / N          CR = N(c) / N"),
          P("The Unsupported Fact Rate is the share of claims that the judge could not verify against the note, and the "
            "Contradiction Rate is the share that the judge considers to conflict with it. CR is a subset of UFR. Both rates "
            "are computed per summary and then summarized across the fifty documents by their mean, standard deviation and "
            "median, so that every document has equal weight regardless of the number of claims it produced.")]
    b += [H3("4.2.5 Statistical Analysis")]
    b += [P("The three conditions are applied to the same fifty documents, so comparisons are paired. For each pair of "
            "conditions and each metric the per-document differences are tested with the two-sided Wilcoxon signed-rank "
            "test [[cite:wilcoxon1945]], which makes no distributional assumption; zero differences are discarded as in the "
            "original procedure. The mean difference is accompanied by a 95 percent percentile bootstrap confidence "
            "interval from 10,000 resamples of the paired differences [[cite:efron1993]]. Two effect sizes are reported: "
            "the standardized mean difference of the paired differences, d(z) [[cite:cohen1988]], and the matched-pairs "
            "rank-biserial correlation, computed as the difference between the proportions of favorable and unfavorable "
            "signed ranks [[cite:kerby2014]]. The share of documents whose metric improved, was unchanged or worsened is also "
            "reported. Six primary tests are performed (three condition pairs times two metrics); Holm's step-down "
            "procedure is applied to control the family-wise error rate across them [[cite:holm1979]], and both raw and "
            "adjusted p-values are reported. Analyses were implemented with SciPy [[cite:virtanen2020]] and scikit-learn "
            "[[cite:pedregosa2011]].")]
    b += [H2("4.3 Approach 1: Zero-Context LLM Summarization (E0)")]
    b += [P("The baseline condition represents the way LLM summarization is most often deployed: the whole document is "
            "placed in the prompt and the model is asked for a summary. GPT-4o-mini [[cite:openai2024]], a cost-efficient "
            "member of the GPT-4o family released in July 2024, is used through the chat completions API with temperature "
            "0.3 and a limit of 450 output tokens. The system prompt instructs the model to act as a medical scribe "
            "assistant, to produce patient-facing summaries \"accurate and grounded only in the provided clinical note\", "
            "and not to \"add or invent information\". The user prompt requests 150 to 250 words in four parts, reason for "
            "visit or diagnosis, key findings, treatment or procedures performed, and follow-up instructions, and asks the "
            "model to avoid medical jargon. The first 4,500 characters of the transcription follow the instruction. The exact "
            "prompts are reproduced in Appendix A."),
          P("Theoretically, a decoder-only LLM produces each token from a distribution conditioned on the prompt and on its "
            "own parameters [[cite:brown2020]]. Faithfulness failures arise when the parametric distribution, shaped by "
            "millions of documents about similar patients, dominates the conditioning on the specific note: the model writes "
            "what discharge instructions usually say rather than what this note says [[cite:ji2023,huang2025]]. E0 measures "
            "how often that happens when the model is merely asked not to invent.")]
    b += [H2("4.4 Approach 2: Retrieval-Augmented Generation (E1)")]
    b += [P("The RAG condition tests whether narrowing the model's view to retrieved excerpts reduces unsupported content. "
            "The source note is segmented into sentences and grouped into consecutive non-overlapping chunks of five "
            "sentences. A retrieval query is formed from the document's description field followed by the first 300 "
            "characters of the note, which simulates the topical intent of a summarizer before writing. Query and chunks are "
            "embedded with all-MiniLM-L6-v2 and the three chunks with the highest cosine similarity are concatenated, "
            "numbered as excerpts, and truncated to 4,000 characters. GPT-4o-mini then receives a system prompt that says "
            "\"Use ONLY the supplied excerpts to write the summary\" and a user prompt with the same four-part structure and "
            "length requirement as E0, but in which the excerpts replace the note. Generation parameters are identical to "
            "E0. [[tab:conditions]] contrasts the conditions."),
          P("Two design decisions distinguish this RAG from the RAG of question-answering systems [[cite:lewis2020,"
            "gao2023rag]]. The retrieval corpus is the document itself, and the generator sees the excerpts *only*. The "
            "second decision is deliberate: if the model also saw the full note, nothing would prevent it from drawing on "
            "unretrieved passages, and the effect of retrieval could not be isolated. It also implies a predictable cost: "
            "content in unretrieved chunks cannot appear in the summary, which the coverage proxy of Section 4.8 measures. "
            "An earlier draft of this work described E1 as receiving the note together with the excerpts; that description "
            "was incorrect, and the variant it describes is implemented as E1b (Section 4.6)."),
          ("table", dict(label="conditions", caption="Generation settings of the five evaluated conditions.",
                         columns=["Condition", "Generator", "Input to the generator", "Retrieval", "Decoding"],
                         widths=[0.8, 1.0, 2.3, 1.5, 0.9], font=9, align=["left", "left", "left", "left", "left"],
                         rows=[["E0", "GPT-4o-mini", "First 4,500 characters of the note; four-part instruction", "none", "T = 0.3, 450 tokens"],
                               ["E1", "GPT-4o-mini", "Top-3 retrieved chunks only (at most 4,000 characters); instruction to use only the excerpts", "5-sentence chunks; query = description + first 300 characters; all-MiniLM-L6-v2 cosine", "T = 0.3, 450 tokens"],
                               ["E2", "none (extractive)", "All source sentences", "centroid ranking by mean cosine similarity; top-5 kept in document order", "deterministic"],
                               ["E1b", "GPT-4o-mini", "Full note plus the same excerpts", "as E1", "as E1"],
                               ["E3", "GPT-4o-mini", "E1 draft; per-claim verification against top-3 sentences from the full note; revision keeps, corrects or removes each claim", "as the judge (k = 3)", "verification T = 0; revision T = 0.3"]]))]
    b += [H2("4.5 Approach 3: Centroid-Based Extractive Summarization (E2)")]
    b += [P("The third approach contains no language model. Following the centroid principle of Radev et al. "
            "[[cite:radev2004]], every source sentence is embedded with all-MiniLM-L6-v2, the cosine similarity between every "
            "pair of sentences is computed, and each sentence is scored by its mean similarity to all other sentences of "
            "the document. The five highest-scoring sentences are returned in their original order as the summary. Documents "
            "with five or fewer sentences are returned whole."),
          P("E2 serves two purposes. As a summarization approach it is the conservative extreme of the faithfulness and "
            "readability trade-off: it cannot fabricate, but it cannot paraphrase into patient-friendly language either, and "
            "it copies the fragments and abbreviations of the note. As a methodological control it is more important still. "
            "Every E2 claim is a verbatim sentence of the note, so a perfect judge would label every E2 claim Supported. Any "
            "Not-Supported or Contradicted label on an E2 claim is a judge error, and the E2 rates therefore estimate the "
            "floor below which UFR and CR cannot be interpreted.")]
    b += [H2("4.6 Approaches 4 and 5: RAG with the Full Note (E1b) and RAG with Chain-of-Verification (E3)")]
    b += [P("Two further conditions extend E1 and share its interfaces, so that their outputs enter the same result files and the "
            "same judge without any other change. E1b gives "
            "GPT-4o-mini the retrieved excerpts *and* the full note, with an instruction to give the excerpts priority; it "
            "separates the effect of focusing attention from the effect of withholding content. E3 implements a factored "
            "Chain-of-Verification [[cite:dhuliawala2024]] on top of E1: the E1 summary is taken as the draft; each of its "
            "claims is verified in a separate prompt against the three source sentences retrieved for that claim from the "
            "full note, with the verifier returning SUPPORTED, NOT SUPPORTED or CONTRADICTED together with a corrected "
            "statement or the instruction REMOVE; and a final prompt rewrites the draft applying every verdict, writing "
            "\"Not stated in the note\" where a section has no supported content. In the revision prompt a claim verified as "
            "SUPPORTED is marked keep unchanged, a claim with a usable correction is marked replace, and a claim without one is "
            "marked remove. Verification uses temperature 0 and revision temperature 0.3. E1b required 50 API calls and E3 about "
            "500; both are scored by the unchanged judge, so any difference from E1 is attributable to the extra note context "
            "(E1b) or to the verification and revision step (E3).")]
    b += [H2("4.7 Validation of the Judge Against Expert Annotations")]
    b += [P("A judge is only useful if its labels track the judgments that matter. The validation study applies the judge, "
            "unchanged, to the 210 expert-annotated summaries of ann-pt-summ (Section 3.4) and compares its labels with the "
            "experts' spans. Each summary is segmented into sentences with character offsets, header lines and sentences "
            "under ten characters are removed exactly as in the main pipeline, and the hospital course serves as the source "
            "document from which evidence sentences are retrieved. A sentence is *expert-flagged* if it overlaps at least one "
            "annotated span by one or more characters. Two judge detectors are evaluated: the *any* detector flags a "
            "sentence when its label is Not-Supported or Contradicted (the UFR definition), and the *contradiction* detector "
            "flags it only when the label is Contradicted (the CR definition)."),
          P("Agreement is summarized with precision, recall, specificity, F1, accuracy, balanced accuracy and Cohen's kappa "
            "[[cite:cohen1960]] at the sentence level, for all summaries, for the doctor-written and LLM-generated subsets, "
            "and for each of the five LLM configurations. Because the judge's threshold is arbitrary, the entailment "
            "threshold is swept from 0.30 to 0.95 and the area under the receiver operating characteriztic curve "
            "[[cite:hanley1982]] is reported for two scores, one minus the entailment probability and the contradiction "
            "probability, which measures how well the judge's continuous outputs separate expert-flagged from unflagged "
            "sentences independently of any threshold. Recall is also reported for each expert label type. At the summary "
            "level, Spearman's rank correlation between the judge's UFR or CR and the fraction of expert-flagged sentences "
            "tests whether the judge at least ranks summaries as the experts do, and the mean judge rates per system are "
            "compared with the mean expert-flag rates per system."),
          P("Finally, five evidence-aggregation variants are compared on the same sentences, all with the same "
            "cross-encoder: the pipeline's rule (maximum over the three retrieved sentences scored separately); the three "
            "retrieved sentences concatenated into a single premise in document order; the five retrieved sentences "
            "concatenated; the maximum over every sentence of the hospital course scored separately, which is the SummaC "
            "zero-shot design [[cite:laban2022]]; and the entire hospital course as one premise, truncated by the model's "
            "512-token limit. For each variant the area under the curve, the agreement statistics at τ = 0.5, the best "
            "attainable F1 and kappa over thresholds, and the summary-level correlation are reported.")]
    b += [H2("4.8 Robustness Analyses")]
    b += [P("Six analyses examine whether the main comparison depends on the judge's design choices and characterize the "
            "content of unsupported claims ([[tab:robustness]]). All run offline on the stored generations."),
          ("table", dict(label="robustness", caption="Robustness and characterization analyses performed on the stored summaries and claim labels.",
                         columns=["Analysis", "What is varied or measured", "Output"],
                         widths=[1.4, 3.0, 2.1], font=9.5, align=["left", "left", "left"],
                         rows=[["Decision threshold", "τ in {0.5, 0.6, 0.7, 0.8, 0.9} applied to the stored probabilities", "UFR, CR per condition; E1 vs E0 and E2 vs E0 tests at each τ"],
                               ["Evidence set size", "k = 1, 3, 5 source sentences retrieved per claim, re-scored", "UFR, CR per condition; tests; agreement of the k = 3 re-run with stored labels"],
                               ["Cleaned evidence", "MTSamples line breaks repaired on the evidence side; E0 and E1 claims re-scored; E2 regenerated from the repaired sentences", "UFR, CR per condition; label transitions; E2 judge errors"],
                               ["Coverage proxy", "Share of source sentences whose best cosine similarity to any claim is at least 0.6 (also 0.5 and 0.7); mean best similarity", "Coverage per condition; paired tests"],
                               ["Negation analysis", "Presence of negation cues (no, not, denies, without, negative, unremarkable, ...) in claims", "Label distribution by negation status; E2 error profile"],
                               ["Error taxonomy", "Keyword rules assign each unsupported E0 and E1 claim to one of eight categories in a fixed priority order; examples reviewed by the author", "Counts per category, condition and label; example claims"]])),
          P("The coverage proxy deserves a word of justification. Without reference summaries there is no direct measure of "
            "omission. The proxy asks, for every sentence of the note, whether some claim of the summary resembles it "
            "closely, and reports the share of sentences for which the answer is yes. It rewards summaries that touch many "
            "parts of the note and penalizes summaries that concentrate on a few passages, which is exactly the risk that "
            "excerpt-only retrieval introduces. The threshold of 0.6 was chosen before the analysis as a value at which "
            "two sentences generally express overlapping content; results at 0.5 and 0.7 are reported to show sensitivity. "
            "The error taxonomy is intentionally simple. Eight categories (medication or dosage; follow-up or scheduling; "
            "generic advice or patient education; diagnosis or assessment; procedure or treatment; findings, examination, "
            "laboratory or imaging; history, symptoms or timeline; other) are assigned by keyword patterns in a fixed "
            "priority order, so that a claim about a medication and a follow-up counts as medication. The author reviewed "
            "the examples of every category; the counts should be read as a coarse characterization, not as a validated "
            "clinical error typology.")]
    b += [H2("4.9 Tools, Software and Computational Environment")]
    b += [P("All experiments were run on an Apple MacBook Air with Apple silicon, on the CPU only; no GPU acceleration was "
            "used. The evaluation of one summary takes a few seconds, the complete offline recomputation of every result in "
            "this thesis about five minutes, and the judge validation with five aggregation variants about fifteen minutes. "
            "[[tab:tools]] lists the software. Models are downloaded once from the Hugging Face hub and cached locally; "
            "every subsequent run is offline except for the generation calls to the OpenAI API."),
          ("table", dict(label="tools", caption="Software, models and versions used.",
                         columns=["Component", "Package or model", "Version"], widths=[2.2, 2.8, 1.5], font=9.5, align=["left", "left", "left"],
                         rows=[["Language", "Python", "3.9.6 (virtual environment)"],
                               ["LLM generator", "OpenAI GPT-4o-mini via openai", "gpt-4o-mini; openai 2.31.0"],
                               ["Sentence segmentation", "spaCy, en_core_web_sm [[cite:honnibal2020]]", "3.7.4"],
                               ["Bi-encoder for retrieval", "sentence-transformers/all-MiniLM-L6-v2 [[cite:reimers2019]]", "sentence-transformers 5.1.2"],
                               ["NLI cross-encoder", "cross-encoder/nli-MiniLM2-L6-H768", "sentence-transformers 5.1.2"],
                               ["Deep learning runtime", "PyTorch", "2.8.0"],
                               ["Statistics", "SciPy [[cite:virtanen2020]]; scikit-learn [[cite:pedregosa2011]]", "1.13.1; 1.6.1"],
                               ["Data handling", "pandas; NumPy", "2.3.3; 1.26.4"],
                               ["Figures", "Matplotlib", "3.9.4"],
                               ["Documents", "python-docx; LibreOffice (PDF conversion)", "1.2.0"]]))]
    return b


def signed_(x):
    return "—" if x is None else f"{x:+.3f}"
