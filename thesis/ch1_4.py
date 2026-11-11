"""ch1_4.py: Chapters 1 to 4 of the thesis (Introduction, Literature Review, Data and Preprocessing, Methodology).

Every number comes from the Results object (thesis/results_loader.py); nothing is typed in by hand.
Citations use [[cite:key]] (thesis/references.py); cross-references use [[tab:label]] / [[fig:label]].
"""
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent
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


def _split_check():
    """Per-condition comparison of the first main-study claim splitter with the corrected one (results/mimic_splitting_check.csv)."""
    import pandas as pd
    f = ROOT / "results" / "mimic_splitting_check.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d[d.variant == "base"].set_index("condition")


def _scan():
    """Totals of the protected-text scan (results/protected_text_scan_summary.csv) as a dict, or None."""
    import pandas as pd
    f = ROOT / "results" / "protected_text_scan_summary.csv"
    return dict(pd.read_csv(f).values.tolist()) if f.exists() else None


def blocks(R: Results) -> list:
    b = []
    sc = _split_check()
    hdr0, hdr1 = R.headers_by_cond.get("E0", 0), R.headers_by_cond.get("E1", 0)
    e0_claims_before = int(R.hdr("E0", "CR", "n_claims_before")); e1_claims_before = int(R.hdr("E1", "CR", "n_claims_before"))
    hc0 = int(R.hdr("E0", "CR", "headers_contradicted")); hc1 = int(R.hdr("E1", "CR", "headers_contradicted"))
    t_cr_before = R.hdr_test("before_filter", "CR"); t_cr_after = R.hdr_test("after_filter", "CR")

    # ══════════════════════════════════ CHAPTER 1 ══════════════════════════════════
    b += [H1("Chapter 1 Introduction"), H2("1.1 Motivation")]
    b += [P("Clinical documentation is written by clinicians for clinicians, and patients often leave a hospital with instructions "
            "they do not fully understand. Large language models (LLMs) can condense such documentation into fluent patient-facing "
            "summaries: in a reader study across four clinical summarization tasks, physicians judged adapted LLMs equivalent to or "
            "better than expert-written summaries [[cite:vanveen2024]], and the models encode substantial clinical knowledge "
            "[[cite:singhal2023,hegselmann2024,adams2021]]. The same models are prone to *hallucination*, fluent text that the input "
            "does not support or that contradicts it [[cite:ji2023,huang2025,maynez2020,kryscinski2019,tam2023]]. In medicine the "
            "consequences are serious: a summary that names a medication never prescribed, invents a follow-up appointment, reverses a "
            "negation or states a diagnosis the clinician only suspected can change what a patient does after leaving care, which is "
            "why hallucination rates must be measured within clinically grounded safety frameworks before deployment "
            "[[cite:asgari2025,pal2023]]. Unsupported statements are not unique to machines: when Hegselmann et al. had medical experts "
            "annotate doctor-written discharge instructions from MIMIC-IV, a substantial share of sentences contained facts the hospital "
            "course did not support [[cite:hegselmann2024]]."),
          P("Measuring hallucination is harder than it appears. ROUGE, BLEU and BERTScore quantify overlap with a reference summary "
            "[[cite:lin2004,papineni2002,zhang2020]]; they were never designed to detect factual errors, they correlate weakly with human "
            "judgments of consistency [[cite:fabbri2021,kryscinski2020,moramarco2022]], and a reference summary is usually not available "
            "for a patient-facing summary of an individual note. What is needed is a *reference-free* procedure that checks each "
            "statement of a summary against the source document itself.")]
    b += [H2("1.2 Problem Statement")]
    b += [P("This thesis addresses two linked problems. The first is a *measurement* problem: how can the unsupported and contradicted "
            "content of an LLM-generated medical summary be quantified automatically, claim by claim, reproducibly and without a "
            "reference summary? Natural language inference (NLI) models, which classify whether a hypothesis is entailed by, "
            "contradicted by, or neutral to a premise, have been proposed as the core of such procedures "
            "[[cite:falke2019,maynez2020,laban2022,honovich2022]], but their validity for patient-facing clinical summaries has not been "
            "established, and an unreliable judge would mislead every conclusion drawn with it. The second is a *mitigation* problem: "
            "does grounding the generator in retrieved passages of the source note, retrieval-augmented generation (RAG) "
            "[[cite:lewis2020]], reduce unsupported content, and at what cost in omitted content? The thesis therefore builds a "
            "claim-level NLI judge, validates it against sentence-level annotations by medical experts [[cite:hegselmann2025data]], "
            "replaces it with the candidate that agrees best with them, and uses that judge in a controlled comparison of five "
            "summarization conditions on real hospital courses. The validation step turns out to be decisive for how every comparison "
            "must be read.")]
    b += [H2("1.3 Research Questions and Hypotheses")]
    b += [P("Three research questions were fixed in the thesis proposal."),
          ("numbers", [
              "**RQ1.** How frequently do statements in patient-facing summaries generated by a modern LLM lack support in, or "
              "contradict, the source clinical note, when measured by a claim-level NLI judge?",
              "**RQ2.** Which kinds of statements are unsupported, and how does the automatic judge itself err when compared with "
              "medical experts?",
              "**RQ3.** To what extent does document-grounded RAG reduce unsupported and contradicted statements relative to full-note "
              "generation without retrieval, how does it compare with a conservative extractive summarizer, and what does it cost in "
              "coverage of the source note?"]),
          P("The pilot study asks these questions of GPT-4o-mini on public notes; the main study asks them of a locally run open-weight "
            "model on MIMIC-IV hospital courses with a judge selected against the expert annotations. Three hypotheses were stated in "
            "advance."),
          ("numbers", [
              "**H1 (measurability and validity).** Full-note LLM summarization without retrieval exhibits a measurable rate of "
              "unsupported facts, and the claim-level NLI judge agrees with medical-expert annotations substantially better than chance.",
              "**H2 (retrieval grounding).** Document-grounded RAG reduces the Unsupported Fact Rate and the Contradiction Rate "
              "relative to full-note generation without retrieval.",
              "**H3 (verification).** A structured verification and revision step (Chain-of-Verification) on top of RAG further reduces "
              "unsupported and contradicted statements (condition E3)."])]
    b += [H2("1.4 Overview of the Approach")]
    b += [P(f"The pilot study summarizes {R.n_docs} public MTSamples transcriptions [[cite:mtsamples]] under five conditions "
            f"([[fig:pipeline]]): full-note LLM summarization without retrieval (E0), RAG from the three most relevant excerpts of the "
            f"note (E1), the same excerpts together with the full note (E1b), RAG followed by Chain-of-Verification of every draft claim "
            f"(E3), and an extractive summarizer that copies the five most representative sentences without any language model (E2); "
            f"because every E2 sentence is verbatim, any unsupported label it receives measures the judge's error. Every summary is "
            f"scored by the same claim-level NLI judge, which segments it into sentence-level claims, retrieves the most similar source "
            f"sentences for each, labels each claim Supported, Not-Supported or Contradicted, and reports two judge-estimated rates per "
            f"summary, the Unsupported Fact Rate (UFR) and the Contradiction Rate (CR). Conditions are compared with paired "
            f"non-parametric tests and bootstrap intervals."),
          P("The pilot judge is validated against 210 patient summaries annotated by two medical experts [[cite:hegselmann2025data]] "
            "and found wanting; a selection study scores six candidates and a seventh adapted with MedNLI [[cite:romanov2018]] against "
            "the same annotations and keeps the best. The main study repeats the five conditions on 110 MIMIC-IV hospital courses with "
            "a model running on the author's computer, requires citations from the retrieval conditions, measures omission against the "
            "discharge instructions the clinicians wrote, and adds the checks a moderate judge demands: a second judge, atomic-fact "
            "scoring, repeated generations, two further generators, length- and instruction-matched variants, an analysis of the "
            "judge-selection data's independence, an external check on clinician-labeled data, and a blind manual audit of the "
            "main-study claims.")]
    b += [H2("1.5 Contributions")]
    b += [("numbers", [
              "An open, reproducible claim-level evaluation pipeline for source-grounded summaries, built from open-weight components "
              "with fixed seeds and pinned dependencies, and the correction of two evaluation artifacts (markdown headers counted as "
              "claims, comma-encoded line breaks) that had inflated previously reported effects.",
              "A validation of an off-the-shelf NLI judge against medical-expert annotations, showing how far it can and cannot be "
              "trusted, and a judge-selection study with nested cross-validation and a patient-disjoint transfer test that estimate how "
              "optimistic the selected judge's agreement is.",
              "A controlled comparison of five summarization conditions on real MIMIC-IV hospital courses with a locally run model, "
              "measuring judge-estimated unsupported content, coverage of the clinician-written instructions, citation coverage, "
              "accuracy and reliability, retrieval ablations and a question-answering pilot.",
              "A blind manual audit of the judge on the main study's own outputs, an external check on clinician-labeled propositions "
              "from another institution, and replication of the main comparison with two further generators.",
              "Open, documented implementations of every condition and study, re-runnable with a single command without sending "
              "protected text to any external service."])]
    b += [H2("1.6 Scope and Delimitations")]
    b += [P("Within each study a single generator is used for all LLM conditions: GPT-4o-mini [[cite:openai2024]] in the pilot on public "
            "MTSamples notes, and Qwen2.5-7B-Instruct, run locally, in the main study on MIMIC-IV courses, because the OpenAI API cannot "
            "receive credentialed MIMIC text. The two studies therefore differ in generator and corpus and are compared qualitatively. "
            "The proposal's question-answering task is implemented as a pilot with three fixed questions per course. Each condition was "
            "generated once per document, with two further generations on 40 courses and two further generators on all courses as "
            "robustness checks (Section 7.10). The thesis makes no claim about the clinical acceptability, safety or usefulness of any "
            "summary: the metrics measure support by the note as judged by an automatic model, and a lower judge-estimated unsupported "
            "fact rate does not by itself make a summary better, because a summary can lower it by saying less.")]
    b += [H2("1.7 Organization of the Thesis")]
    b += [P("Chapter 2 reviews the literature, Chapter 3 the data and preprocessing, and Chapter 4 the methodology. Chapter 5 reports the "
            "pilot study, Chapter 6 the selection of a valid judge and its checks, Chapter 7 the main study, Chapter 8 the discussion and "
            "limitations, and Chapter 9 the conclusions. The appendices contain the prompts, the code repository and the audit protocol.")]

    # ══════════════════════════════════ CHAPTER 2 ══════════════════════════════════
    b += [H1("Chapter 2 Literature Review"), H2("2.1 Hallucination in Natural Language Generation")]
    b += [P("Hallucination describes output that is fluent but unfaithful to its input. Maynez et al. distinguished *intrinsic* "
            "hallucinations, which misrepresent information in the source, from *extrinsic* ones, which add information the source does "
            "not contain, and found both frequent in abstractive summarization even for models with strong ROUGE scores "
            "[[cite:maynez2020]]; Ji et al. surveyed definitions, causes (noisy training data, exposure bias, decoding, and parametric "
            "knowledge that overrides the input), metrics and mitigations [[cite:ji2023]]; and Huang et al. separate *factuality* "
            "hallucinations, which conflict with world knowledge, from *faithfulness* hallucinations, which conflict with the given "
            "context [[cite:huang2025]]. Source-grounded summarization of a clinical note is a faithfulness problem: a statement that is "
            "medically plausible but absent from the note is still an error. Unfaithful content has historically been common, about "
            "thirty percent of summaries from the models of 2019 [[cite:kryscinski2019]] and still frequent for instruction-tuned LLMs "
            "[[cite:tam2023]], and extrinsic content is not always wrong: a patient summary that adds standard safety advice is "
            "extrinsic yet appropriate, a question of intended use to which Section 8.2 returns.")]
    b += [H2("2.2 Limitations of Overlap-Based Evaluation")]
    b += [P("ROUGE, BLEU and BERTScore measure n-gram or embedding overlap with a reference summary "
            "[[cite:lin2004,papineni2002,zhang2020]] and assume that a good reference exists and that similarity to it indicates quality. "
            "Neither holds for faithfulness: a summary can reproduce the vocabulary of a note while inverting a negation or changing a "
            "dose. Such metrics correlated weakly with human ratings of consistency in SummEval [[cite:fabbri2021]], a dedicated "
            "consistency classifier detected errors they missed [[cite:kryscinski2020]], and in clinical note generation they correlated "
            "only weakly to moderately with clinician judgments [[cite:moramarco2022]]; clinical deployments need hallucination rates in "
            "clinically meaningful units within a safety framework [[cite:asgari2025]], which motivates reference-free, source-grounded "
            "evaluation.")]
    b += [H2("2.3 Automatic Faithfulness Evaluation")]
    b += [P("Three families of reference-free faithfulness metrics exist. *Entity-based* methods compare the entities and numbers in the "
            "summary with those in the source [[cite:goodrich2019,nan2021]]; *question-answering* methods such as QAGS, FEQA and "
            "QuestEval generate questions from the summary and answer them from the source [[cite:wang2020qags,durmus2020,scialom2021]]; "
            "*entailment-based* methods use an NLI classifier to decide whether the source entails the summary. Early NLI models could not "
            "re-rank summaries by correctness [[cite:falke2019]], but entailment probabilities correlated with human faithfulness "
            "judgments better than other measures [[cite:maynez2020]], and Laban et al. identified granularity as decisive: NLI on whole "
            "documents fails, whereas splitting document and summary into sentences and aggregating sentence-pair scores (SummaC) "
            "detects inconsistency well [[cite:laban2022]]. The TRUE meta-evaluation confirmed that large NLI models fine-tuned on diverse "
            "consistency data are among the strongest reference-free metrics [[cite:honovich2022]], AlignScore unified NLI, QA and "
            "paraphrase signals [[cite:zha2023]], FActScore decomposes outputs into atomic facts [[cite:min2023]], and SelfCheckGPT "
            "detects hallucination without a source from the inconsistency of sampled generations [[cite:manakul2023]]."),
          P("The judge built here is entailment-based and follows SummaC's sentence-level design, with two differences that matter: the "
            "premise is the few most similar source sentences rather than the whole document, and the pilot's classifier is a small "
            "distilled cross-encoder (Section 2.4); Section 5.3 tests the consequences and Chapter 6 replaces it with larger candidates. "
            "Two recent lines of work bear on that choice. Grounding-specific checkers such as MiniCheck match GPT-4 on the "
            "LLM-AggreFact benchmark at a fraction of the cost [[cite:tang2024minicheck]]; and mounting evidence shows that even the best "
            "checkers are unreliable instruments: Godbole and Jia found five recent factuality metrics disagreeing with each other and "
            "penalizing paraphrase and distant evidence, and recommend validating any metric in the target domain [[cite:godbole2025]], "
            "while on FaithBench most state-of-the-art detectors, GPT-4o included, are close to 50 percent accurate "
            "[[cite:bao2025faithbench]]. These findings make the validation step of this thesis a requirement rather than an extra.")]
    b += [H2("2.4 Natural Language Inference Models")]
    b += [P("NLI asks whether a premise entails, contradicts or is neutral to a hypothesis; SNLI [[cite:bowman2015]] and MultiNLI "
            "[[cite:williams2018]] made it the standard sentence-understanding benchmark, and transformer encoders "
            "[[cite:vaswani2017,devlin2019]] fine-tuned on them reach high in-distribution accuracy. Two architectures govern pairwise "
            "classification: a *bi-encoder* embeds each sentence independently so that a collection can be pre-computed and searched "
            "[[cite:reimers2019]], and a *cross-encoder* processes both sentences jointly, which is far more accurate for entailment but "
            "must run once per pair [[cite:reimers2019,thakur2021]]. The pipeline uses both in their natural roles, a bi-encoder "
            "(all-MiniLM-L6-v2) to retrieve evidence and a cross-encoder (nli-MiniLM2-L6-H768) to classify, both MiniLM distillations "
            "that run quickly on a CPU [[cite:wang2020minilm]]. Clinical text is a known challenge: Romanov and Shivade built MedNLI from "
            "MIMIC notes and showed that general-domain NLI models transfer poorly, because clinical entailment depends on "
            "abbreviations, negation, temporal qualifiers and medical knowledge [[cite:romanov2018]]. The judge used here was not "
            "trained on clinical data, so its agreement with medical experts is an empirical question, and Chapter 6 also tests whether "
            "fine-tuning on MedNLI closes the gap.")]
    b += [H2("2.5 Large Language Models for Clinical Summarization")]
    b += [P("Large language models summarize without task-specific training [[cite:brown2020]] and encode clinical knowledge sufficient "
            "to pass medical licensing questions [[cite:singhal2023]]. Adams et al. laid the groundwork for hospital-course summarization "
            "from MIMIC-III [[cite:adams2021]]; Van Veen et al. found in a blinded reader study that the best adapted model was preferred "
            "to or judged equivalent to medical experts in most comparisons [[cite:vanveen2024]]; and Tang et al. documented factual "
            "inconsistency and over-generalization as the main failure modes of LLM summaries of medical evidence [[cite:tang2023]]. Most "
            "relevant here, Hegselmann et al. generated patient-facing summaries from the hospital course of MIMIC-IV discharge summaries, "
            "had two medical experts annotate unsupported facts in doctor-written and LLM-generated summaries, and showed that cleaning "
            "the training data yields models whose summaries contain fewer unsupported facts than the doctors' own instructions "
            "[[cite:hegselmann2024]]; GPT-4 prompted to mark unsupported facts was their best automatic detector, yet reached a span-level "
            "F1 of only 0.16 to 0.20. Their annotation dataset, ann-pt-summ, is the reference standard of this thesis "
            "[[cite:hegselmann2025data]]. Closest to the judge problem, Chung et al. built VeriFact, which splits a generated brief "
            "hospital course into propositions, retrieves the most relevant facts from the patient's whole record, and asks a "
            "70-billion-parameter model whether each is supported; on 13,290 clinician-annotated propositions it agreed with an "
            "adjudicated clinician reference more often than individual clinicians agreed with each other [[cite:chung2026verifact]], on "
            "a server with four data-center GPUs, whereas the judges compared here are limited to models that run on a laptop.")]
    b += [H2("2.6 Retrieval-Augmented Generation")]
    b += [P("Retrieval-augmented generation places passages retrieved from an index in the generator's context so that the output is "
            "conditioned on explicit evidence [[cite:lewis2020,guu2020]]; dense retrieval with learned encoders [[cite:karpukhin2020]] "
            "replaced sparse retrieval in many systems, and engineering knowledge covers chunking, the number of passages and re-ranking "
            "[[cite:gao2023rag]]. In medicine, RAG improved accuracy on medical question answering by up to eighteen percent "
            "[[cite:xiong2024]], and a retrieval-grounded clinical assistant produced answers that clinicians judged more factual and "
            "safer [[cite:zakka2024]]. The RAG studied here differs in one respect: the retrieval corpus is the source note itself, and "
            "the purpose is not to add knowledge but to focus the generator on the passages relevant to the requested sections, in the "
            "expectation that a model shown only relevant excerpts fabricates less. The obvious risk is omission, since whatever the "
            "retriever does not return cannot be summarized; the proposal identified this trade-off, Section 4.8 defines a coverage proxy "
            "for it, and the main study measures coverage of the instructions the treating clinician actually wrote.")]
    b += [H2("2.7 Verification and Revision Methods")]
    b += [P("A complementary strategy checks the generator's own output. Chain-of-Verification has the model draft a response, plan "
            "verification questions, answer them independently of the draft and revise; factored variants that answer each question in "
            "a separate prompt reduced hallucination most [[cite:dhuliawala2024]]. RARR researches evidence for each claim and revises "
            "the text to be attributable to it [[cite:gao2023rarr]], and SelfCheckGPT detects unreliable sentences from the disagreement "
            "among sampled generations [[cite:manakul2023]]. Retrieval constrains what the model sees and verification catches what it "
            "nevertheless invents; condition E3 implements a factored CoVe on top of RAG, verifying each draft claim against evidence "
            "retrieved for it from the full note.")]
    b += [H2("2.8 Datasets for Clinical Summarization Research")]
    b += [P("MIMIC-IV is a de-identified electronic health record database from Beth Israel Deaconess Medical Center "
            "[[cite:johnson2023mimic]], and MIMIC-IV-Note adds its free-text discharge summaries [[cite:johnson2023note]]; both are "
            "distributed through PhysioNet [[cite:goldberger2000]] under a credentialing process with a signed data use agreement. The "
            "ann-pt-summ dataset derived from MIMIC-IV-Note contains one hundred doctor-written and one hundred LLM-generated patient "
            "summaries with expert span annotations of unsupported facts [[cite:hegselmann2025data]]. MTSamples is a public collection of "
            "roughly five thousand sample medical transcriptions created for transcriptionist training [[cite:mtsamples]]; its documents "
            "are realistic in structure and vocabulary but are not real patient records, and because the corpus is public it may be in "
            "the pre-training data of commercial LLMs.")]
    b += [H2("2.9 Summary of the Research Gap")]
    b += [P("The literature establishes that LLMs hallucinate, that overlap metrics cannot detect it, that sentence-level NLI can detect "
            "inconsistency when granularity and aggregation are handled carefully, that general-domain NLI models transfer imperfectly to "
            "clinical text, and that retrieval and verification are promising mitigations. Missing is a single study that applies a "
            "claim-level NLI judge to patient-facing summaries in a controlled comparison of full-note generation, document-grounded RAG "
            "and an extractive reference, quantifies the coverage cost of retrieval, validates the judge against medical-expert "
            "annotations before drawing conclusions, and selects a better judge by that validation and repeats the comparison on real "
            "hospital courses against what the treating clinician wrote. This thesis fills that gap.")]

    # ══════════════════════════════════ CHAPTER 3 ══════════════════════════════════
    b += [H1("Chapter 3 Data and Preprocessing"), H2("3.1 Overview of the Data Sources")]
    b += [P("Four data sources play distinct roles ([[tab:datasources]]). MTSamples supplies the fifty notes of the pilot study. The "
            "ann-pt-summ expert annotations supply the reference standard for validating and selecting the judge, and their 110 hospital "
            "courses, each paired with the discharge instructions written by the treating clinician, are the documents of the main "
            "study. MIMIC-IV-Note, from which those courses are drawn, and MedNLI, used to adapt a candidate judge, were obtained under "
            "PhysioNet credentialing (Section 3.7)."),
          ("table", dict(label="datasources", caption="Data sources used in this thesis, their access conditions and their roles.",
                         columns=["Source", "Access", "Content", "Size", "Role in this thesis"],
                         widths=[1.1, 1.0, 2.1, 0.8, 1.5], font=9.5, align=["left", "left", "left", "center", "left"],
                         rows=[["MTSamples [[cite:mtsamples]]", "Public", f"{R.total_rows:,} sample medical transcriptions, "
                                f"{R.n_specialties} specialty categories", "17 MB", f"Generation and comparison of E0, E1, E1b, E2 and E3 on {R.n_docs} sampled notes"],
                               ["ann-pt-summ v1.0.1 [[cite:hegselmann2025data]]", "PhysioNet credentialed, DUA", "210 patient summaries with expert span "
                                "annotations of unsupported facts (100 doctor-written, 100 LLM-generated, 10 validation)", "2.3 MB used",
                                "Validation and selection of the judge; its 110 hospital courses and clinician-written instructions are the main-study documents (local models only)"],
                               ["MIMIC-IV-Note v2.2 [[cite:johnson2023note]]", "PhysioNet credentialed, DUA", "331,794 de-identified discharge summaries and "
                                "2.3 million radiology reports", "1.8 GB", "Source of the ann-pt-summ hospital courses; the 110 annotated courses are the documents of the main study"],
                               ["MedNLI v1.0.0 [[cite:romanov2018]]", "PhysioNet credentialed, DUA", "14,049 clinician-written premise and hypothesis pairs from MIMIC-III notes (11,232 train, 1,395 dev, 1,422 test), labels entailment, contradiction, neutral", "1.5 MB",
                                "Adaptation of the candidate judge to clinical language (Chapter 6)"]]))]
    b += [H2("3.2 MTSamples")]
    b += [P(f"MTSamples is a single comma-separated file with one row per transcription [[cite:mtsamples]]; the copy used here has "
            f"{R.total_rows:,} rows in {R.n_specialties} specialty categories. Two categories resemble the documents from which "
            f"patient-facing summaries are written in practice, Discharge Summaries and Consult (History and Physical) notes; after "
            f"removing empty transcriptions, {R.n_eligible} documents were eligible and fifty were drawn with seed 42 ({R.n_consult} "
            f"consultations, {R.n_discharge} discharge summaries; [[tab:mtstats]]). Two properties matter for the experiments: the "
            f"transcriptions are heavily structured, with upper-case section headings and many terse negations (\"No nausea, vomiting, or "
            f"diarrhea\"), and the public file has lost its line breaks, so that wherever the source had a new line the file has a comma "
            f"(\"CHIEF COMPLAINT:,  Foul-smelling urine...,HISTORY OF PRESENT ILLNESS:,\"), which Section 3.5 handles."),
          ("table", dict(label="mtstats", caption="Characteristics of the fifty sampled MTSamples documents.",
                         columns=["Characteristic", "Value"], widths=[3.4, 3.1], font=10, align=["left", "left"],
                         rows=[["Eligible documents (two note types, non-empty)", f"{R.n_eligible} ({R.n_consult_eligible} consultation, {R.n_discharge_eligible} discharge)"],
                               ["Sampled documents", f"{R.n_docs} ({R.n_consult} consultation, {R.n_discharge} discharge), seed 42"],
                               ["Source length, words: mean (SD)", f"{R.src_words['mean']:.0f} ({R.src_words['sd']:.0f})"],
                               ["Source length, words: minimum / median / maximum", f"{R.src_words['min']:.0f} / {R.src_words['median']:.0f} / {R.src_words['max']:.0f}"],
                               ["Characters passed to the E0 generator", "first 4,500 characters of the note"],
                               ["Characters of retrieved excerpts passed to the E1 generator", "at most 4,000"]]))]
    b += [H2("3.3 MIMIC-IV-Note")]
    b += [P("MIMIC-IV-Note version 2.2 contains 331,794 de-identified discharge summaries and about 2.3 million radiology reports from "
            "the Beth Israel Deaconess Medical Center [[cite:johnson2023note,johnson2023mimic]], with protected health information "
            "replaced by placeholders. Credentialed access was approved on April 28, 2026, and the files were verified against the "
            "published checksums. Because the notes are real patient records, the data use agreement prohibits sharing them with third "
            "parties, including commercial API providers, so the pilot ran on MTSamples with GPT-4o-mini and every experiment on "
            "MIMIC-derived text uses models that run on the author's computer (Sections 3.7 and 4.10).")]
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
    b += [P("The ann-pt-summ dataset [[cite:hegselmann2025data]] was created for the study of Hegselmann et al. [[cite:hegselmann2024]]. "
            "Its source texts are the Brief Hospital Course sections of MIMIC-IV discharge summaries (at most 4,000 characters) and its "
            "summaries are the corresponding Discharge Instructions, written by the discharging clinician or generated by one of five LLM "
            "configurations (Llama-2 70B before and after training-data cleaning, GPT-4 zero-shot, and GPT-4 with two prompting variants). "
            "Two medical experts independently marked every span not supported by the hospital course and reconciled their annotations; "
            f"each agreed span carries character offsets and one of the labels in [[tab:expertlabels]]. The portion used here comprises "
            f"100 doctor-written summaries, 100 LLM-generated summaries (20 courses times five systems) and 10 validation summaries, with "
            f"{n_spans} annotated spans, every offset verified against its quoted text. In the main study the 110 courses are the source "
            "documents and the clinician-written instructions the reference against which omission is measured: a summary that omits "
            "what the clinician chose to tell the patient has omitted something a clinician judged important. (The PhysioNet download "
            "was incomplete; the fourteen annotation files, stored at the start of the archive, were extracted sequentially and each "
            "matched its published SHA-256 digest, and the missing portion is training data this thesis does not use.)"),
          ("table", dict(label="expertlabels", caption="Expert annotation labels in ann-pt-summ and the number of agreed spans of each type in the 210 summaries used. Descriptions paraphrase the label names of the published annotation protocol.",
                         columns=["Label", "Spans", "Description"], widths=[1.7, 0.7, 4.1], font=9.5, align=["left", "center", "left"], rows=lab_rows))]
    b += [H2("3.5 Preprocessing Pipeline")]
    b += [P("Preprocessing determines what the judge sees, and two of its steps changed the conclusions of the study. In order:"),
          ("numbers", [
              "**Filtering and sampling.** MTSamples rows of the two note types with a non-empty transcription; fifty sampled with seed "
              "42, keeping the description field, from which the E1 retrieval query is built.",
              "**Truncation.** The E0 prompt receives the first 4,500 characters of the note and the E1 prompt at most 4,000 characters of "
              "excerpts; the judge's source sentences come from the complete note.",
              "**Sentence segmentation.** spaCy's en_core_web_sm pipeline [[cite:honnibal2020]]; sentences shorter than ten characters are "
              "discarded as residual punctuation or numbering.",
              "**Chunking for retrieval (E1 only).** Consecutive, non-overlapping chunks of five sentences.",
              "**Removal of section headers from summaries.** Bold markdown headers such as \"**Key Findings:**\" contain no verifiable "
              "proposition and are removed by a pattern for bold text, markdown headings and short title-case phrases ending in a colon "
              "(Section 3.6 quantifies why).",
              "**Removal of explicit abstentions.** \"Not stated in the note.\" is an abstention, not a claim about the patient; such lines "
              "are excluded from the claim set and counted separately (Section 5.4 shows that the judge labels almost all of them as "
              "contradictions).",
              "**Repair of comma-encoded line breaks (ablation only).** A cleaning function restores the line breaks MTSamples replaced with "
              "commas; because the summaries were generated from the unrepaired text, the repair is applied only on the evidence side and "
              "reported as an ablation (Section 4.8).",
              "**Character offsets for validation.** Every sentence of an expert-annotated summary is stored with its offsets so that it "
              "can be matched against the experts' spans."]
              + ([(f"**Citation markers (main study).** The retrieval conditions write excerpt numbers such as \"[2]\" after their "
                   f"sentences, before or after the final period. Splitting the raw text, as the first version of the evaluation did, "
                   f"discarded a marker before the period as a short fragment and credited a marker after it to the next sentence: "
                   f"{pct0(sc.loc['E1', 'markers_dropped_share'])} of E1's {int(sc.loc['E1', 'marker_groups'])} citation groups were "
                   f"discarded and {pct0(sc.loc['E1', 'markers_moved_share'])} moved, and {pct0(sc.loc['E1b', 'markers_dropped_share'])} "
                   f"and {pct0(sc.loc['E1b', 'markers_moved_share'])} of E1b's {int(sc.loc['E1b', 'marker_groups'])}. The corrected "
                   f"procedure removes the markers before segmentation and assigns each to the sentence that contains it.")] if sc is not None and {"E1", "E1b"} <= set(sc.index) else [])
              + ([(f"**Headers merged into sentences (main study).** The local model's bold or # headers are sometimes joined by the "
                   f"splitter to the sentence that follows, escaping the filter above: {pct(sc.loc['E0', 'merged_header_share'])} of the "
                   f"E0 claims and {pct(sc.loc['E3', 'merged_header_share'])} of the E3 claims carried a merged header in the first version "
                   f"of the evaluation. Such headers are now stripped from every claim; all main-study results use the corrected claims, "
                   f"which change the rates slightly and the citation results substantially (Section 7.6).")] if sc is not None and {"E0", "E3"} <= set(sc.index) else []))]
    b += [H2("3.6 Why Preprocessing Matters: Two Artifacts")]
    b += [P(f"The original pipeline treated every spaCy sentence of a summary as a claim, including markdown headers: {hdr0} of the "
            f"{e0_claims_before} E0 claims and {hdr1} of the {e1_claims_before} E1 claims were header lines, and the NLI model labeled "
            f"{hc0} of the E0 headers and {hc1} of the E1 headers as Contradicted. Because the full-note condition emits about four times "
            f"as many headers as the RAG condition, part of the contradiction-rate difference first reported between E0 and E1 was an "
            f"artifact of formatting ([[tab:headerfilter]]): removing header lines shrinks the paired E1 versus E0 difference in CR from "
            f"{signed_(t_cr_before.delta_mean)} ({fp(t_cr_before.p)}) to {signed_(t_cr_after.delta_mean)} ({fp(t_cr_after.p)}). The second "
            f"artifact is the comma-encoded line break of Section 3.2, which produces evidence fragments such as \",CARDIOVASCULAR: "
            f",RESPIRATORY:\" that the extractive baseline copies into its summaries; the cleaned-evidence ablation of Section 5.4 "
            f"measures how much of the judge's error on verbatim sentences it explains."),
          ("table", dict(label="headerfilter", caption="Effect of removing markdown header lines from the claim set. Means are per-document means over the fifty documents.",
                         columns=["Condition", "Claim rows before", "Header lines", "Claims after", "Headers labeled Contradicted", "Mean CR before", "Mean CR after", "Mean UFR before", "Mean UFR after"],
                         widths=[0.8, 0.75, 0.7, 0.7, 0.9, 0.7, 0.7, 0.7, 0.7], font=9,
                         rows=[[c, str(int(R.hdr(c, "CR", "n_claims_before"))), str(int(R.hdr(c, "CR", "n_headers"))), str(int(R.hdr(c, "CR", "n_claims_after"))),
                                str(int(R.hdr(c, "CR", "headers_contradicted"))), f3(R.hdr(c, "CR", "mean_before")), f3(R.hdr(c, "CR", "mean_after")),
                                f3(R.hdr(c, "UFR", "mean_before")), f3(R.hdr(c, "UFR", "mean_after"))] for c in R.conds]))]
    b += [H2("3.7 Ethics, Data Governance and Reproducibility")]
    b += [P("The study involves no interaction with human subjects and no identifiable patient data. MTSamples is a public corpus of "
            "sample transcriptions that were never real patient records. MIMIC-IV-Note, ann-pt-summ, MedNLI and VeriFact-BHC are "
            "de-identified under the HIPAA Safe Harbor standard [[cite:hipaa]] and were obtained through PhysioNet's credentialed-access "
            "process (CITI training completed April 10, 2026; credentialing approved April 28, 2026; each data use agreement accepted "
            "before download). Under those agreements the data may not be shared, redistributed, used in any attempt at re-identification "
            "or transmitted to third parties, so no MIMIC-derived text was ever sent to the OpenAI API: the judge validation and "
            "selection, the MedNLI adaptation, the external check and the entire main study run with models on the author's computer, "
            "outputs that contain MIMIC-derived text are written to a directory excluded from version control, only aggregate statistics "
            "are published, and the credentialed files are stored on a single encrypted computer. The pilot's summaries were produced "
            "with the OpenAI API from public text, which raises a validity rather than a privacy concern (Section 8.7): the corpus may "
            "be in the model's pre-training data. Reproducibility rests on fixed seeds, pinned package versions, storage of every "
            "generated summary and claim label, and a recomputation script that reproduces every pilot number without any API or model "
            "call; for the main study the aggregate result files are published and the scripts regenerate them on any machine that holds "
            "the credentialed data (Appendix B).")]
    scan = _scan()
    if scan is not None:
        n_src, n_pilot = int(scan["files with a source-text match"]), int(scan["of which pilot result files"])
        b += [P("An automated check confirms that the repository contains no protected text: every credentialed text and every "
                "MIMIC-derived output kept privately is cut into overlapping eight-word sequences, and every tracked file, including the "
                "text of the PDF, Word and PowerPoint files, is searched for them, ignoring sequences that also occur in public MTSamples. "
                + (f"No tracked file shares a sequence with the credentialed texts. " if n_src == 0 else
                   f"The {int(scan['distinct shingles shared with the source texts'])} distinct sequences shared with the credentialed "
                   f"texts all occur in {'pilot result files' if n_pilot == n_src else 'the files listed in results/protected_text_scan.csv'}, "
                   f"written by GPT-4o-mini from public notes without access to the protected data, and are generic patient-instruction "
                   f"phrases such as \"follow up with your primary care provider\"; none occurs in more than "
                   f"{int(scan['largest number of source documents containing one of them'])} credentialed documents. ")
                + (f"The same check over all {int(scan['file versions in the history scanned'])} file versions in the repository's history "
                   f"finds matches only in earlier versions of the pilot result files. "
                   if "file versions in the history scanned" in scan and int(scan["history versions with a source-text match"]) == int(scan["of which versions of pilot result files"]) else "")
                + "The check, check_no_protected_text.py, prints only counts and is part of the repository.")]

    b += [H1("Chapter 4 Methodology"), H2("4.1 Overview")]
    b += [P("The methodology has two layers ([[fig:pipeline]]): a *generation layer* that produces one summary per document under each "
            "of five conditions, and an *evaluation layer* that applies the same claim-level NLI judge to every summary, so that "
            "differences between conditions cannot come from differences in measurement. A validation study applies the judge to "
            "expert-annotated summaries, robustness analyses vary its design choices, and Sections 4.9 to 4.11 extend the design to "
            "the judge selection study, the main study on MIMIC-IV hospital courses and the question-answering pilot. UFR and CR are "
            "judge-estimated throughout; only the expert annotations (Section 4.7) and the manual audit (Section 4.10.5) provide "
            "human-confirmed labels."),
          ("figure", dict(label="pipeline", path=f"{FIG}/fig_pipeline.png", width=6.5,
                          caption="Overview of the experimental pipeline. Each source note is summarized under five conditions; every summary is scored by the same claim-level NLI judge, whose evidence sentences are retrieved from the same note."))]
    b += [H2("4.2 The Claim-Level Evaluation Framework"), H3("4.2.1 Claim Segmentation")]
    b += [P("A summary is decomposed into claims at sentence granularity with spaCy; sentences under ten characters, section header "
            "lines and explicit abstention lines are removed (Section 3.5), and in the main study merged headers are stripped and "
            "citation markers are removed before splitting (Section 4.10.3). A sentence-level claim is not an atomic fact: one "
            "patient-facing sentence can combine a diagnosis, a medication with its dose and an instruction, and a single label hides "
            "which part failed. Atomic facts in the sense of FActScore [[cite:min2023]] avoid this ambiguity but need an extra "
            "generation step whose errors propagate into the metric, and decomposition does not help reliably [[cite:wanner2024]]. "
            "Sentences are therefore the primary unit, because they match the granularity of the expert annotations, and atomic-fact "
            "scoring is an additional analysis (Sections 6.6 and 7.10).")]
    b += [H3("4.2.2 Evidence Retrieval")]
    b += [P("For each claim c the source note is segmented into sentences, claim and sentences are embedded with the all-MiniLM-L6-v2 "
            "bi-encoder [[cite:reimers2019,wang2020minilm]], and the evidence set is the k = 3 source sentences with the highest "
            "cosine similarity to the claim:"),
          ("eq", "sim(c, s) = (e(c) · e(s)) / (‖e(c)‖ ‖e(s)‖),     E(c) = top-3 sentences by sim(c, s)"),
          P("Retrieval keeps the premise short and relevant, following the granularity argument of Laban et al. [[cite:laban2022]], "
            "and makes the judge's cost linear in the number of claims. Whether three sentences are enough, and whether they should be "
            "scored separately or together, is examined in Sections 5.3 and 5.4.")]
    b += [H3("4.2.3 Natural Language Inference Classification")]
    b += [P("Each (evidence sentence, claim) pair is passed to cross-encoder/nli-MiniLM2-L6-H768, a six-layer MiniLM cross-encoder "
            "fine-tuned on SNLI and MultiNLI [[cite:bowman2015,williams2018,wang2020minilm]], with the evidence as premise and the claim "
            "as hypothesis; its three logits (contradiction, entailment, neutral) are converted to probabilities. The claim's "
            "entailment score p(e) is the maximum entailment probability over its three pairs and its contradiction score p(c) the "
            "maximum contradiction probability, and the label follows a symmetric rule with threshold τ = 0.5:"),
          ("eq", "Supported if p(e) ≥ τ and p(e) > p(c);   Contradicted if p(c) ≥ τ and p(c) > p(e);   Not-Supported otherwise"),
          P("Not-Supported therefore covers both a claim the model finds neutral to every evidence sentence and one whose probabilities "
            "are both below the threshold. The threshold is varied in Section 5.4.")]
    b += [H3("4.2.4 Metrics")]
    b += [P("Two rates are computed for every summary from its N claims, of which N(c) are labeled Contradicted and N(n) Not-Supported:"),
          ("eq", "UFR = (N(c) + N(n)) / N          CR = N(c) / N"),
          P("The Unsupported Fact Rate is the share of claims the judge could not verify against the note and the Contradiction Rate "
            "the share it considers to conflict with it; CR is a subset of UFR, and both are computed per summary and then summarized "
            "across documents so that every document has equal weight. Both are judge-estimated: they count the claims an automatic "
            "judge fails to verify, including its false positives and excluding the unsupported claims it misses, and they are "
            "distinguished throughout from expert-confirmed rates (the expert annotations) and manually confirmed rates (the audit). "
            "Because a rate is a share of claims, it is reported with the number of claims, the supported and unsupported counts per "
            "summary and, in the main study, coverage: a summary can lower its rate by saying less, and a lower rate is not by itself "
            "a better summary.")]
    b += [H3("4.2.5 Statistical Analysis")]
    b += [P("All conditions are applied to the same documents, so comparisons are paired. Per-document differences are tested with "
            "the two-sided Wilcoxon signed-rank test [[cite:wilcoxon1945]]; the mean difference carries a 95 percent percentile "
            "bootstrap interval from 10,000 resamples [[cite:efron1993]]; the standardized mean difference d(z) [[cite:cohen1988]], "
            "the matched-pairs rank-biserial correlation [[cite:kerby2014]] and the share of documents improved, unchanged or worsened "
            "are reported. Holm's step-down procedure [[cite:holm1979]] controls the family-wise error rate: in the pilot over all "
            "twenty pairwise tests, in the main study over the primary family (E1, E1b, E2 and E3 against E0 on UFR, CR and coverage, "
            "twelve tests) and within each measure for the retrieval ablations. A result is called significant only after the "
            "adjustment, and an unadjusted p-value below 0.05 is called suggestive. A difference that is not significant is not "
            "evidence of equivalence: where two conditions are compared on coverage, the interval of differences compatible with the "
            "data is reported and, as a post hoc check, whether it lies within a margin of five percentage points (two one-sided tests "
            "at the 5 percent level with the bootstrap 90 percent interval). The unit of analysis is the hospital course; in the main "
            "study every course comes from a different patient and admission (Section 6.5). Analyses use SciPy [[cite:virtanen2020]] "
            "and scikit-learn [[cite:pedregosa2011]].")]
    b += [H2("4.3 Approach 1: Full-Note LLM Summarization Without Retrieval (E0)")]
    b += [P("The baseline represents the way LLM summarization is most often deployed: the whole document is placed in the prompt and "
            "the model is asked for a summary. GPT-4o-mini [[cite:openai2024]] is called with temperature 0.3 and at most 450 output "
            "tokens; the system prompt asks for patient-facing summaries \"accurate and grounded only in the provided clinical note\" "
            "without adding or inventing information, and the user prompt asks for 150 to 250 words in four parts (reason for visit, key "
            "findings, treatments, follow-up) without jargon, followed by the first 4,500 characters of the note (prompts in Appendix "
            "A). Faithfulness failures arise when the model's parametric knowledge of what discharge instructions usually say dominates "
            "the conditioning on the specific note [[cite:ji2023,huang2025]]; E0 measures how often that happens when the model is merely "
            "asked not to invent.")]
    b += [H2("4.4 Approach 2: Retrieval-Augmented Generation (E1)")]
    b += [P("E1 tests whether narrowing the model's view to retrieved excerpts reduces unsupported content. The note is split into "
            "consecutive five-sentence chunks, a query is formed from the document's description field and the first 300 characters of "
            "the note, query and chunks are embedded with all-MiniLM-L6-v2, and the three most similar chunks, numbered as excerpts and "
            "truncated to 4,000 characters, replace the note in a prompt that says \"Use ONLY the supplied excerpts\"; everything else is "
            "as in E0 ([[tab:conditions]]). Unlike the RAG of question-answering systems [[cite:lewis2020,gao2023rag]], the retrieval "
            "corpus is the document itself and the generator sees the excerpts only, so that the effect of retrieval can be isolated; "
            "the predictable cost is that content in unretrieved chunks cannot appear in the summary, which the coverage proxy of "
            "Section 4.8 measures, and the variant that also sees the note is E1b (Section 4.6)."),
          ("table", dict(label="conditions", caption="Generation settings of the five evaluated conditions.",
                         columns=["Condition", "Generator", "Input to the generator", "Retrieval", "Decoding"],
                         widths=[0.8, 1.0, 2.3, 1.5, 0.9], font=9, align=["left", "left", "left", "left", "left"],
                         rows=[["E0", "GPT-4o-mini", "First 4,500 characters of the note; four-part instruction", "none", "T = 0.3, 450 tokens"],
                               ["E1", "GPT-4o-mini", "Top-3 retrieved chunks only (at most 4,000 characters); instruction to use only the excerpts", "5-sentence chunks; query = description + first 300 characters; all-MiniLM-L6-v2 cosine", "T = 0.3, 450 tokens"],
                               ["E2", "none (extractive)", "All source sentences", "centroid ranking by mean cosine similarity; top-5 kept in document order", "deterministic"],
                               ["E1b", "GPT-4o-mini", "Full note plus the same excerpts", "as E1", "as E1"],
                               ["E3", "GPT-4o-mini", "E1 draft; per-claim verification against top-3 sentences from the full note; revision keeps, corrects or removes each claim", "as the judge (k = 3)", "verification T = 0; revision T = 0.3"]]))]
    b += [H2("4.5 Approach 3: Centroid-Based Extractive Summarization (E2)")]
    b += [P("E2 contains no language model. Following the centroid principle of Radev et al. [[cite:radev2004]], every source sentence "
            "is embedded with all-MiniLM-L6-v2 and scored by its mean cosine similarity to all other sentences, and the five "
            "highest-scoring sentences are returned in document order. As a summarizer it is the conservative extreme: it cannot "
            "fabricate, but it cannot paraphrase into patient-friendly language either. As a methodological control it matters more: "
            "every E2 claim is a verbatim sentence of the note, so any Not-Supported or Contradicted label on it is a judge error, and "
            "the E2 rates measure the judge's error on verbatim extracted text. They are an empirical reference, not a general lower "
            "bound, because the judge's errors on paraphrased, generated or negated sentences can be more or less frequent than on "
            "verbatim ones.")]
    b += [H2("4.6 Approaches 4 and 5: RAG with the Full Note (E1b) and RAG with Chain-of-Verification (E3)")]
    b += [P("Two further conditions extend E1 and share its interfaces. E1b gives the model the retrieved excerpts *and* the full note, "
            "with an instruction to give the excerpts priority, which separates the effect of focusing attention from the effect of "
            "withholding content. E3 implements a factored Chain-of-Verification [[cite:dhuliawala2024]] on top of E1: each claim of the "
            "E1 draft is verified in a separate prompt against the three source sentences retrieved for it from the full note (verdict "
            "SUPPORTED, NOT SUPPORTED or CONTRADICTED, with a corrected statement or REMOVE), and a final prompt rewrites the draft "
            "applying every verdict, writing \"Not stated in the note\" where a section has no supported content; verification runs "
            "at temperature 0 and revision at 0.3, about ten model calls per note. Both are scored by the unchanged judge, so any "
            "difference from E1 is attributable to the extra note context (E1b) or to verification and revision (E3).")]
    b += [H2("4.7 Validation of the Judge Against Expert Annotations")]
    b += [P("The validation study applies the judge, unchanged, to the 210 expert-annotated summaries of ann-pt-summ (Section 3.4), "
            "segmented and filtered exactly as in the main pipeline, with the hospital course as the source. A sentence is "
            "*expert-flagged* if it overlaps an annotated span; the *any* detector flags a sentence labeled Not-Supported or "
            "Contradicted (the UFR definition) and the *contradiction* detector only a Contradicted one. Agreement is reported as "
            "precision, recall, specificity, F1 and Cohen's kappa [[cite:cohen1960]] at the sentence level, overall and by subset, "
            "with the threshold swept from 0.30 to 0.95 and the area under the ROC curve [[cite:hanley1982]] as a threshold-free "
            "measure, and at the summary level as Spearman's correlation between the judge's rates and the experts' flagged share. "
            "Five evidence-aggregation variants are compared with the same cross-encoder: the pipeline's maximum over three retrieved "
            "sentences, three or five sentences concatenated, the maximum over every sentence of the course (the SummaC zero-shot "
            "design [[cite:laban2022]]), and the whole course as one premise. Sentences of one summary share a course and the 100 "
            "LLM-generated summaries come from only 20 courses, so intervals for the selected judge are bootstrapped over courses "
            "(Section 6.5).")]
    b += [H2("4.8 Robustness Analyses")]
    b += [P("Six offline analyses examine whether the pilot comparison depends on the judge's design and characterize the unsupported "
            "claims: the decision threshold τ from 0.5 to 0.9; one, three or five evidence sentences per claim; the evidence text with "
            "MTSamples' line breaks repaired, with E2 regenerated from repaired sentences; a coverage proxy; a negation analysis; and a "
            "keyword-assisted error taxonomy. Without reference summaries there is no direct measure of omission, so the coverage proxy "
            "counts the share of source sentences that some claim resembles closely (cosine similarity at least 0.6, chosen before the "
            "analysis, with 0.5 and 0.7 for sensitivity), which penalizes summaries that concentrate on a few passages. The taxonomy "
            "assigns each unsupported claim to one of eight categories by keyword patterns in a fixed priority order; the counts are a "
            "coarse characterization, not a validated clinical error typology.")]
    b += [H2("4.9 Judge Selection Study")]
    b += [P("The judge selection study scores six candidate judges, all runnable on a laptop, against the same 1,781 expert-annotated "
            "sentences with the protocol of Section 4.7: the pilot judge; two general-domain DeBERTa-v3-large NLI models, one trained "
            "on MultiNLI, FEVER-NLI, ANLI, LingNLI and WANLI and one from the sentence-transformers cross-encoder collection "
            "[[cite:reimers2019]]; two MiniCheck classifiers (DeBERTa-v3-large and RoBERTa-large) trained to decide whether a sentence "
            "is grounded in a document [[cite:tang2024minicheck]]; and Bespoke-MiniCheck-7B [[cite:bespoke2024]], a seven-billion-parameter "
            "language model fine-tuned for the same decision that leads the LLM-AggreFact leaderboard, served locally in eight-bit "
            "precision and answering yes or no, with the probability of \"Yes\" as its support score. Each candidate runs in two evidence "
            "modes, the three retrieved sentences concatenated and the whole hospital course (in windows of at most 400 tokens for the "
            "encoders, taking the maximum over windows). Thresholds are never tuned on the sentences they are evaluated on: for the "
            "LLM-generated summaries the threshold that maximizes kappa is chosen on the doctor-written ones and vice versa. The "
            "strongest general model is then fine-tuned on MedNLI [[cite:romanov2018]] for one epoch (learning rate 2e-5, effective batch "
            "16, sequence length 128, Adafactor [[cite:shazeer2018]] with the embedding matrix frozen to fit beside the language-model "
            "server, seed 42), and the checkpoint with the best MedNLI development accuracy is scored like the other candidates. The "
            "decision rule, fixed before the comparison, selects the candidate and mode with the highest kappa on all sentences."),
          P("Because the winner is chosen on the sentences on which its agreement is reported, that agreement is optimistic. Three "
            "analyses quantify this (Section 6.5): the whole selection procedure is cross-validated in five folds of courses, repeated "
            "twenty times; a patient-disjoint transfer test selects on the doctor-written instructions, whose courses are the main-study "
            "documents, and evaluates on the LLM-generated summaries of other patients, and vice versa; and agreement intervals are "
            "bootstrapped over courses and reported per source of summaries. Patient and admission identity was verified by matching "
            "every course to its MIMIC-IV-Note discharge note.")]
    b += [H2("4.10 Main Study on MIMIC-IV Hospital Courses"), H3("4.10.1 Generation")]
    b += [P("The main study repeats the five conditions on the 110 hospital courses of ann-pt-summ with three changes dictated by the "
            "data use agreement and the proposal. The generator is Qwen2.5-7B-Instruct [[cite:qwen2025]], quantized to four bits and "
            "served on the author's computer through an OpenAI-compatible local interface, so the pilot code runs unchanged while no "
            "protected text leaves the machine. The prompts ask for a four-part patient-facing summary (why the patient was in the "
            "hospital, key findings, treatments, medications and follow-up) and allow \"Not stated in the note.\" for a section "
            "without support, and the retrieval conditions must cite, after every sentence, the excerpt numbers it is based on. "
            "Because the courses have no description field, the retrieval query is the four section topics followed by the first 300 "
            "characters of the course. Decoding uses temperature 0.3 and at most 450 tokens.")]
    b += [H3("4.10.2 What Differs Between the Conditions")]
    b += [P("The conditions are meant to isolate retrieval and verification, but they differ in more than one variable ([[tab:confounds]]). "
            "Excerpt-only retrieval changes the content the model sees and adds the citation instruction, which may itself keep the "
            "model closer to its sources; verification adds model calls and deletes statements, so it shortens the summary; and shorter "
            "summaries may simply omit the sentences the judge most often fails to verify. Three variants separate these factors on the "
            "same courses: E1 and E1b without the citation instruction, and E0 asked for about 150 words, the mean length of E1, as a "
            "length-matched baseline. E2 cannot paraphrase and serves as a reference for the judge's errors, not as a competitor on "
            "usefulness."),
          ("table", dict(label="confounds", caption="Variables that differ between the main-study conditions and the variants that separate them. All LLM conditions use the same generator and decoding (Qwen2.5-7B-Instruct, temperature 0.3, at most 450 output tokens).",
                         columns=["Condition", "Source text shown to the model", "Citation instruction", "Length instruction", "Model calls", "Other differences"],
                         widths=[1.05, 1.55, 0.8, 0.85, 0.6, 1.65], font=8.5, align=["left"] * 6,
                         rows=[["E0", "Whole course (at most 6,000 characters)", "No", "150 to 250 words", "1", "Abstention allowed"],
                               ["E0, length-matched", "Whole course", "No", "About 150 words", "1", "Isolates the length instruction"],
                               ["E1", "Three retrieved five-sentence chunks (at most 4,000 characters)", "Yes", "150 to 250 words", "1", "Told to use only the excerpts"],
                               ["E1, no citations", "As E1", "No", "150 to 250 words", "1", "Isolates retrieval from the citation instruction"],
                               ["E1b", "The same chunks and the whole course", "Yes", "150 to 250 words", "1", "Told to give the excerpts priority"],
                               ["E1b, no citations", "As E1b", "No", "150 to 250 words", "1", "Isolates the citation instruction for E1b"],
                               ["E3", "E1 draft; three retrieved course sentences per claim for verification", "Draft only", "150 to 250 words", "About 10", "Verification at temperature 0; revision deletes, replaces or keeps each claim"],
                               ["E2", "All course sentences", "Not applicable", "Five sentences", "0", "Verbatim extraction; no language model"]]))]
    b += [H3("4.10.3 Claims, Citations and Measures")]
    b += [P("Every summary, and the clinician-written discharge instructions as a human reference, is scored by the judge selected in "
            "Chapter 6 with its selected evidence mode and threshold. Claims use the pilot's segmentation with two corrections: citation "
            "markers are removed before splitting and assigned to the sentence that contains them (a marker after the final period "
            "belongs to that sentence), and header lines merged into the following sentence are stripped (Section 3.5). Five measures "
            "are added to UFR and CR. *Coverage of the clinician's instructions* is the share of the clinician's sentences that the "
            "summary supports, obtained by running the judge in the reverse direction with the summary as premise and each instruction "
            "sentence as hypothesis; it replaces the pilot's embedding proxy with a reference-based omission measure. *Citation "
            "coverage* is the share of claims with at least one valid citation, *citation accuracy* the share of cited claims whose cited "
            "excerpts support the claim under the judge, and *citation reliability* their product, the share of all claims with a valid "
            "and supporting citation. *Abstentions* are counted per summary. Tests follow Section 4.2.5.")]
    b += [H3("4.10.4 Additional Conditions and Robustness Analyses")]
    b += [P("Six retrieval ablations vary one setting of E1 at a time (chunks of three or eight sentences, two or five retrieved chunks, "
            "BM25 lexical retrieval [[cite:robertson2009]], and no citation instruction), each compared with the base configuration with "
            "Holm adjustment within each measure; the length-matched E0 and E1b without citations are compared with their base conditions "
            "the same way. Seven further analyses test the conclusions: (i) scoring at the pilot's threshold of 0.5; (ii) a second judge, "
            "the strongest candidate of a different model family, with the shares flagged by both judges and by either reported as a "
            "conservative and an inclusive rule, alternative estimates rather than bounds because both judges can share an error; (iii) "
            "atomic facts, with every claim split by the local model (prompt in Appendix A), every fact scored, and a sentence flagged "
            "when its weakest fact is; (iv) half-overlapping evidence windows, to count flags caused by evidence split across windows; "
            "(v) repeated generation of E0, E1, E1b and E3 on a random subset of 40 courses (seed 42), twice more with fixed seeds; (vi) "
            "retrieval failures of the verifier, scoring every rejected E1 draft claim against the whole course and against the "
            "verifier's three retrieved sentences; and (vii) two further generators, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, "
            "regenerating the four text conditions on all courses under the same prompts, decoding, retrieval, judge and tests.")]
    b += [H3("4.10.5 Manual Audit")]
    b += [P("A blind manual audit checks whether the selected judge remains reliable on the summaries this generator wrote. Sixty of "
            "the 110 courses were drawn at random (seed 7); within them, for each of E0, E1, E1b and E3, up to 30 claims the judge "
            "flagged and 30 it accepted were drawn, together with 30 cited claims of E1 and of E1b with their excerpts. A local labeling "
            "tool shows one course at a time with its sampled claims in random order and hides the judge's label and the condition. Each "
            "claim is labeled supported, partly supported (several facts, at least one unsupported), not supported, or unsure, with the "
            "type of problem recorded; each citation is labeled by whether the cited excerpt supports the claim; the protocol is in "
            "Appendix C. Partly and not supported count as unsupported. Because flagged and accepted claims were sampled in fixed numbers, "
            "each label is weighted by its stratum's share of the condition's claims, which gives a manually confirmed unsupported share "
            "per condition that requires no assumption about the judge, and, against the judge's labels for the same claims, the judge's "
            "precision, false omission rate, sensitivity, specificity and kappa on these outputs. Intervals come from 2,000 bootstrap "
            "resamples of courses, and for each pair of conditions the manual difference is compared with the judge's. The audit checks "
            "textual support by the course, not clinical correctness, and was performed by one annotator, the author, who is not a "
            "clinician."),
          P("The audit was labeled in two blind passes. A review of the first pass found paraphrased claims labeled partly supported "
            "against the protocol's paraphrase rule, so the rubric was clarified (Appendix C), the sample was extended by 105 claims "
            "from the same courses under the same design (seed 8; 340 claims and 60 citation checks in all), and the author relabeled "
            "the full sample with the first pass's answers hidden. Chapter 7 reports the second pass and the agreement between the "
            "passes; both label files are preserved.")]
    b += [H2("4.11 Question-Answering Pilot")]
    b += [P("The proposal's second task is piloted with three fixed questions per course, about medications after discharge, follow-up "
            "and warning signs. The local model answers each in at most three sentences from the full course (QA-E0) or from the three "
            "chunks retrieved with the question as query (QA-E1), replying \"Not stated in the note.\" when the course is silent. Per "
            "condition and question type the pilot reports the abstention rate, the UFR and CR of answered questions under the judge, "
            "the share of answer sentences supported by the clinician's own instructions, and, over all questions, the shares abstained "
            "on, answered with every sentence supported, and answered with an unsupported sentence. There are no clinician-written gold "
            "answers, so the pilot is exploratory: it shows how abstention and grounding interact, not whether the answers are correct.")]
    b += [H2("4.12 Tools, Software and Computational Environment")]
    b += [P("All experiments ran on an Apple MacBook Air (Apple M5, 10 cores, 24 GB unified memory); the pilot pipeline on the CPU, the "
            "candidate judges, the MedNLI fine-tuning and the local language model on the GPU through PyTorch's Metal backend and Apple's "
            "MLX framework. Recomputing every pilot result takes about five minutes, scoring one large candidate judge against the expert "
            "annotations about an hour, and generating the 1,210 main-study summaries roughly seven hours. [[tab:tools]] lists the "
            "software; every run is offline except the pilot's generation calls to the OpenAI API."),
          ("table", dict(label="tools", caption="Software, models and versions used.",
                         columns=["Component", "Package or model", "Version"], widths=[2.2, 2.8, 1.5], font=9.5, align=["left", "left", "left"],
                         rows=[["Language", "Python", "3.9.6 (pipeline) and 3.13 (local model server)"],
                               ["LLM generator, pilot study", "OpenAI GPT-4o-mini via openai", "gpt-4o-mini; openai 2.31.0"],
                               ["LLM generator, main study", "Qwen2.5-7B-Instruct, 4-bit, served locally with mlx-lm", "mlx 0.32; mlx-lm server on localhost"],
                               ["Candidate judges", "DeBERTa-v3-large NLI models and MiniCheck DeBERTa/RoBERTa-large (transformers); Bespoke-MiniCheck-7B, 8-bit, served locally with MLX", "transformers 4.57.6; mlx-lm 0.31"],
                               ["Lexical retrieval (ablation)", "rank-bm25", "0.2.2"],
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
    return "n/a" if x is None else f"{x:+.3f}"
