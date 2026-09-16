"""references.py — Reference database (IEEE style) and in-text citation resolver.

Write [[cite:key]] or [[cite:key1,key2]] in any text block.  resolve_citations() numbers the
references in order of first appearance and returns the formatted reference list.
"""
import re

REFS = {
 "ji2023": 'Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, "Survey of hallucination in natural language generation," *ACM Computing Surveys*, vol. 55, no. 12, art. 248, pp. 1–38, 2023, doi: 10.1145/3571730.',
 "huang2025": 'L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu, "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *ACM Transactions on Information Systems*, vol. 43, no. 2, art. 42, 2025, doi: 10.1145/3703155.',
 "maynez2020": 'J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, "On faithfulness and factuality in abstractive summarization," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 1906–1919, doi: 10.18653/v1/2020.acl-main.173.',
 "lin2004": 'C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Text Summarization Branches Out*, Barcelona, Spain, 2004, pp. 74–81. [Online]. Available: https://aclanthology.org/W04-1013',
 "papineni2002": 'K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A method for automatic evaluation of machine translation," in *Proc. 40th Annual Meeting of the Association for Computational Linguistics*, 2002, pp. 311–318, doi: 10.3115/1073083.1073135.',
 "zhang2020": 'T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in *Proc. International Conference on Learning Representations*, 2020, doi: 10.48550/arXiv.1904.09675.',
 "fabbri2021": 'A. R. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher, and D. Radev, "SummEval: Re-evaluating summarization evaluation," *Transactions of the Association for Computational Linguistics*, vol. 9, pp. 391–409, 2021, doi: 10.1162/tacl_a_00373.',
 "kryscinski2019": 'W. Kryściński, N. S. Keskar, B. McCann, C. Xiong, and R. Socher, "Neural text summarization: A critical evaluation," in *Proc. EMNLP-IJCNLP*, 2019, pp. 540–551, doi: 10.18653/v1/D19-1053.',
 "kryscinski2020": 'W. Kryściński, B. McCann, C. Xiong, and R. Socher, "Evaluating the factual consistency of abstractive text summarization," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2020, pp. 9332–9346, doi: 10.18653/v1/2020.emnlp-main.750.',
 "goodrich2019": 'B. Goodrich, V. Rao, P. J. Liu, and M. Saleh, "Assessing the factual accuracy of generated text," in *Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2019, pp. 166–175, doi: 10.1145/3292500.3330955.',
 "nan2021": 'F. Nan, R. Nallapati, Z. Wang, C. N. dos Santos, H. Zhu, D. Zhang, K. McKeown, and B. Xiang, "Entity-level factual consistency of abstractive text summarization," in *Proc. 16th Conference of the European Chapter of the Association for Computational Linguistics*, 2021, pp. 2727–2733, doi: 10.18653/v1/2021.eacl-main.235.',
 "wang2020qags": 'A. Wang, K. Cho, and M. Lewis, "Asking and answering questions to evaluate the factual consistency of summaries," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 5008–5020, doi: 10.18653/v1/2020.acl-main.450.',
 "durmus2020": 'E. Durmus, H. He, and M. Diab, "FEQA: A question answering evaluation framework for faithfulness assessment in abstractive summarization," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 5055–5070, doi: 10.18653/v1/2020.acl-main.454.',
 "scialom2021": 'T. Scialom, P.-A. Dray, S. Lamprier, B. Piwowarski, J. Staiano, A. Wang, and P. Gallinari, "QuestEval: Summarization asks for fact-based evaluation," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2021, pp. 6594–6604, doi: 10.18653/v1/2021.emnlp-main.529.',
 "falke2019": 'T. Falke, L. F. R. Ribeiro, P. A. Utama, I. Dagan, and I. Gurevych, "Ranking generated summaries by correctness: An interesting but challenging application for natural language inference," in *Proc. 57th Annual Meeting of the Association for Computational Linguistics*, 2019, pp. 2214–2220, doi: 10.18653/v1/P19-1213.',
 "laban2022": 'P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," *Transactions of the Association for Computational Linguistics*, vol. 10, pp. 163–177, 2022, doi: 10.1162/tacl_a_00453.',
 "honovich2022": 'O. Honovich, R. Aharoni, J. Herzig, H. Taitelbaum, D. Kukliansy, V. Cohen, T. Scialom, I. Szpektor, A. Hassidim, and Y. Matias, "TRUE: Re-evaluating factual consistency evaluation," in *Proc. 2022 Conference of the North American Chapter of the Association for Computational Linguistics*, 2022, pp. 3905–3920, doi: 10.18653/v1/2022.naacl-main.287.',
 "zha2023": 'Y. Zha, Y. Yang, R. Li, and Z. Hu, "AlignScore: Evaluating factual consistency with a unified alignment function," in *Proc. 61st Annual Meeting of the Association for Computational Linguistics*, 2023, pp. 11328–11348, doi: 10.18653/v1/2023.acl-long.634.',
 "min2023": 'S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. W. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2023, pp. 12076–12100, doi: 10.18653/v1/2023.emnlp-main.741.',
 "tam2023": 'D. Tam, A. Mascarenhas, S. Zhang, S. Kwan, M. Bansal, and C. Raffel, "Evaluating the factual consistency of large language models through news summarization," in *Findings of the Association for Computational Linguistics: ACL 2023*, 2023, pp. 5220–5255, doi: 10.18653/v1/2023.findings-acl.322.',
 "manakul2023": 'P. Manakul, A. Liusie, and M. J. F. Gales, "SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2023, pp. 9004–9017, doi: 10.18653/v1/2023.emnlp-main.557.',
 "bowman2015": 'S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning, "A large annotated corpus for learning natural language inference," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2015, pp. 632–642, doi: 10.18653/v1/D15-1075.',
 "williams2018": 'A. Williams, N. Nangia, and S. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," in *Proc. 2018 Conference of the North American Chapter of the Association for Computational Linguistics*, 2018, pp. 1112–1122, doi: 10.18653/v1/N18-1101.',
 "romanov2018": 'A. Romanov and C. Shivade, "Lessons from natural language inference in the clinical domain," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2018, pp. 1586–1596, doi: 10.18653/v1/D18-1187.',
 "shazeer2018": 'N. Shazeer and M. Stern, "Adafactor: Adaptive learning rates with sublinear memory cost," in *Proc. 35th International Conference on Machine Learning*, vol. 80, 2018, pp. 4596–4604, doi: 10.48550/arXiv.1804.04235.',
 "vaswani2017": 'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 5998–6008, doi: 10.48550/arXiv.1706.03762.',
 "devlin2019": 'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 2019, pp. 4171–4186, doi: 10.18653/v1/N19-1423.',
 "brown2020": 'T. B. Brown et al., "Language models are few-shot learners," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 1877–1901, doi: 10.48550/arXiv.2005.14165.',
 "reimers2019": 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992, doi: 10.18653/v1/D19-1410.',
 "thakur2021": 'N. Thakur, N. Reimers, J. Daxenberger, and I. Gurevych, "Augmented SBERT: Data augmentation method for improving bi-encoders for pairwise sentence scoring tasks," in *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics*, 2021, pp. 296–310, doi: 10.18653/v1/2021.naacl-main.28.',
 "wang2020minilm": 'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 5776–5788, doi: 10.48550/arXiv.2002.10957.',
 "vanveen2024": 'D. Van Veen, C. Van Uden, L. Blankemeier, J.-B. Delbrouck, A. Aali, C. Bluethgen, A. Pareek, M. Polacin, E. P. Reis, A. Seehofnerová, N. Rohatgi, P. Hosamani, W. Collins, N. Ahuja, C. P. Langlotz, J. Hom, S. Gatidis, J. Pauly, and A. S. Chaudhari, "Adapted large language models can outperform medical experts in clinical text summarization," *Nature Medicine*, vol. 30, pp. 1134–1142, 2024, doi: 10.1038/s41591-024-02855-5.',
 "asgari2025": 'E. Asgari, N. Montaña-Brown, M. Dubois, S. Khalil, J. Balloch, J. A. Yeung, and D. Pimenta, "A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation," *npj Digital Medicine*, vol. 8, art. 274, 2025, doi: 10.1038/s41746-025-01670-7.',
 "hegselmann2024": 'S. Hegselmann, Z. Shen, F. Gierse, M. Agrawal, D. Sontag, and X. Jiang, "A data-centric approach to generate faithful and high quality patient summaries with large language models," in *Proc. Conference on Health, Inference, and Learning*, PMLR vol. 248, 2024, pp. 339–379. [Online]. Available: https://proceedings.mlr.press/v248/hegselmann24a.html',
 "hegselmann2025data": 'S. Hegselmann, Z. Shen, F. Gierse, M. Agrawal, D. Sontag, and X. Jiang, "Medical expert annotations of unsupported facts in doctor-written and LLM-generated patient summaries (version 1.0.1)," PhysioNet, 2025, doi: 10.13026/gedc-j464.',
 "tang2023": 'L. Tang, Z. Sun, B. Idnay, J. G. Nestor, A. Soroush, P. A. Elias, Z. Xu, Y. Ding, G. Durrett, J. F. Rousseau, C. Weng, and Y. Peng, "Evaluating large language models on medical evidence summarization," *npj Digital Medicine*, vol. 6, art. 158, 2023, doi: 10.1038/s41746-023-00896-7.',
 "adams2021": 'G. Adams, E. Alsentzer, M. Ketenci, J. Zucker, and N. Elhadad, "What\'s in a summary? Laying the groundwork for advances in hospital-course summarization," in *Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics*, 2021, pp. 4794–4811, doi: 10.18653/v1/2021.naacl-main.382.',
 "moramarco2022": 'F. Moramarco, A. Papadopoulos Korfiatis, M. Perera, D. Juric, J. Flann, E. Reiter, A. Belz, and A. Savkov, "Human evaluation and correlation with automatic metrics in consultation note generation," in *Proc. 60th Annual Meeting of the Association for Computational Linguistics*, 2022, pp. 5739–5754, doi: 10.18653/v1/2022.acl-long.394.',
 "singhal2023": 'K. Singhal et al., "Large language models encode clinical knowledge," *Nature*, vol. 620, pp. 172–180, 2023, doi: 10.1038/s41586-023-06291-2.',
 "pal2023": 'A. Pal, L. K. Umapathi, and M. Sankarasubbu, "Med-HALT: Medical domain hallucination test for large language models," in *Proc. 27th Conference on Computational Natural Language Learning*, 2023, pp. 314–334, doi: 10.18653/v1/2023.conll-1.21.',
 "lewis2020": 'P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 9459–9474, doi: 10.48550/arXiv.2005.11401.',
 "guu2020": 'K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang, "REALM: Retrieval-augmented language model pre-training," in *Proc. 37th International Conference on Machine Learning*, PMLR vol. 119, 2020, pp. 3929–3938, doi: 10.48550/arXiv.2002.08909.',
 "karpukhin2020": 'V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, "Dense passage retrieval for open-domain question answering," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2020, pp. 6769–6781, doi: 10.18653/v1/2020.emnlp-main.550.',
 "gao2023rag": 'Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," 2023, doi: 10.48550/arXiv.2312.10997.',
 "xiong2024": 'G. Xiong, Q. Jin, Z. Lu, and A. Zhang, "Benchmarking retrieval-augmented generation for medicine," in *Findings of the Association for Computational Linguistics: ACL 2024*, 2024, pp. 6233–6251, doi: 10.18653/v1/2024.findings-acl.372.',
 "zakka2024": 'C. Zakka et al., "Almanac — Retrieval-augmented language models for clinical medicine," *NEJM AI*, vol. 1, no. 2, 2024, doi: 10.1056/AIoa2300068.',
 "dhuliawala2024": 'S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu, X. Li, A. Celikyilmaz, and J. Weston, "Chain-of-Verification reduces hallucination in large language models," in *Findings of the Association for Computational Linguistics: ACL 2024*, 2024, pp. 3563–3578, doi: 10.18653/v1/2024.findings-acl.212.',
 "gao2023rarr": 'L. Gao, Z. Dai, P. Pasupat, A. Chen, A. T. Chaganty, Y. Fan, V. Zhao, N. Lao, H. Lee, D.-C. Juan, and K. Guu, "RARR: Researching and revising what language models say, using language models," in *Proc. 61st Annual Meeting of the Association for Computational Linguistics*, 2023, pp. 16477–16508, doi: 10.18653/v1/2023.acl-long.910.',
 "johnson2023mimic": 'A. E. W. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout, S. Horng, T. J. Pollard, S. Hao, B. Moody, B. Gow, L.-w. H. Lehman, L. A. Celi, and R. G. Mark, "MIMIC-IV, a freely accessible electronic health record dataset," *Scientific Data*, vol. 10, art. 1, 2023, doi: 10.1038/s41597-022-01899-x.',
 "johnson2023note": 'A. Johnson, T. Pollard, S. Horng, L. A. Celi, and R. Mark, "MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2)," PhysioNet, 2023, doi: 10.13026/1n74-ne17.',
 "goldberger2000": 'A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. Ch. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000, doi: 10.1161/01.CIR.101.23.e215.',
 "mtsamples": 'MTSamples, "Transcribed medical transcription sample reports and examples." [Online]. Available: https://mtsamples.com (accessed Sep. 2026). Kaggle mirror: T. Boyle, "Medical transcriptions," 2018. [Online]. Available: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions',
 "openai2024": 'OpenAI, "GPT-4o mini: Advancing cost-efficient intelligence," Jul. 2024. [Online]. Available: https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/ (accessed Sep. 2026).',
 "honnibal2020": 'M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, "spaCy: Industrial-strength natural language processing in Python," 2020, doi: 10.5281/zenodo.1212303.',
 "pedregosa2011": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html',
 "virtanen2020": 'P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.',
 "wilcoxon1945": 'F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.',
 "efron1993": 'B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993, doi: 10.1201/9780429246593.',
 "cohen1988": 'J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum Associates, 1988.',
 "kerby2014": 'D. S. Kerby, "The simple difference formula: An approach to teaching nonparametric correlation," *Comprehensive Psychology*, vol. 3, art. 1, 2014, doi: 10.2466/11.IT.3.1.',
 "holm1979": 'S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. [Online]. Available: https://www.jstor.org/stable/4615733',
 "cohen1960": 'J. Cohen, "A coefficient of agreement for nominal scales," *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960, doi: 10.1177/001316446002000104.',
 "hanley1982": 'J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982, doi: 10.1148/radiology.143.1.7063747.',
 "radev2004": 'D. R. Radev, H. Jing, M. Styś, and D. Tam, "Centroid-based summarization of multiple documents," *Information Processing and Management*, vol. 40, no. 6, pp. 919–938, 2004, doi: 10.1016/j.ipm.2003.10.006.',
 "tang2024minicheck": 'L. Tang, P. Laban, and G. Durrett, "MiniCheck: Efficient fact-checking of LLMs on grounding documents," in *Proc. Conference on Empirical Methods in Natural Language Processing*, 2024, pp. 8818–8847, doi: 10.18653/v1/2024.emnlp-main.499.',
 "qwen2025": 'Qwen Team, "Qwen2.5 technical report," 2025, doi: 10.48550/arXiv.2412.15115.',
 "robertson2009": 'S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009, doi: 10.1561/1500000019.',
 "hipaa": 'U.S. Department of Health and Human Services, "Standards for privacy of individually identifiable health information: De-identification of protected health information, 45 C.F.R. § 164.514," *Code of Federal Regulations*, 2013. [Online]. Available: https://www.ecfr.gov/current/title-45/section-164.514',
}

_CITE = re.compile(r"\[\[cite:([A-Za-z0-9_,\- ]+)\]\]")
_TEXT_KEYS = ("caption", "note")


def resolve_citations(blocks):
    """Replace [[cite:...]] with IEEE numbers in order of first appearance. Returns (blocks, entries)."""
    order = []

    def num(key):
        key = key.strip()
        if key not in REFS:
            raise KeyError(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def sub(text):
        def rep(m):
            keys = [k for k in m.group(1).split(",") if k.strip()]
            return ", ".join(f"[{num(k)}]" for k in keys)
        return _CITE.sub(rep, text)

    out = []
    for b in blocks:
        kind = b[0]
        if kind in ("h1", "h2", "h3", "p", "pni", "eq"):
            out.append((kind, sub(b[1])))
        elif kind in ("bullets", "numbers"):
            out.append((kind, [sub(x) for x in b[1]]))
        elif kind in ("table", "figure"):
            d = dict(b[1])
            for k in _TEXT_KEYS:
                if d.get(k):
                    d[k] = sub(d[k])
            if kind == "table":
                d["rows"] = [[sub(str(c)) for c in row] for row in d["rows"]]
                d["columns"] = [sub(str(c)) for c in d["columns"]]
            out.append((kind, d))
        else:
            out.append(b)
    entries = [f"[{i + 1}] {REFS[k]}" for i, k in enumerate(order)]
    return out, entries
