import re
import json
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from models import Review


def _strip_bullets(text: str) -> str:
    return (text or "").strip().lstrip("-•*\t ")


def _normalize_phrase(phrase: str) -> str:
    t = (phrase or "").strip().lower()
    # English normalizations
    t = re.sub(r"\b(arriv(e|ed|es|ing))\s+(late|delayed|with delay)\b", "late delivery", t)
    t = re.sub(r"\b(delivery\s+(was\s+)?)?(late|delayed)\b", "late delivery", t)
    t = re.sub(r"\b(damag(e|ed|es|ing))\s+(package|packaging|box|item|product)\b", "damaged packaging", t)
    t = re.sub(r"\b(broken|cracked)\s+(item|product|screen|device)\b", "broken product", t)
    t = re.sub(r"\b(bad|poor)\s+quality\b", "poor quality", t)
    t = re.sub(r"\b(high|expensive)\s+price\b", "high price", t)
    t = re.sub(r"\b(rude|unhelpful)\s+(support|customer service|service|courier|driver)\b", "poor service", t)
    t = re.sub(r"\b(long|slow)\s+(response|reply|shipping|delivery)\b", "slow service", t)
    t = re.sub(r"\b(missing|lost)\s+(parts|item|package|order)\b", "missing items", t)
    # Russian normalizations
    t = re.sub(r"\b(доставк[аи])\s+(опоздал[аи]|поздн[ао])\b", "опоздание доставки", t)
    t = re.sub(r"\b(опоздал[аи])\s+доставк[аи]\b", "опоздание доставки", t)
    t = re.sub(r"\b(порв[ао]н(ая|ы)|помят(ая|ы)|поврежд(ен|ена|ены))\s+(упаковк[аи]|коробк[аи])\b", "поврежденная упаковка", t)
    t = re.sub(r"\b(сломан(а|о|ы)|трещин[аы])\b", "сломанный товар", t)
    t = re.sub(r"\b(плох(ое|ой)|низкое)\s+качеств[оа]\b", "плохое качество", t)
    t = re.sub(r"\b(высокая|завышенная|большая)\s+цен[аы]\b", "высокая цена", t)
    t = re.sub(r"\b(груб(ая|ый)|хамств(о|о)|невежлив(ая|ый))\s+(поддержк[аи]|курьер|сервис)\b", "плохой сервис", t)
    t = re.sub(r"\b(долго|медленн[оая])\s+(отвечают|ответ|доставка|поддержка)\b", "медленный сервис", t)
    t = re.sub(r"\b(нет|отсутствуют|пропал(и)?)\s+(детал(и|ей)|товар|части|комплектующ(ие|их))\b", "отсутствуют комплектующие", t)
    # Cleanup duplicated spaces
    t = re.sub(r"\s+", " ", t).strip()
    # Capitalize first letter for display
    if t:
        t = t[0].upper() + t[1:]
    return t


def _clean_insight_list(items: List[str], polarity: str) -> List[str]:
    cleaned: List[str] = []
    seen: set = set()
    for raw in items:
        t = _strip_bullets(str(raw))
        t = re.sub(r"[\"'`\-•*]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        norm = t.lower()
        junk = {"arrived", "really", "all", "item", "price", "poor", "bad"} if polarity == "negative" else {"good", "great", "nice", "price"}
        if norm in junk:
            continue
        norm = _normalize_phrase(norm)
        if not norm:
            continue
        if len(norm.split()) < 2:
            continue
        # Drop common incomplete starters
        if norm.startswith("not worth"):
            norm = "poor value for money"
        if norm.startswith("чтобы "):
            norm = re.sub(r"^чтобы\s+было?\s+", "", norm).strip()
            if not norm or len(norm.split()) < 1:
                continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(norm)
        if len(cleaned) >= 10:
            break
    return cleaned


def _parse_insights_from_text(text: str) -> Dict[str, List[str]]:
    problems: List[str] = []
    strengths: List[str] = []
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            p = obj.get("problems") or obj.get("issues") or []
            s = obj.get("strengths") or obj.get("pros") or []
            problems = [str(x).strip() for x in p if str(x).strip()]
            strengths = [str(x).strip() for x in s if str(x).strip()]
    except Exception:
        problems = []
        strengths = []
    if not problems and not strengths:
        lines = [l.strip(" -*•\t") for l in text.splitlines()]
        mode = None
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["problems", "issues", "недостат", "минус"]):
                mode = "problems"
                continue
            if any(k in low for k in ["strengths", "pros", "достоин", "плюс"]):
                mode = "strengths"
                continue
            if mode == "problems" and len(line) >= 3:
                problems.append(line)
            elif mode == "strengths" and len(line) >= 3:
                strengths.append(line)
    return {"problems": problems[:10], "strengths": strengths[:10]}


def extract_insights_ai(reviews: List[Review], token: str, model: Optional[str] = None) -> Dict[str, List[str]]:
    from services import _hf_text2text  # avoid circular import at module import time
    if not token:
        return {"problems": [], "strengths": []}
    model_name = model or "google/flan-t5-large"

    snippets = []
    total_chars = 0
    for r in reviews:
        t = (r.text or "").strip()
        if not t:
            continue
        t = t.replace("\n", " ")
        if len(t) > 400:
            t = t[:400] + "…"
        if total_chars + len(t) > 6000:
            break
        snippets.append(f"- {t}")
        total_chars += len(t)
    has_cyr = any("\u0400" <= ch <= "\u04FF" for ch in " ".join(snippets))
    if has_cyr:
        system = (
            "Извлеки ключевые ПРОБЛЕМЫ и ДОСТОИНСТВА из отзывов ниже. "
            "Сгруппируй по смыслу, избегай повторов, используй краткие пунктов списки. "
            "Ответ верни в JSON с полями problems (список строк) и strengths (список строк)."
        )
        prompt = system + "\nОтзывы:\n" + "\n".join(snippets)
    else:
        system = (
            "Extract the key PROBLEMS and STRENGTHS from the reviews below. "
            "Group similar points, avoid duplicates, use concise bullet-like phrases. "
            "Return JSON with fields problems (list of strings) and strengths (list of strings)."
        )
        prompt = system + "\nReviews:\n" + "\n".join(snippets)
    try:
        gen = _hf_text2text(prompt, model_name, token, max_new_tokens=384)
        parsed = _parse_insights_from_text(gen)
        cleaned = {
            "problems": _clean_insight_list(parsed.get("problems", []), polarity="negative"),
            "strengths": _clean_insight_list(parsed.get("strengths", []), polarity="positive"),
        }
        return cleaned
    except Exception:
        return {"problems": [], "strengths": []}


def extract_insights_heuristic(reviews: List[Review]) -> Dict[str, List[str]]:
    pos_texts = [r.text for r in reviews if (r.sentiment or "").startswith("pos")]
    neg_texts = [r.text for r in reviews if (r.sentiment or "").startswith("neg")]

    stop = set([
        # EN
        "and","the","to","of","a","in","for","is","on","it","was","with","this","that","are","be","as","have","but","not","or","very","i","we","you","they","them","their","our","my","your","me","he","she","his","her","its","at","by","from","so","if","because","also","even","still","all","really","just","one","get","got","would","could","should","arrived","item","thing","stuff","lot","bit","quite","pretty",
        # RU
        "и","в","на","это","как","что","не","с","за","от","очень","бы","то","по","у","из","а","но","до","при","для","же","так","он","она","они","мы","вы","мне","нам","вас","их","там","тут","ещё","еще","очень","всего","все","реально","просто","вот","типа","короче","товар","вещь","предмет",
    ])

    def top_phrases(texts: List[str], n: int = 8) -> List[str]:
        if not texts:
            return []
        vec = TfidfVectorizer(
            max_features=8000,
            ngram_range=(2, 3),
            min_df=1,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        X = vec.fit_transform([(t or "").lower() for t in texts])
        vocab = vec.get_feature_names_out()
        scores = X.mean(axis=0).A1
        pairs = [(vocab[i], scores[i]) for i in range(len(vocab))]
        def ok(term: str) -> bool:
            parts = term.split()
            if any(len(w) <= 2 or w.isdigit() or w in stop for w in parts):
                return False
            if len(set(parts)) == 1:
                return False
            if not any(re.search(r"[a-zA-Zа-яА-Я]", w) for w in parts):
                return False
            return True
        pairs = [(t, s) for (t, s) in pairs if ok(t)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        # Diversity-aware selection
        result: List[str] = []
        result_norm: List[str] = []
        seen_keys: set = set()
        for term, _ in pairs:
            norm = _normalize_phrase(term)
            if not norm:
                continue
            generic = {"Really", "All", "Item", "Товар", "Просто", "Очень"}
            if norm in generic:
                continue
            key = " ".join(sorted(set(norm.lower().split())))
            if key in seen_keys:
                continue
            words = set(norm.lower().split())
            too_similar = False
            for existing in result_norm:
                w2 = set(existing.lower().split())
                overlap = len(words & w2) / max(1, len(words | w2))
                if overlap >= 0.6:
                    too_similar = True
                    break
            if too_similar:
                continue
            seen_keys.add(key)
            result.append(norm)
            result_norm.append(norm)
            if len(result) >= n:
                break
        return result

    problems = top_phrases(neg_texts, n=3)
    strengths = top_phrases(pos_texts, n=3)
    return {"problems": problems, "strengths": strengths}


def extract_key_insights(reviews: List[Review], max_items: int = 3) -> Dict[str, List[str]]:
    pos_docs = [r.text for r in reviews if (r.sentiment or "").startswith("pos")]
    neg_docs = [r.text for r in reviews if (r.sentiment or "").startswith("neg")]
    if not pos_docs and not neg_docs:
        return {"problems": [], "strengths": []}

    def build_candidates(texts: List[str]) -> List[str]:
        if not texts:
            return []
        vec = TfidfVectorizer(
            max_features=12000,
            ngram_range=(2, 3),
            min_df=2,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        try:
            X = vec.fit_transform([(t or "").lower() for t in texts])
        except ValueError:
            return []
        vocab = vec.get_feature_names_out()
        scores = X.mean(axis=0).A1
        pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        cands: List[str] = []
        seen: set = set()
        for term, _ in pairs:
            norm = _normalize_phrase(term)
            if not norm or len(norm.split()) < 2:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cands.append(norm)
            if len(cands) >= 200:
                break
        return cands

    pos_cands = build_candidates(pos_docs)
    neg_cands = build_candidates(neg_docs)

    try:
        sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def cluster_and_label(texts: List[str], cands: List[str]) -> List[str]:
        if not texts:
            return []
        docs = [t for t in texts if t and t.strip()]
        if not docs:
            return []
        doc_emb = sbert.encode(docs, normalize_embeddings=True)
        k = 2 if len(docs) < 30 else 3
        if len(docs) < 2:
            k = 1
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(doc_emb)
            centers = km.cluster_centers_
        except Exception:
            labels = np.zeros(len(docs), dtype=int)
            centers = np.mean(doc_emb, axis=0, keepdims=True)
            k = 1

        sizes = [int(np.sum(labels == i)) for i in range(k)]
        order = np.argsort(sizes)[::-1]

        cand_list = cands[:200] if cands else []
        if not cand_list:
            return extract_insights_heuristic([Review(text=t) for t in texts])["problems"][:max_items]
        cand_emb = sbert.encode(cand_list, normalize_embeddings=True)

        selected: List[str] = []
        seen: set = set()
        for idx in order:
            centroid = centers[idx:idx+1]
            sims = cosine_similarity(centroid, cand_emb)[0]
            ranked = np.argsort(sims)[::-1]
            for ridx in ranked[:30]:
                cand = cand_list[ridx]
                key = cand.lower()
                if key in seen:
                    continue
                diverse = True
                for ex in selected:
                    w1 = set(cand.lower().split())
                    w2 = set(ex.lower().split())
                    if len(w1 & w2) / max(1, len(w1 | w2)) >= 0.6:
                        diverse = False
                        break
                if not diverse:
                    continue
                seen.add(key)
                selected.append(cand)
                break
            if len(selected) >= max_items:
                break
        return selected[:max_items]

    problems = cluster_and_label(neg_docs, neg_cands)
    strengths = cluster_and_label(pos_docs, pos_cands)

    problems = _clean_insight_list(problems, polarity="negative")[:max_items]
    strengths = _clean_insight_list(strengths, polarity="positive")[:max_items]
    return {"problems": problems, "strengths": strengths}


def _rule_based_clarify(phrase: str, target_lang: str) -> str:
    p = phrase.strip()
    low = p.lower()
    # English rules
    if target_lang == "en":
        if low.startswith("not worth") or "not worth" in low:
            return "Poor value for money"
        if "arrived damaged" in low or "item arrived damaged" in low or "damaged on arrival" in low:
            return "Damaged on arrival"
        if "perfectly will" in low or "will order" in low or "order again" in low:
            return "Will order again"
        if "too dim" in low or "make it brighter" in low or "brighter" in low:
            return "Low brightness"
    # Russian rules
    if target_lang == "ru":
        if low.startswith("не стоит") or "не стоит своих денег" in low or "not worth" in low:
            return "Не стоит своих денег"
        if "чтобы" in low and "ярч" in low:
            return "Недостаточная яркость"
        if "яркост" in low and ("мала" in low or "низка" in low or "слаб" in low or "тускл" in low):
            return "Слабая яркость"
        if "поврежд" in low and ("товар" in low or "упаков" in low):
            return "Повреждение при доставке"
        if "буду заказывать" in low or "закажу ещё" in low or "ещё закажу" in low or "повторно" in low:
            return "Буду заказывать снова"
    return p


def clarify_insights(reviews: List[Review], insights: Dict[str, List[str]], hf_token: Optional[str] = None) -> Dict[str, List[str]]:
    # Choose target language by majority of reviews
    ru_count = sum(1 for r in reviews if (r.language or "") == "ru")
    en_count = sum(1 for r in reviews if (r.language or "") == "en")
    target_lang = "ru" if ru_count >= en_count else "en"

    def apply_rules(items: List[str]) -> List[str]:
        out: List[str] = []
        seen: set = set()
        for it in items:
            lab = _rule_based_clarify(it, target_lang)
            lab = _normalize_phrase(lab)
            lab = re.sub(r"\s+", " ", lab).strip()
            if not lab:
                continue
            key = lab.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(lab)
        return out

    problems = apply_rules(insights.get("problems", []))
    strengths = apply_rules(insights.get("strengths", []))

    if not hf_token:
        return {"problems": problems, "strengths": strengths}

    # If token provided, ask model to rewrite for clarity
    try:
        from services import _hf_text2text  # local import to avoid cycles
        if target_lang == "ru":
            prompt = (
                "Переформулируй пункты ниже в короткие, ясные формулировки (2–5 слов), "
                "без лишних глаголов, без 'чтобы', без неполных оборотов. "
                "Верни JSON с полями problems и strengths (списки строк).\n\n"
            )
        else:
            prompt = (
                "Rewrite the items below into short, clear labels (2–5 words), "
                "no filler verbs, no incomplete phrases. Return JSON with fields problems and strengths.\n\n"
            )
        items_block = json.dumps({"problems": problems, "strengths": strengths}, ensure_ascii=False)
        gen = _hf_text2text(prompt + items_block, model="google/flan-t5-large", token=hf_token, max_new_tokens=256)
        obj = json.loads(re.search(r"\{[\s\S]*\}", gen).group(0))
        p2 = [str(x).strip() for x in obj.get("problems", []) if str(x).strip()]
        s2 = [str(x).strip() for x in obj.get("strengths", []) if str(x).strip()]
        # final cleanup
        p2 = _clean_insight_list(p2, polarity="negative")[:3]
        s2 = _clean_insight_list(s2, polarity="positive")[:3]
        return {"problems": p2 or problems, "strengths": s2 or strengths}
    except Exception:
        return {"problems": problems, "strengths": strengths}


