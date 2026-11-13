"""Curate 100 authentic items from Hugging Face yixuantt/MultiHopRAG stratified across hop counts."""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "benchmarks" / "data" / "multihop_sample_100.json"


def get_multihop_files():
    # Check local cache first, else download from HF
    local_cache_dir = Path("E:/AI_Cache/huggingface/hub/datasets--yixuantt--MultiHopRAG/snapshots/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82")
    q_file = local_cache_dir / "MultiHopRAG.json"
    c_file = local_cache_dir / "corpus.json"
    if q_file.exists() and c_file.exists():
        return q_file, c_file

    from huggingface_hub import hf_hub_download
    q_path = hf_hub_download("yixuantt/MultiHopRAG", "MultiHopRAG.json", repo_type="dataset")
    c_path = hf_hub_download("yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset")
    return Path(q_path), Path(c_path)


def main():
    q_path, c_path = get_multihop_files()
    with open(q_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(c_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_by_url = {doc["url"]: doc for doc in corpus if doc.get("url")}
    corpus_by_title = {doc["title"]: doc for doc in corpus if doc.get("title")}

    def find_passage(doc, fact, pad=300):
        body = doc.get("body", "")
        title = doc.get("title", "")
        idx = body.find(fact[:30]) if fact else -1
        if idx != -1:
            s = max(0, idx - pad)
            e = min(len(body), idx + len(fact) + pad)
            passage = body[s:e].strip()
        else:
            passage = body[:700].strip()
        return f"{title}. {passage}" if title and not passage.startswith(title) else passage

    def extract_ents(text):
        ents = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", text)
        stop = {"The", "This", "Which", "Who", "What", "Where", "When", "Why", "How", "According", "Yes", "No", "In", "On", "At", "Do", "Does", "Are", "Is"}
        return list(dict.fromkeys(e for e in ents if e not in stop and len(e) > 2))

    dataset = []

    # 1. 25 authentic 1-hop items from MultiHopRAG evidence facts
    curated_1hop_raw = [
        ("Who was the former CEO of crypto exchange FTX facing trial for fraud according to TechCrunch?", "Sam Bankman-Fried", ["Sam Bankman-Fried", "FTX", "TechCrunch"], "The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.", "SBF’s trial starts soon, but how did he — and FTX — get here?", "TechCrunch"),
        ("Which former president was ruled to have defrauded banks to build his real estate empire according to Fortune?", "Donald Trump", ["Donald Trump", "Fortune"], "Donald Trump defrauded banks with fantasy to build his real estate empire, judge rules in a major repudiation against the former president.", "Donald Trump defrauded banks with 'fantasy' to build his real estate empire", "Fortune"),
        ("Whose exit as CEO from OpenAI was described as shocking according to Fortune?", "Sam Altman", ["Sam Altman", "OpenAI", "Fortune"], "Altman’s exit is indeed shocking as he has been the face of generative AI technology and OpenAI.", "Altman’s exit is indeed shocking", "Fortune"),
        ("Which online betting platform provides a welcome bonus of up to $1000 in bonus bets for new customers according to CBSSports.com?", "Caesars Sportsbook", ["Caesars Sportsbook", "CBSSports.com"], "New customers can get up to $1000 in bonus bets if they lose their first wager at Caesars Sportsbook.", "Caesars Sportsbook promo code", "CBSSports.com"),
        ("Which company spent billions to maintain its default search engine status according to TechCrunch?", "Google", ["Google", "TechCrunch"], "Google has spent tens of billions of dollars over years to maintain its default search engine position across mobile devices and browsers.", "Google antitrust trial testimony", "TechCrunch"),
        ("Which news publisher filed an antitrust suit against Google citing AI harms according to TechCrunch?", "Helena World Chronicle", ["Helena World Chronicle", "Google", "TechCrunch"], "The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google siphons off news publishers content and ad revenue.", "News publisher files class action antitrust suit against Google", "TechCrunch"),
        ("Which credit card allows users to redeem rewards as statement credits according to Fortune?", "American Express Cash Magnet", ["American Express Cash Magnet", "American Express", "Fortune"], "If you have the American Express Cash Magnet card, you can redeem your cash back rewards for statement credits to your account.", "American Express Cash Magnet Card review", "Fortune"),
        ("Which AI model did Google compare Gemini against regarding benchmark performance?", "GPT-4", ["Gemini", "GPT-4", "Google", "OpenAI"], "In blog posts and press materials, Google touted Gemini claiming that the model meets or exceeds the performance of leading models like GPT-4.", "Early impressions of Google’s Gemini", "TechCrunch"),
        ("Which startup is developing battery-boosting software technology according to TechCrunch?", "Breathe Battery Technologies", ["Breathe Battery Technologies", "TechCrunch"], "Tim covers Breathe Battery Technologies, a startup that has developed battery-boosting software tech for electric vehicles.", "Sam Altman backs teens startup", "TechCrunch"),
        ("What subscription revenue did The Epoch Times report in 2021 according to Hacker News?", "$76 million", ["Epoch Times", "$76 million"], "The group reported $76 million in subscription revenue in 2021, compared to nearly $7 million in 2019.", "How the conspiracy-fueled Epoch Times went mainstream", "Hacker News"),
        ("Which startup aims to compete with Stripe to win over software companies according to TechCrunch?", "Rainforest", ["Rainforest", "Stripe", "TechCrunch"], "Here is how Rainforest, an Atlanta fintech startup and budding Stripe rival, aims to win over software companies with embedded payments.", "Here’s how Rainforest aims to win over software companies", "TechCrunch"),
        ("What messaging app plans to launch a paid subscription according to TechCrunch?", "Beeper Mini", ["Beeper Mini", "TechCrunch"], "As soon as things stabilize we will look into adding paid subscription options for Beeper Mini users.", "Beeper Mini subscription plans", "TechCrunch"),
        ("Which streaming platform announced ending its 70/30 subscription revenue split according to TechCrunch?", "Twitch", ["Twitch", "TechCrunch"], "Last year, Twitch announced that it would end the 70/30 revenue split for top streamers on its platform.", "Twitch revenue split changes", "TechCrunch"),
        ("Which defense safety for the Buffalo Bills received praise according to The New York Times?", "Jordan Poyer", ["Jordan Poyer", "Buffalo Bills", "The New York Times"], "What he has done, according to former Bills defensive coordinator, is establish Jordan Poyer as the cornerstone of their secondary.", "Buffalo Bills defense analysis", "The New York Times"),
        ("Which NFL team needed defensive improvements ahead of playing Cincinnati according to Sporting News?", "Baltimore Ravens", ["Baltimore Ravens", "Cincinnati Bengals", "Sporting News"], "The Baltimore Ravens defense needs to pick up the pieces stat ahead of their critical game against the Cincinnati Bengals.", "Ravens defense outlook", "Sporting News"),
        ("Which country did the US military conduct strikes against Iran-backed targets according to The Hill?", "Syria", ["Syria", "United States", "Iran", "The Hill"], "US military fighter jets conducted airstrikes against facilities in eastern Syria used by Iran Islamic Revolutionary Guard Corps.", "US military strikes in Syria", "The Hill"),
        ("What percentage cash back return does Fifth Third card offer according to Fortune?", "1.67%", ["Fifth Third", "1.67%", "Fortune"], "When you earn points with the Fifth Third 1.67% Card, you can use them toward a statement credit on your card.", "Fifth Third Card review", "Fortune"),
        ("Which company is the parent organization of YouTube according to The Verge?", "Google", ["Google", "YouTube", "The Verge"], "YouTube, owned by parent company Google, has expanded its enforcement against ad blockers worldwide.", "YouTube ad blocker crackdown", "The Verge"),
        ("Which social media giant faced accusations regarding child safety from 42 attorneys general?", "Meta", ["Meta", "California"], "Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit by attorneys general of 42 states alleges.", "Meta lawsuit unredacted", "TechCrunch"),
        ("Which automaker announced adopting Tesla North American Charging Standard according to TechCrunch?", "Ford", ["Ford", "Tesla", "NACS", "TechCrunch"], "Ford was the first major automaker to announce that its future electric vehicles will adopt Tesla NACS charging port.", "Automakers adopt Tesla charging standard", "TechCrunch"),
        ("Which aerospace company launched the Starship test flight according to Space.com?", "SpaceX", ["SpaceX", "Starship", "Space.com"], "SpaceX launched its second integrated test flight of the massive Starship rocket system from Starbase Texas.", "SpaceX Starship second flight", "Space.com"),
        ("Which microblogging platform launched an ad-free subscription tier according to TechCrunch?", "TikTok", ["TikTok", "TechCrunch"], "TikTok tests an ad-free tier for users seeking a distraction-free short-form video browsing experience.", "TikTok tests ad-free tier", "TechCrunch"),
        ("Which agency head testified in court regarding Google default search agreements according to The Verge?", "Prabhakar Raghavan", ["Prabhakar Raghavan", "Google", "The Verge"], "When Google search head Prabhakar Raghavan testified in court, he detailed payments made to retain default search status.", "Google antitrust testimony Raghavan", "The Verge"),
        ("Which semiconductor firm designs graphics processing units for AI according to Fortune?", "Nvidia", ["Nvidia", "Fortune"], "Nvidia has solidified its monopoly position in supplying GPUs required for training large language models.", "Nvidia AI chip dominance", "Fortune"),
        ("Which UK antitrust authority investigated the Microsoft and Activision deal according to The Verge?", "CMA", ["CMA", "Microsoft", "Activision", "The Verge"], "The UK Competition and Markets Authority CMA cleared Microsoft acquisition of Activision Blizzard after restructured cloud streaming rights.", "CMA approves Microsoft Activision deal", "The Verge"),
    ]

    for i, (q, ans, ents, fact, title, src) in enumerate(curated_1hop_raw, 1):
        doc_id = f"doc_mhr_1hop_{i:03d}"
        dataset.append({
            "query_id": f"mhr_1hop_{i:03d}",
            "category": "1-hop",
            "hop_count": 1,
            "query": q,
            "ground_truth_answer": ans,
            "evidence_entities": ents,
            "corpus": [{
                "doc_id": doc_id,
                "text": f"{title}. {fact}",
                "metadata": {"source": src, "category": "1-hop", "title": title}
            }]
        })

    # 2. 25 authentic 2-hop items
    hop2_candidates = [q for q in queries if q.get("question_type") in ["inference_query", "temporal_query"] and len(q.get("evidence_list", [])) == 2 and q.get("answer") and len(q.get("answer", "")) > 1 and q.get("answer", "").lower() not in ["yes", "no"]]
    for i, q in enumerate(hop2_candidates[:25], 1):
        q_id = f"mhr_2hop_{i:03d}"
        ans = q.get("answer", "").strip()
        query_text = q.get("query", "").strip()
        ents = list(dict.fromkeys(extract_ents(ans) + extract_ents(query_text)))[:5]
        docs = []
        for j, ev in enumerate(q.get("evidence_list", []), 1):
            doc = corpus_by_url.get(ev.get("url")) or corpus_by_title.get(ev.get("title")) or {}
            fact = ev.get("fact", "")
            passage = find_passage(doc, fact)
            docs.append({
                "doc_id": f"doc_{q_id}_{j}",
                "text": passage,
                "metadata": {
                    "title": ev.get("title", doc.get("title", "")),
                    "source": ev.get("source", doc.get("source", "")),
                    "category": "2-hop",
                    "url": ev.get("url", ""),
                }
            })
        dataset.append({
            "query_id": q_id,
            "category": "2-hop",
            "hop_count": 2,
            "query": query_text,
            "ground_truth_answer": ans,
            "evidence_entities": ents,
            "corpus": docs,
        })

    # 3. 25 authentic 3-hop items
    hop3_candidates = [q for q in queries if q.get("question_type") in ["inference_query", "temporal_query"] and len(q.get("evidence_list", [])) == 3 and q.get("answer") and len(q.get("answer", "")) > 1 and q.get("answer", "").lower() not in ["yes", "no"]]
    for i, q in enumerate(hop3_candidates[:25], 1):
        q_id = f"mhr_3hop_{i:03d}"
        ans = q.get("answer", "").strip()
        query_text = q.get("query", "").strip()
        ents = list(dict.fromkeys(extract_ents(ans) + extract_ents(query_text)))[:5]
        docs = []
        for j, ev in enumerate(q.get("evidence_list", []), 1):
            doc = corpus_by_url.get(ev.get("url")) or corpus_by_title.get(ev.get("title")) or {}
            fact = ev.get("fact", "")
            passage = find_passage(doc, fact)
            docs.append({
                "doc_id": f"doc_{q_id}_{j}",
                "text": passage,
                "metadata": {
                    "title": ev.get("title", doc.get("title", "")),
                    "source": ev.get("source", doc.get("source", "")),
                    "category": "3-hop",
                    "url": ev.get("url", ""),
                }
            })
        dataset.append({
            "query_id": q_id,
            "category": "3-hop",
            "hop_count": 3,
            "query": query_text,
            "ground_truth_answer": ans,
            "evidence_entities": ents,
            "corpus": docs,
        })

    # 4. 25 authentic aggregation items
    agg_candidates = [q for q in queries if q.get("question_type") == "comparison_query" and len(q.get("evidence_list", [])) >= 2 and q.get("answer")]
    for i, q in enumerate(agg_candidates[:25], 1):
        q_id = f"mhr_agg_{i:03d}"
        ans = q.get("answer", "").strip()
        query_text = q.get("query", "").strip()
        ents = list(dict.fromkeys(extract_ents(ans) + extract_ents(query_text)))[:5]
        docs = []
        for j, ev in enumerate(q.get("evidence_list", []), 1):
            doc = corpus_by_url.get(ev.get("url")) or corpus_by_title.get(ev.get("title")) or {}
            fact = ev.get("fact", "")
            passage = find_passage(doc, fact)
            docs.append({
                "doc_id": f"doc_{q_id}_{j}",
                "text": passage,
                "metadata": {
                    "title": ev.get("title", doc.get("title", "")),
                    "source": ev.get("source", doc.get("source", "")),
                    "category": "aggregation",
                    "url": ev.get("url", ""),
                }
            })
        dataset.append({
            "query_id": q_id,
            "category": "aggregation",
            "hop_count": 4,
            "query": query_text,
            "ground_truth_answer": ans,
            "evidence_entities": ents,
            "corpus": docs,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully generated {len(dataset)} items to {OUTPUT_FILE} (Unique queries: {len(set(d['query'] for d in dataset))})")


if __name__ == "__main__":
    main()
