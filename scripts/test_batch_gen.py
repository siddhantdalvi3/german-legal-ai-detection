"""Test batch generation speed vs single-call."""
import json
import time
import subprocess
import sys

OLLAMA_API = "http://localhost:11434/api/generate"

BATCH_PROMPT = """Du bist Jurist im Bundesministerium. Schreibe 25 kurze, in sich abgeschlossene juristische Texte auf Deutsch (je 1-2 Sätze). Jeder Text behandelt ein anderes Thema. Trenne die Texte mit "---".

Themen:
1. Die Voraussetzungen einer wirksamen Willenserklärung
2. Die Haftung des Verkäufers für Sachmängel nach § 437 BGB
3. Die Anfechtung eines Verwaltungsakts nach § 48 VwVfG
4. Die Grundsätze der Verhältnismäßigkeit im öffentlichen Recht
5. Die Rechtsprechung des BVerfG zur Meinungsfreiheit
6. Die Anforderungen an eine ordnungsgemäße Klageerhebung
7. Die Voraussetzungen des polizeilichen Notstandes
8. Die Rechtsfolgen einer nichtigen Ehe nach § 1313 BGB
9. Die Vergabe öffentlicher Aufträge nach Vergaberecht
10. Die Haftung des Staates nach § 839 BGB
11. Die Auslegung von Steuergesetzen nach der AO
12. Die Voraussetzungen der einstweiligen Anordnung nach § 123 VwGO
13. Die Rechtsstellung des Betriebsrats nach BetrVG
14. Die Voraussetzungen der Pfändung von Arbeitseinkommen
15. Die Wirksamkeit von AGB im Rechtsverkehr
16. Die Haftung des GmbH-Geschäftsführers bei Insolvenzverschleppung
17. Die Anforderungen an einen wirksamen Arbeitsvertrag
18. Die Voraussetzungen der Enteignung nach Art. 14 GG
19. Die Rechtsmittel gegen einen belastenden VA
20. Die Grundsätze der DSGVO
21. Die Auswirkungen des § 242 StGB auf die Polizeiarbeit
22. Die Anforderungen an eine wirksame Kündigung
23. Die Rechtsprechung zur Störerhaftung im Internetrecht
24. Die Regeln zur Geschwindigkeitsüberschreitung
25. Die Voraussetzungen der Unterlassungsklage"""


def test_model(model: str, temperature: float = 0.7):
    print(f"\n=== Testing {model} (temp={temperature}) ===")
    
    import requests
    
    # Test 1: batch generation
    print("Batch generation (25 topics in 1 call)...")
    payload = {
        "model": model,
        "prompt": BATCH_PROMPT,
        "temperature": temperature,
        "stream": False,
    }
    start = time.time()
    resp = requests.post(OLLAMA_API, json=payload, timeout=600)
    elapsed = time.time() - start
    text = resp.json()["response"].strip()
    parts = [p.strip() for p in text.split("---") if p.strip()]
    
    # count non-empty sentences
    total_sents = 0
    for p in parts:
        sents = [s.strip() for s in p.replace("\n", " ").split(".") if len(s.strip()) > 20]
        total_sents += len(sents)
    
    print(f"  Response parts (splits on ---): {len(parts)}")
    print(f"  Estimated sentences: {total_sents}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Effective rate: {total_sents/elapsed:.1f} sent/s ({elapsed/total_sents:.1f} s/sent)")
    print(f"  First 100 chars: {text[:100]!r}...")
    
    # Test 2: single call (current approach)
    print("\nSingle generation (1 topic, current approach)...")
    single_prompt = "Du bist Jurist im Bundesministerium. Schreibe einen formellen juristischen Text auf Deutsch über die Voraussetzungen einer wirksamen Willenserklärung im Bürgerlichen Recht. Verwende präzise juristische Fachsprache. Der Text soll 5-10 Sätze lang sein."
    payload2 = {
        "model": model,
        "prompt": single_prompt,
        "temperature": temperature,
        "stream": False,
    }
    start2 = time.time()
    resp2 = requests.post(OLLAMA_API, json=payload2, timeout=120)
    elapsed2 = time.time() - start2
    text2 = resp2.json()["response"].strip()
    sents2 = [s.strip() for s in text2.replace("\n", " ").split(".") if len(s.strip()) > 20]
    print(f"  Sentences: {len(sents2)}")
    print(f"  Time: {elapsed2:.1f}s")
    print(f"  Rate: {len(sents2)/elapsed2:.1f} sent/s ({elapsed2/len(sents2):.1f} s/sent)")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["qwen2.5:7b", "gemma3:12b"]
    for m in models:
        test_model(m)
