import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = 'reports/category_probes_20260524_224530.json'
with open(path, encoding='utf-8') as f:
    data = json.load(f)

print(f"Overall category score: {data['category_score']}")
print()

math_cats = {'arithmetic','algebra','geometry','statistics','unit_conversion',
             'word_problems','trigonometry','calculus','advanced_geometry'}

for cat_result in data['category_results']:
    cat = cat_result['category']
    score = cat_result['score']
    for qr in cat_result.get('question_results', []):
        resp = qr.get('response', '')
        resp_short = resp[:250].replace('\n', ' ')
        rule = qr.get('rule_score')
        print(f"[{cat}] score={score} rule={rule}")
        print(f"  Q: {qr.get('question','')[:90]}")
        print(f"  A: {resp_short}")
        print()
