import json, sys

filename = "pcb_defect_001.jpg" # ex: pcb_defect_001.jpg

with open('/srv/groups/group7/data/pcb-defects2/annotation/_annotations.coco.json') as f:
    data = json.load(f)

cat_map = {c['id']: c['name'] for c in data['categories']}
img = next(i for i in data['images'] if i['file_name'] == filename)
anns = [a for a in data['annotations'] if a['image_id'] == img['id']]

print(f"image_id: {img['id']}")
for a in anns:
    print(f"  defect: {cat_map[a['category_id']]:20s}  bbox: {a['bbox']}")