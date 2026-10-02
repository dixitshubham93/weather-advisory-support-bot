import re
content = open('backend/nodes.py', encoding='utf-8').read()
found = re.findall(r'"SOP-\d+"|\'SOP-\d+\'', content)
print('Found in nodes.py:', found)
print('---')
# Show any lines containing SOP-
for i, line in enumerate(content.split('\n'), 1):
    if 'SOP-' in line:
        print(f"  Line {i}: {line.rstrip()}")
