"""Extract text from the user's Response Letter docx (docx = zip of XML)."""
import re
import zipfile

p = r'd:\project ML\Response Letter.docx'
with zipfile.ZipFile(p) as z:
    xml = z.read('word/document.xml').decode('utf-8')

# paragraph-level split, then strip tags
paras = re.split(r'</w:p>', xml)
out = []
for para in paras:
    texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', para)
    line = ''.join(texts).strip()
    if line:
        out.append(line)
print(f'--- {len(out)} non-empty paragraphs ---')
for i, line in enumerate(out):
    print(f'[{i}] {line}')
