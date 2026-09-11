"""Build portable HTML with a short main narrative and collapsible supplement."""
from pathlib import Path
import base64,re
from markdown_it import MarkdownIt
root=Path(__file__).resolve().parent
md=MarkdownIt('commonmark',{'html':True}).enable('table')
html=md.render((root/'report.md').read_text())
html+='<details><summary>Supplementary figures, numerical details, and reproduction</summary>'+md.render((root/'supplement.md').read_text())+'</details>'
def embed(m):
 return 'src="data:image/png;base64,'+base64.b64encode((root/m.group(1)).read_bytes()).decode()+'"'
html=re.sub(r'src="(figures/[^\"]+\.png)"',embed,html)
css='''body{font:17px/1.65 Georgia,serif;color:#182330;margin:45px auto;max-width:1040px;padding:0 30px}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.2;color:#133b52}h1{font-size:36px}h2{margin-top:2.4em;font-size:25px}img{width:100%;height:auto;margin:16px 0 0}table{border-collapse:collapse;width:100%;font:14px/1.45 system-ui,sans-serif;margin:25px 0}td,th{padding:10px 12px;border-bottom:1px solid #ced9df;text-align:left}th{background:#ecf3f7}a{color:#00638e}code{font-size:.86em;background:#f0f3f5;padding:2px 4px}pre{padding:16px;background:#f0f3f5;overflow:auto}p:has(>em:only-child){font-size:14px;line-height:1.5;color:#425363}details{margin-top:50px;border-top:1px solid #ced9df;padding-top:20px}summary{font:18px system-ui,sans-serif;cursor:pointer;color:#133b52}@media print{body{font-size:11pt;max-width:none;margin:0}h2{break-after:avoid}img,table{break-inside:avoid}}'''
(root/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>When does delegator scaling help?</title><style>'+css+'</style><main>'+html+'</main></html>')
print(root/'report.html')
