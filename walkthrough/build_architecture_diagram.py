"""Draw the selected E08 pipeline precisely. Requires reportlab and pypdfium2."""
from pathlib import Path
import math
from reportlab.graphics.shapes import Drawing,Rect,Line,Polygon,String
from reportlab.graphics import renderPDF,renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'walkthrough'

def main():
    fonts=Path('/System/Library/Fonts/Supplemental')
    if (fonts/'Arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('Arial',str(fonts/'Arial.ttf')))
        pdfmetrics.registerFont(TTFont('ArialBold',str(fonts/'Arial Bold.ttf')))
        regular,bold='Arial','ArialBold'
    else:regular,bold='Helvetica','Helvetica-Bold'
    d=Drawing(1920,1080)
    ink=HexColor('#172C43');muted=HexColor('#43556B');edge=HexColor('#59738F')
    d.add(Rect(0,0,1920,1080,fillColor=HexColor('#F7F9FC'),strokeColor=None))
    def text(x,y,s,size=28,b=False,color=ink,anchor='start'):
        d.add(String(x,y,s,fontName=bold if b else regular,fontSize=size,fillColor=color,textAnchor=anchor))
    def box(x,y,w,h,title,lines,fill='#FFFFFF'):
        d.add(Rect(x,y,w,h,rx=16,ry=16,fillColor=HexColor(fill),strokeColor=HexColor('#A6B6C8'),strokeWidth=2))
        text(x+22,y+h-40,title,30,True)
        for i,line in enumerate(lines):text(x+22,y+h-83-i*35,line,25,color=muted)
    def arrow(points):
        for a,b in zip(points,points[1:]):d.add(Line(*a,*b,strokeColor=edge,strokeWidth=4))
        a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=16;width=8
        p1=(b[0]-length*math.cos(angle)+width*math.sin(angle),b[1]-length*math.sin(angle)-width*math.cos(angle))
        p2=(b[0]-length*math.cos(angle)-width*math.sin(angle),b[1]-length*math.sin(angle)+width*math.cos(angle))
        d.add(Polygon([*b,*p1,*p2],fillColor=edge,strokeColor=edge))
    text(70,989,'Final retrieval architecture',49,True)
    text(70,938,'Selected E08 system   |   SemEval 2026 Task 8 Subtask A',29,color=muted)
    box(70,445,290,190,'Input',['Latest user question','Earlier conversation','Assigned collection'])
    box(440,415,420,250,'Prepare two queries',['Original latest question','Standalone rewrite','Qwen2.5-0.5B-Instruct'],'#EAF1F9')
    box(1000,650,330,175,'SPLADE search',['Learned term weights','Both query views'],'#EAF1F9')
    box(1000,285,330,175,'Dense search',['BGE meaning vectors','Both query views'],'#EAF1F9')
    box(1450,465,400,195,'Combine rankings',['Reciprocal rank fusion','Up to 4 ranked lists','Keep 100 candidates'],'#EAF1F9')
    box(1450,170,400,155,'Return 10 passages',['Ranked original passage IDs'])
    arrow([(360,540),(440,540)])
    arrow([(860,540),(925,540),(925,737),(1000,737)])
    arrow([(860,540),(925,540),(925,372),(1000,372)])
    arrow([(1330,737),(1390,737),(1390,563),(1450,563)])
    arrow([(1330,372),(1390,372),(1390,563),(1450,563)])
    arrow([(1650,465),(1650,325)])
    text(70,91,'Each query view searches only the assigned collection; each search returns up to 100 candidates.',25,color=muted)
    text(70,48,'The selected system retrieves passages. BM25 and rerankers were tested as separate comparisons.',25,color=muted)
    pdf=OUT/'ARCHITECTURE.pdf';svg=OUT/'ARCHITECTURE.svg';png=OUT/'ARCHITECTURE.png'
    renderPDF.drawToFile(d,str(pdf));renderSVG.drawToFile(d,str(svg))
    p=pdfium.PdfDocument(pdf);p[0].render(scale=1).to_pil().save(png)
    print(png)

if __name__=='__main__':main()
