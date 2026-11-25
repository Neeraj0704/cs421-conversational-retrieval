"""Typeset the reviewed report source as a clean, self-contained PDF."""
from pathlib import Path
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,CondPageBreak

ROOT=Path(__file__).resolve().parents[1]

def main():
    font_root=Path('/System/Library/Fonts/Supplemental')
    if (font_root/'Arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('ReportArial',str(font_root/'Arial.ttf')))
        pdfmetrics.registerFont(TTFont('ReportArialBold',str(font_root/'Arial Bold.ttf')))
        regular,bold='ReportArial','ReportArialBold'
    else:regular,bold='Helvetica','Helvetica-Bold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportBody',fontName=regular,fontSize=11,leading=14.0,spaceAfter=7,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle('ReportTitle',fontName=bold,fontSize=21,leading=24,spaceAfter=10,keepWithNext=True))
    styles.add(ParagraphStyle('ReportH1',fontName=bold,fontSize=14,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle('ReportH2',fontName=bold,fontSize=11.5,leading=14,spaceBefore=9,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle('ReportMeta',fontName=regular,fontSize=10,leading=12,spaceAfter=2))
    styles.add(ParagraphStyle('ReportCaption',fontName=bold,fontSize=9,leading=12,spaceBefore=5,spaceAfter=5,keepWithNext=False))
    styles.add(ParagraphStyle('ReportCell',fontName=regular,fontSize=9.5,leading=12))
    styles.add(ParagraphStyle('ReportCellCentre',parent=styles['ReportCell'],alignment=1))
    styles.add(ParagraphStyle('ReportCellHead',parent=styles['ReportCell'],fontName=bold))
    styles.add(ParagraphStyle('ReportRef',fontName=regular,fontSize=9,leading=12,spaceAfter=6,splitLongWords=True))
    lines=(ROOT/'report/FINAL_PROJECT_REPORT.md').read_text().splitlines();story=[];i=0;tn=0
    captions={1:'Table 1  Passage collections used in the study',2:'Table 2  Components and fixed settings',3:'Table 3  All validation comparisons',4:'Table 4  Local final comparisons on 332 judged tasks',5:'Table 5  Published final task results from the overview'}
    widths={1:[82,333,85],2:[78,182,240],3:[43,381,76],4:[177,80.75,80.75,80.75,80.75],5:[320,90,90]}
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(Paragraph(escape(line[2:]),styles['ReportTitle']));i+=1;continue
        if line.startswith('## '):
            story.extend([CondPageBreak(90),Paragraph(escape(line[3:]),styles['ReportH1'])]);i+=1;continue
        if line.startswith('### '):
            story.extend([CondPageBreak(65),Paragraph(escape(line[4:]),styles['ReportH2'])]);i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i].strip());i+=1
            rows=[[c.strip() for c in s.strip('|').split('|')] for s in [block[0],*block[2:]]];tn+=1
            story.extend([CondPageBreak(90),Paragraph(captions[tn],styles['ReportCaption'])])
            cells=[]
            for ri,row in enumerate(rows):
                cells.append([Paragraph(escape(c),styles['ReportCellHead'] if ri==0 else styles['ReportCellCentre'] if (tn in [1,3] and ci==len(row)-1) or (tn in [4,5] and ci>0) else styles['ReportCell']) for ci,c in enumerate(row)])
            table=Table(cells,colWidths=widths[tn],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,9)]);continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|')):parts.append(lines[i].strip());i+=1
        text=' '.join(parts)
        style='ReportRef' if text.startswith('[') else 'ReportMeta' if text in ['Neeraj Vijayakumar Pattanashetti','CS 421 Natural Language Processing','November 25 2026'] else 'ReportBody'
        story.append(Paragraph(escape(text),styles[style]))
    def decorate(canvas,doc):
        canvas.setFillColor(colors.black);canvas.setFont(regular,8)
        if doc.page>1:canvas.drawString(56,766,'CS 421   Conversational passage retrieval')
        canvas.drawRightString(556,25,f'CS 421  |  {doc.page}')
    target=ROOT/'report/FINAL_PROJECT_REPORT.pdf'
    SimpleDocTemplate(str(target),pagesize=(612,792),leftMargin=56,rightMargin=56,topMargin=52,bottomMargin=50,title='Improving conversational passage retrieval',author='Neeraj Vijayakumar Pattanashetti').build(story,onFirstPage=decorate,onLaterPages=decorate)
    print(target)

if __name__=='__main__':main()
