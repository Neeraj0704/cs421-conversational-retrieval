"""Build the rubric-aligned Word report from the reviewed Markdown source.

Render with the document skill's render_docx.py and promote its PDF after visual QA.
Only presentation artifacts are changed; frozen retrieval/evaluation code stays intact.
"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]

def set_attr(parent,tag,attrs):
    e=parent.find(qn(tag))
    if e is None:e=OxmlElement(tag);parent.append(e)
    for k,v in attrs.items():e.set(qn(k),str(v))
    return e

def main():
    doc=Document()
    sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.72);sec.bottom_margin=Inches(.70)
    sec.left_margin=Inches(.78);sec.right_margin=Inches(.78)
    sec.header_distance=Inches(.3);sec.footer_distance=Inches(.3)
    normal=doc.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(11)
    normal.font.color.rgb=RGBColor(0,0,0)
    normal.paragraph_format.line_spacing=1.1
    normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',21),('Heading 1',14),('Heading 2',11.5)]:
        style=doc.styles[name];style.font.name='Arial';style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0)
        style.font.bold=True
        fonts=style.element.get_or_add_rPr().find(qn('w:rFonts'))
        if fonts is not None:
            for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
                fonts.attrib.pop(qn('w:'+attr),None)
        style.paragraph_format.keep_with_next=True
        style.paragraph_format.space_before=Pt(13 if name!='Title' else 0)
        style.paragraph_format.space_after=Pt(7)
        style.paragraph_format.line_spacing=1.04
    title_pr=doc.styles['Title'].element.get_or_add_pPr()
    borders=title_pr.find(qn('w:pBdr'))
    if borders is not None:title_pr.remove(borders)
    sec.different_first_page_header_footer=True
    header=sec.header.paragraphs[0];header.text='CS 421   Conversational passage retrieval'
    header.style='Normal'
    for r in header.runs:r.font.size=Pt(8);r.font.color.rgb=RGBColor(0,0,0)
    for footer in [sec.footer,sec.first_page_footer]:
        p=footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        r=p.add_run('CS 421  |  ');r.font.size=Pt(8)
        r=p.add_run();f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');r._r.addnext(f)
    doc.core_properties.title='Improving conversational passage retrieval'
    doc.core_properties.author='Neeraj Vijayakumar Pattanashetti'
    doc.core_properties.subject='SemEval 2026 Task 8 Subtask A experiment report'
    doc.core_properties.keywords='conversational retrieval, MTRAGEval, SPLADE, dense retrieval'
    lines=(ROOT/'report/FINAL_PROJECT_REPORT.md').read_text().splitlines()
    i=0; tables=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- pagebreak -->':doc.add_page_break();i+=1;continue
        if line.startswith('# '):doc.add_paragraph(line[2:],style='Title');i+=1;continue
        if line.startswith('## '):doc.add_paragraph(line[3:],style='Heading 1');i+=1;continue
        if line.startswith('### '):doc.add_paragraph(line[4:],style='Heading 2');i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip());i+=1
            rows=[[x.strip() for x in s.strip('|').split('|')] for s in [block[0],*block[2:]]]
            tables+=1
            captions={1:'Table 1  All validation comparisons',2:'Table 2  Local final comparisons on 332 judged tasks',3:'Table 3  Published final task scores from the overview'}
            cap=doc.add_paragraph(captions[tables]);cap.paragraph_format.keep_with_next=True
            for r in cap.runs:r.font.size=Pt(11);r.font.bold=True
            n=len(rows[0]);widths={1:[.60,5.29,1.05],2:[2.46,1.12,1.12,1.12,1.12],3:[4.44,1.25,1.25]}[tables]
            table=doc.add_table(rows=1,cols=n);table.autofit=False;table.alignment=WD_TABLE_ALIGNMENT.CENTER
            for col,w in zip(table.columns,widths):col.width=Inches(w)
            tblpr=table._tbl.tblPr
            border=OxmlElement('w:tblBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:
                e=OxmlElement('w:'+edge)
                for key,value in [('val','single'),('sz','4'),('color','D9D9D9')]:e.set(qn('w:'+key),value)
                border.append(e)
            tblpr.append(border)
            margins=OxmlElement('w:tblCellMar')
            for edge,w in [('top',55),('bottom',55),('left',90),('right',90)]:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),str(w));e.set(qn('w:type'),'dxa');margins.append(e)
            tblpr.append(margins)
            for ri,values in enumerate(rows):
                row=table.rows[0] if ri==0 else table.add_row()
                if ri==0:set_attr(row._tr.get_or_add_trPr(),'w:tblHeader',{'w:val':'true'})
                set_attr(row._tr.get_or_add_trPr(),'w:cantSplit',{})
                for ci,(cell,value,w) in enumerate(zip(row.cells,values,widths)):
                    cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    if ri==0:set_attr(cell._tc.get_or_add_tcPr(),'w:shd',{'w:fill':'E8EDF2'})
                    p=cell.paragraphs[0]
                    p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
                    p.paragraph_format.line_spacing=1.06
                    p.paragraph_format.keep_with_next=False
                    p.alignment=WD_ALIGN_PARAGRAPH.CENTER if (tables==1 and ci==n-1) or (tables in [2,3] and ci>0) else WD_ALIGN_PARAGRAPH.LEFT
                    r=p.add_run(value);r.font.name='Arial';r.font.size=Pt(11);r.font.bold=ri==0;r.font.color.rgb=RGBColor(0,0,0)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
            continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','<!--')):
            paragraph.append(lines[i].strip());i+=1
        p=doc.add_paragraph(' '.join(paragraph))
        if re.match(r'^\[\d+\]',line):
            p.paragraph_format.space_after=Pt(6)
            for r in p.runs:r.font.size=Pt(11)
        elif line in ['Neeraj Vijayakumar Pattanashetti','CS 421 Natural Language Processing','November 25 2026']:
            p.paragraph_format.space_after=Pt(2)
            for r in p.runs:r.font.size=Pt(10)
    target=ROOT/'report/FINAL_PROJECT_REPORT.docx';doc.save(target)
    print(target)

if __name__=='__main__':main()
