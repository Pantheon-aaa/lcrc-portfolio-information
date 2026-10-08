"""Create a standalone, editable TeX rendering of the reviewed English notes."""
from pathlib import Path
import re
ROOT=Path(__file__).parent

def escape(s):
    chars={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
           '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    s=''.join(chars.get(c,c) for c in s)
    return re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m[1]+'}',s)

def run():
    source=(ROOT/'docs/THEORY_NOTES.md').read_text(encoding='utf8')
    head=r'''\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,mathtools}
\usepackage[margin=0.85in]{geometry}
\usepackage[colorlinks=true,linkcolor=blue,hypertexnames=false]{hyperref}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\allowdisplaybreaks
\title{Information-Dependent Boundary Response\\\large Conditional Tail Portfolio Decisions: Research Derivations}
\author{Local research draft}
\date{October 8, 2026}
\begin{document}
\maketitle
'''
    parts=re.split(r'(\$\$.*?\$\$)',source,flags=re.S);body=[]
    for part in parts:
        if part.startswith('$$'):
            math=part[2:-2].strip()
            body.append('\\begin{equation}\n'+math+'\n\\end{equation}\n' if r'\tag{' in math else '\\[\n'+math+'\n\\]\n')
        else:
            for line in part.splitlines():
                if line.startswith('# '):continue
                if line.startswith('### '):body.append('\\subsection*{'+escape(line[4:])+'}')
                elif line.startswith('## '):body.append('\\section{'+escape(re.sub(r'^\d+\.\s*','',line[3:]))+'}')
                else:body.append(escape(line))
    target=ROOT/'docs/THEORY_NOTES.tex';target.write_text(head+'\n'.join(body)+'\n\\end{document}\n',encoding='utf8')
    print(target)

if __name__=='__main__':run()
