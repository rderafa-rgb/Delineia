# -*- coding: utf-8 -*-
"""
PDF Generator - Sistema Delinéia
Gera relatório PDF personalizado com todos os elementos da trilha de aprendizagem.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable
)
from reportlab.lib import colors
from io import BytesIO
import textwrap
import re

# ==================== FUNÇÃO LIMPEZA ========================

def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    
    # 1. Tenta manter UTF-8 (aceita acentos e símbolos comuns)
    try:
        # Se o ReportLab aceitar, ótimo. 
        # Mas para evitar quadrados/erros, removemos caracteres fora do Basic Multilingual Plane (onde vivem os emojis)
        # Emojis geralmente estão acima de U+FFFF
        return re.sub(r'[^\u0000-\uFFFF]', '', text)
    except:
        # Fallback seguro: remove caracteres não-latinos mas sem deixar '?' (usando ignore)
        return text.encode('latin-1', 'ignore').decode('latin-1')


# ==================== FUNÇÕES AUXILIARES ====================

def wrap_text(text: str, width: int = 90) -> str:
    """Quebra texto em linhas com largura máxima."""
    lines = []
    for line in text.split('\n'):
        if line.strip():
            lines.append(textwrap.fill(line, width=width))
        else:
            lines.append('')
    return '\n'.join(lines)


def clean_markdown_for_pdf(text: str) -> str:
    """Remove/converte markdown para tags ReportLab."""
    if not text:
        return ""
    
    # Bold: **texto** → <b>texto</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Italic: *texto* → <i>texto</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    
    # Backticks: `texto` → <font name="Courier">texto</font>
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    
    # Remove headers markdown
    text = text.replace('###', '').replace('##', '').replace('#', '')
    
    # Remove linhas com apenas "---"
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    
    # Múltiplas quebras → máximo 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def split_glossary_entries(glossary_text: str) -> list:
    """Divide glossário em entradas individuais."""
    glossary_text = glossary_text.strip()
    entries = []
    
    # Padrão 1: número + ponto + espaços + **
    pattern1 = re.split(r'(?=\d+\.\s+\*\*)', glossary_text)
    if len(pattern1) > 1:
        entries = [e.strip() for e in pattern1 if e.strip() and (re.match(r'^\d+\.', e) or '**' in e)]
    
    # Padrão 2 (fallback): quebra por número + ponto no início de linha
    if not entries:
        pattern2 = re.split(r'(?=^\d+\.)', glossary_text, flags=re.MULTILINE)
        entries = [e.strip() for e in pattern2 if e.strip()]
    
    # Padrão 3 (fallback): quebra por dupla linha
    if not entries:
        pattern3 = glossary_text.split('\n\n')
        entries = [e.strip() for e in pattern3 if e.strip() and ('**' in e or re.match(r'^\d+\.', e))]
    
    return entries if entries else [glossary_text]


def format_badge_for_pdf(badge: str) -> str:
    """
    Formata badge para exibição no PDF.
    Os badges podem vir com ou sem emoji do session_state.
    Exemplos de entrada: "🎯 Explorador" ou "Explorador"
    """
    # Se já tem emoji, retorna como está
    if any(c in badge for c in ['🎯', '🔬', '🧩', '🏆', '💎', '🧭', '📐', '⭐']):
        return badge
    
    # Caso contrário, adiciona emoji
    badge_emojis = {
        'explorador': '🎯',
        'exploradora': '🎯',
        'pesquisador': '🔬',
        'pesquisadora': '🔬',
        'seletor': '🧩',
        'seletora': '🧩',
        'delineador': '🏆',
        'delineadora': '🏆',
        'avaliador': '💎',
        'avaliadora': '💎'
    }
    
    badge_lower = badge.lower().strip()
    emoji = badge_emojis.get(badge_lower, '🏅')
    return f"{emoji} {badge}"

# ==================== ESTILOS ====================

def create_styles():
    """Cria e retorna todos os estilos do documento."""
    styles = getSampleStyleSheet()
    
    custom_styles = {
        'title': ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        
        'subtitle': ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER,
            spaceAfter=20
        ),
        
        'heading': ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ),
        
        'subheading': ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#374151'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ),
        
        'body': ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14
        ),
        
        'body_left': ParagraphStyle(
            'BodyLeft',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=8,
            leading=14
        ),
        
        'glossary_entry': ParagraphStyle(
            'GlossaryEntry',
            parent=styles['BodyText'],
            fontSize=9,
            alignment=TA_LEFT,
            spaceAfter=10,
            spaceBefore=4,
            leading=13,
            leftIndent=10
        ),
        
        'code': ParagraphStyle(
            'Code',
            parent=styles['Code'],
            fontSize=8,
            leftIndent=10,
            rightIndent=10,
            spaceAfter=10,
            spaceBefore=8,
            backColor=colors.HexColor('#f3f4f6'),
            borderColor=colors.HexColor('#d1d5db'),
            borderWidth=1,
            borderPadding=8,
            fontName='Courier',
            leading=12
        ),
        
        'badge': ParagraphStyle(
            'Badge',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#059669'),
            fontName='Helvetica-Bold'
        ),
        
        'footer': ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ),
        
        'caption': ParagraphStyle(
            'Caption',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER,
            spaceAfter=6
        ),
        
        'highlight': ParagraphStyle(
            'Highlight',
            parent=styles['Normal'],
            fontSize=10,
            backColor=colors.HexColor('#ecfdf5'),
            borderColor=colors.HexColor('#10b981'),
            borderWidth=1,
            borderPadding=8,
            leading=14
        )
    }
    
    return custom_styles

# ==================== GERADOR PRINCIPAL ====================

def generate_pdf_report(
    form_data: dict,
    result: dict,
    selected_concepts: list = None,
    suggested_keywords: list = None,
    suggested_strings: dict = None,
    badges: list = None
) -> bytes:
    """
    Gera PDF completo do relatório Delinéia.
    
    Parâmetros:
    -----------
    form_data : dict
        Dados do formulário inicial (nome, email, tema, questão, palavras-chave, timestamp)
    result : dict
        Resultados do processamento (full_report, glossary, graph_interpretation, etc.)
    selected_concepts : list
        Lista de conceitos selecionados pelo usuário
    suggested_keywords : list
        Lista de dicts com sugestões de palavras-chave [{term_pt, term_en, description}, ...]
    suggested_strings : dict
        Dict com strings de busca {ampla: {string, objetivo}, focada: {...}}
    badges : list
        Lista de distintivos conquistados
    
    Retorna:
    --------
    bytes : Conteúdo do PDF
    """
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    styles = create_styles()
    story = []
    
    # Valores padrão
    selected_concepts = selected_concepts or []
    suggested_keywords = suggested_keywords or []
    suggested_strings = suggested_strings or {}
    badges = badges or []
    
    # ==================== 1. CAPA ====================
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Delinéia", styles['title']))
    story.append(Paragraph("Relatório de Delineamento de Escopo Temático", styles['subtitle']))
    story.append(Spacer(1, 0.5*cm))
    
    # Info do aluno
    info_data = [
        ['Participante:', form_data.get('nome', 'N/A')],
        ['E-mail:', form_data.get('email', 'N/A')],
        ['Data:', form_data.get('timestamp', 'N/A')]
    ]
    
    info_table = Table(info_data, colWidths=[3.5*cm, 13.5*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Distintivos conquistados (se houver)
    if badges:
        badges_text = "Distintivos: " + " | ".join([format_badge_for_pdf(b) for b in badges])
        story.append(Paragraph(badges_text, styles['badge']))
        story.append(Spacer(1, 0.3*cm))
    
    # ==================== 2. DADOS FORNECIDOS ====================
    story.append(Paragraph("Dados do Projeto", styles['heading']))
    
    story.append(Paragraph(f"<b>Tema:</b> {form_data.get('tema', 'N/A')}", styles['body_left']))
    story.append(Paragraph(f"<b>Questão de pesquisa:</b> {form_data.get('questao', 'N/A')}", styles['body_left']))
    story.append(Paragraph(f"<b>Palavras-chave iniciais:</b> {form_data.get('palavras_chave', 'N/A')}", styles['body_left']))
    
    # Nível de confiança (se disponível)
    confianca = form_data.get('confianca', form_data.get('nivel_confianca', ''))
    if confianca:
        story.append(Paragraph(f"<b>Nível de confiança inicial:</b> {confianca}", styles['body_left']))
    
    # ==================== 3. AVALIAÇÃO DO PROJETO ====================
    story.append(Paragraph("Avaliação do Projeto", styles['heading']))
    
    report_text = clean_markdown_for_pdf(result.get('full_report', 'Não disponível'))
    paragraphs = report_text.split('\n\n')
    
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['body']))
    
    # ==================== 4. MÉTRICAS DA BUSCA ====================
    story.append(Paragraph("Métricas da Análise Bibliométrica", styles['heading']))
    
    graph_stats = result.get('graph_stats', {})
    
    metrics_data = [
        ['Métrica', 'Contagem'],
        ['Artigos analisados', str(result.get('articles_count', 0))],
        ['Conceitos identificados', str(graph_stats.get('nodes', 0))],
        ['Coocorrências', str(graph_stats.get('edges', 0))]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[10*cm, 7*cm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.5*cm))
    
    # ==================== 5. GRAFO DE COOCORRÊNCIAS ====================
    story.append(Paragraph("Grafo de Coocorrência de Conceitos", styles['heading']))
    
    story.append(Paragraph(
        "O grafo abaixo ilustra as relações entre os principais conceitos identificados na busca por literatura. "
        "Os nós representam conceitos e as arestas indicam a coocorrência entre eles nos artigos científicos recuperados.",
        styles['body']
    ))
    
    if result.get('visualization_path'):
        try:
            img = Image(result['visualization_path'], width=16*cm, height=12*cm)
            story.append(img)
            
            # Legenda do grafo
            n_concepts = len(selected_concepts) if selected_concepts else graph_stats.get('nodes', 0)
            story.append(Paragraph(
                f"Grafo com {n_concepts} conceitos selecionados",
                styles['caption']
            ))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Erro ao carregar visualização: {str(e)}", styles['body']))
    else:
        story.append(Paragraph("⚠️ Visualização não disponível", styles['body']))
    
    # ==================== 6. CONCEITOS SELECIONADOS ====================
    if selected_concepts:
        story.append(Paragraph("Conceitos Selecionados", styles['heading']))
        
        story.append(Paragraph(
            f"Você selecionou <b>{len(selected_concepts)}</b> conceitos para delinear o escopo do seu projeto:",
            styles['body']
        ))
        
        # Criar tabela de conceitos
        concepts_per_row = 3
        concept_rows = []
        for i in range(0, len(selected_concepts), concepts_per_row):
            row = selected_concepts[i:i+concepts_per_row]
            # Preencher com vazios se necessário
            while len(row) < concepts_per_row:
                row.append('')
            concept_rows.append(row)
        
        if concept_rows:
            concepts_table = Table(concept_rows, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
            concepts_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfdf5')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#10b981')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#065f46'))
            ]))
            story.append(concepts_table)
    
    # ==================== 7. GLOSSÁRIO ====================
    story.append(Paragraph("Glossário de Conceitos", styles['heading']))
    
    glossary_text = result.get('glossary', 'Não disponível')
    
    if glossary_text and glossary_text != 'Não disponível' and 'Poucos conceitos' not in glossary_text:
        glossary_clean = clean_markdown_for_pdf(glossary_text)
        entries = split_glossary_entries(glossary_clean)
        
        if entries and len(entries) > 1:
            for entry in entries:
                if entry.strip():
                    story.append(Paragraph(entry.strip(), styles['glossary_entry']))
        else:
            story.append(Paragraph(glossary_clean, styles['body']))
    else:
        story.append(Paragraph(glossary_text or "Glossário não disponível", styles['body']))
    
    # ==================== 8. INTERPRETAÇÃO DO GRAFO ====================
    story.append(Paragraph("Interpretação Personalizada do Grafo", styles['heading']))
    
    interpretation_text = clean_markdown_for_pdf(result.get('graph_interpretation', 'Não disponível'))
    interp_paragraphs = interpretation_text.split('\n\n')
    
    for para in interp_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['body']))
    
    # ==================== 9. SUGESTÕES DE PALAVRAS-CHAVE ====================
    if suggested_keywords:
        story.append(Paragraph("Sugestões de Palavras-chave", styles['heading']))
        
        story.append(Paragraph(
            "Com base na análise do seu projeto, sugerimos as seguintes palavras-chave para enriquecer sua busca:",
            styles['body']
        ))
        
        for i, kw in enumerate(suggested_keywords, 1):
            term_pt = kw.get('term_pt', 'N/A')
            term_en = kw.get('term_en', 'N/A')
            description = kw.get('description', '')
            
            entry_text = f"<b>{i}. {term_pt}</b> ({term_en})"
            if description:
                entry_text += f" — {description}"
            
            story.append(Paragraph(entry_text, styles['glossary_entry']))
    
    # ==================== 10. CHAVES DE BUSCA SUGERIDAS ====================
    if suggested_strings:
        story.append(Paragraph("Chaves de Busca Sugeridas", styles['heading']))
        
        story.append(Paragraph(
            "As chaves abaixo foram elaboradas para diferentes estratégias de busca bibliográfica:",
            styles['body']
        ))
        
        string_order = ['ampla', 'focada']
        string_labels = {
            'ampla': ('Chave de busca ampla (exploratória)', 'Usa operadores OR para cobrir o máximo de variações dos conceitos.'),
            'focada': ('Chave de busca focada (conceitos centrais)', 'Cruza os conceitos mais importantes usando AND para alta precisão.'),            
        }
        
        for key in string_order:
            if key in suggested_strings:
                string_data = suggested_strings[key]
                label, desc = string_labels.get(key, (key.title(), ''))
                
                story.append(Paragraph(f"<b>{label}</b>", styles['subheading']))
                story.append(Paragraph(f"<i>{desc}</i>", styles['caption']))
                
                # String em si
                string_text = string_data.get('string', 'N/A')
                wrapped = wrap_text(string_text, 75)
                story.append(Paragraph(
                    f'<font name="Courier" size="8">{wrapped}</font>',
                    styles['code']
                ))
                
                # Objetivo
                objetivo = string_data.get('objetivo', '')
                if objetivo:
                    story.append(Paragraph(f"<b>Objetivo:</b> {objetivo}", styles['body']))
                
                story.append(Spacer(1, 0.3*cm))
    
    # ==================== 11. TRANSPARÊNCIA ====================
    story.append(Paragraph("Transparência: chave de busca usada", styles['heading']))
    
    story.append(Paragraph(
        "Para garantir a reprodutibilidade desta análise, apresentamos a chave de busca original utilizada na consulta à base OpenAlex:",
        styles['body']
    ))
    
    search_string = result.get('search_string', 'N/A')
    wrapped_string = wrap_text(search_string, 75)
    
    story.append(Paragraph(
        f'<font name="Courier" size="8">{wrapped_string}</font>',
        styles['code']
    ))
    
    # Objetivo da busca original
    search_objective = result.get('search_objective', '')
    if search_objective:
        obj_clean = clean_markdown_for_pdf(search_objective)
        story.append(Paragraph(f"<b>Objetivo:</b> {obj_clean}", styles['body']))
    
    # ==================== 12. RODAPÉ ====================
    story.append(Spacer(1, 1*cm))
    
    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#d1d5db'),
        spaceBefore=10,
        spaceAfter=10
    ))
    
    story.append(Paragraph(
        "Relatório gerado pelo Sistema Delinéia",
        styles['footer']
    ))
    story.append(Paragraph(
        "https://delineia.streamlit.app",
        styles['footer']
    ))
        
    # ==================== BUILD ====================
    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        print(f"❌ Erro na geração do PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # PDF de erro
        buffer = BytesIO()
        simple_doc = SimpleDocTemplate(buffer, pagesize=A4)
        error_story = [
            Paragraph("Erro na Geração do Relatório", styles['title']),
            Spacer(1, 1*cm),
            Paragraph(f"Ocorreu um erro: {str(e)}", styles['body']),
            Paragraph("Por favor, tente novamente ou entre em contato com o suporte.", styles['body'])
        ]
        simple_doc.build(error_story)
        return buffer.getvalue()

def generate_comparison_pdf(
    form_data: dict,
    metrics: dict,
    meta_antigo: dict,
    meta_novo: dict,
    analise_ia: str = None
) -> bytes:
    """
    Gera PDF do relatório de comparação de delineamentos com mapas hierárquicos.
    """
    import graphviz
    import tempfile
    import os
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    styles = create_styles()
    story = []
    
    # ==================== FUNÇÃO AUXILIAR: GERAR MAPA GRAPHVIZ ====================
    def create_hierarchical_map_image(concepts: list, nodes_info: dict, color_scheme: str = "blue") -> str:
        """Gera imagem PNG do mapa hierárquico e retorna o caminho do arquivo."""
        if not concepts:
            return None
        
        # Classificar por nível
        levels_6 = {i: [] for i in range(6)}
        for c in concepts[:100]:  # Limita a 100 conceitos
            if c in nodes_info:
                try:
                    lvl = int(float(nodes_info[c].get('level', 5)))
                    lvl = min(max(lvl, 0), 5)
                    levels_6[lvl].append((c, nodes_info[c].get('score', 0)))
                except:
                    levels_6[5].append((c, 0))
            else:
                levels_6[5].append((c, 0))
        
        # Ordenar por score e pegar top 5 por nível
        for lvl in levels_6:
            levels_6[lvl] = sorted(levels_6[lvl], key=lambda x: x[1], reverse=True)[:5]
        
        # Esquemas de cores
        if color_scheme == "green":
            cores = ["#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a"]
        elif color_scheme == "red":
            cores = ["#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#ef4444", "#dc2626"]
        else:  # blue
            cores = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb"]
        
        labels = ["L0: Raiz", "L1: Área", "L2: Campo", "L3: Subcampo", "L4: Tópico", "L5: Específico"]
        
        # Criar grafo
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='TB', bgcolor='white', dpi='150')
        dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='9', margin='0.1,0.05')
        
        total = 0
        niveis_com_dados = []
        
        for lvl in range(6):
            if levels_6[lvl]:
                niveis_com_dados.append(lvl)
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    for c, _ in levels_6[lvl]:
                        node_label = f"{c}\n({labels[lvl]})"
                        s.node(c, label=node_label, fillcolor=cores[lvl])
                        total += 1
        
        # Conexões entre níveis
        for i in range(len(niveis_com_dados) - 1):
            lvl1, lvl2 = niveis_com_dados[i], niveis_com_dados[i+1]
            if levels_6[lvl1] and levels_6[lvl2]:
                dot.edge(levels_6[lvl1][0][0], levels_6[lvl2][0][0], style='dashed', color='#94a3b8', arrowhead='none')
        
        if total == 0:
            return None
        
        # Salvar como PNG temporário
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        dot.render(temp_path.replace('.png', ''), format='png', cleanup=True)
        return temp_path
    
    # ==================== RECUPERAR NODES_INFO ====================
    nodes_info = {}
    # Tenta recuperar de meta_novo ou meta_antigo
    # (Os nodes_info não estão nos metadados do Google Sheets, então usamos um dicionário vazio)
    # TODO: Passar nodes_info como parâmetro da função se disponível
    
    # ==================== 1. CAPA ====================
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Delinéia", styles['title']))
    story.append(Paragraph("Relatório de Comparação de Delineamentos", styles['subtitle']))
    story.append(Spacer(1, 0.4*cm))
    
    # Info do aluno
    info_data = [
        ['Participante:', clean_text(form_data.get('nome', 'N/A'))],
        ['E-mail:', clean_text(form_data.get('email', 'N/A'))],
        ['Data:', form_data.get('timestamp', 'N/A')]
    ]
    
    info_table = Table(info_data, colWidths=[3.5*cm, 13.5*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # ==================== 2. MÉTRICAS ====================
    story.append(Paragraph("Resultados da Comparação", styles['heading']))
    
    jaccard = metrics.get('jaccard', 0)
    qtd_novos = len(metrics.get('exclusivos_novos', []))
    qtd_antigos = len(metrics.get('exclusivos_antigos', []))
    qtd_comuns = len(metrics.get('comuns', []))
    qtd_total = metrics.get('qtd_2', 0)
    
    metricas_data = [
        ['Métrica', 'Valor'],
        ['Similaridade (Jaccard)', f'{jaccard*100:.1f}%'],
        ['Tamanho do Vocabulário (B)', f'{qtd_total} conceitos'],
        ['Conceitos Novos', str(qtd_novos)],
        ['Conceitos Removidos', str(qtd_antigos)],
        ['Núcleo Estável', str(qtd_comuns)]
    ]
    
    metricas_table = Table(metricas_data, colWidths=[8*cm, 8*cm])
    metricas_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
    ]))
    story.append(metricas_table)
    story.append(Spacer(1, 0.4*cm))
    
    # ==================== 3. CONTEXTO DOS DELINEAMENTOS ====================
    story.append(Paragraph("Contexto dos Delineamentos", styles['heading']))
    
    story.append(Paragraph("<b>Delineamento A (Anterior)</b>", styles['body_left']))
    story.append(Paragraph(f"Tema: {clean_text(meta_antigo.get('aluno_tema', 'N/A'))}", styles['body_left']))
    story.append(Paragraph(f"Questão: {clean_text(meta_antigo.get('aluno_questao', 'N/A'))}", styles['body_left']))
    story.append(Spacer(1, 0.2*cm))
    
    story.append(Paragraph("<b>Delineamento B (Atual)</b>", styles['body_left']))
    story.append(Paragraph(f"Tema: {clean_text(meta_novo.get('aluno_tema', 'N/A'))}", styles['body_left']))
    story.append(Paragraph(f"Questão: {clean_text(meta_novo.get('aluno_questao', 'N/A'))}", styles['body_left']))
    story.append(Spacer(1, 0.4*cm))
    
    # ==================== 4. CONCEITOS NOVOS ====================
    novos = metrics.get('exclusivos_novos', [])
    if novos:
        story.append(Paragraph(f"Conceitos Novos ({len(novos)} termos)", styles['heading']))
        novos_text = ", ".join(novos[:40])
        if len(novos) > 40:
            novos_text += f"... e mais {len(novos)-40} conceitos."
        story.append(Paragraph(clean_text(novos_text), styles['body']))
        story.append(Spacer(1, 0.3*cm))
    
    # ==================== 5. CONCEITOS REMOVIDOS ====================
    antigos = metrics.get('exclusivos_antigos', [])
    if antigos:
        story.append(Paragraph(f"Conceitos Removidos ({len(antigos)} termos)", styles['heading']))
        antigos_text = ", ".join(antigos[:40])
        if len(antigos) > 40:
            antigos_text += f"... e mais {len(antigos)-40} conceitos."
        story.append(Paragraph(clean_text(antigos_text), styles['body']))
        story.append(Spacer(1, 0.3*cm))
    
    # ==================== 6. NÚCLEO ESTÁVEL ====================
    comuns = metrics.get('comuns', [])
    if comuns:
        story.append(Paragraph(f"Núcleo Estável ({len(comuns)} termos)", styles['heading']))
        comuns_text = ", ".join(comuns[:40])
        if len(comuns) > 40:
            comuns_text += f"... e mais {len(comuns)-40} conceitos."
        story.append(Paragraph(clean_text(comuns_text), styles['body']))
        story.append(Spacer(1, 0.3*cm))
    
    # ==================== 7. PARECER DA IA ====================
    if analise_ia:
        story.append(PageBreak())
        story.append(Paragraph("Parecer da Orientação Artificial", styles['heading']))
        
        analise_limpa = clean_markdown_for_pdf(analise_ia)
        paragrafos = analise_limpa.split('\n\n')
        for p in paragrafos:
            p = p.strip()
            if p:
                if re.match(r'^\d+\.', p):
                    story.append(Paragraph(clean_text(p), styles['body_left']))
                else:
                    story.append(Paragraph(clean_text(p), styles['body']))
                story.append(Spacer(1, 0.15*cm))
    
    # ==================== 8. RODAPÉ INSTITUCIONAL COM CC ====================
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#d1d5db')))
    story.append(Spacer(1, 0.2*cm))
    
    # Tenta incluir imagem CC
    cc_image_path = "assets/cc_by_nc_nd.png"
    try:
        if os.path.exists(cc_image_path):
            cc_img = Image(cc_image_path, width=2.5*cm, height=0.9*cm)
            
            rodape_data = [
                [cc_img, Paragraph(
                    "<b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br/>"
                    "Pesquisa de Doutorado - PPGIE / UFRGS<br/>"
                    "<font size='7'>© 2025 Rafael Antunes dos Santos - Licenciado sob CC BY-NC-ND 4.0</font>",
                    styles['footer']
                )]
            ]
            rodape_table = Table(rodape_data, colWidths=[3*cm, 14*cm])
            rodape_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),
            ]))
            story.append(rodape_table)
        else:
            raise FileNotFoundError("CC image not found")
    except:
        rodape_text = """
        <b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br/>
        Pesquisa de Doutorado - PPGIE / UFRGS<br/>
        <font size="8">Delinéia © 2025 by Rafael Antunes dos Santos is licensed under CC BY-NC-ND 4.0</font>
        """
        story.append(Paragraph(rodape_text, styles['footer']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==================== FUNÇÃO DE COMPATIBILIDADE ====================

def generate_pdf_report_legacy(form_data: dict, result: dict) -> bytes:
    """
    Função de compatibilidade com assinatura antiga.
    Chama a função principal com parâmetros vazios para os novos campos.
    """
    return generate_pdf_report(
        form_data=form_data,
        result=result,
        selected_concepts=None,
        suggested_keywords=None,
        suggested_strings=None,
        badges=None
    )