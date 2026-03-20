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
    
    # 1. Backticks PRIMEIRO: `texto` → <font name="Courier">texto</font>
    # Assim isolamos os termos técnicos de busca antes de processar o resto
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)

    # 2. Bold: **texto** → <b>texto</b>
    # O [^<>]+ impede que a regex quebre as tags <font> geradas acima
    text = re.sub(r'\*\*([^<>]+?)\*\*', r'<b>\1</b>', text)
    
    # 3. Italic: *texto* → <i>texto</i>
    # O (?=\S) e (?<=\S) exigem que o texto itálico comece e termine colado em letras.
    # O [^*<>]+ impede parear truncamentos (ex: Entomolog*) com outros asteriscos distantes.
    text = re.sub(r'\*(?=\S)([^*<>]+?)(?<=\S)\*', r'<i>\1</i>', text)
    
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

def create_institutional_footer(styles) -> list:
    """
    Cria rodapé institucional com licença CC para os PDFs.
    Retorna lista de elementos para adicionar ao story.
    """
    import os
    
    footer_elements = []
    footer_elements.append(Spacer(1, 0.5*cm))
    footer_elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#d1d5db')))
    footer_elements.append(Spacer(1, 0.2*cm))
    
    # Tenta carregar imagem CC
    cc_path = "assets/cc_by_nc_nd.png"
    
    try:
        if os.path.exists(cc_path):
            cc_img = Image(cc_path, width=2.2*cm, height=0.77*cm)
            
            footer_text = Paragraph(
                "<b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br/>"
                "Pesquisa de Doutorado - PPGIE / UFRGS<br/>"
                "<font size='7'>© 2025 Rafael Antunes dos Santos - Licenciado sob CC BY-NC-ND 4.0</font>",
                styles['footer']
            )
            
            footer_table = Table(
                [[cc_img, footer_text]],
                colWidths=[2.5*cm, 14.5*cm]
            )
            footer_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),
                ('LEFTPADDING', (1, 0), (1, 0), 10),
            ]))
            footer_elements.append(footer_table)
        else:
            raise FileNotFoundError()
    except:
        footer_elements.append(Paragraph(
            "<b>Delinéia</b> - Sistema de Apoio ao Delineamento de Escopo Temático<br/>"
            "Pesquisa de Doutorado - PPGIE / UFRGS<br/>"
            "<font size='8'>© 2025 Rafael Antunes dos Santos - CC BY-NC-ND 4.0</font>",
            styles['footer']
        ))
    
    return footer_elements

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
    story.extend(create_institutional_footer(styles))
        
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
    analise_ia: str = None,
    nodes_info: dict = None
) -> bytes:
    """
    Gera PDF do relatório de comparação de delineamentos com mapas hierárquicos.
    """
    import tempfile
    import os
    
    buffer = BytesIO()
    nodes_info = nodes_info or {}
    
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

    # Variáveis para arquivos temporários
    mapa_novos = None
    mapa_antigos = None
    mapa_comuns = None
    
    # ==================== FUNÇÃO: GERAR MAPA HIERÁRQUICO ====================
    def create_hierarchical_map(concepts: list, color_scheme: str = "blue") -> str:
        """Gera imagem PNG do mapa hierárquico usando Graphviz."""
        if not concepts or len(concepts) == 0:
            return None
        
        try:
            import graphviz
        except ImportError:
            return None
        
        # Classificar por nível
        levels_6 = {i: [] for i in range(6)}
        for c in concepts:
            if c in nodes_info:
                try:
                    lvl = int(float(nodes_info[c].get('level', 5)))
                    lvl = min(max(lvl, 0), 5)
                    score = nodes_info[c].get('score', 0)
                    levels_6[lvl].append((c, score))
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
        
        labels = ["L0", "L1", "L2", "L3", "L4", "L5"]
        
        # Criar grafo
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='TB', bgcolor='white', dpi='120', size='7,5')
        dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='9', margin='0.1,0.05')
        
        total = 0
        niveis_com_dados = []
        
        for lvl in range(6):
            if levels_6[lvl]:
                niveis_com_dados.append(lvl)
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    for c, _ in levels_6[lvl]:
                        node_label = f"{c[:25]}..." if len(c) > 25 else c
                        node_label = f"{node_label}\n({labels[lvl]})"
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
        try:
            with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
                temp_path = f.name
            
            dot.render(temp_path, format='png', cleanup=True)
            return temp_path + '.png'
        except Exception as e:
            print(f"Erro ao gerar mapa: {e}")
            return None
    
    # ==================== 1. CAPA ====================
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Delinéia", styles['title']))
    story.append(Paragraph("Relatório de Comparação de Delineamentos", styles['subtitle']))
    story.append(Spacer(1, 0.3*cm))
    
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
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))
    
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
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 3. CONTEXTO ====================
    story.append(Paragraph("Contexto dos Delineamentos", styles['heading']))
    
    story.append(Paragraph("<b>Delineamento A (Anterior)</b>", styles['body_left']))
    story.append(Paragraph(f"Tema: {clean_text(meta_antigo.get('aluno_tema', 'N/A'))}", styles['body_left']))
    story.append(Paragraph(f"Questão de pesquisa: {clean_text(meta_antigo.get('aluno_questao', 'N/A'))}", styles['body_left']))
    story.append(Spacer(1, 0.15*cm))
    
    story.append(Paragraph("<b>Delineamento B (Atual)</b>", styles['body_left']))
    story.append(Paragraph(f"Tema: {clean_text(meta_novo.get('aluno_tema', 'N/A'))}", styles['body_left']))
    story.append(Paragraph(f"Questão de pesquisa: {clean_text(meta_novo.get('aluno_questao', 'N/A'))}", styles['body_left']))
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 4. LEGENDA ====================
    story.append(Paragraph("Legenda: Níveis de Abstração (OpenAlex)", styles['heading']))
    
    legenda_texto = """
    <b>L0 (Raiz):</b> Grandes áreas do conhecimento (ex: Medicine, Psychology)<br/>
    <b>L1 (Area):</b> Disciplinas amplas (ex: Biology, Education)<br/>
    <b>L2 (Campo):</b> Campos de estudo (ex: Genetics, Pedagogy)<br/>
    <b>L3 (Subcampo):</b> Especializações (ex: Molecular biology)<br/>
    <b>L4 (Topico):</b> Tópicos específicos (ex: Gene expression)<br/>
    <b>L5 (Especifico):</b> Termos muito específicos (ex: CRISPR, PCR)<br/><br/>
    <i>Níveis baixos (L0-L2) = conceitos abrangentes</i><br/>
    <i>Níveis altos (L4-L5) = conceitos específicos</i><br/>
    <i>Conceitos são introduzidos nos mapas hierárquicos segundo a declaração de relevância presente em Score.</i>
    """
    story.append(Paragraph(legenda_texto, styles['body_left']))
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 5. CONCEITOS NOVOS ====================
    novos = metrics.get('exclusivos_novos', [])
    if novos:
        story.append(Paragraph(f"Conceitos Novos ({len(novos)} termos)", styles['heading']))
        
        mapa_novos = create_hierarchical_map(novos, "green")
        if mapa_novos and os.path.exists(mapa_novos):
            try:
                story.append(Image(mapa_novos, width=16*cm, height=10*cm, kind='proportional'))
                story.append(Paragraph(f"<i>Top 5 conceitos de {len(novos)} novidades, por nível de abstração.</i>", styles['footer']))
            except:
                pass
        story.append(Spacer(1, 0.2*cm))
    
    # ==================== 6. CONCEITOS REMOVIDOS ====================
    antigos = metrics.get('exclusivos_antigos', [])
    if antigos:
        story.append(Paragraph(f"Conceitos Removidos ({len(antigos)} termos)", styles['heading']))
        
        mapa_antigos = create_hierarchical_map(antigos, "red")
        if mapa_antigos and os.path.exists(mapa_antigos):
            try:
                story.append(Image(mapa_antigos, width=16*cm, height=10*cm, kind='proportional'))
                story.append(Paragraph(f"<i>Top 5 conceitos de {len(antigos)} removidos, por nível de abstração.</i>", styles['footer']))
            except:
                pass
        story.append(Spacer(1, 0.2*cm))
    
    # ==================== 7. NÚCLEO ESTÁVEL ====================
    comuns = metrics.get('comuns', [])
    if comuns:
        story.append(Paragraph(f"Núcleo Estavel ({len(comuns)} termos)", styles['heading']))
        
        mapa_comuns = create_hierarchical_map(comuns, "blue")
        if mapa_comuns and os.path.exists(mapa_comuns):
            try:
                story.append(Image(mapa_comuns, width=16*cm, height=10*cm, kind='proportional'))
                story.append(Paragraph(f"<i>Top 5 conceitos de {len(comuns)} do núcleo estável, por nível de abstração.</i>", styles['footer']))
            except:
                pass
        story.append(Spacer(1, 0.2*cm))
    
    # ==================== 7. PARECER DA IA ====================
    if analise_ia:
        story.append(Paragraph("Parecer da Orientação Artificial", styles['heading']))
        
        analise_limpa = clean_markdown_for_pdf(analise_ia)
        paragrafos = analise_limpa.split('\n\n')
        for p in paragrafos:
            p = p.strip()
            if p:
                story.append(Paragraph(clean_text(p), styles['body']))
                story.append(Spacer(1, 0.1*cm))
    
    # ==================== 8. RODAPÉ ====================
    story.extend(create_institutional_footer(styles))
    
    # Build PDF
    doc.build(story)

    # Limpar arquivos temporários após o build
    for temp_file in [mapa_novos, mapa_antigos, mapa_comuns]:
        try:
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass
    
    buffer.seek(0)
    return buffer.getvalue()

def generate_evaluation_pdf(form_data: dict, avaliacao_data: dict) -> bytes:
    """
    Gera PDF com as respostas do formulário de avaliação.
    Medida de transparência e ética.
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
    
    # ==================== 1. CAPA ====================
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Delinéia", styles['title']))
    story.append(Paragraph("Comprovante de Avaliação", styles['subtitle']))
    story.append(Spacer(1, 0.3*cm))
    
    # Info do participante
    info_data = [
        ['Participante:', clean_text(form_data.get('nome', 'N/A'))],
        ['E-mail:', clean_text(form_data.get('email', 'N/A'))],
        ['Data:', avaliacao_data.get('timestamp', 'N/A')[:19].replace('T', ' ')]
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
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))
    
    # ==================== 2. PERGUNTAS LIKERT ====================
    story.append(Paragraph("Respostas do Formulário de Avaliação", styles['heading']))
    
    # Mapeamento das perguntas
    perguntas_likert = {
        'q1': "F2.1. Usar o Delinéia melhora a minha capacidade de escolha de palavras-chave",
        'q2': "F2.2. Usar o Delinéia aumenta minha produtividade na definição do projeto",
        'q3': "F2.3. O Delinéia é útil para delimitar meu projeto de pesquisa",
        'q4': "F2.4. O Delinéia me ajuda a posicionar meu projeto na literatura",
        'q5': "F2.5. O Delinéia é fácil de usar",
        'q6': "F2.6. A interação com o Delinéia é clara e compreensível",
        'q7': "F2.7. A navegação entre as funcionalidades é intuitiva",
        'q8': "F2.8. As análises e sugestões são relevantes para meu projeto",
        'q9': "F2.9. A avaliação gerada pela IA é construtiva",
        'q10': "F2.10. As chaves de busca oferecidas são precisas para o meu tema",
        'q11': "F2.11. O grafo de coocorrências me ajudou a visualizar relações",
        'q12': "F2.12. O Delinéia me ajudou a formular perguntas mais precisas",
        'q13': "F2.13. O relatório em PDF é adequado para apresentar ao orientador",
        'q14': "F2.14. O tempo gasto compensa os resultados obtidos",
        'q15': "F2.15. Eu pretendo usar o Delinéia em projetos futuros",
        'q16': "F2.16. Eu usaria o Delinéia em diferentes fases da pesquisa",
        'q17': "F2.17. Eu confio nas análises geradas pelo Delinéia",
        'q18': "F2.18. Eu me sinto confortável em basear decisões com os resultados",
        'q19': "F2.19. O design da interface é agradável",
        'q20': "F2.20. O tempo de processamento do relatório foi adequado",
    }
    
    # Tabela de respostas Likert
    likert_data = [['Pergunta', 'Resposta']]
    for key, pergunta in perguntas_likert.items():
        resposta = avaliacao_data.get(key, '-')
        likert_data.append([pergunta, resposta])
    
    likert_table = Table(likert_data, colWidths=[13*cm, 4*cm])
    likert_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(likert_table)
    story.append(Spacer(1, 0.4*cm))
    
    # ==================== 3. NPS ====================
    story.append(Paragraph("Net Promoter Score (NPS)", styles['heading']))
    nps = avaliacao_data.get('nps', 0)
    nps_cat = avaliacao_data.get('nps_category', '-')
    story.append(Paragraph(f"<b>F2.21. Recomendação:</b> {nps}/10 ({nps_cat})", styles['body_left']))
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 4. AUTOAVALIAÇÃO ====================
    story.append(Paragraph("Autoavaliação de Segurança", styles['heading']))
    q26 = avaliacao_data.get('q26', '-')
    story.append(Paragraph(f"<b>F2.26. Nível de segurança após o uso:</b> {q26}", styles['body_left']))
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 5. PERFIL ====================
    story.append(Paragraph("Perfil do Participante", styles['heading']))
    
    perfil_data = [
        ['F2.27. Nível acadêmico:', avaliacao_data.get('q27', '-')],
        ['F2.28. Experiência com bibliometria:', avaliacao_data.get('q28', '-')],
        ['F2.29. Área do conhecimento:', avaliacao_data.get('q29', '-')],
        ['F2.30. Tempo de uso:', avaliacao_data.get('q30', '-')],
    ]
    
    perfil_table = Table(perfil_data, colWidths=[7*cm, 10*cm])
    perfil_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    story.append(perfil_table)
    story.append(Spacer(1, 0.3*cm))
    
    # ==================== 6. RESPOSTAS ABERTAS ====================
    story.append(Paragraph("Respostas Abertas", styles['heading']))
    
    abertas = {
        'q22': "F2.22. O que você mais gostou no Delinéia?",
        'q23': "F2.23. O que poderia ser melhorado?",
        'q24': "F2.24. Funcionalidades que gostaria de ver no futuro:",
        'q25': "F2.25. Como usou/pretende usar os resultados na pesquisa?",
    }
    
    for key, pergunta in abertas.items():
        resposta = avaliacao_data.get(key, '-') or '-'
        story.append(Paragraph(f"<b>{pergunta}</b>", styles['body_left']))
        story.append(Paragraph(clean_text(resposta), styles['body']))
        story.append(Spacer(1, 0.2*cm))
    
    # ==================== 7. CONSENTIMENTO ====================
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Consentimento", styles['heading']))
    
    tcle = "Aceito" if avaliacao_data.get('tcle_aceite', False) else "Não aceito"
    continuidade = "Sim" if avaliacao_data.get('aceite_continuidade', False) else "Não"
    
    story.append(Paragraph(f"<b>TCLE:</b> {tcle}", styles['body_left']))
    story.append(Paragraph(f"<b>Aceite para continuidade da pesquisa:</b> {continuidade}", styles['body_left']))
    
    # ==================== 8. RODAPÉ ====================
    story.extend(create_institutional_footer(styles))
    
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