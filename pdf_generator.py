# -*- coding: utf-8 -*-

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
import textwrap
import re

def wrap_text(text: str, width: int = 90) -> str:
    """Quebra texto"""
    lines = []
    for line in text.split('\n'):
        if line.strip():
            lines.append(textwrap.fill(line, width=width))
        else:
            lines.append('')
    return '\n'.join(lines)

def clean_markdown_for_pdf(text: str) -> str:
    """Remove markdown"""
    if not text:
        return ""

    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # Headers
    text = text.replace('###', '').replace('##', '').replace('#', '')

    # Múltiplas quebras
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

def split_glossary_entries(glossary_text: str) -> list:
    """
    Divide glossário com MÚLTIPLOS PADRÕES
    """
    glossary_text = glossary_text.strip()
    entries = []

    # PADRÃO 1: número + ponto + espaços + **
    pattern1 = re.split(r'(?=\d+\.\s+\*\*)', glossary_text)
    if len(pattern1) > 1:
        entries = [e.strip() for e in pattern1 if e.strip() and (re.match(r'^\d+\.', e) or '**' in e)]

    # PADRÃO 2 (fallback): quebra por número + ponto no início de linha
    if not entries:
        pattern2 = re.split(r'(?=^\d+\.)', glossary_text, flags=re.MULTILINE)
        entries = [e.strip() for e in pattern2 if e.strip()]

    # PADRÃO 3 (fallback): quebra por dupla linha
    if not entries:
        pattern3 = glossary_text.split('\n\n')
        entries = [e.strip() for e in pattern3 if e.strip() and ('**' in e or re.match(r'^\d+\.', e))]

    return entries if entries else [glossary_text]

def generate_pdf_report(form_data: dict, result: dict) -> bytes:
    """
    Gera PDF LINEAR

    ORDEM FINAL:
    1. Dados
    2. Avaliação
    3. "Com base nas palavras-chave..."
    4. "Então, elaborou-se..."
    5. Métricas
    6. Grafo
    7. Glossário
    8. Interpretação
    9. String
    10. Objetivo
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # ==================== ESTILOS ====================
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=24,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=16
    )

    glossary_entry_style = ParagraphStyle(
        'GlossaryEntry',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=12,
        spaceBefore=4,
        leading=14
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=12,
        spaceBefore=12,
        backColor=colors.HexColor('#f3f4f6'),
        borderColor=colors.HexColor('#d1d5db'),
        borderWidth=1,
        borderPadding=10,
        fontName='Courier',
        leading=13
    )

    # ==================== CONSTRUIR ====================
    story = []

    # 1. CAPA
    story.append(Paragraph("Delineamento por palavras-chave", title_style))
    story.append(Paragraph("Relatório produzido por Inteligência Artificial", subtitle_style))
    story.append(Spacer(1, 0.5*cm))

    # 2. INFO ALUNO
    info_table = Table([
        ['Aluno:', form_data['nome']],
        ['E-mail:', form_data['email']],
        ['Data:', form_data['timestamp']]
    ], colWidths=[4*cm, 13*cm])

    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))

    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))

    # 3. DADOS FORNECIDOS
    story.append(Paragraph("Dados fornecidos", heading_style))
    story.append(Paragraph(f"<b>Tema:</b> {form_data['tema']}", body_style))
    story.append(Paragraph(f"<b>Questão:</b> {form_data['questao']}", body_style))
    story.append(Paragraph(f"<b>Palavras-chave:</b> {form_data['palavras_chave']}", body_style))

    # 4. AVALIAÇÃO DO PROJETO
    story.append(Paragraph("Avaliação do projeto", heading_style))

    report_text = clean_markdown_for_pdf(result.get('full_report', 'Não disponível'))
    paragraphs = report_text.split('\n\n')

    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    story.append(Spacer(1, 0.6*cm))

    # 5. PARÁGRAFO: "Com base nas palavras-chave..."
    transicao_texto = "Com base nas palavras-chave fornecidas, desenvolvemos uma string de busca."
    story.append(Paragraph(transicao_texto, body_style))
    story.append(Spacer(1, 0.4*cm))

    # 6. PARÁGRAFO: "Então, elaborou-se..."
    grafo_texto = """Então, elaborou-se um modelo de visualização. Por favor, considere as palavras-chave disponíveis no grafo e avalie se elas fazem sentido para a formulação de suas estratégias de buscas. O exercício de visualização é uma abstração sobre o tema e pode dizer algo para o delineamento do escopo temático do seu projeto de pesquisa. O modelo é uma rede de coocorrências de conceitos que ilustra os principais termos usados para descrever a literatura científica. Os termos aparecem em língua inglesa, e elaborou-se um pequeno glossário."""
    story.append(Paragraph(grafo_texto, body_style))
    story.append(Spacer(1, 0.6*cm))

    # 7. MÉTRICAS
    metrics_table = Table([
        ['Métrica', 'Valor'],
        ['Artigos analisados', str(result.get('articles_count', 0))],
        ['Conceitos identificados', str(result['graph_stats']['nodes'])],
        ['Conexões no grafo', str(result['graph_stats']['edges'])]
    ], colWidths=[10*cm, 7*cm])

    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
    ]))

    story.append(metrics_table)
    story.append(Spacer(1, 0.6*cm))

    # 8. GRAFO
    story.append(Paragraph("Rede de Coocorrência de Conceitos", heading_style))

    if result.get('visualization_path'):
        try:
            story.append(Image(result['visualization_path'], width=17*cm, height=12.75*cm))
            story.append(Spacer(1, 0.6*cm))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Erro ao carregar: {str(e)}", body_style))
    else:
        story.append(Paragraph("⚠️ Visualização não disponível", body_style))

    # 9. GLOSSÁRIO
    story.append(Paragraph("Glossário de Conceitos", heading_style))

    glossary_text = result.get('glossary', 'Não disponível')

    if glossary_text and glossary_text != 'Não disponível' and 'Poucos conceitos' not in glossary_text:
        glossary_clean = clean_markdown_for_pdf(glossary_text)
        entries = split_glossary_entries(glossary_clean)

        if entries and len(entries) > 1:
            for entry in entries:
                if entry.strip():
                    story.append(Paragraph(entry.strip(), glossary_entry_style))
                    story.append(Spacer(1, 0.25*cm))
        else:
            # Fallback
            story.append(Paragraph(glossary_clean, body_style))
    else:
        story.append(Paragraph(glossary_text, body_style))

    # 10. INTERPRETAÇÃO
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Interpretação do Grafo", heading_style))

    interpretation_text = clean_markdown_for_pdf(result.get('graph_interpretation', 'Não disponível'))
    interp_paragraphs = interpretation_text.split('\n\n')

    for para in interp_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    story.append(Spacer(1, 0.6*cm))

    # 11. STRING
    story.append(Paragraph("String de Busca Utilizada", heading_style))

    search_string = result.get('search_string', 'N/A')
    wrapped_string = wrap_text(search_string, 80)

    story.append(Paragraph(
        f'<font name="Courier" size="9">{wrapped_string}</font>',
        code_style
    ))

    story.append(Spacer(1, 0.4*cm))

    # 12. OBJETIVO
    search_objective = result.get('search_objective', '')

    if search_objective:
        obj_clean = clean_markdown_for_pdf(search_objective)
        obj_paragraphs = obj_clean.split('\n\n')

        for para in obj_paragraphs:
            if para.strip():
                # Adicionar label no primeiro
                if para == obj_paragraphs[0] and not para.lower().startswith('objetivo'):
                    para_final = f"<b>Objetivo:</b> {para}"
                else:
                    para_final = para

                story.append(Paragraph(para_final, body_style))

    # 13. RODAPÉ
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "___",
        ParagraphStyle('Line', parent=styles['Normal'], alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        "Relatório gerado pelo Sistema Delinéia",
        ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
    ))

    # ==================== BUILD ====================
    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

        # PDF erro
        buffer = BytesIO()
        simple_doc = SimpleDocTemplate(buffer, pagesize=A4)
        error_story = [
            Paragraph("Erro na Geração", title_style),
            Spacer(1, 1*cm),
            Paragraph(f"Erro: {str(e)}", body_style)
        ]
        simple_doc.build(error_story)
        return buffer.getvalue()
