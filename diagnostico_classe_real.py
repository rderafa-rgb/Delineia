# -*- coding: utf-8 -*-
"""
DIAGNÓSTICO DA CLASSE GEMINI (Simulação do Streamlit)
Executa a função exata que o botão do app chama.
"""
import os
import sys

# 1. Simular o st.secrets ANTES de importar o pipeline
# Isso engana a classe para ela achar que está no Streamlit
import streamlit as st
if not hasattr(st, "secrets"):
    # Tenta pegar da variável de ambiente ou pede input
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY não encontrada nas variáveis de ambiente.")
        api_key = input("Cole sua API KEY aqui: ").strip()
    
    # Mock do objeto secrets
    st.secrets = {"GEMINI_API_KEY": api_key}

# 2. Agora importa a classe real do seu projeto
try:
    print("📥 Importando research_pipeline...")
    from research_pipeline import GeminiQueryGenerator
    print("✅ Importação com sucesso.")
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)

# 3. Dados Simulados (Exatamente como sai do export_utils)
metrics_teste = {
    'jaccard': 0.383,
    'qtd_1': 400,
    'qtd_2': 660,
    'exclusivos_antigos': ['Action research', 'Adult Learning', 'Autonomy'],
    'exclusivos_novos': ['21st century skills', 'Active learning', 'Augmented reality'],
    'comuns': ['Literacy', 'Pedagogy', 'Psychology']
}

# 4. Execução do Teste
print("\n🤖 Iniciando teste da função 'generate_evolution_analysis'...")

try:
    # Instancia a classe
    generator = GeminiQueryGenerator()
    
    # Chama a função (testando com Genero)
    print("⏳ Enviando prompt para o Gemini...")
    resultado = generator.generate_evolution_analysis(
        metrics=metrics_teste, 
        nome_aluno="Emilli", 
        genero="Feminino" # Testando o parâmetro que faltava
    )
    
    print("\n" + "="*50)
    print("RESULTADO GERADO PELA CLASSE:")
    print("="*50)
    print(resultado)
    print("="*50)
    print("\n✅ TESTE BEM SUCEDIDO: A lógica da classe está perfeita.")

except Exception as e:
    print(f"\n❌ ERRO DURANTE A EXECUÇÃO: {e}")
    import traceback
    traceback.print_exc()