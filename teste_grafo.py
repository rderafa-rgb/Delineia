import streamlit as st
import os

st.title("Teste de Diagnóstico Graphviz")

# 1. Mostra o PATH atual para vermos se a pasta bin está lá
st.write("### 1. Verificação de Caminho")
path_atual = os.environ.get('PATH', '')
st.code(path_atual)

# 2. Tenta desenhar um gráfico simples
st.write("### 2. Teste de Renderização")
try:
    graph = """
    digraph {
        rankdir=LR;
        A [label="Funcionou!", style=filled, fillcolor=green];
        B [label="Graphviz", style=filled, fillcolor=lightblue];
        B -> A;
    }
    """
    st.graphviz_chart(graph) # Sem parâmetros extras para testar o básico
    st.success("✅ O Graphviz está funcionando perfeitamente!")
except Exception as e:
    st.error(f"❌ Falha crítica: {e}")