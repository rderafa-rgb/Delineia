# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from typing import List


class ThematicMapAnalyzer:
    """
    Analisador de Mapa Temático baseado em *Strategic Diagram*

    Referências:
    - Aria, M., & Cuccurullo, C. (2017). *bibliometrix: An R-tool for comprehensive
      science mapping analysis.* Journal of Informetrics, 11(4), 959–975.
    - He, Q. (1999). *Knowledge discovery through co-word analysis.* Library Trends,
      48(1), 133–159.
    """

    def __init__(self, graph: nx.Graph, concepts_lists: List[List[str]]):
        """
        Args:
            graph: grafo de coocorrências (NetworkX)
            concepts_lists: lista de listas de conceitos por artigo
        """
        self.G = graph
        self.concepts_lists = concepts_lists
        self.clusters = []
        self.cluster_metrics = None

    def detect_clusters(self, method: str = "louvain", min_size: int = 3):
        """
        Detecta clusters (comunidades) no grafo.
        """
        if self.G is None or len(self.G.nodes()) == 0:
            self.clusters = []
            return self.clusters

        from networkx.algorithms import community

        if method == "louvain":
            # NetworkX não traz Louvain “original” por padrão,
            # usamos greedy_modularity_communities como aproximação
            clusters_raw = list(community.greedy_modularity_communities(self.G))
        else:
            clusters_raw = list(community.greedy_modularity_communities(self.G))

        self.clusters = [set(c) for c in clusters_raw if len(c) >= min_size]
        return self.clusters

    def _build_cooccurrence_matrix(self) -> pd.DataFrame:
        """
        Constrói matriz de coocorrência a partir do grafo.
        Usa o peso da aresta como frequência de coocorrência.
        """
        concepts = sorted(self.G.nodes())
        matrix = pd.DataFrame(0.0, index=concepts, columns=concepts)

        for u, v, data in self.G.edges(data=True):
            w = data.get("weight", 1.0)
            matrix.loc[u, v] = w
            matrix.loc[v, u] = w

        return matrix

    @staticmethod
    def _density_for_cluster(cluster_nodes, matrix: pd.DataFrame) -> float:
        """
        Densidade = média dos links internos do cluster.
        """
        cluster_nodes = list(cluster_nodes)
        if len(cluster_nodes) < 2:
            return 0.0

        values = []
        for i, ci in enumerate(cluster_nodes):
            for j, cj in enumerate(cluster_nodes):
                if i < j:
                    values.append(matrix.loc[ci, cj])

        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _centrality_for_cluster(cluster_nodes, all_nodes, matrix: pd.DataFrame) -> float:
        """
        Centralidade = soma dos links externos do cluster.
        """
        cluster_nodes = set(cluster_nodes)
        values = []
        for ci in cluster_nodes:
            for cj in all_nodes:
                if cj not in cluster_nodes:
                    values.append(matrix.loc[ci, cj])

        return float(np.sum(values)) if values else 0.0

    def analyze_clusters(self):
        """
        Calcula densidade, centralidade e métricas normalizadas
        para cada cluster detectado.
        """
        if not self.clusters:
            self.cluster_metrics = []
            return self.cluster_metrics

        matrix = self._build_cooccurrence_matrix()
        all_nodes = list(matrix.index)

        data = []
        for idx, cluster in enumerate(self.clusters, start=1):
            density = self._density_for_cluster(cluster, matrix)
            centrality = self._centrality_for_cluster(cluster, all_nodes, matrix)

            # Principais conceitos = top 5 pelo grau interno
            subG = self.G.subgraph(cluster)
            degs = sorted(subG.degree(), key=lambda x: x[1], reverse=True)
            main_concepts = [n for n, _ in degs[:5]]

            data.append(
                {
                    "nome": f"Cluster {idx}",
                    "conceitos": sorted(cluster),
                    "densidade": density,
                    "centralidade": centrality,
                    "tamanho": len(cluster),
                    "conceitos_principais": ", ".join(main_concepts),
                }
            )

        df = pd.DataFrame(data)

        # Normalização: centralizar na mediana e dividir pelo máximo absoluto
        for col in ["densidade", "centralidade"]:
            median = df[col].median()
            centered = df[col] - median
            max_abs = centered.abs().max()
            if max_abs == 0:
                norm = 0
            else:
                norm = centered / max_abs
            df[f"{col}_norm"] = norm

        self.cluster_metrics = df
        return self.cluster_metrics

    @staticmethod
    def classify_quadrant(centrality_norm: float, density_norm: float) -> str:
        """
        Classifica cluster em quadrante do mapa temático.
        """
        if centrality_norm >= 0 and density_norm >= 0:
            return "Q1: Motor Themes"
        elif centrality_norm >= 0 and density_norm < 0:
            return "Q2: Basic/Transversal Themes"
        elif centrality_norm < 0 and density_norm >= 0:
            return "Q3: Niche Themes"
        else:
            return "Q4: Emerging/Declining Themes"

    def create_strategic_diagram(self, width: int = 900, height: int = 700) -> go.Figure:
        """
        Cria figura Plotly do mapa temático.
        """
        if self.cluster_metrics is None or len(self.cluster_metrics) == 0:
            fig = go.Figure()
            fig.update_layout(
                title="🗺️ Mapa Temático (nenhum cluster encontrado)",
                xaxis_title="Centralidade",
                yaxis_title="Densidade",
            )
            return fig

        df = self.cluster_metrics.copy()

        fig = go.Figure()

        # Cores por quadrante
        color_map = {
            "Q1: Motor Themes": "rgba(56, 168, 75, 0.8)",          # verde
            "Q2: Basic/Transversal Themes": "rgba(255, 193, 7, 0.8)",  # amarelo
            "Q3: Niche Themes": "rgba(93, 95, 222, 0.8)",          # azul
            "Q4: Emerging/Declining Themes": "rgba(220, 53, 69, 0.8)",  # vermelho
        }

        # Quadrante de cada cluster
        df["Quadrante"] = df.apply(
            lambda row: self.classify_quadrant(
                row["centralidade_norm"], row["densidade_norm"]
            ),
            axis=1,
        )

        for _, row in df.iterrows():
            centrality = row["centralidade_norm"]
            density = row["densidade_norm"]
            size = row["tamanho"]
            color = color_map.get(row["Quadrante"], "rgba(150,150,150,0.7)")

            hover_text = (
                f"<b>{row['nome']}</b><br>"
                f"Quadrante: {row['Quadrante']}<br>"
                f"Tamanho: {row['tamanho']} conceitos<br>"
                f"Densidade: {row['densidade']:.3f} "
                f"({row['densidade_norm']:.2f})<br>"
                f"Centralidade: {row['centralidade']:.3f} "
                f"({row['centralidade_norm']:.2f})<br>"
                f"Principais conceitos: {row['conceitos_principais']}"
            )

            fig.add_trace(
                go.Scatter(
                    x=[centrality],
                    y=[density],
                    mode="markers+text",
                    marker=dict(
                        size=float(np.sqrt(size) * 15.0),
                        color=color,
                        opacity=0.7,
                        line=dict(width=2, color="white"),
                    ),
                    text=row["nome"],
                    textposition="top center",
                    textfont=dict(size=10, color="black"),
                    hovertext=hover_text,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        # Linhas centrais (medianas normalizadas = 0)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=2)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=2)

        # Blocos de anotação dos quadrantes
        annotations = [
            {
                "x": 0.6,
                "y": 0.6,
                "text": "<b>Motor Themes</b><br>(Central + Desenvolvido)",
                "bgcolor": "rgba(100, 200, 100, 0.3)",
            },
            {
                "x": 0.6,
                "y": -0.6,
                "text": "<b>Basic Themes</b><br>(Central + Subdesenvolvido)",
                "bgcolor": "rgba(255, 220, 100, 0.3)",
            },
            {
                "x": -0.6,
                "y": 0.6,
                "text": "<b>Niche Themes</b><br>(Periférico + Desenvolvido)",
                "bgcolor": "rgba(150, 150, 220, 0.3)",
            },
            {
                "x": -0.6,
                "y": -0.6,
                "text": "<b>Emerging/Declining</b><br>(Periférico + Subdesenvolvido)",
                "bgcolor": "rgba(200, 100, 100, 0.3)",
            },
        ]

        for ann in annotations:
            fig.add_annotation(
                x=ann["x"],
                y=ann["y"],
                text=ann["text"],
                showarrow=False,
                font=dict(size=11, color="black"),
                bgcolor=ann["bgcolor"],
                borderpad=4,
            )

        fig.update_layout(
            title={
                "text": "🗺️ Mapa Temático (Strategic Diagram)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            xaxis_title="<b>Centralidade</b> (Relevância global) →",
            yaxis_title="<b>Densidade</b> (Coesão interna) →",
            width=width,
            height=height,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="gray",
                gridcolor="lightgray",
                range=[-1.2, 1.2],
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="gray",
                gridcolor="lightgray",
                range=[-1.2, 1.2],
            ),
        )

        return fig

    def get_interpretation_table(self) -> pd.DataFrame:
        """
        Retorna tabela interpretativa dos clusters.
        """
        if self.cluster_metrics is None or len(self.cluster_metrics) == 0:
            return pd.DataFrame()

        df = self.cluster_metrics.copy()

        df["Quadrante"] = df.apply(
            lambda row: self.classify_quadrant(
                row["centralidade_norm"], row["densidade_norm"]
            ),
            axis=1,
        )

        interpretations = {
            "Q1: Motor Themes": "Temas centrais e bem desenvolvidos. Estruturam o campo de pesquisa.",
            "Q2: Basic/Transversal Themes": "Temas centrais, mas subdesenvolvidos. Tópicos transversais ou básicos.",
            "Q3: Niche Themes": "Temas periféricos, porém bem desenvolvidos. Nichos especializados.",
            "Q4: Emerging/Declining Themes": "Temas periféricos e subdesenvolvidos. Emergentes ou em declínio.",
        }

        df["Interpretação"] = df["Quadrante"].map(interpretations)

        result = df[
            ["nome", "Quadrante", "tamanho", "conceitos_principais", "Interpretação"]
        ].rename(
            columns={
                "nome": "Cluster",
                "tamanho": "Nº Conceitos",
                "conceitos_principais": "Principais Conceitos",
            }
        )

        return result
