"""
Purga e resguardo (purging / embargo) na partição temporal: quantifica o vazamento residual de rótulo.

Problema: a partição 80/20 corta pela DATA DO PEDIDO, mas o rótulo (atraso) só é conhecido na ENTREGA.
Um pedido de treino feito perto do corte pode ter sido entregue já no período de teste; nesse caso, no
instante do corte o rótulo dele ainda era desconhecido, mas o modelo é treinado com ele.

Configurações (o conjunto de TESTE é sempre o mesmo do experimento principal):
  base      : treino completo (experimento principal)
  purga     : remove do treino pedidos com ORDER_DELIVERED_AT > corte (rótulo ainda desconhecido no corte)
  purga+15  : purga + remove pedidos feitos nos últimos 15 dias antes do corte (resguardo)
  purga+30  : idem, 30 dias
  purga+60  : idem, 60 dias
O corte é a data do primeiro pedido do conjunto de teste.

Mesmos hiperparâmetros, sementes, partição e bootstrap pareado do restante do projeto.
Para cada cenário (A e B) e modelo: ROC-AUC, diferença para a configuração base e diferença A - B.

Uso (na pasta do projeto, com o venv ativo):
    python purga_resguardo.py

Saídas: assets/purga_resguardo.csv        (ROC-AUC e diferenças, com IC 95%)
        assets/purga_resguardo_treino.csv (tamanho e taxa de atraso do treino por configuração)
        assets/purga_resguardo_saida.txt  (cópia de tudo o que o script imprime no terminal,
                                           inclusive quantos pedidos de treino foram entregues
                                           depois do corte)
Não altera nenhuma tabela nem nenhum arquivo existente; só lê GOLD_DATASET_A/B e GOLD_PEDIDOS_VENDEDOR.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from script_ml import conectar_snowflake, extrair_dados
from bootstrap_pareado import (particoes, bootstrap, amostrador_linhas, amostrador_vendedores,
                               _linha, MODELOS, CHAVE, TARGET, COL_DATA)
from avaliacao_cruzada import criar_modelos, FEATURES

CONFIGS = [('base', None, False), ('purga', 0, True), ('purga+15', 15, True),
           ('purga+30', 30, True), ('purga+60', 60, True)]   # (nome, dias de resguardo, aplica purga)


def anexar_entrega(df, ev):
    out = df.merge(ev[CHAVE + ['ORDER_DELIVERED_AT']], on=CHAVE, how='left')
    assert len(out) == len(df), "A junção com a base de eventos duplicou ou perdeu linhas"
    assert out['ORDER_DELIVERED_AT'].notna().all(), "Há pedidos sem data de entrega"
    return out


def filtrar_treino(treino, corte, dias, purga):
    if not purga:
        return treino
    manter = pd.to_datetime(treino['ORDER_DELIVERED_AT']) <= corte
    if dias:
        manter &= pd.to_datetime(treino[COL_DATA]) <= corte - pd.Timedelta(days=dias)
    return treino[manter]


def executar(df_a, df_b, ev, n_boot=1000):
    dados = {}
    for cen, df in (('A', df_a), ('B', df_b)):
        treino, teste = particoes(anexar_entrega(df, ev))
        dados[cen] = (treino, teste)
    ya = dados['A'][1][TARGET].astype(int).to_numpy()
    assert np.array_equal(ya, dados['B'][1][TARGET].astype(int).to_numpy())
    assert dados['A'][1][CHAVE].equals(dados['B'][1][CHAVE]), "Conjuntos de teste diferentes entre A e B"
    y = ya
    teste_seller = dados['A'][1]['SELLER_ID']
    corte = pd.to_datetime(dados['A'][1][COL_DATA]).min()
    print(f"\nCorte (primeiro pedido do teste): {corte}")

    # Diagnóstico: quantos pedidos de treino têm rótulo desconhecido no corte
    tr = dados['A'][0]
    tarde = pd.to_datetime(tr['ORDER_DELIVERED_AT']) > corte
    print(f"Treino: {len(tr)} pedidos; {int(tarde.sum())} ({100 * tarde.mean():.2f}%) foram entregues "
          f"depois do corte (rótulo desconhecido no corte).")

    probs, tamanhos = {}, []
    for nome, dias, purga in CONFIGS:
        for cen in ('A', 'B'):
            treino, teste = dados[cen]
            tr_f = filtrar_treino(treino, corte, dias, purga)
            X_tr = tr_f[FEATURES].astype(float).fillna(0)
            y_tr = tr_f[TARGET].astype(int).to_numpy()
            X_te = teste[FEATURES].astype(float).fillna(0)
            tamanhos.append(dict(config=nome, cenario=cen, n_treino=len(tr_f),
                                 taxa_atraso_treino=y_tr.mean()))
            for mod, modelo in criar_modelos(y_tr).items():
                modelo.fit(X_tr, y_tr)
                probs[(f'{nome}|{cen}', mod)] = modelo.predict_proba(X_te)[:, 1]
            print(f"  {nome} / Cenário {cen}: treino com {len(tr_f)} pedidos")

    est = {k: roc_auc_score(y, p) for k, p in probs.items()}
    nomes = [c[0] for c in CONFIGS]
    linhas = []
    for amost_nome, amostrar in (('linhas', amostrador_linhas(len(y))),
                                 ('vendedores', amostrador_vendedores(teste_seller))):
        print(f"\nBootstrap por {amost_nome}...")
        auc, _ = bootstrap(y, probs, amostrar, n_boot=n_boot)
        for mod in MODELOS:
            for cfg in nomes:
                for cen in ('A', 'B'):
                    k = (f'{cfg}|{cen}', mod)
                    linhas.append(_linha('ROC-AUC', amost_nome, f'{mod}: {cfg} (Cenário {cen})', est[k], auc[k], p=False))
                    if cfg != 'base':
                        kb = (f'base|{cen}', mod)
                        linhas.append(_linha('ROC-AUC', amost_nome, f'{mod}: {cfg} - base (Cenário {cen})',
                                             est[k] - est[kb], auc[k] - auc[kb]))
                ka, kb_ = (f'{cfg}|A', mod), (f'{cfg}|B', mod)
                linhas.append(_linha('ROC-AUC', amost_nome, f'{mod}: A - B ({cfg})',
                                     est[ka] - est[kb_], auc[ka] - auc[kb_]))
    return pd.DataFrame(linhas), pd.DataFrame(tamanhos)


class Tee:
    """Escreve no terminal e no arquivo de saída ao mesmo tempo."""
    def __init__(self, *fluxos):
        self.fluxos = fluxos

    def write(self, s):
        for fl in self.fluxos:
            fl.write(s)

    def flush(self):
        for fl in self.fluxos:
            fl.flush()


if __name__ == "__main__":
    os.makedirs('assets', exist_ok=True)
    arq_saida = open('assets/purga_resguardo_saida.txt', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, arq_saida)
    ctx = conectar_snowflake()
    print("Baixando datasets e base de eventos...")
    df_a = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_A_LEAKAGE")
    df_b = extrair_dados(ctx, "SELECT * FROM GOLD_DATASET_B_POINT_IN_TIME")
    ev = extrair_dados(ctx, "SELECT ORDER_ID, SELLER_ID, ORDER_DELIVERED_AT FROM GOLD_PEDIDOS_VENDEDOR")
    os.makedirs('assets', exist_ok=True)

    tab, tam = executar(df_a, df_b, ev)
    tab.to_csv('assets/purga_resguardo.csv', index=False)
    tam.to_csv('assets/purga_resguardo_treino.csv', index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 80)
    print("\n=== Tamanho do treino por configuração ===")
    print(tam.round(4).to_string(index=False))
    print("\n=== Resultados (reamostragem por linhas) ===")
    print(tab[tab.amostragem == 'linhas'].drop(columns=['amostragem', 'p_bilateral_aprox', 'n_replicas'])
          .round(4).to_string(index=False))
    print("\nFinalizado! CSV salvo em assets/purga_resguardo.csv")
