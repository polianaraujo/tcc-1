-- Verificação rápida após o `dbt run` (rodar no Snowsight, database TCC_LAKEHOUSE).
-- Valores de referência obtidos reproduzindo o pipeline em pandas com os CSVs do Olist.

USE SCHEMA TCC_LAKEHOUSE.GOLD;

-- 1) Tamanho e taxa de atraso dos datasets  (esperado: 97.817 linhas, 6.548 atrasos ≈ 6,69%)
SELECT 'A' AS cenario, COUNT(*) AS linhas, SUM(target_houve_atraso) AS atrasos,
       ROUND(AVG(target_houve_atraso), 4) AS taxa,
       COUNT(DISTINCT order_id || '|' || seller_id) AS chaves_unicas      -- deve ser = linhas
FROM gold_dataset_a_leakage
UNION ALL
SELECT 'B', COUNT(*), SUM(target_houve_atraso), ROUND(AVG(target_houve_atraso), 4),
       COUNT(DISTINCT order_id || '|' || seller_id)
FROM gold_dataset_b_point_in_time;

-- 2) Efeito da correção do rótulo: atrasos por timestamp x por data  (esperado: 7.845 x 6.548)
SELECT SUM(IFF(order_delivered_at > order_estimated_delivery_at, 1, 0)) AS atrasos_criterio_antigo,
       SUM(is_delayed)                                                  AS atrasos_criterio_novo
FROM gold_pedidos_vendedor
WHERE order_delivered_at IS NOT NULL;

-- 3) Linhas do tempo com uma versão por (vendedor, instante)  (as duas colunas devem ser iguais)
SELECT 'volume' AS linha_do_tempo, COUNT(*) AS versoes, COUNT(DISTINCT seller_id || '|' || feature_timestamp) AS unicas FROM gold_fs_seller_volume
UNION ALL SELECT 'atraso', COUNT(*), COUNT(DISTINCT seller_id || '|' || feature_timestamp) FROM gold_fs_seller_delay
UNION ALL SELECT 'nota',   COUNT(*), COUNT(DISTINCT seller_id || '|' || feature_timestamp) FROM gold_fs_seller_review;
-- esperado ≈ 99.891 / 97.775 / 99.069 versões

-- 4) Teste de consistência temporal do Cenário B: nenhum pedido pode "ver" um atraso
--    de pedido que ainda não tinha sido entregue no instante da compra.  (esperado: 0)
SELECT COUNT(*) AS violacoes
FROM gold_dataset_b_point_in_time b
WHERE b.seller_delay_count > (
    SELECT COALESCE(SUM(p.is_delayed), 0)
    FROM gold_pedidos_vendedor p
    WHERE p.seller_id = b.seller_id
      AND p.order_delivered_at < b.data_do_pedido
);

-- 5) Mesmo teste para o VOLUME: o volume recebido deve ser exatamente o número de pedidos do
--    vendedor com compra ESTRITAMENTE anterior à data do pedido.  (esperado: 0)
SELECT COUNT(*) AS violacoes
FROM gold_dataset_b_point_in_time b
WHERE b.seller_order_volume <> (
    SELECT COUNT(*)
    FROM gold_pedidos_vendedor p
    WHERE p.seller_id = b.seller_id
      AND p.order_purchase_at < b.data_do_pedido
);

-- 6) Mesmo teste para a NOTA MÉDIA: deve ser a média das notas cujas avaliações foram RESPONDIDAS
--    estritamente antes da data do pedido (0 se não houver nenhuma).  (esperado: 0)
SELECT COUNT(*) AS violacoes
FROM gold_dataset_b_point_in_time b
WHERE ABS(b.seller_avg_review_score - (
    SELECT COALESCE(AVG(p.review_score), 0)
    FROM gold_pedidos_vendedor p
    WHERE p.seller_id = b.seller_id
      AND p.review_score IS NOT NULL
      AND p.review_answered_at < b.data_do_pedido
)) > 1e-6;

-- 7) CONTROLE POSITIVO: as mesmas consultas 4, 5 e 6, trocando gold_dataset_b_point_in_time por
--    gold_dataset_a_leakage, devem retornar muitas violações (o Cenário A usa o estado final do
--    vendedor). Para o atraso, espera-se ≈ 79,3 mil (81,06% de 97.817, Tabela de diferenças).
--    Isso mostra que os testes são capazes de detectar vazamento.

-- 8) Quantos registros do Cenário B têm SELLER_AVG_REVIEW_SCORE = 0?  Como as notas vão de 1 a 5, a
--    média só é 0 quando o vendedor não tem nenhuma avaliação respondida antes da compra (o 0 vem do
--    COALESCE do modelo). Resultado por conjunto, com o mesmo corte 80/20 usado nos experimentos.
WITH ordenado AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY data_do_pedido, order_id, seller_id) AS rn,
           COUNT(*)     OVER ()                                              AS n
    FROM gold_dataset_b_point_in_time
)
SELECT IFF(rn <= FLOOR(n * 0.8), 'treino', 'teste')                          AS conjunto,
       COUNT(*)                                                              AS registros,
       SUM(IFF(seller_avg_review_score = 0, 1, 0))                           AS nota_zero,
       ROUND(100 * AVG(IFF(seller_avg_review_score = 0, 1, 0)), 2)           AS pct_nota_zero
FROM ordenado
GROUP BY 1
ORDER BY 1 DESC;
