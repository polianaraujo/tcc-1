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
