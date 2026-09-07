{{ config(
    materialized='table',
    schema='gold'
) }}

WITH eventos_pedidos AS (
    SELECT 
        order_id,
        seller_id,
        feature_timestamp AS data_do_pedido,
        is_delayed AS target_houve_atraso
    FROM {{ ref('gold_seller_features') }}
)

SELECT 
    e.order_id,
    e.data_do_pedido,
    e.target_houve_atraso,
    -- COALESCE garante que, se for a PRIMEIRA venda do vendedor, as features comecem zeradas em vez de null
    COALESCE(f.seller_order_volume, 0) AS seller_order_volume,
    COALESCE(f.seller_delay_count, 0) AS seller_delay_count,
    COALESCE(f.seller_avg_review_score, 0) AS seller_avg_review_score
FROM eventos_pedidos e
-- Apontamos para a tabela com a linha do tempo real do negócio
ASOF JOIN {{ ref('gold_seller_features') }} f 
    -- O segredo do Point-in-Time: buscar o estado ESTRITAMENTE ANTERIOR (>) ao pedido
    -- Isso impede que o atraso do pedido atual vaze para a própria predição dele
    MATCH_CONDITION (e.data_do_pedido > f.feature_timestamp)
    ON e.seller_id = f.seller_id