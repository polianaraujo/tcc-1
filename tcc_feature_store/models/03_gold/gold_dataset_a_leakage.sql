{{ config(
    materialized='table',
    schema='gold'
) }}

WITH estado_atual_vendedor AS (
    -- Pegamos a última "foto" do vendedor ignorando o tempo
    SELECT 
        seller_id,
        seller_order_volume,
        seller_delay_count,
        seller_avg_review_score
    FROM {{ ref('gold_seller_features') }}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY feature_timestamp DESC) = 1
)

SELECT 
    o.order_id,
    o.feature_timestamp AS data_do_pedido,
    o.is_delayed AS target_houve_atraso, 
    -- Features vindas do futuro!
    v.seller_order_volume,
    v.seller_delay_count,
    v.seller_avg_review_score
FROM {{ ref('gold_seller_features') }} o
JOIN estado_atual_vendedor v ON o.seller_id = v.seller_id