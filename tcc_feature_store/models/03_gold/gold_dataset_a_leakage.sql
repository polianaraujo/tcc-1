-- gold_dataset_a_leakage.sql
-- Cenário A (com vazamento): cada pedido recebe o ÚLTIMO estado conhecido do
-- vendedor em cada linha do tempo, ignorando quando o pedido aconteceu.

{{ config(materialized='table', schema='gold') }}

WITH eventos AS (
    -- População: pedidos entregues (únicos com rótulo observável)
    SELECT order_id, seller_id, order_purchase_at, is_delayed
    FROM {{ ref('gold_pedidos_vendedor') }}
    WHERE order_delivered_at IS NOT NULL
),

ultimo_volume AS (
    SELECT seller_id, seller_order_volume
    FROM {{ ref('gold_fs_seller_volume') }}
    -- (seller_id, feature_timestamp) é único: o desempate é determinístico
    QUALIFY ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY feature_timestamp DESC) = 1
),

ultimo_atraso AS (
    SELECT seller_id, seller_delay_count
    FROM {{ ref('gold_fs_seller_delay') }}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY feature_timestamp DESC) = 1
),

ultima_nota AS (
    SELECT seller_id, seller_avg_review_score
    FROM {{ ref('gold_fs_seller_review') }}
    QUALIFY ROW_NUMBER() OVER (PARTITION BY seller_id ORDER BY feature_timestamp DESC) = 1
)

SELECT
    e.order_id,
    e.seller_id,
    e.order_purchase_at AS data_do_pedido,
    e.is_delayed        AS target_houve_atraso,
    -- Features vindas do futuro!
    COALESCE(v.seller_order_volume, 0)     AS seller_order_volume,
    COALESCE(d.seller_delay_count, 0)      AS seller_delay_count,
    COALESCE(r.seller_avg_review_score, 0) AS seller_avg_review_score
FROM eventos e
LEFT JOIN ultimo_volume v ON e.seller_id = v.seller_id
LEFT JOIN ultimo_atraso d ON e.seller_id = d.seller_id
LEFT JOIN ultima_nota   r ON e.seller_id = r.seller_id
