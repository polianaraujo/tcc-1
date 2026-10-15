-- gold_dataset_b_point_in_time.sql
-- Cenário B (point-in-time): cada pedido recebe, em CADA linha do tempo, o estado
-- do vendedor conhecido ESTRITAMENTE ANTES do instante da compra.
-- Volume é conhecido na compra, atraso na entrega e nota na resposta da avaliação.

{{ config(materialized='table', schema='gold') }}

WITH eventos AS (
    SELECT order_id, seller_id, order_purchase_at AS data_do_pedido, is_delayed AS target_houve_atraso
    FROM {{ ref('gold_pedidos_vendedor') }}
    WHERE order_delivered_at IS NOT NULL
),

com_volume AS (
    SELECT e.*, v.seller_order_volume
    FROM eventos e
    ASOF JOIN {{ ref('gold_fs_seller_volume') }} v
        MATCH_CONDITION (e.data_do_pedido > v.feature_timestamp)
        ON e.seller_id = v.seller_id
),

com_atraso AS (
    SELECT c.*, d.seller_delay_count
    FROM com_volume c
    ASOF JOIN {{ ref('gold_fs_seller_delay') }} d
        MATCH_CONDITION (c.data_do_pedido > d.feature_timestamp)
        ON c.seller_id = d.seller_id
),

com_nota AS (
    SELECT c.*, r.seller_avg_review_score
    FROM com_atraso c
    ASOF JOIN {{ ref('gold_fs_seller_review') }} r
        MATCH_CONDITION (c.data_do_pedido > r.feature_timestamp)
        ON c.seller_id = r.seller_id
)

SELECT
    order_id,
    seller_id,
    data_do_pedido,
    target_houve_atraso,
    -- Sem estado anterior (vendedor sem histórico conhecido) => 0
    COALESCE(seller_order_volume, 0)     AS seller_order_volume,
    COALESCE(seller_delay_count, 0)      AS seller_delay_count,
    COALESCE(seller_avg_review_score, 0) AS seller_avg_review_score
FROM com_nota
